"""Read all Q9 R4 JSON and tiny weights-only CPU checkpoints; no GPU needed."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ENGINES = ("fsdp1_flat", "fsdp1_orig", "fsdp2")
SHAPES = {
    "embedding.weight": [31, 16],
    "up.weight": [24, 16],
    "up.bias": [24],
    "down.weight": [8, 24],
    "down.bias": [8],
}
STAGES = (
    ("q09_roundtrip_4_to_2_save_w4", "save", 4, 4, "roundtrip_4_to_2"),
    ("q09_roundtrip_4_to_2_load_w2_v9", "load", 2, 4, "roundtrip_4_to_2"),
    ("q09_roundtrip_2_to_4_save_w2_v9", "save", 2, 2, "roundtrip_2_to_4"),
    ("q09_roundtrip_2_to_4_load_w4_v9", "load", 4, 2, "roundtrip_2_to_4"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_row(row: dict, action: str, world: int, origin: int, rank: int) -> dict:
    checks = {
        "identity": row["rank"] == rank
        and row["world_size"] == world
        and row["origin_world_size"] == origin
        and row["action"] == action,
        "status": row["status"] == "PASS",
        "implementation": row["implementation"] in ENGINES,
        "scope": "model only" in row["scope"]
        and "no sharded optimizer-state resume" in row["scope"],
    }
    if action == "load":
        observed = {check["name"]: check for check in row["checks"]}
        expected = {f"restored_parameter/{name}" for name in SHAPES} | {
            "state_dict_keys",
            "restored_forward",
        }
        checks["complete_checks"] = set(observed) == expected
        checks["parameters_exact"] = all(
            observed.get(f"restored_parameter/{name}", {}).get("pass") is True
            and observed.get(f"restored_parameter/{name}", {}).get("max_abs") == 0.0
            for name in SHAPES
        )
        checks["keys_and_forward"] = all(
            observed.get(name, {}).get("pass") is True
            for name in ("state_dict_keys", "restored_forward")
        )
        checks["forward_finite"] = math.isfinite(
            observed.get("restored_forward", {}).get("max_abs", float("nan"))
        )
        observation = row.get("input_state_api_observation", {})
        checks["isolated_cpu_reference"] = (
            observation.get("reference_device") == "cpu"
            and observation.get("reference_is_separate_from_api_input") is True
        )
        checks["mapping_keys"] = observation.get("mapping_keys_unchanged") is True
        before, after = observation.get("before", {}), observation.get("after", {})
        checks["before_cpu_shape_dtype"] = set(before) == set(SHAPES) and all(
            before[name]["device"] == "cpu"
            and before[name]["shape"] == shape
            and before[name]["dtype"] == "torch.float32"
            for name, shape in SHAPES.items()
        )
        checks["after_cuda_shape_dtype"] = set(after) == set(SHAPES) and all(
            after[name]["device"] == f"cuda:{rank}"
            and after[name]["shape"] == shape
            and after[name]["dtype"] == "torch.float32"
            for name, shape in SHAPES.items()
        )
        checks["mapping_mutation_observed"] = set(
            observation.get("changed_device_names", [])
        ) == set(SHAPES)
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def source_guard(path: Path) -> dict:
    module = ast.parse(path.read_text(encoding="utf-8"))
    roundtrip = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "roundtrip"
    )
    errors = [
        node
        for node in ast.walk(roundtrip)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "error_row"
        and isinstance(node.args[0], ast.JoinedStr)
    ]
    strict = len(errors) == 1 and all(
        isinstance(arg, ast.Constant) and arg.value == 0.0 for arg in errors[0].args[-2:]
    )
    setters = [
        node
        for node in ast.walk(roundtrip)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set_model_state_dict"
    ]
    independent_input = len(setters) == 1 and any(
        keyword.arg == "model_state_dict"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "api_input_state"
        for keyword in setters[0].keywords
    )
    return {
        "sha256_local": sha(path),
        "strict_parameter_atol_rtol_zero": strict,
        "setter_receives_separate_api_input": independent_input,
        "status": "PASS" if strict and independent_input else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q09-root", required=True, type=Path)
    parser.add_argument(
        "--training-source",
        type=Path,
        default=Path(__file__).parents[1] / "cases/q09_param_representation.py",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("fresh output required; earlier R4 failure preserved")
    rows, provenance, environments = [], {}, []
    for stage, action, world, origin, direction in STAGES:
        for rank in range(world):
            path = args.q09_root / stage / f"results.rank{rank}.json"
            environment_path = path.with_name(f"environment.rank{rank}.json")
            results = json.loads(path.read_text(encoding="utf-8"))
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            environments.append(environment)
            for source in (path, environment_path):
                provenance[str(source)] = {
                    "sha256_local": sha(source),
                    "bytes": source.stat().st_size,
                }
            if len(results) != 3 or {row["implementation"] for row in results} != set(ENGINES):
                raise ValueError("missing/duplicate R4 engine result")
            for row in results:
                audit = verify_row(row, action, world, origin, rank)
                rows.append(
                    {
                        "stage": stage,
                        "direction": direction,
                        "action": action,
                        "rank": rank,
                        "world": world,
                        "origin_world": origin,
                        "engine": row["implementation"],
                        "status": audit["status"],
                        "audit_checks": audit["checks"],
                        "checkpoint_bytes": row["checkpoint_bytes"],
                        "max_parameter_error": max(
                            (
                                check["max_abs"]
                                for check in row["checks"]
                                if check["name"].startswith("restored_parameter/")
                            ),
                            default=None,
                        ),
                        "forward_error": next(
                            (
                                check["max_abs"]
                                for check in row["checks"]
                                if check["name"] == "restored_forward"
                            ),
                            None,
                        ),
                        "state_api_timings": row["state_api_timings"],
                        "entire_action_wall_ms": row["entire_action_wall_ms"],
                        "result_path": str(path),
                    }
                )
    checkpoint_rows = []
    for direction, origin in (("roundtrip_4_to_2", 4), ("roundtrip_2_to_4", 2)):
        for engine in ENGINES:
            path = args.q09_root / direction / "checkpoint" / f"{engine}.pt"
            if path.stat().st_size > 1024**2:
                raise ValueError("R4 tiny model checkpoint exceeds bounded read size")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            tensors = checkpoint["model"]
            checks = {
                "origin_world": checkpoint["world_size"] == origin,
                "seed": checkpoint["seed"] == 1234,
                "model_only": checkpoint["optimizer_state_saved"] is False,
                "keys": set(tensors) == set(SHAPES),
                "cpu_fp32_shapes": all(
                    tensors[name].device.type == "cpu"
                    and tensors[name].dtype == torch.float32
                    and list(tensors[name].shape) == shape
                    for name, shape in SHAPES.items()
                ),
                "finite": all(bool(torch.isfinite(tensor).all()) for tensor in tensors.values()),
                "numel": sum(tensor.numel() for tensor in tensors.values()) == 1104,
            }
            provenance[str(path)] = {"sha256_local": sha(path), "bytes": path.stat().st_size}
            checkpoint_rows.append(
                {
                    "direction": direction,
                    "engine": engine,
                    "path": str(path),
                    "checks": checks,
                    "status": "PASS" if all(checks.values()) else "FAIL",
                    **provenance[str(path)],
                }
            )
    source = source_guard(args.training_source)
    versions_equal = (
        len(
            {
                (env["torch_version"], env["torch_git_version"], tuple(env["nccl_version"]))
                for env in environments
            }
        )
        == 1
    )
    summary = {
        "status": "PASS"
        if len(rows) == 36
        and all(row["status"] == "PASS" for row in rows + checkpoint_rows)
        and source["status"] == "PASS"
        and versions_equal
        else "FAIL",
        "rank_actions": len(rows),
        "load_rank_actions": sum(row["action"] == "load" for row in rows),
        "parameter_restore_checks": sum(row["action"] == "load" for row in rows) * len(SHAPES),
        "rows": rows,
        "checkpoints": checkpoint_rows,
        "training_source": source,
        "versions_equal": versions_equal,
        "scope": "Independent 4-to-2 and 2-to-4 FULL model checkpoints only; "
        "not optimizer resume or production DCP sharded storage",
        "timing_limit": "Single ordered run; first-engine cold initialization/communication "
        "may be included. No speed ranking. Save IO timer includes "
        "GPU-to-CPU conversion and serialization.",
    }
    args.output_dir.mkdir(parents=True)

    def dump(path: Path, value) -> None:
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    dump(args.output_dir / "q09_roundtrip_audit.json", summary)
    with (args.output_dir / "rank_action_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value) if isinstance(value, dict) else value
                for key, value in row.items()
            }
            for row in rows
        )
    dump(
        args.output_dir / "analysis_manifest.json",
        {
            "argv": sys.argv,
            "script_sha256": sha(Path(__file__)),
            "input_provenance_local": provenance,
            "remote_delivery_hash_match": "NOT_VERIFIED_BY_THIS_ANALYSIS",
            "gpu_executed_by_this_audit": False,
        },
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rank_actions": len(rows),
                "parameter_restore_checks": summary["parameter_restore_checks"],
                "output": str(args.output_dir),
            }
        )
    )
    return 0 if summary["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
