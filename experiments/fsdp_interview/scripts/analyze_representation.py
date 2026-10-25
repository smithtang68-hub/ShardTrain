"""Read-only Q9/Q10 result audit and plots; never opens an actual GPU trace.

Only supplied result/environment/step JSON and derived profile summaries are read.
Missing inputs are NOT_RUN. Correctness-only and profiled runs are excluded from
timing comparisons even when their training summary permits a timing claim.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

IMPLEMENTATIONS = ("fsdp1_flat", "fsdp1_orig", "fsdp2")
CONTROL_KEYS = (
    "seq_len",
    "microbatch",
    "accumulation_steps",
    "layers",
    "hidden",
    "heads",
    "ffn",
    "vocab",
    "seed",
    "atol",
    "rtol",
)
POLICY_KEYS = (
    "parameter_count",
    "precision_policy",
    "parameter_reshard_policy",
    "gradient_accumulation_policy",
    "input_policy",
    "timing_policy",
    "torch_version",
    "torch_git_version",
    "nccl_version",
    "world_size",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path, provenance: dict):
    if not path.is_file():
        return None
    if path.suffix.lower() != ".json":
        raise ValueError(f"Only result JSON accepted; actual traces are forbidden: {path}")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"Result JSON is unexpectedly large; refusing potential raw trace: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "traceEvents" in data:
        raise ValueError("Chrome trace JSON is forbidden; provide an already-derived summary")
    provenance[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return data


def dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def csv_file(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            for row in rows
        )


def verdict(states: list[str]) -> str:
    if not states:
        return "NOT_RUN"
    if all(state == "NOT_RUN" for state in states):
        return "NOT_RUN"
    if "FAIL" in states:
        return "FAIL"
    if any(state != "PASS" for state in states):
        return "PARTIAL"
    return "PASS"


def audit_q9(directories: list[str], expected_world: int, provenance: dict) -> dict:
    experiments = defaultdict(list)
    missing = []
    source_dirs = []
    environments = {}
    for directory in directories:
        root = Path(directory)
        paths = sorted(root.glob("results.rank*.json"))
        if not paths:
            missing.append(
                {"directory": directory, "status": "NOT_RUN", "reason": "No rank result files"}
            )
            continue
        source_dirs.append(directory)
        for path in paths:
            rank_suffix = path.name.split("rank", 1)[1].split(".", 1)[0]
            environment = load_json(root / f"environment.rank{rank_suffix}.json", provenance)
            environments[str(path)] = environment
            data = load_json(path, provenance)
            if not isinstance(data, list):
                missing.append(
                    {
                        "path": str(path),
                        "status": "NOT_RUN",
                        "reason": "Not a distributed GPU result list",
                    }
                )
                continue
            for item in data:
                experiments[item["experiment_id"]].append({**item, "result_path": str(path)})
    matrix = []
    for implementation in IMPLEMENTATIONS:
        for case in ("R1", "R2", "R3"):
            name = f"Q9_{case}_{implementation}"
            items = experiments.get(name, [])
            ranks = [item["rank"] for item in items]
            row = {
                "case": case,
                "implementation": implementation,
                "experiment_id": name,
                "ranks": sorted(ranks),
                "expected_ranks": list(range(expected_world)),
                "status": "NOT_RUN" if not items else verdict([item["status"] for item in items]),
                "paths": [item["result_path"] for item in items],
                "training_supported_at_tested_granularity": True,
            }
            if items and (sorted(ranks) != list(range(expected_world))):
                row["status"] = "PARTIAL"
                row["reason"] = "Missing or duplicate rank result records"
            row["recorded_tolerances"] = [
                {
                    "rank": item["rank"],
                    **{
                        key: (environments[item["result_path"]] or {}).get("config", {}).get(key)
                        for key in ("atol", "rtol")
                    },
                }
                for item in items
            ]
            row["tolerances_not_relaxed"] = bool(items) and all(
                entry["atol"] is not None
                and entry["rtol"] is not None
                and 0 <= entry["atol"] <= 3e-6
                and 0 <= entry["rtol"] <= 3e-5
                for entry in row["recorded_tolerances"]
            )
            if items and any(environments[item["result_path"]] is None for item in items):
                row["status"] = "NOT_RUN"
                row["reason"] = "Required rank environment not available to audit tolerances"
            elif items and not row["tolerances_not_relaxed"]:
                row["status"] = "FAIL"
            rejected = bool(items) and all(
                item.get("outcome") == "EXPECTED_LIMITATION" for item in items
            )
            row["expected_constructor_rejection"] = rejected
            if rejected:
                row["training_supported_at_tested_granularity"] = False
                row["exception_messages"] = [item.get("exception_message") for item in items]
            checks = [check for item in items for check in item.get("checks", [])]
            row["check_count"] = len(checks)
            row["failed_checks"] = [check for check in checks if not check["pass"]]
            if row["failed_checks"]:
                row["status"] = "FAIL"
            if items and not rejected and any(not item.get("checks") for item in items):
                row["status"] = "FAIL"
            row["max_abs_numerical_error"] = max(
                (check.get("max_abs") or 0 for check in checks), default=None
            )
            if case == "R3" and implementation == "fsdp1_flat":
                row["wrapping_workaround"] = bool(items) and all(
                    "single_root_flat_optimizer_group_limitation" in item for item in items
                )
            if case == "R2" and not rejected and items:
                frozen = []
                for item in items:
                    by_name = {check["name"]: check for check in item.get("checks", [])}
                    gradient = by_name.get("gradient/embedding.weight", {})
                    boundary = by_name.get("frozen_optimizer_boundary/embedding.weight", {})
                    unchanged = by_name.get("frozen_embedding_unchanged", {})
                    update = by_name.get("updated_parameter/embedding.weight", {})
                    representation = gradient.get("observed_representation")
                    strict_zero = representation == "None" or (
                        representation == "full_observer_zero_tensor_view"
                        and gradient.get("nonzero_elements") == 0
                        and gradient.get("finite") is True
                        and gradient.get("shape") == gradient.get("expected_shape") == [31, 16]
                    )
                    standard = (
                        gradient.get("pass") is True
                        and strict_zero
                        and boundary.get("pass") is True
                        and boundary.get("gradient_is_none") is True
                        and boundary.get("excluded_from_optimizer") is True
                        and unchanged.get("pass") is True
                        and update.get("pass") is True
                        and update.get("max_abs") == 0.0
                    )
                    frozen.append(
                        {
                            "rank": item["rank"],
                            "status": "PASS" if standard else "FAIL",
                            "gradient_observer": gradient,
                            "optimizer_boundary": boundary,
                            "frozen_bitwise_unchanged": unchanged.get("pass"),
                            "frozen_update_max_abs": update.get("max_abs"),
                        }
                    )
                row["strict_frozen_standard_audit"] = frozen
                if any(item["status"] != "PASS" for item in frozen):
                    row["status"] = "FAIL"
            if items:
                row["parameter_representation"] = [
                    {
                        "rank": item["rank"],
                        "named_parameter_count": len(
                            rows := [
                                p
                                for p in item.get("parameter_metadata", [])
                                if p["phase"] == "after_wrap"
                            ]
                        ),
                        "flat_leaves": sum(p["is_flat_parameter"] for p in rows),
                        "dtensor_leaves": sum(p["is_dtensor"] for p in rows),
                        "original_identity_preserved": sum(
                            p["same_identity_as_original"] for p in rows
                        ),
                        "local_shape_examples": {p["name"]: p["local_shape"] for p in rows},
                        "exposed_shape_examples": {p["name"]: p["shape"] for p in rows},
                        "dtensor_logical_global_shapes": {
                            p["name"]: p["shape"] for p in rows if p["is_dtensor"]
                        },
                        "original_shape_examples": {p["name"]: p["original_shape"] for p in rows},
                        "flat_parameter_mapping": [
                            {
                                "flat_name": p["name"],
                                "original_names": p["flat_original_names"],
                                "original_shapes": p["flat_original_shapes"],
                            }
                            for p in rows
                            if p["is_flat_parameter"]
                        ],
                        "internal_flat_handle_count": len(
                            item.get("internal_flat_handles_after_wrap", [])
                        ),
                        "sharded_state_types": sorted(
                            set(
                                p["type"]
                                for p in item.get(
                                    "sharded_state_dict_representation_after_update", []
                                )
                            )
                        ),
                    }
                    for item in items
                ]
            matrix.append(row)
    return {
        "status": verdict([row["status"] for row in matrix]),
        "matrix": matrix,
        "missing": missing,
        "source_directories": source_dirs,
        "scope": "Small FP32 FSDP-only parameter representation and numerical correctness",
        "frozen_standard": (
            "Observer None or exact zero finite original-shape Tensor; "
            "optimizer boundary grad=None, excluded from optimizer, weight bitwise unchanged; "
            "trainable tolerances unchanged"
        ),
        "warning": (
            "EXPECTED_LIMITATION means rejection observed, not successful training; "
            "wrapping workaround is not equivalent performance"
        ),
    }


def q10_control(environment: dict) -> dict:
    return {
        "config": {key: environment.get("config", {}).get(key) for key in CONTROL_KEYS},
        "policy": {key: environment.get(key) for key in POLICY_KEYS},
    }


def audit_q10_correctness(directories: list[str], expected_world: int, provenance: dict) -> dict:
    rows = []
    for directory in directories:
        root = Path(directory)
        rank_rows = []
        implementation = None
        for rank in range(expected_world):
            env = load_json(root / f"environment.rank{rank}.json", provenance)
            data = load_json(root / f"correctness.rank{rank}.json", provenance)
            if env is None or data is None:
                rank_rows.append({"rank": rank, "status": "NOT_RUN"})
                continue
            implementation = env["config"]["implementation"]
            checks = data.get("checks", [])
            categories = {}
            for prefix in ("output/", "loss", "gradient/", "update/"):
                selected = [check for check in checks if check["name"].startswith(prefix)]
                categories[prefix] = {
                    "count": len(selected),
                    "max_abs": max((c.get("max_abs") or 0 for c in selected), default=None),
                    "pass": bool(selected) and all(c["pass"] for c in selected),
                }
            status = (
                "PASS"
                if data.get("status") == "PASS"
                and all(c["pass"] for c in checks)
                and all(c["pass"] for c in categories.values())
                else "FAIL"
            )
            rank_rows.append(
                {
                    "rank": rank,
                    "status": status,
                    "checks": len(checks),
                    "categories": categories,
                    "max_abs": max((c.get("max_abs") or 0 for c in checks), default=None),
                    "failed_checks": [c for c in checks if not c["pass"]],
                    "controls": q10_control(env),
                    "environment_path": str(root / f"environment.rank{rank}.json"),
                }
            )
        controls = [row["controls"] for row in rank_rows if "controls" in row]
        aligned = bool(controls) and all(value == controls[0] for value in controls)
        row = {
            "directory": directory,
            "implementation": implementation,
            "ranks": rank_rows,
            "status": verdict([row["status"] for row in rank_rows]),
            "controls_equal_across_ranks": aligned,
        }
        if not aligned and controls:
            row["status"] = "FAIL"
        rows.append(row)
    controls = [row["ranks"][0].get("controls") for row in rows if row["status"] == "PASS"]
    paired = {row["implementation"] for row in rows if row["status"] == "PASS"} == {
        "fsdp1",
        "fsdp2",
    }
    aligned = paired and bool(controls) and all(value == controls[0] for value in controls)
    status = verdict([row["status"] for row in rows])
    if paired and not aligned:
        status = "FAIL"
    elif rows and not paired and status == "PASS":
        status = "PARTIAL"
    return {
        "status": status,
        "runs": rows,
        "controls_equal_across_implementations": aligned,
        "physical_execution_claim": (
            "None: JSON checks prove numerical accumulation; "
            "not physical streams or per-microstep RS counts"
        ),
        "timing_claim": "None: correctness-only one-step timing excluded",
    }


def source_semantics(source: Path) -> dict:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    policies = []
    gradient_sync_calls = []
    no_sync_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "fully_shard":
                target = ast.unparse(node.args[0])
                kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
                policies.append(
                    {"target": target, "reshard_after_forward": kw.get("reshard_after_forward")}
                )
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "set_requires_gradient_sync":
                    gradient_sync_calls.append(ast.unparse(node))
                if node.func.attr == "no_sync":
                    no_sync_found = True
    expected = (
        {entry["target"]: entry["reshard_after_forward"] for entry in policies}
        == {"block": "True", "model": "False"}
        and no_sync_found
        and "model.set_requires_gradient_sync(sync)" in gradient_sync_calls
    )
    return {
        "path": str(source),
        "sha256": sha(source),
        "status": "PASS" if expected else "UNQUALIFIED_SOURCE",
        "fully_shard_policies": policies,
        "fsdp1_no_sync_call_present": no_sync_found,
        "fsdp2_gradient_sync_calls": gradient_sync_calls,
        "interpretation": (
            "Child block True/root model False; FSDP1 source retains root storage. "
            "Both public accumulation APIs suppress first-microstep reduction"
        )
        if expected
        else "UNQUALIFIED_SOURCE: inspect changed root/child/sync policy",
        "scope": (
            "Static source/API policy plus dense accumulation check; "
            "GPU dependencies require announced trace analysis"
        ),
    }


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * p / 100
    lower, upper = math.floor(location), math.ceil(location)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def audit_timing(directories: list[str], expected_world: int, provenance: dict) -> dict:
    runs = []
    for directory in directories:
        root = Path(directory)
        envs = [
            load_json(root / f"environment.rank{rank}.json", provenance)
            for rank in range(expected_world)
        ]
        summaries = load_json(root / "summary.global.json", provenance)
        if any(env is None for env in envs) or summaries is None:
            runs.append(
                {
                    "directory": directory,
                    "status": "NOT_RUN",
                    "reason": "Missing rank environments/global summary",
                }
            )
            continue
        env = envs[0]
        config = env["config"]
        if (
            config.get("profile")
            or config.get("correctness")
            or config.get("warmup", 0) < 10
            or config.get("steps", 0) < 30
        ):
            runs.append(
                {
                    "directory": directory,
                    "status": "EXCLUDED",
                    "reason": "Profile/correctness or insufficient steady-state warmup/steps",
                }
            )
            continue
        by_rank = []
        for rank in range(expected_world):
            data = load_json(root / f"steps.rank{rank}.json", provenance)
            by_rank.append([row for row in data or [] if row["phase"] == "measured"])
        if any(len(rows) != config["steps"] for rows in by_rank):
            runs.append(
                {
                    "directory": directory,
                    "status": "PARTIAL",
                    "reason": "Incomplete measured rank steps",
                }
            )
            continue
        common = q10_control(env)
        if any(q10_control(value) != common for value in envs):
            runs.append(
                {"directory": directory, "status": "FAIL", "reason": "Rank controls differ"}
            )
            continue
        times = [
            max(rows[index]["step_time_ms"] for rows in by_rank) for index in range(config["steps"])
        ]
        numerical = all(
            row["status"] == "PASS" and row["finite_gradients"] for rows in by_rank for row in rows
        )
        summary_verified = (
            summaries.get("status") == "PASS"
            and not summaries.get("profile")
            and len(summaries.get("max_rank_step_ms", [])) == len(times)
            and all(
                math.isclose(actual, recorded, rel_tol=1e-10, abs_tol=1e-8)
                for actual, recorded in zip(times, summaries["max_rank_step_ms"], strict=True)
            )
        )
        runs.append(
            {
                "directory": directory,
                "implementation": config["implementation"],
                "status": "PASS" if numerical and summary_verified else "FAIL",
                "summary_recomputed_from_all_ranks": summary_verified,
                "controls": common,
                "steps": len(times),
                "max_rank_step_ms": times,
                "mean_ms": statistics.mean(times),
                "p50_ms": statistics.median(times),
                "p95_ms": percentile(times, 95),
                "unique_tokens": by_rank[0][0]["unique_training_tokens_per_step"],
                "peak_allocated_bytes": max(
                    row["peak_allocated_bytes"] for rows in by_rank for row in rows
                ),
                "peak_reserved_bytes": max(
                    row["peak_reserved_bytes"] for rows in by_rank for row in rows
                ),
            }
        )
    grouped = defaultdict(list)
    for row in runs:
        if row["status"] == "PASS":
            grouped[row["implementation"]].append(row)
    aggregated = []
    for implementation, repeats in grouped.items():
        controls = repeats[0]["controls"]
        aligned = all(row["controls"] == controls for row in repeats)
        values = [value for row in repeats for value in row["max_rank_step_ms"]]
        unique = repeats[0]["unique_tokens"]
        aggregated.append(
            {
                "implementation": implementation,
                "status": "PASS" if aligned and len(repeats) >= 3 else "PARTIAL",
                "repeats": len(repeats),
                "steps": len(values),
                "controls": controls,
                "mean_ms": statistics.mean(values),
                "p50_ms": statistics.median(values),
                "p95_ms": percentile(values, 95),
                "repeat_p50_range_ms": [
                    min(row["p50_ms"] for row in repeats),
                    max(row["p50_ms"] for row in repeats),
                ],
                "unique_tokens_per_step": unique,
                "tokens_per_s": unique * 1000 / statistics.mean(values),
                "peak_allocated_GiB": max(row["peak_allocated_bytes"] for row in repeats) / 2**30,
                "peak_reserved_GiB": max(row["peak_reserved_bytes"] for row in repeats) / 2**30,
            }
        )
    cross_aligned = len(aggregated) == 2 and aggregated[0]["controls"] == aggregated[1]["controls"]
    return {
        "status": verdict([row["status"] for row in aggregated]) if aggregated else "NOT_RUN",
        "runs": runs,
        "aggregate": aggregated,
        "strict_comparison_controls_equal": cross_aligned,
        "scope": (
            "Max rank synchronized full optimizer step; independent repeats; "
            "Q10 medium FP32 FSDP-only"
        ),
    }


def plots(output: Path, q9: dict, timing: dict) -> dict:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {"status": "NOT_RUN", "reason": str(exc)}
    result = {"status": "PASS", "files": []}
    figure, axis = plt.subplots(figsize=(8, 3.2))
    axis.set_xlim(-0.5, 2.5)
    axis.set_ylim(2.5, -0.5)
    axis.set_xticks(range(3), ["R1 representation", "R2 mixed frozen", "R3 optimizer groups"])
    axis.set_yticks(range(3), ["FSDP1 flat", "FSDP1 orig", "FSDP2 DTensor"])
    colors = {"PASS": "#d7ebdf", "FAIL": "#f1d4d6", "NOT_RUN": "#eeeeee", "PARTIAL": "#f6e7bb"}
    for row in q9["matrix"]:
        y, x = IMPLEMENTATIONS.index(row["implementation"]), int(row["case"][1:]) - 1
        label = row["status"]
        color = colors.get(label, "#eeeeee")
        if row.get("expected_constructor_rejection"):
            label, color = "Expected rejection", "#f6e7bb"
        elif row.get("wrapping_workaround"):
            label = f"{label}\nseparate group wraps"
        axis.add_patch(plt.Rectangle((x - 0.49, y - 0.49), 0.98, 0.98, color=color))
        axis.text(x, y, label, ha="center", va="center", fontsize=9)
    axis.set_title("Q9 observed compatibility; rejection does not mean training PASS")
    figure.tight_layout()
    for extension in ("png", "svg"):
        filename = f"q9_compatibility.{extension}"
        figure.savefig(output / filename, dpi=180)
        result["files"].append(filename)
    plt.close(figure)
    if timing["strict_comparison_controls_equal"] and all(
        row["status"] == "PASS" for row in timing["aggregate"]
    ):
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        rows = sorted(timing["aggregate"], key=lambda row: row["implementation"])
        names = [row["implementation"] for row in rows]
        centers = [row["p50_ms"] for row in rows]
        axes[0].bar(names, centers, color=["#567b9e", "#5e9980"])
        axes[0].scatter(
            names, [row["p95_ms"] for row in rows], marker="_", s=350, color="#252525", label="P95"
        )
        axes[0].set_ylabel("max-rank step time (ms)")
        axes[0].legend()
        axes[1].bar(names, [row["peak_reserved_GiB"] for row in rows], color=["#567b9e", "#5e9980"])
        axes[1].set_ylabel("peak reserved GiB")
        figure.suptitle("Q10 FP32 medium Transformer, FSDP-only; 3 independent repeats")
        figure.tight_layout()
        for extension in ("png", "svg"):
            filename = f"q10_fsdp_only_timing.{extension}"
            figure.savefig(output / filename, dpi=180)
            result["files"].append(filename)
        plt.close(figure)
    else:
        result["q10_timing_plot"] = "NOT_RUN: requires matching successful 3-repeat timing inputs"
    return result


def markdown(output: Path, q9: dict, q10: dict, timing: dict, source: dict):
    lines = [
        "# Q9 / Q10 结果审计",
        "",
        "只读取数值结果、环境、step JSON；没有读取实际 NSYS/SQLite/Chrome trace。",
        "",
        "## Q9 参数表示",
        "",
        "| Case | Implementation | Status | Ranks | Numerical checks | Note |",
        "|---|---|---|---|---|---|",
    ]
    for row in q9["matrix"]:
        note = (
            "Expected constructor rejection; not trained"
            if row.get("expected_constructor_rejection")
            else (
                "Separate homogeneous FSDP group wrapping" if row.get("wrapping_workaround") else ""
            )
        )
        lines.append(
            f"| {row['case']} | {row['implementation']} | {row['status']} | {row['ranks']} "
            f"| {row['check_count']} | {note} |"
        )
    lines += [
        "",
        q9["frozen_standard"],
        "",
        "![Q9 compatibility](q9_compatibility.png)",
        "",
        "## Q10 数值正确性",
        "",
        "| Input directory | Implementation | Status | Checks/rank | Max absolute error |",
        "|---|---|---|---|---|",
    ]
    for row in q10["runs"]:
        counts = [r.get("checks") for r in row["ranks"]]
        maximum = max((r.get("max_abs") or 0 for r in row["ranks"]), default=None)
        lines.append(
            f"| {row['directory']} | {row['implementation']} | {row['status']} "
            f"| {counts} | {maximum} |"
        )
    lines += [
        "",
        f"跨实现控制变量一致：{q10['controls_equal_across_implementations']}。",
        "",
        "Root/child 与 no_sync 依据静态训练源码/API和实际 metadata 核对；"
        "完整 accumulation dense oracle 验证数学行为。",
        "这些证据不能证明实际 RS 次数、物理流依赖或 overlap；需已公告的 trace 关联分析。",
        "",
        f"训练源码 SHA256：`{source['sha256']}`。",
        "",
        "## Q10 正式计时",
        "",
        "| Implementation | Status | Repeats/steps | P50 ms | P95 ms "
        "| Tokens/s | Peak reserved GiB |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in timing["aggregate"]:
        lines.append(
            f"| {row['implementation']} | {row['status']} | {row['repeats']}/{row['steps']} "
            f"| {row['p50_ms']:.3f} | {row['p95_ms']:.3f} | {row['tokens_per_s']:.1f} "
            f"| {row['peak_reserved_GiB']:.3f} |"
        )
    if not timing["aggregate"]:
        lines.append("| — | NOT_RUN | — | — | — | — | — |")
    lines += [
        "",
        "正确性单步、profile、有不足 warmup/steps 的输入均不用于速度比较。",
        "Q10 是同条件 FP32 中型 FSDP-only；不能与全 Qwen FP16 FSDP+TP/SP 直接排名。",
        "",
    ]
    (output / "representation_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dirs", nargs="*", default=[])
    parser.add_argument("--q10-correctness-dirs", nargs="*", default=[])
    parser.add_argument("--q10-timing-dirs", nargs="*", default=[])
    parser.add_argument("--profile-analysis", nargs="*", default=[])
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument(
        "--q10-training-source",
        default=str(Path(__file__).parents[1] / "cases" / "q10_fsdp_only_benchmark.py"),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {}
    q9 = audit_q9(args.result_dirs, args.expected_world_size, provenance)
    q10 = audit_q10_correctness(args.q10_correctness_dirs, args.expected_world_size, provenance)
    timing = audit_timing(args.q10_timing_dirs, args.expected_world_size, provenance)
    source = source_semantics(Path(args.q10_training_source))
    profiles = []
    for filename in args.profile_analysis:
        path = Path(filename)
        if path.suffix != ".json" or not any(
            word in path.name for word in ("analysis", "summary", "audit")
        ):
            raise ValueError(
                "--profile-analysis accepts already-derived named JSON summaries; "
                "actual traces forbidden"
            )
        data = load_json(path, provenance)
        profiles.append(
            {
                "path": filename,
                "status": "NOT_RUN" if data is None else "DERIVED_INPUT_PRESENT",
                "scope": (
                    "Parent-provided derivative; "
                    "no actual trace opened or independently revalidated"
                ),
                "derived_summary": data,
            }
        )
    dump(output / "q9_representation_audit.json", q9)
    dump(output / "q10_correctness_audit.json", q10)
    dump(output / "q10_timing_audit.json", timing)
    dump(output / "q10_source_semantics.json", source)
    dump(output / "profile_derived_inputs.json", profiles)
    dump(output / "input_provenance.json", provenance)
    csv_file(output / "q9_compatibility_matrix.csv", q9["matrix"])
    csv_file(output / "q10_timing_matrix.csv", timing["aggregate"])
    dump(output / "plot_status.json", plots(output, q9, timing))
    markdown(output, q9, q10, timing, source)
    dump(
        output / "analysis_manifest.json",
        {
            "argv": sys.argv,
            "config": vars(args),
            "analyzer_sha256": sha(Path(__file__)),
            "q9_status": q9["status"],
            "q10_correctness_status": q10["status"],
            "q10_timing_status": timing["status"],
            "actual_trace_read": False,
            "source_training_sha256": source["sha256"],
        },
    )
    print(
        json.dumps(
            {
                "q9": q9["status"],
                "q10_correctness": q10["status"],
                "q10_timing": timing["status"],
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
