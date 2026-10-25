"""NCCL-tests references on the actual 4-GPU, FSDP and TP physical pairs.

AllGather -b/-e is full output bytes; ReduceScatter is full input bytes.
Both out-of-place and in-place measurements and #wrong are preserved separately.
This is an isolated bandwidth reference, not training critical-path measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
from pathlib import Path

import owned_stage
from build_nccl_tests import (
    configure_owner,
    digest,
    inside,
    nccl_link_path,
    stage,
    version_probe,
)

GROUPS = {
    "all4": (0, 1, 2, 3),
    "fsdp02": (0, 2),
    "fsdp13": (1, 3),
    "tp01": (0, 1),
    "tp23": (2, 3),
}
DTYPES = {
    "float32": "float",
    "torch.float32": "float",
    "float": "float",
    "fp32": "float",
    "float16": "half",
    "torch.float16": "half",
    "half": "half",
    "fp16": "half",
}
CONTROLS = (
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_MAX_CTAS",
    "NCCL_MIN_CTAS",
    "NCCL_NCHANNELS",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_TESTS_MIN_BW",
)


def parse_output(text: str) -> tuple[list[dict], dict]:
    result = []
    for line in text.splitlines():
        if not re.match(r"^\s*\d+\s+\d+\s+(?:float|half)\s+", line):
            continue
        values = line.split()
        if len(values) != 13:
            raise ValueError(f"Unexpected pinned nccl-tests row layout: {line}")
        common = {
            "reported_size_bytes": int(values[0]),
            "count_elements": int(values[1]),
            "dtype": values[2],
            "redop": values[3],
            "root": int(values[4]),
        }
        for mode, offset in (("out_of_place", 5), ("in_place", 9)):
            result.append(
                {
                    **common,
                    "mode": mode,
                    "time_us": float(values[offset]),
                    "algorithm_bandwidth_GB_s": float(values[offset + 1]),
                    "bus_bandwidth_GB_s": float(values[offset + 2]),
                    "wrong_elements": int(values[offset + 3]),
                }
            )
    footer = re.search(r"#\s*Out of bounds values\s*:\s*(\d+)\s+(OK|FAILED)", text)
    check = {
        "footer_found": footer is not None,
        "out_of_bounds_values": int(footer[1]) if footer else None,
        "footer_status": footer[2] if footer else None,
        "rows": len(result),
        "all_wrong_zero": bool(result) and all(row["wrong_elements"] == 0 for row in result),
        "valid_measurements": bool(result)
        and all(
            math.isfinite(row["time_us"])
            and row["time_us"] > 0
            and math.isfinite(row["algorithm_bandwidth_GB_s"])
            and row["algorithm_bandwidth_GB_s"] >= 0
            and math.isfinite(row["bus_bandwidth_GB_s"])
            and row["bus_bandwidth_GB_s"] >= 0
            for row in result
        ),
    }
    check["status"] = (
        "PASS"
        if footer
        and footer[1] == "0"
        and footer[2] == "OK"
        and check["all_wrong_zero"]
        and check["valid_measurements"]
        else "FAIL"
    )
    return result, check


def exact_points(path: str | None) -> list[dict]:
    if not path:
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "nccl_test_message_sizes" in raw:
        raw = raw["nccl_test_message_sizes"]
    if "nccl_tests_message_sizes" in raw:
        raw = raw["nccl_tests_message_sizes"]
    points = []
    for collective, values in raw.items():
        if collective not in {"fsdp_ag", "fsdp_rs", "sp_ag", "sp_rs"}:
            continue
        for value in values:
            dtype = DTYPES.get(value["dtype"])
            if dtype is None:
                raise ValueError(f"Unsupported actual message dtype {value['dtype']}")
            size = value.get("nccl_tests_size_bytes")
            if size is None:
                size = value[
                    "output_bytes_per_rank" if collective.endswith("ag") else "input_bytes_per_rank"
                ]
            if not isinstance(size, int) or size <= 0:
                raise ValueError(f"Invalid message size: {size}")
            points.append(
                {
                    "collective": "all_gather" if collective.endswith("ag") else "reduce_scatter",
                    "role": "fsdp" if collective.startswith("fsdp") else "tp",
                    "dtype": dtype,
                    "size": size,
                    "source_collective": collective,
                    "source": value,
                }
            )
    return points


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--project-dir")
    parser.add_argument("--build-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="nccl_tests_reference")
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=list(GROUPS))
    parser.add_argument("--message-sizes")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--min-bytes", default="4M")
    parser.add_argument("--max-bytes", default="1G")
    parser.add_argument("--sp-min-bytes", default="1M")
    parser.add_argument("--sp-max-bytes", default="16M")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-sp-sweep", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--stage-prefix", default="nccl_tests_run")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def make_plan(args) -> list[dict]:
    plan = []
    for repetition in range(args.repeat):
        if not args.skip_sweep:
            for group in args.groups:
                for collective in ("all_gather", "reduce_scatter"):
                    plan.append(
                        {
                            "group": group,
                            "collective": collective,
                            "dtype": "float",
                            "min_bytes": args.min_bytes,
                            "max_bytes": args.max_bytes,
                            "repetition": repetition,
                            "kind": "parameter_size_sweep",
                        }
                    )
        if not args.skip_sp_sweep:
            for group in args.groups:
                if not group.startswith("tp"):
                    continue
                for collective in ("all_gather", "reduce_scatter"):
                    plan.append(
                        {
                            "group": group,
                            "collective": collective,
                            "dtype": "half",
                            "min_bytes": args.sp_min_bytes,
                            "max_bytes": args.sp_max_bytes,
                            "repetition": repetition,
                            "kind": "sp_activation_size_sweep",
                        }
                    )
        for point in exact_points(args.message_sizes):
            for group in args.groups:
                if not group.startswith(point["role"]):
                    continue
                # Dedupe identical source messages, preserving source metadata in plan.
                entry = {
                    "group": group,
                    "collective": point["collective"],
                    "dtype": point["dtype"],
                    "min_bytes": str(point["size"]),
                    "max_bytes": str(point["size"]),
                    "repetition": repetition,
                    "kind": "actual_training_message",
                    "requested_exact_bytes": point["size"],
                    "source_collective": point["source_collective"],
                    "source_message": point["source"],
                }
                key = (group, point["collective"], point["dtype"], point["size"], repetition)
                if not any(
                    tuple(
                        p.get(k)
                        for k in (
                            "group",
                            "collective",
                            "dtype",
                            "requested_exact_bytes",
                            "repetition",
                        )
                    )
                    == key
                    for p in plan
                ):
                    plan.append(entry)
    return plan


def main() -> int:
    args = parse_args()
    if args.repeat < 1 or args.iters < 1 or args.warmup < 1:
        raise ValueError("Repeat/iters/warmup must be positive")
    run = configure_owner(args.run_root, args.project_dir)
    output = inside(args.output_dir, run)
    if output.exists():
        raise FileExistsError(f"Refusing to replace NCCL-tests output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    plan = make_plan(args)
    manifest = {
        "status": "NOT_RUN",
        "argv": sys.argv,
        "config": vars(args),
        "plan": plan,
        "source_links": {
            "official": "https://github.com/NVIDIA/nccl-tests/tree/v2.15.0",
            "message_and_bandwidth": "https://github.com/NVIDIA/nccl-tests/blob/v2.15.0/doc/PERFORMANCE.md",
        },
        "scope": (
            "Isolated AG/RS bandwidth reference, not actual training link bytes or critical path"
        ),
        "process_layout": "MPI=0, one process/one CPU test thread controls all visible GPUs",
        "time_unit": "us per official output header; both out-of-place/in-place preserved",
        "alignment_rule": (
            "Pinned tests round each per-rank send/recv count down to 16-byte alignment"
        ),
        "completed": [],
    }
    path = output / "run_manifest.json"
    owned_stage.write_json(path, manifest)
    if args.plan_only:
        return 0
    if platform.system() != "Linux":
        raise RuntimeError("Owned CUDA test stages require Linux")
    rows = []
    try:
        build_manifest = inside(args.build_manifest, run)
        build = json.loads(build_manifest.read_text())
        if build["status"] != "PASS":
            raise RuntimeError("NCCL-tests build did not pass")
        nccl = version_probe(Path(build["config"]["nccl_home"]), build["config"]["expected_nccl"])
        if nccl["library_sha256"] != build["nccl"]["library_sha256"]:
            raise RuntimeError("NCCL library changed after the recorded build")
        if os.environ.get("LD_PRELOAD"):
            raise RuntimeError("LD_PRELOAD prevents a guaranteed NCCL library selection")
        manifest["removed_inherited_tuning"] = {
            key: os.environ.pop(key) for key in CONTROLS if key in os.environ
        }
        environment = {
            "LD_LIBRARY_PATH": build["link_environment"]["LD_LIBRARY_PATH"],
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,ENV",
        }
        manifest.update(
            build_manifest=str(build_manifest),
            build_manifest_sha256=digest(build_manifest),
            nccl=nccl,
            tests_commit=build["commit"],
            environment=environment,
        )
        for binary_name, binary in build["binaries"].items():
            executable = Path(binary["path"])
            if digest(executable) != binary["sha256"]:
                raise RuntimeError(f"Binary changed after build: {executable}")
            _, log = stage(
                args.stage_prefix + "_ldd_" + binary_name, ["ldd", executable], 60, environment
            )
            if str(nccl_link_path((log / "stdout.log").read_text())) != nccl["library"]:
                raise RuntimeError("NCCL-tests executable would load a different shared library")
        for index, entry in enumerate(plan):
            visible = GROUPS[entry["group"]]
            binary_name = entry["collective"] + "_perf"
            command = [
                build["binaries"][binary_name]["path"],
                "-b",
                entry["min_bytes"],
                "-e",
                entry["max_bytes"],
                "-f",
                "2",
                "-g",
                str(len(visible)),
                "-t",
                "1",
                "-w",
                str(args.warmup),
                "-n",
                str(args.iters),
                "-d",
                entry["dtype"],
                "-c",
                "1",
            ]
            name = f"{args.stage_prefix}_{index:03d}_{entry['group']}_{entry['collective']}"
            record, log = stage(
                name,
                command,
                args.timeout_seconds,
                {
                    **environment,
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, visible)),
                },
            )
            parsed, check = parse_output((log / "stdout.log").read_text())
            if check["status"] != "PASS":
                raise RuntimeError(f"NCCL-tests correctness not verified: {name}: {check}")
            reported_versions = sorted(
                set(
                    re.findall(
                        r"NCCL version\s+(\d+\.\d+\.\d+)",
                        (log / "stdout.log").read_text() + (log / "stderr.log").read_text(),
                    )
                )
            )
            if any(value != build["config"]["expected_nccl"] for value in reported_versions):
                raise RuntimeError(f"Unexpected runtime NCCL version banner: {reported_versions}")
            item = {
                **entry,
                "stage": name,
                "execution": record,
                "check": check,
                "physical_devices": visible,
                "command": command,
                "runtime_nccl_version_banners": reported_versions,
            }
            manifest["completed"].append(item)
            for row in parsed:
                rows.append(
                    {
                        "run_id": args.run_id,
                        "experiment_id": name,
                        "group": entry["group"],
                        "physical_devices": ",".join(map(str, visible)),
                        "world_size": len(visible),
                        "collective": entry["collective"],
                        "repetition": entry["repetition"],
                        "kind": entry["kind"],
                        "requested_exact_bytes": entry.get("requested_exact_bytes"),
                        "exact_count_shape_supported": (
                            entry["requested_exact_bytes"] % (16 * len(visible)) == 0
                            if "requested_exact_bytes" in entry
                            else None
                        ),
                        "exact_requested_size_reported": (
                            row["reported_size_bytes"] == entry["requested_exact_bytes"]
                            if "requested_exact_bytes" in entry
                            else None
                        ),
                        **row,
                        "status": "PASS",
                        "stdout_path": str(log / "stdout.log"),
                    }
                )
            owned_stage.write_json(path, manifest)
            owned_stage.write_json(output / "metrics.json", rows)
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["status"])
            writer.writeheader()
            writer.writerows(rows)
        manifest["status"] = "PASS" if plan else "NOT_RUN"
        manifest["metric_rows"] = len(rows)
    except Exception as exc:
        manifest.update(status="FAIL", exception_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        owned_stage.write_json(path, manifest)
    print(
        json.dumps(
            {"status": manifest["status"], "metrics_rows": len(rows), "output_dir": str(output)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
