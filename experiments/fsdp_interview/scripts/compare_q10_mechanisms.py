"""Compare already-derived official Q10 evidence; never open raw traces.

Timing results come only from three independent non-profile repeats. Trace
comparisons describe residency, operation counts and exact source scopes; they
do not attribute the full speed difference to streams or CPU rate limiting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def csv_write(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def read_run(root: Path, provenance: dict) -> dict:
    paths = [
        root / name
        for name in (
            "q10_trace_audit.json",
            "operations.csv",
            "cpu_synchronization_apis.csv",
            "step_metrics.csv",
            "analysis_manifest.json",
        )
    ]
    for path in paths:
        provenance[str(path)] = {"bytes": path.stat().st_size, "sha256_local": sha(path)}
    audit = json.loads(paths[0].read_text(encoding="utf-8"))
    source_path = Path(audit["source_sqlite"])
    engine = (
        "fsdp1"
        if "_fsdp1" in source_path.parent.name
        else "fsdp2"
        if "_fsdp2" in source_path.parent.name
        else "UNKNOWN"
    )
    operations = csv_rows(paths[1])
    synchronization = csv_rows(paths[2])
    metrics = csv_rows(paths[3])
    return {
        "root": str(root),
        "engine": engine,
        "audit": audit,
        "operations": operations,
        "synchronization": synchronization,
        "metrics": metrics,
    }


def build_matrix(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    matrix, source_matrix = [], []
    for run in runs:
        for step in run["audit"]["steps"]:
            rank, index = step["rank"], step["step"]
            selected = [
                row
                for row in run["operations"]
                if row["rank"] == str(rank) and row["step"] == str(index)
            ]
            finite_labels = {
                label
                for row in selected
                for label in json.loads(row["source_nvtx_labels"])
                if label.startswith("aten::isfinite")
            }
            cpu = next(
                row
                for row in run["metrics"]
                if row["rank"] == str(rank) and row["step"] == str(index)
            )
            matrix.append(
                {
                    "engine": run["engine"],
                    "rank": rank,
                    "step": index,
                    "all_gather_kernels": step["kernel_counts"].get("AllGather", 0),
                    "reduce_scatter_kernels": step["kernel_counts"].get("ReduceScatter", 0),
                    "first_microstep_rs": step["accumulation_trace_check"]["first_microstep_rs"],
                    "final_microstep_rs": step["accumulation_trace_check"]["final_microstep_rs"],
                    "no_sync_count_check_pass": step["accumulation_trace_check"]["pass"],
                    "ag_streams": step["streams_by_family"].get("AllGather", []),
                    "rs_streams": step["streams_by_family"].get("ReduceScatter", []),
                    "non_nccl_streams": step["streams_by_family"].get("NON_NCCL_KERNEL", []),
                    "non_nccl_kernels": step["kernel_counts"].get("NON_NCCL_KERNEL", 0),
                    "optimizer_kernels": step["optimizer_non_nccl_kernel_count"],
                    "unique_isfinite_source_operators": len(finite_labels),
                    "cpu_isfinite_scope_count": next(
                        (
                            row["count"]
                            for row in run["audit"].get("cpu_isfinite_scopes", [])
                            if row["rank"] == rank and row["step"] == index
                        ),
                        None,
                    ),
                    "ag_rs_residency_intersection_ms": step["ag_rs_overlap_ms"],
                    "ag_non_nccl_residency_intersection_ms": step["ag_non_nccl_overlap_ms"],
                    "rs_non_nccl_residency_intersection_ms": step["rs_non_nccl_overlap_ms"],
                    "profile_cpu_scope_ms": float(cpu["window_ms"]),
                    "unknown_micro_nccl": step["unknown_micro_nccl_kernels"],
                    "pg_and_generic_target_role": "UNKNOWN",
                }
            )
        groups = defaultdict(list)
        for row in run["operations"]:
            for label in json.loads(row["source_nvtx_labels"]):
                name = label.split(",", 1)[0]
                if name.startswith(("FullyShardedDataParallel.", "FSDP::", "fsdp::")):
                    groups[(row["rank"], name, row["operation_family"], row["stream_id"])].append(
                        row
                    )
        source_matrix.extend(
            {
                "engine": run["engine"],
                "rank": key[0],
                "captured_source_scope": key[1],
                "operation_family": key[2],
                "physical_stream": key[3],
                "kernel_count": len(rows),
                "evidence": "exact API correlation; complete same-thread source NVTX containment",
                "pg_role": "UNKNOWN",
            }
            for key, rows in groups.items()
        )
    return matrix, source_matrix


def plots(output: Path, runs: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"AllGather": "#2470a2", "ReduceScatter": "#d17724", "NON_NCCL_KERNEL": "#777777"}
    figure, axes = plt.subplots(len(runs), 1, figsize=(13, 4 * len(runs)), squeeze=False)
    for axis, run in zip(axes.flat, runs, strict=True):
        selected = [row for row in run["operations"] if row["rank"] == "0" and row["step"] == "0"]
        metric = next(row for row in run["metrics"] if row["rank"] == "0" and row["step"] == "0")
        start = int(metric["start_ns"])
        streams = sorted({int(row["stream_id"]) for row in selected})
        for position, stream in enumerate(streams):
            for family, color in colors.items():
                intervals = [
                    ((int(row["gpu_start_ns"]) - start) / 1e6, float(row["duration_ms"]))
                    for row in selected
                    if int(row["stream_id"]) == stream and row["operation_family"] == family
                ]
                axis.broken_barh(intervals, (position - 0.25, 0.5), facecolors=color)
        axis.set_yticks(list(range(len(streams))), [f"CUDA stream {stream}" for stream in streams])
        axis.set_title(f"{run['engine']}, rank 0 / step 0: observed kernel residency")
        axis.set_xlabel("ms from this capture's step CPU start (independent clocks/captures)")
        axis.grid(axis="x", alpha=0.2)
    figure.legend(
        handles=[Patch(color=color, label=family) for family, color in colors.items()],
        loc="lower center",
        ncol=3,
    )
    figure.suptitle(
        "Q10 mechanism evidence from profiles: not steady-state timing or speed ranking"
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    figure.savefig(output / "q10_mechanism_temporal.png", dpi=160)
    figure.savefig(output / "q10_mechanism_temporal.svg")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dirs", required=True, type=Path, nargs=2)
    parser.add_argument("--timing-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("choose a fresh comparison output")
    provenance = {}
    runs = [read_run(root, provenance) for root in args.trace_dirs]
    if {run["engine"] for run in runs} != {"fsdp1", "fsdp2"}:
        raise ValueError("requires exactly one FSDP1 and one FSDP2 derived audit")
    timing = json.loads(args.timing_audit.read_text(encoding="utf-8"))
    provenance[str(args.timing_audit)] = {
        "sha256_local": sha(args.timing_audit),
        "bytes": args.timing_audit.stat().st_size,
    }
    if timing["status"] != "PASS" or not timing["strict_comparison_controls_equal"]:
        raise ValueError("formal non-profile timing/source controls are not verified")
    matrix, source_matrix = build_matrix(runs)
    summary = {
        "status": "PASS"
        if len(matrix) == 16
        and all(
            row["no_sync_count_check_pass"] and row["unknown_micro_nccl"] == 0 for row in matrix
        )
        else "PARTIAL",
        "formal_timing": timing["aggregate"],
        "profile_matrix": matrix,
        "cpu_synchronization": [
            {
                "engine": run["engine"],
                "rank": rank,
                "api_counts": dict(
                    Counter(
                        row["api_name"]
                        for row in run["synchronization"]
                        if row["rank"] == str(rank)
                    )
                ),
                "event_sync_cpu_ms": sum(
                    float(row["duration_ms"])
                    for row in run["synchronization"]
                    if row["rank"] == str(rank) and "EventSynchronize" in row["api_name"]
                ),
            }
            for run in runs
            for rank in range(4)
        ],
        "event_metadata": [
            {
                "engine": run["engine"],
                "event_generation_guard": run["audit"].get(
                    "event_generation_guard", "NOT_PRESENT_IN_THIS_DERIVED_VERSION"
                ),
                "edge_status_counts": dict(
                    Counter(
                        {
                            status: sum(
                                row["count"]
                                for row in run["audit"]["event_edges"]
                                if row["status"] == status
                            )
                            for status in {row["status"] for row in run["audit"]["event_edges"]}
                        }
                    )
                ),
            }
            for run in runs
        ],
        "scope_limits": [
            "PASS covers complete step/microstep count evidence plus "
            "independently checked formal timing controls.",
            "Unknown PG/target roles, missing event metadata and tensor lifetimes "
            "are not converted into PASS claims.",
            "Profile durations are not used to rank performance; "
            "only six non-profile independent timing runs are.",
            "Observed optimizer/isfinite count differences identify benchmark pathways, "
            "not a causal attribution of the full speed difference.",
            "Medium FP32 FSDP-only results cannot be speed-ranked against "
            "full Qwen FP16 TP/SP ShardTrain.",
        ],
    }
    args.output_dir.mkdir(parents=True)
    dump(args.output_dir / "q10_comparison_audit.json", summary)
    csv_write(args.output_dir / "phase_stream_matrix.csv", matrix)
    csv_write(args.output_dir / "source_scope_stream_matrix.csv", source_matrix)
    plots(args.output_dir, runs)
    dump(
        args.output_dir / "analysis_manifest.json",
        {
            "argv": sys.argv,
            "script_sha256": sha(Path(__file__)),
            "input_provenance_local": provenance,
            "raw_trace_read": False,
            "remote_delivery_hash_match": "NOT_VERIFIED_BY_THIS_ANALYSIS",
        },
    )
    print(
        json.dumps(
            {"status": summary["status"], "output": str(args.output_dir), "rank_steps": len(matrix)}
        )
    )
    return 0 if summary["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
