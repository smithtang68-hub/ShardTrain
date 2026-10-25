"""Summarize actual nccl-tests metrics and exact training-payload references.

Never mix ideal Ring byte models, derived nccl-tests bus bandwidth and measured
training NCCL residency. Reference matches require members, dtype, collective,
full array bytes and actual exact-size status; no nearest size interpolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.fsdp_interview.scripts.analyze_communication import (  # noqa: E402
    csv_write,
    percentile,
)

DTYPES = {"torch.float32": "float", "torch.float16": "half", "torch.float64": "double"}


def compare_training(metrics, trace_paths):
    references = defaultdict(list)
    for row in metrics:
        if (
            row["kind"] != "actual_training_message"
            or row["mode"] != "out_of_place"
            or row["status"] != "PASS"
        ):
            continue
        if not row.get("exact_count_shape_supported") or not row.get(
            "exact_requested_size_reported"
        ):
            continue
        members = tuple(int(value) for value in row["physical_devices"].split(","))
        references[(members, row["collective"], row["dtype"], row["reported_size_bytes"])].append(
            row
        )
    result = []
    for path in trace_paths:
        data = json.loads(Path(path).read_text())
        groups = defaultdict(list)
        for row in data["collectives"]:
            if row["trace_status"] != "MATCHED" or row["family"] not in {
                "fsdp_parameter_ag",
                "fsdp_gradient_rs",
            }:
                continue
            collective = "all_gather" if row["family"] == "fsdp_parameter_ag" else "reduce_scatter"
            size = row["output_bytes"] if collective == "all_gather" else row["input_bytes"]
            key = (
                row["rank"],
                tuple(row["group_members"]),
                collective,
                DTYPES.get(row["dtype"]),
                size,
                row["phase"],
                row.get("group_role"),
            )
            groups[key].append(row["gpu_kernel_residency_union_ms"])
        for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
            rank, members, collective, dtype, size, phase, role = key
            reference = references.get((members, collective, dtype, size), [])
            reference_time = (
                statistics.mean(row["time_us"] for row in reference) / 1000 if reference else None
            )
            result.append(
                {
                    "source_trace": data["source_sqlite"],
                    "source_analysis": str(path),
                    "rank": rank,
                    "members": members,
                    "collective": collective,
                    "dtype": dtype,
                    "full_array_size_bytes": size,
                    "launch_phase": phase,
                    "group_role": role,
                    "training_calls": len(values),
                    "training_p50_residency_ms": percentile(values, 0.5),
                    "training_p95_residency_ms": percentile(values, 0.95),
                    "isolated_exact_nccl_tests_mean_ms": reference_time,
                    "reference_experiment_ids": [row["experiment_id"] for row in reference],
                    "reference_stdout_paths": [row["stdout_path"] for row in reference],
                    "status": "EXACT_CONDITIONAL_REFERENCE" if reference else "UNMATCHED_REFERENCE",
                    "scope": "same payload/dtype/members; "
                    "algorithm/protocol/skew/compute/profile differ",
                }
            )
    return result


def plots(output, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    for kind, filename, title in (
        (
            "parameter_size_sweep",
            "q6_nccl_bus_bandwidth.png",
            "NCCL-tests parameter array sweep, FP32",
        ),
        (
            "sp_activation_size_sweep",
            "q6_sp_reference_bandwidth.png",
            "NCCL-tests activation reference sweep, dtype as recorded",
        ),
    ):
        groups = defaultdict(list)
        for row in metrics:
            if row["kind"] == kind and row["mode"] == "out_of_place" and row["status"] == "PASS":
                groups[(row["group"], row["collective"], row["dtype"], row["world_size"])].append(
                    row
                )
        if not groups:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for (group, collective, dtype, world), rows in sorted(groups.items()):
            axis = axes[0 if collective == "all_gather" else 1]
            rows.sort(key=lambda row: row["reported_size_bytes"])
            axis.plot(
                [row["reported_size_bytes"] / 2**20 for row in rows],
                [row["bus_bandwidth_GB_s"] for row in rows],
                "o-",
                markersize=3,
                label=f"{group}: {world} ranks, {dtype}",
            )
        for axis, label in zip(axes, ("AllGather", "ReduceScatter"), strict=True):
            axis.set(
                xscale="log",
                xlabel="Full reported array (MiB)",
                ylabel="NCCL-tests derived bus BW (GB/s)",
                title=label,
            )
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
        figure.suptitle(title + "; isolated benchmark, not training wire bandwidth", fontsize=10)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=170)
        plt.close(figure)
        names.append(filename)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trace-json", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metrics = json.loads(Path(args.metrics).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose fresh NCCL-reference output")
    output.mkdir(parents=True, exist_ok=True)
    exact = [
        row
        for row in metrics
        if row["kind"] == "actual_training_message" and row["mode"] == "out_of_place"
    ]
    comparisons = compare_training(metrics, args.trace_json)
    status = (
        "PASS"
        if manifest["status"] == "PASS"
        and metrics
        and all(row["status"] == "PASS" and row["wrong_elements"] == 0 for row in metrics)
        else "FAIL"
    )
    result = {
        "status": status,
        "source_metrics": str(args.metrics),
        "source_manifest": str(args.manifest),
        "metrics_sha256": hashlib.sha256(Path(args.metrics).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "completed_experiments": len({row["experiment_id"] for row in metrics}),
        "metric_rows": len(metrics),
        "wrong_elements": sum(row["wrong_elements"] for row in metrics),
        "exact_messages": exact,
        "training_reference": comparisons,
        "plot_paths": plots(output, metrics),
        "limits": [
            "Bus bandwidth is NCCL-tests algorithm model, not hardware wire counter.",
            "Training residency includes peer wait, dependencies, compute "
            "and profile perturbation.",
            "Reference requires exact member/dtype/size match; SP sweep may have different dtype.",
            "One benchmark repetition with 10 warmup/20 iterations is not three independent runs.",
            "Raw stdout remains at recorded paths; metric JSON is the actual saved parser output.",
        ],
    }
    csv_write(output / "exact_training_messages.csv", exact)
    csv_write(output / "training_reference.csv", comparisons)
    (output / "nccl_reference_analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "experiments": result["completed_experiments"],
                "rows": len(metrics),
                "comparisons": len(comparisons),
                "plots": result["plot_paths"],
            }
        )
    )


if __name__ == "__main__":
    main()
