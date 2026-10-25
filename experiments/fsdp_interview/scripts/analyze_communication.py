"""Summarize completed Q5/Q6/Q12 measurements; never fabricate missing data.

Training steps carry global tokens and max-rank duration on every rank. Count
each optimizer step once, retaining per-rank peaks and independent-run variance.
Profile data is kept separate from unprofiled throughput. Source trace names are
written into all derived tables. This script does not read or modify raw SQLite.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def series_label(payload):
    if payload.get("injection"):
        injection = payload["injection"]
        if injection["mode"] == "gpu":
            return f"Q12_gpu_{injection['gemm_size']}x{injection['gemm_repeats']}"
        return f"Q12_cpu_{injection['delay_ms']}ms"
    implementation = payload.get("implementation", "unknown")
    if implementation == "native":
        return "native/matched"
    policy = payload.get("policy", "unknown")
    if policy == "C0":
        policy = "C0_guarded" if payload.get("guarded_baseline") else "C0_fast"
    recorder = payload.get("effective_event_recorder", payload.get("event_recorder", False))
    if recorder and not payload.get("profile"):
        policy = policy.removesuffix("_fast") + "_recorded"
    return f"{implementation}/{policy}"


def training_run(directory):
    directory = Path(directory)
    paths = sorted(directory.glob("benchmark.rank*.json"))
    if not paths:
        raise ValueError(f"No rank benchmark files in {directory}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    payloads.sort(key=lambda value: value["rank"])
    first = payloads[0]
    world = first["world_size"]
    if {value["rank"] for value in payloads} != set(range(world)) or len(payloads) != world:
        raise ValueError(f"Missing/duplicate rank benchmark files in {directory}")
    steps = first.get("steps", [])
    status = "PASS" if all(value["status"] == "PASS" for value in payloads) else "FAIL"
    issues = []
    for value in payloads:
        if value.get("workload_hash") != first.get("workload_hash"):
            raise ValueError(f"Rank workload divergence in {directory}")
        if len(value.get("steps", [])) != len(steps):
            raise ValueError(f"Rank step-count divergence in {directory}")
        if value.get("dataset", {}).get("tokens_sha256") != first.get("dataset", {}).get(
            "tokens_sha256"
        ):
            raise ValueError(f"Rank dataset divergence in {directory}")
        for baseline, candidate in zip(steps, value.get("steps", []), strict=True):
            for key in ("optimizer_step", "measured_step", "valid_tokens", "unique_input_tokens"):
                if baseline.get(key) != candidate.get(key):
                    raise ValueError(f"Rank global step/token divergence in {directory}: {key}")
            if not math.isclose(baseline["step_time_ms"], candidate["step_time_ms"], abs_tol=1e-5):
                raise ValueError(f"Rank maximum-duration divergence in {directory}")
    if not steps:
        status = "NOT_RUN"
    times = [row["step_time_ms"] for row in steps]
    if any(not math.isfinite(value) or value <= 0 for value in times):
        raise ValueError(f"Invalid measured time in {directory}")
    valid_tokens = [row.get("unique_training_tokens", row["valid_tokens"]) for row in steps]
    input_tokens = [row["unique_input_tokens"] for row in steps]
    finite = all(
        row.get("finite_gradients", False) for value in payloads for row in value.get("steps", [])
    )
    if not finite:
        status = "FAIL"
    if first.get("profile"):
        issues.append("PROFILED: excluded from primary throughput aggregation")
    diagnostic = bool(
        first.get("gemm_shape_diagnostic")
        or first.get("diagnostic_shapes")
        or first.get("performance_claim_allowed") is False
    )
    performance_claim_allowed = not first.get("profile") and not diagnostic
    if diagnostic:
        issues.append("DIAGNOSTIC: excluded from primary throughput aggregation")
    if first.get("event_recorder") and not first.get("profile"):
        issues.append("RECORDER_ON: do not mix with recorder-off primary timing")
    if not first.get("source_manifest_sha256"):
        issues.append("source_manifest_sha256 absent; require command/source manifest provenance")
    # This fingerprint excludes engine and schedule but includes actual input/config.
    fingerprint = (first.get("workload_hash"), first.get("dataset", {}).get("tokens_sha256"))
    per_rank = []
    for value in payloads:
        row_times = [row["rank_step_time_ms"] for row in value.get("steps", [])]
        injection = value.get("instrumentation", {}).get("injection")
        per_rank.append(
            {
                "rank": value["rank"],
                "p50_rank_step_ms": percentile(row_times, 0.5),
                "p95_rank_step_ms": percentile(row_times, 0.95),
                "peak_allocated_bytes": value.get("peak_allocated_bytes"),
                "peak_reserved_bytes": value.get("peak_reserved_bytes"),
                "parameter_shard_bytes": value.get("ownership_after", {}).get(
                    "parameter_shard_bytes"
                ),
                "gradient_shard_bytes": value.get("ownership_after", {}).get(
                    "gradient_shard_bytes"
                ),
                "injection_hit_count": injection.get("hit_count") if injection else None,
            }
        )
    injection = first.get("injection")
    target_hits = None
    expected_hits = None
    if injection:
        target = next(value for value in payloads if value["rank"] == injection["rank"])
        target_hits = target.get("instrumentation", {}).get("injection", {}).get("hit_count")
        expected_hits = len(steps)
        if target_hits != expected_hits:
            status = "FAIL"
            issues.append(f"Injection hit count {target_hits}, expected {expected_hits}")
    seconds = sum(times) / 1000
    summary = {
        "directory": str(directory),
        "experiment_id": first.get("experiment_id"),
        "series": series_label(first),
        "implementation": first.get("implementation"),
        "policy": first.get("policy"),
        "guarded_baseline": first.get("guarded_baseline", False),
        "status": status,
        "profile": bool(first.get("profile")),
        "diagnostic_shapes": diagnostic,
        "performance_claim_allowed": performance_claim_allowed,
        "world_size": world,
        "workload_hash": first.get("workload_hash"),
        "dataset_sha256": first.get("dataset", {}).get("tokens_sha256"),
        "source_manifest_sha256": first.get("source_manifest_sha256"),
        "git_commit": first.get("git_commit"),
        "fingerprint": fingerprint,
        "seq_len": first.get("seq_len"),
        "microbatch": first.get("microbatch"),
        "accumulation_steps": first.get("accumulation_steps"),
        "fsdp_group_size": first.get("fsdp_group_size"),
        "tp_group_size": first.get("tp_group_size"),
        "sp_group_size": first.get("sp_group_size"),
        "precision": first.get("precision"),
        "steps": len(steps),
        "mean_step_ms": statistics.mean(times) if times else None,
        "p50_step_ms": percentile(times, 0.5),
        "p95_step_ms": percentile(times, 0.95),
        "min_step_ms": min(times) if times else None,
        "max_step_ms": max(times) if times else None,
        "training_tokens_per_s": sum(valid_tokens) / seconds if seconds else None,
        "input_tokens_per_s": sum(input_tokens) / seconds if seconds else None,
        "unique_training_tokens_per_step": valid_tokens[0]
        if valid_tokens and len(set(valid_tokens)) == 1
        else None,
        "unique_input_tokens_per_step": input_tokens[0]
        if input_tokens and len(set(input_tokens)) == 1
        else None,
        "peak_allocated_bytes": max(value.get("peak_allocated_bytes", 0) for value in payloads),
        "peak_reserved_bytes": max(value.get("peak_reserved_bytes", 0) for value in payloads),
        "minimum_endpoint_free_bytes": min(
            (
                row["physical_free_bytes_endpoint"]
                for value in payloads
                for row in value.get("steps", [])
            ),
            default=None,
        ),
        "finite_gradient_status": "PASS" if finite else "FAIL",
        "numerical_correctness_scope": (
            "Full-run finite/update check; separate smoke oracle required for numerical equivalence"
        ),
        "mfu_percent": first.get("performance", {}).get("mfu_percent")
        if performance_claim_allowed
        else None,
        "injection": injection,
        "target_injection_hits": target_hits,
        "expected_injection_hits": expected_hits,
        "rank_metrics": per_rank,
        "issues": issues,
    }
    records = [
        {
            "directory": str(directory),
            "series": summary["series"],
            "profile": summary["profile"],
            "performance_claim_allowed": performance_claim_allowed,
            "status": status,
            "workload_hash": summary["workload_hash"],
            "dataset_sha256": summary["dataset_sha256"],
            **row,
        }
        for row in steps
    ]
    return summary, records


def timing_aggregation(summaries, steps):
    eligible = {
        row["directory"]: row
        for row in summaries
        if row["status"] == "PASS"
        and not row["profile"]
        and row.get("performance_claim_allowed", True)
    }
    groups = defaultdict(list)
    for row in steps:
        if row["directory"] in eligible:
            groups[(row["workload_hash"], row["dataset_sha256"], row["series"])].append(row)
    result = []
    for (workload, dataset, series), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        times = [row["step_time_ms"] for row in rows]
        directories = sorted({row["directory"] for row in rows})
        run_means = [eligible[directory]["mean_step_ms"] for directory in directories]
        seconds = sum(times) / 1000
        result.append(
            {
                "workload_hash": workload,
                "dataset_sha256": dataset,
                "series": series,
                "status": "PASS",
                "independent_runs": len(directories),
                "optimizer_steps": len(rows),
                "mean_step_ms": statistics.mean(times),
                "p50_step_ms": percentile(times, 0.5),
                "p95_step_ms": percentile(times, 0.95),
                "minimum_step_ms": min(times),
                "maximum_step_ms": max(times),
                "run_mean_min_ms": min(run_means),
                "run_mean_max_ms": max(run_means),
                "run_mean_stdev_ms": statistics.stdev(run_means) if len(run_means) > 1 else None,
                "training_tokens_per_s": sum(
                    row.get("unique_training_tokens", row["valid_tokens"]) for row in rows
                )
                / seconds,
                "input_tokens_per_s": sum(row["unique_input_tokens"] for row in rows) / seconds,
                "peak_allocated_GiB": max(
                    eligible[directory]["peak_allocated_bytes"] for directory in directories
                )
                / 2**30,
                "peak_reserved_GiB": max(
                    eligible[directory]["peak_reserved_bytes"] for directory in directories
                )
                / 2**30,
                "directories": directories,
                "token_counting": (
                    "global token count from one rank result per optimizer step; "
                    "no TP/FSDP rank duplication"
                ),
                "comparison_scope": (
                    "C0_guarded/C1/C2/C3 for Q5; C0_fast/native for engine comparison; "
                    "Q12 delays against CPU0"
                ),
            }
        )
    return result


def _trace_metadata(path):
    for directory in (Path(path).parent, Path(path).parent.parent):
        candidate = directory / "benchmark.rank0.json"
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def trace_summary(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = _trace_metadata(path)
    collectives = data.get("collectives", [])
    if metadata:
        label = series_label(metadata)
    elif collectives:
        label = (
            f"{collectives[0].get('implementation', 'shardtrain')}/"
            f"{collectives[0]['policy']} (guard metadata absent)"
        )
    else:
        label = "unattributed mechanism trace"
    source = data["source_sqlite"]
    steps = [
        {"source_trace": source, "source_analysis": str(path), "series": label, **row}
        for row in data.get("step_metrics", [])
    ]
    skew = [{"source_trace": source, "series": label, **row} for row in data.get("group_skew", [])]
    payload_groups = defaultdict(list)
    for row in collectives:
        if row.get("trace_status") != "MATCHED":
            continue
        key = (
            row["rank"],
            row["family"],
            row["phase"],
            row["unit_id"],
            row["input_bytes"],
            row["output_bytes"],
            row["dtype"],
            tuple(row["group_members"]),
            row.get("group_role"),
        )
        payload_groups[key].append(row["gpu_kernel_residency_union_ms"])
    durations = []
    for key, values in sorted(payload_groups.items(), key=lambda item: str(item[0])):
        rank, family, phase, unit, inputs, outputs, dtype, members, role = key
        durations.append(
            {
                "source_trace": source,
                "series": label,
                "rank": rank,
                "family": family,
                "phase": phase,
                "unit_id": unit,
                "input_bytes": inputs,
                "output_bytes": outputs,
                "dtype": dtype,
                "group_members": list(members),
                "group_role": role,
                "sample_count": len(values),
                "mean_kernel_residency_ms": statistics.mean(values),
                "p50_kernel_residency_ms": percentile(values, 0.5),
                "p95_kernel_residency_ms": percentile(values, 0.95),
                "scope": "same rank/PG/unit/phase/payload only; NCCL residency includes waiting",
            }
        )
    summary = {
        "source_analysis": str(path),
        "source_trace": source,
        "series": label,
        "profiled_rank_step_windows": len(steps),
        "captured_collectives": len(collectives),
        "matched_collectives": sum(row.get("trace_status") == "MATCHED" for row in collectives),
        "unmatched_collectives": sum(row.get("trace_status") != "MATCHED" for row in collectives),
        "limits": data.get("scope_limits", []),
        "status": "ANALYZED" if steps else "NO_STEP_WINDOWS",
    }
    return summary, steps, skew, durations


def volume_summary(directory):
    directory = Path(directory)
    result = json.loads(
        (directory / "communication_volume_summary.json").read_text(encoding="utf-8")
    )
    inventory = json.loads((directory / "collective_inventory.json").read_text(encoding="utf-8"))
    per_rank_step = defaultdict(lambda: defaultdict(float))
    count_per_rank_step = defaultdict(lambda: defaultdict(int))
    for row in inventory:
        family = row["family"]
        if family == "tp_norm_allreduce":
            family = (
                "tp_norm_gradient_allreduce"
                if row.get("group_role") == "norm_gradient_bucket"
                else "tp_replicated_gradient_allreduce"
                if row.get("group_role") in {"tp_replicated_gradient", "replicated_gradient"}
                else "tp_norm_allreduce_unclassified"
            )
        per_rank_step[(row["rank"], row["measured_step"])][family] += row["modeled_ring_send_bytes"]
        count_per_rank_step[(row["rank"], row["measured_step"])][family] += row["collective_count"]
    families = sorted({family for values in per_rank_step.values() for family in values})
    rows = []
    for family in families:
        values = [group.get(family, 0) for group in per_rank_step.values()]
        counts = [group.get(family, 0) for group in count_per_rank_step.values()]
        rows.append(
            {
                "directory": str(directory),
                "family": family,
                "rank_optimizer_step_samples": len(values),
                "mean_collectives_per_rank_step": statistics.mean(counts) if counts else None,
                "mean_modeled_ring_send_bytes_per_rank_step": statistics.mean(values)
                if values
                else None,
                "minimum_modeled_ring_send_bytes_per_rank_step": min(values) if values else None,
                "maximum_modeled_ring_send_bytes_per_rank_step": max(values) if values else None,
                "scope": (
                    "per-rank one-direction ideal Ring send model; not measured link/HBM bytes"
                ),
            }
        )
    return {"directory": str(directory), **result}, rows


def nccl_test_rows(path):
    """Parse standard 13-column nccl-tests output; retain raw source and wrong counts."""
    result = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        pieces = line.split()
        if len(pieces) != 13 or not pieces[0].isdigit() or not pieces[1].isdigit():
            continue
        try:
            values = [float(pieces[index]) for index in (5, 6, 7, 9, 10, 11)]
            wrong = [int(pieces[index]) for index in (8, 12)]
        except ValueError:
            continue
        for index, mode in enumerate(("out_of_place", "in_place")):
            result.append(
                {
                    "source_log": str(path),
                    "source_line": line_number,
                    "size_bytes": int(pieces[0]),
                    "count": int(pieces[1]),
                    "dtype": pieces[2],
                    "reduction_op": pieces[3],
                    "root": pieces[4],
                    "mode": mode,
                    "time_us": values[index * 3],
                    "algorithm_bandwidth_GB_s": values[index * 3 + 1],
                    "bus_bandwidth_GB_s": values[index * 3 + 2],
                    "wrong_elements": wrong[index],
                    "status": "PASS" if wrong[index] == 0 else "FAIL",
                    "scope": (
                        "nccl-tests derived bus BW; PG/topology from command/environment evidence"
                    ),
                }
            )
    return result


def csv_write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target, fieldnames=sorted({key for row in rows for key in row}) or ["status"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def figures(output, aggregate, trace_steps, skew, volumes, bandwidth):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    groups = defaultdict(list)
    for row in aggregate:
        groups[(row["workload_hash"], row["dataset_sha256"])].append(row)
    for index, (_, rows) in enumerate(sorted(groups.items(), key=lambda item: str(item[0]))):
        q5 = [
            row
            for row in rows
            if row["series"]
            in {"shardtrain/C0_guarded", "shardtrain/C1", "shardtrain/C2", "shardtrain/C3"}
        ]
        if q5:
            q5.sort(key=lambda row: row["series"])
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            labels = [row["series"].split("/", 1)[1] for row in q5]
            axes[0].bar(labels, [row["p50_step_ms"] for row in q5])
            axes[0].scatter(
                labels, [row["p95_step_ms"] for row in q5], color="black", marker="_", label="P95"
            )
            axes[0].set_ylabel("Max-rank optimizer step (ms)")
            axes[0].legend()
            axes[1].bar(labels, [row["training_tokens_per_s"] for row in q5])
            axes[1].set_ylabel("Unique valid next-token labels / s")
            axes[2].bar(labels, [row["peak_reserved_GiB"] for row in q5])
            axes[2].set_ylabel("Worst-rank peak reserved (GiB)")
            fig.suptitle("Q5 controlled schedule: unprofiled measurements, guarded C0")
            fig.tight_layout()
            name = f"q5_controlled_{index}.png"
            fig.savefig(output / name, dpi=170)
            plt.close(fig)
            paths.append(name)
        q12 = [row for row in rows if row["series"].startswith("Q12_cpu_")]
        if q12:
            q12.sort(key=lambda row: int(row["series"].split("_")[-1].removesuffix("ms")))
            delays = [int(row["series"].split("_")[-1].removesuffix("ms")) for row in q12]
            fig, axis = plt.subplots(figsize=(6.5, 4))
            axis.plot(delays, [row["p50_step_ms"] for row in q12], "o-", label="P50")
            axis.plot(delays, [row["p95_step_ms"] for row in q12], "s--", label="P95")
            axis.set(
                xlabel="Injected CPU delay before one collective (ms)", ylabel="Optimizer step (ms)"
            )
            axis.legend()
            axis.set_title("Q12 one injection per optimizer step (microstep 0)")
            fig.tight_layout()
            name = f"q12_delay_step_time_{index}.png"
            fig.savefig(output / name, dpi=170)
            plt.close(fig)
            paths.append(name)
    if trace_steps:
        grouped = defaultdict(list)
        for row in trace_steps:
            grouped[row["series"]].append(row)
        labels = sorted(grouped)
        fig, axis = plt.subplots(figsize=(9, 4))
        axis.bar(
            labels,
            [
                statistics.mean(row["compute_nccl_temporal_overlap_ms"] for row in grouped[label])
                for label in labels
            ],
        )
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylabel("Mean per-rank-window residency intersection (ms)")
        axis.set_title("Profile temporal compute/NCCL overlap; not saved time")
        fig.tight_layout()
        name = "profile_temporal_overlap.png"
        fig.savefig(output / name, dpi=170)
        plt.close(fig)
        paths.append(name)
    if volumes:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        names = {
            "fsdp_gradient_rs": "FSDP RS",
            "fsdp_parameter_ag": "FSDP AG",
            "sp_activation_ag": "SP AG",
            "sp_activation_rs": "SP RS",
            "tp_norm_gradient_allreduce": "TP norm AR",
            "tp_replicated_gradient_allreduce": "TP replicated grad AR",
        }
        labels = [names.get(row["family"], row["family"]) for row in volumes]
        axes[0].bar(
            labels, [row["mean_modeled_ring_send_bytes_per_rank_step"] / 2**20 for row in volumes]
        )
        axes[1].bar(labels, [row["mean_collectives_per_rank_step"] for row in volumes])
        for axis in axes:
            axis.tick_params(axis="x", rotation=25)
        axes[0].set_ylabel("Ideal Ring send / rank / step (MiB)")
        axes[1].set_ylabel("Collective calls / rank / optimizer step")
        fig.suptitle("Q6 traffic and actual call counts; small bytes do not imply cheap calls")
        fig.tight_layout()
        name = "q6_modeled_traffic.png"
        fig.savefig(output / name, dpi=170)
        plt.close(fig)
        paths.append(name)
    if bandwidth:
        by_log = defaultdict(list)
        for row in bandwidth:
            if row["status"] == "PASS" and row["mode"] == "out_of_place":
                by_log[row["source_log"]].append(row)
        fig, axis = plt.subplots(figsize=(9, 4))
        for label, rows in sorted(by_log.items()):
            rows.sort(key=lambda row: row["size_bytes"])
            axis.plot(
                [row["size_bytes"] / 2**20 for row in rows],
                [row["bus_bandwidth_GB_s"] for row in rows],
                "o-",
                label=Path(label).parent.name + "/" + Path(label).name,
            )
        axis.set(
            xlabel="Full NCCL-tests array size (MiB)",
            ylabel="Derived NCCL-tests bus bandwidth (GB/s)",
            xscale="log",
        )
        axis.legend(fontsize=7)
        axis.set_title("Q6 microbenchmark; topology/PG in raw command evidence")
        fig.tight_layout()
        name = "q6_nccl_tests_bandwidth.png"
        fig.savefig(output / name, dpi=170)
        plt.close(fig)
        paths.append(name)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-dirs", nargs="*", default=[])
    parser.add_argument("--trace-json", nargs="*", default=[])
    parser.add_argument("--volume-dirs", nargs="*", default=[])
    parser.add_argument("--nccl-logs", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    (
        summaries,
        steps,
        trace_summaries,
        trace_steps,
        skew,
        durations,
        volume_summaries,
        volumes,
        bandwidth,
    ) = [], [], [], [], [], [], [], [], []
    for directory in args.timing_dirs:
        summary, rows = training_run(directory)
        summaries.append(summary)
        steps.extend(rows)
    aggregate = timing_aggregation(summaries, steps)
    for path in args.trace_json:
        summary, windows, skews, timing = trace_summary(path)
        trace_summaries.append(summary)
        trace_steps.extend(windows)
        skew.extend(skews)
        durations.extend(timing)
    for directory in args.volume_dirs:
        summary, rows = volume_summary(directory)
        volume_summaries.append(summary)
        volumes.extend(rows)
    for path in args.nccl_logs:
        bandwidth.extend(nccl_test_rows(path))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "training_runs": summaries,
        "training_steps": steps,
        "timing_aggregate": aggregate,
        "trace_summary": trace_summaries,
        "trace_rank_steps": trace_steps,
        "collective_duration_by_payload": durations,
        "rank_collective_skew": skew,
        "modeled_volume": volumes,
        "nccl_tests_bandwidth": bandwidth,
    }
    for name, rows in tables.items():
        csv_write(output / f"{name}.csv", rows)
    report = {
        **tables,
        "volume_summaries": volume_summaries,
        "plot_paths": []
        if args.no_plots
        else figures(output, aggregate, trace_steps, skew, volumes, bandwidth),
        "missing_inputs": {
            "unprofiled_timing": not aggregate,
            "trace_analysis": not trace_summaries,
            "volume_inventory": not volume_summaries,
            "nccl_microbenchmark": not bandwidth,
        },
        "scope": "Actual supplied files only; missing input means NOT_RUN, not zero measurement",
    }
    (output / "communication_analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "training_runs": len(summaries),
                "aggregate_series": len(aggregate),
                "trace_files": len(trace_summaries),
                "plots": report["plot_paths"],
                "missing_inputs": report["missing_inputs"],
            }
        )
    )


if __name__ == "__main__":
    main()
