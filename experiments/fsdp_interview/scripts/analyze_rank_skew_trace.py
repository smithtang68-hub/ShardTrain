"""Audit selected Q12 injections using exact, already derived NSYS evidence.

All inter-rank API/start/end comparisons use one NSYS report clock. The recorded
injection wall duration uses its own CPU clock and is never subtracted from NSYS.
No trace is discovered automatically: callers must explicitly supply an announced
trace's derived JSON. Profile step time is diagnostic, not steady-state timing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.fsdp_interview.scripts.analyze_communication import csv_write  # noqa: E402
from experiments.fsdp_interview.scripts.parse_trace import duration_ms  # noqa: E402


def group_key(row):
    return (
        row["process_group_label"],
        tuple(row["group_members"]),
        row["collective_sequence"],
    )


def injection_gpu_evidence(data, selected):
    """Match full label, same-thread contained launch API and exact kernel record.

    Returned kernels include auxiliary work. Their union is not a single GEMM's
    duration or an independently calibrated GPU delay.
    """
    rank = selected["rank"]
    label = f"interview/rank_gpu_injection/rank{rank}/U{selected['unit_id']}"
    windows = [
        row
        for row in data.get("step_windows", [])
        if row["rank"] == rank and row["step"] == selected.get("optimizer_step", selected["step"])
    ]
    scopes = [
        scope
        for scope in data.get("auxiliary_nvtx_ranges", [])
        if scope["label"] == label
        and any(
            scope["pid"] == window["pid"]
            and window["start_ns"] <= scope["start"] <= scope["end"] <= window["end_ns"]
            for window in windows
        )
    ]
    if len(scopes) != 1:
        return {"status": "UNMATCHED", "scope_candidates": len(scopes), "kernels": []}
    scope = scopes[0]
    kernels = []
    for kernel in data.get("kernels", []):
        if kernel["pid"] != scope["pid"] or label not in kernel.get("enclosing_nvtx_labels", []):
            continue
        apis = kernel.get("launch_apis", [])
        exact = [
            api
            for api in apis
            if api["globalTid"] == scope["globalTid"]
            and scope["start"] <= api["start"] <= api["end"] <= scope["end"]
        ]
        if not exact or kernel["is_nccl"]:
            continue
        # Multi-table runtime/driver rows may describe the same launch. Do not
        # choose one if any correlated API is outside this exact scope/thread.
        if len(exact) != len(apis):
            continue
        kernels.append(kernel)
    return {
        "status": "MATCHED" if kernels else "UNMATCHED",
        "scope_candidates": 1,
        "scope": scope,
        "kernel_count": len(kernels),
        "gpu_start_ns": min((row["gpu_start_ns"] for row in kernels), default=None),
        "gpu_end_ns": max((row["gpu_end_ns"] for row in kernels), default=None),
        "gpu_kernel_residency_union_ms": duration_ms(
            [(row["gpu_start_ns"], row["gpu_end_ns"]) for row in kernels]
        ),
        "stream_ids": sorted({row["stream_id"] for row in kernels}),
        "kernels": kernels,
        "scope_limit": "all exact non-NCCL injection kernels, including auxiliary kernels",
    }


def analyze(data, following_count=12):
    collectives = data.get("collectives", [])
    groups = defaultdict(list)
    for row in collectives:
        groups[group_key(row)].append(row)
    selected = [row for row in collectives if row.get("rank_injection")]
    candidates = [row for row in collectives if row.get("rank_injection_candidate")]
    target_results, propagation = [], []
    for target in selected:
        members = target["group_members"]
        peer_rows = groups[group_key(target)]
        precise = (
            len(peer_rows) == len(members)
            and {row["rank"] for row in peer_rows} == set(members)
            and all(row.get("trace_status") == "MATCHED" for row in peer_rows)
            and all(row.get("cpu_nsys_submit_ns") is not None for row in peer_rows)
        )
        own_rows = [
            row
            for row in collectives
            if row["rank"] == target["rank"]
            and row.get("channel") == target.get("channel")
            and row.get("trace_status") == "MATCHED"
            and row.get("cpu_nsys_submit_ns") is not None
            and row["cpu_nsys_submit_ns"] < target.get("cpu_nsys_submit_ns", 0)
        ]
        previous = max(own_rows, key=lambda row: row["cpu_nsys_submit_ns"], default=None)
        comparisons = []
        if precise:
            for peer in peer_rows:
                if peer["rank"] == target["rank"]:
                    continue
                comparisons.append(
                    {
                        "peer_rank": peer["rank"],
                        "target_minus_peer_api_ms": (
                            target["cpu_nsys_submit_ns"] - peer["cpu_nsys_submit_ns"]
                        )
                        / 1e6,
                        "target_minus_peer_gpu_start_ms": (
                            target["gpu_start_timestamp"] - peer["gpu_start_timestamp"]
                        )
                        / 1e6,
                        "target_minus_peer_gpu_end_ms": (
                            target["gpu_end_timestamp"] - peer["gpu_end_timestamp"]
                        )
                        / 1e6,
                        "target_kernel_residency_ms": target["gpu_kernel_residency_union_ms"],
                        "peer_kernel_residency_ms": peer["gpu_kernel_residency_union_ms"],
                        "target_api_to_gpu_start_ms": (
                            target["gpu_start_timestamp"] - target["cpu_nsys_submit_ns"]
                        )
                        / 1e6,
                        "peer_api_to_gpu_start_ms": (
                            peer["gpu_start_timestamp"] - peer["cpu_nsys_submit_ns"]
                        )
                        / 1e6,
                    }
                )
        previous_brief = None
        if previous:
            previous_brief = {
                key: previous.get(key)
                for key in (
                    "runtime_op_id",
                    "unit_id",
                    "phase",
                    "collective_sequence",
                    "cpu_nsys_submit_ns",
                    "gpu_start_timestamp",
                    "gpu_end_timestamp",
                    "stream_ids",
                )
            }
            previous_brief["remaining_gpu_ms_at_target_api"] = (
                max(0, previous["gpu_end_timestamp"] - target["cpu_nsys_submit_ns"]) / 1e6
            )
        target_results.append(
            {
                "status": "MATCHED" if precise else "UNMATCHED",
                "process_group_label": target["process_group_label"],
                "group_members": members,
                "collective_sequence": target["collective_sequence"],
                "target_rank": target["rank"],
                "unit_id": target["unit_id"],
                "runtime_microstep": target["runtime_microstep"],
                "runtime_op_id": target["runtime_op_id"],
                "injection": target["rank_injection"],
                "per_rank": peer_rows,
                "signed_comparisons": comparisons,
                "preceding_same_channel": previous_brief,
                "gpu_injection": injection_gpu_evidence(data, target)
                if target["rank_injection"]["mode"] == "gpu"
                else {"status": "NOT_APPLICABLE"},
            }
        )
        ordered = sorted(
            [
                (key, rows)
                for key, rows in groups.items()
                if key[:2] == group_key(target)[:2]
                and key[2] >= target["collective_sequence"]
                and rows[0].get("runtime_microstep") == target["runtime_microstep"]
            ],
            key=lambda pair: pair[0][2],
        )[:following_count]
        for key, rows in ordered:
            lookup = {row["rank"]: row for row in rows}
            for peer in members:
                if peer == target["rank"]:
                    continue
                own, other = lookup.get(target["rank"]), lookup.get(peer)
                valid = (
                    len(rows) == len(members)
                    and set(lookup) == set(members)
                    and own is not None
                    and other is not None
                    and all(row.get("trace_status") == "MATCHED" for row in rows)
                    and all(row.get("cpu_nsys_submit_ns") is not None for row in rows)
                )
                propagation.append(
                    {
                        "target_sequence": target["collective_sequence"],
                        "sequence": key[2],
                        "process_group_label": key[0],
                        "group_members": list(key[1]),
                        "target_rank": target["rank"],
                        "peer_rank": peer,
                        "unit_id": rows[0]["unit_id"],
                        "phase": rows[0]["phase"],
                        "kind": rows[0]["collective_type"],
                        "status": "MATCHED" if valid else "UNMATCHED",
                        **{
                            name: (own[field] - other[field]) / 1e6 if valid else None
                            for name, field in (
                                ("target_minus_peer_api_ms", "cpu_nsys_submit_ns"),
                                ("target_minus_peer_gpu_start_ms", "gpu_start_timestamp"),
                                ("target_minus_peer_gpu_end_ms", "gpu_end_timestamp"),
                            )
                        },
                    }
                )
    return {
        "source_sqlite": data["source_sqlite"],
        "selected_injections": len(selected),
        "candidate_count": len(candidates),
        "skipped_candidates": sum(
            row["rank_injection_candidate"]["status"] == "SKIPPED" for row in candidates
        ),
        "matched_targets": sum(row["status"] == "MATCHED" for row in target_results),
        "targets": target_results,
        "propagation": propagation,
        "scope_limits": [
            "Positive signed skew means target rank is later than this exact same-PG peer",
            "Injection CPU wall duration is recorded independently; no mixed-clock subtraction",
            "API-to-GPU delay includes queued work/dependencies and is not alone a CPU bottleneck",
            "NCCL kernel residency includes peer waiting; not pure transfer duration",
            "Following operation skew shows propagation, not exclusive causal attribution",
            "One profiled step does not establish steady-state throughput or delay response",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--following-count", type=int, default=12)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Fresh output directory required: {output}")
    results = []
    for path in args.trace_json:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result = analyze(data, args.following_count)
        result["source_analysis"] = str(Path(path).resolve())
        results.append(result)
    output.mkdir(parents=True)
    (output / "rank_skew_analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    propagation = []
    for result in results:
        for target in result["targets"]:
            for comparison in target["signed_comparisons"]:
                rows.append(
                    {
                        "source_sqlite": result["source_sqlite"],
                        "mode": target["injection"]["mode"],
                        "requested_cpu_delay_ms": target["injection"].get("cpu_delay_ms"),
                        "cpu_injection_wall_ms": target["injection"].get("cpu_injection_wall_ms"),
                        **{
                            key: target[key]
                            for key in (
                                "status",
                                "target_rank",
                                "unit_id",
                                "runtime_microstep",
                                "collective_sequence",
                            )
                        },
                        **comparison,
                    }
                )
        propagation.extend(
            {"source_sqlite": result["source_sqlite"], **row} for row in result["propagation"]
        )
    csv_write(output / "selected_injections.csv", rows)
    csv_write(output / "following_collectives.csv", propagation)
    print(json.dumps({"traces": len(results), "selected": len(rows), "output_dir": str(output)}))


if __name__ == "__main__":
    main()
