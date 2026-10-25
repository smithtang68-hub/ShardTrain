"""Summarize exact derived communication trace evidence, without role guesses.

Unit-adjacent comparisons use metadata unit/microstep/phase, not time-nearest
matching. Non-NCCL kernel intersections are residency and include packing/copy;
they are not automatically current-layer GEMMs or measured saved time.
"""

from __future__ import annotations

import argparse
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
from experiments.fsdp_interview.scripts.parse_trace import duration_ms, intersection  # noqa: E402


def event_intervals(rows):
    return [(row["gpu_start_timestamp"], row["gpu_end_timestamp"]) for row in rows]


def schedule_summary(data):
    rows = [row for row in data["collectives"] if row["trace_status"] == "MATCHED"]
    by_rank = defaultdict(list)
    for row in rows:
        by_rank[row["rank"]].append(row)
    phase_rows, unit_rows, stream_rows = [], [], []
    for rank, own in sorted(by_rank.items()):
        pid = next(row["pid"] for row in data["step_windows"] if row["rank"] == rank)
        compute = [
            (row["gpu_start_ns"], row["gpu_end_ns"])
            for row in data["kernels"]
            if row["pid"] == pid and not row["is_nccl"]
        ]
        for phase in ("FWD", "BWD"):
            selected = [row for row in own if row["phase"] == phase]

            def family(name, selected=selected):
                return event_intervals([row for row in selected if row["family"] == name])

            ag, rs, sp_ag = (
                family("fsdp_parameter_ag"),
                family("fsdp_gradient_rs"),
                family("sp_activation_ag"),
            )
            norm = event_intervals(
                [
                    row
                    for row in selected
                    if row["group_role"] == "norm_gradient_bucket"
                    and row["family"] == "tp_norm_allreduce"
                ]
            )
            ag_rs = intersection(ag, rs)
            phase_rows.append(
                {
                    "rank": rank,
                    "phase": phase,
                    "fsdp_ag_sp_ag_ms": duration_ms(intersection(ag, sp_ag)),
                    "fsdp_ag_rs_ms": duration_ms(ag_rs),
                    "fsdp_ag_rs_norm_ms": duration_ms(intersection(ag_rs, norm)),
                    "fsdp_ag_rs_non_nccl_ms": duration_ms(intersection(ag_rs, compute)),
                    "scope": "kernel residency intersection, not critical path or saved time",
                }
            )
        for micro in sorted(
            {row["runtime_microstep"] for row in own if row["phase"] in {"FWD", "BWD"}}
        ):
            active = [row for row in own if row["runtime_microstep"] == micro]
            for unit in sorted({row["unit_id"] for row in active if row["unit_id"] is not None}):
                # Each target AG is paired with the structurally adjacent unit;
                # unrelated overlapping kernels are excluded from this evidence.
                fwd_ag = [
                    row
                    for row in active
                    if row["unit_id"] == unit
                    and row["phase"] == "FWD"
                    and row["family"] == "fsdp_parameter_ag"
                ]
                fwd_sp = [
                    row
                    for row in active
                    if row["unit_id"] == unit - 1
                    and row["phase"] == "FWD"
                    and row["family"] == "sp_activation_ag"
                ]
                if fwd_ag and fwd_sp:
                    overlaps = intersection(event_intervals(fwd_ag), event_intervals(fwd_sp))
                    unit_rows.append(
                        {
                            "rank": rank,
                            "runtime_microstep": micro,
                            "phase": "FWD",
                            "target_ag_unit": unit,
                            "current_unit": unit - 1,
                            "previous_rs_unit": None,
                            "fsdp_ag_sp_ag_ms": duration_ms(overlaps),
                            "fsdp_ag_rs_norm_ms": None,
                            "target_ag_opids": [row["runtime_op_id"] for row in fwd_ag],
                            "intersection_intervals_ns": overlaps,
                        }
                    )
                bwd_ag = [
                    row
                    for row in active
                    if row["unit_id"] == unit
                    and row["phase"] == "BWD"
                    and row["family"] == "fsdp_parameter_ag"
                ]
                previous_rs = [
                    row
                    for row in active
                    if row["unit_id"] == unit + 2
                    and row["phase"] == "BWD"
                    and row["family"] == "fsdp_gradient_rs"
                ]
                current_norm = [
                    row
                    for row in active
                    if row["unit_id"] == unit + 1
                    and row["phase"] == "BWD"
                    and row["group_role"] == "norm_gradient_bucket"
                    and row["family"] == "tp_norm_allreduce"
                ]
                if bwd_ag and previous_rs and current_norm:
                    overlaps = intersection(
                        intersection(event_intervals(bwd_ag), event_intervals(previous_rs)),
                        event_intervals(current_norm),
                    )
                    unit_rows.append(
                        {
                            "rank": rank,
                            "runtime_microstep": micro,
                            "phase": "BWD",
                            "target_ag_unit": unit,
                            "current_unit": unit + 1,
                            "previous_rs_unit": unit + 2,
                            "fsdp_ag_sp_ag_ms": None,
                            "fsdp_ag_rs_norm_ms": duration_ms(overlaps),
                            "fsdp_ag_rs_norm_non_nccl_ms": duration_ms(
                                intersection(overlaps, compute)
                            ),
                            "target_ag_opids": [row["runtime_op_id"] for row in bwd_ag],
                            "intersection_intervals_ns": overlaps,
                            "scope": "adjacent metadata units; "
                            "non-NCCL not proven current-unit compute",
                        }
                    )
        maps = defaultdict(list)
        for row in own:
            role = row.get("group_role") or row["family"]
            maps[
                (
                    row["channel"],
                    row["logical_stream_ptr"],
                    row["actual_submission_stream_ptr"],
                    tuple(row["stream_ids"]),
                    role,
                )
            ].append(row)
        for key, values in sorted(maps.items(), key=lambda item: str(item[0])):
            channel, logical, actual, nsys, role = key
            stream_rows.append(
                {
                    "rank": rank,
                    "channel": channel,
                    "logical_stream_pointer": logical,
                    "actual_submission_stream_pointer": actual,
                    "same_pointer": logical == actual,
                    "nsys_stream_ids": nsys,
                    "role": role,
                    "collective_count": len(values),
                }
            )
    skew_groups = defaultdict(list)
    for row in data["group_skew"]:
        if row["status"] == "MATCHED":
            skew_groups[
                (row["process_group_label"], tuple(row["group_members"]), row["kind"], row["phase"])
            ].append(row)
    skew_rows = []
    for key, values in sorted(skew_groups.items(), key=lambda item: str(item[0])):
        label, members, kind, phase = key
        skew_rows.append(
            {
                "process_group_label": label,
                "members": members,
                "kind": kind,
                "phase": phase,
                "collectives": len(values),
                **{
                    f"{metric}_{stat}_ms": percentile([row[metric] for row in values], quantile)
                    for metric in (
                        "nsys_cpu_launch_submit_skew_ms",
                        "gpu_start_skew_ms",
                        "gpu_end_skew_ms",
                    )
                    for stat, quantile in (("p50", 0.5), ("p95", 0.95), ("max", 1.0))
                },
            }
        )
    gaps = defaultdict(list)
    for row in data["gpu_idle_gaps"]:
        gaps[(row["rank"], row["interpretation"])].append(row)
    gap_rows = [
        {
            "rank": rank,
            "interpretation": kind,
            "gaps_ge_100us": len(values),
            "sum_gap_ms": sum(row["gap_ms"] for row in values),
            "p50_gap_ms": percentile([row["gap_ms"] for row in values], 0.5),
            "p95_gap_ms": percentile([row["gap_ms"] for row in values], 0.95),
            "maximum_gap_ms": max(row["gap_ms"] for row in values),
        }
        for (rank, kind), values in sorted(gaps.items())
    ]
    apis = {}
    for row in data["kernels"]:
        for api in row["launch_apis"]:
            apis[(row["pid"], api["api_table"], api["api_rowid"])] = api
    api_rows = defaultdict(list)
    for (pid, _, _), api in apis.items():
        api_rows[(pid, api["api_name"])].append((api["end"] - api["start"]) / 1e6)
    api_summary = [
        {
            "pid": pid,
            "name": name,
            "samples": len(values),
            "mean_api_ms": statistics.mean(values),
            "p50_api_ms": percentile(values, 0.5),
            "p95_api_ms": percentile(values, 0.95),
            "maximum_api_ms": max(values),
        }
        for (pid, name), values in sorted(api_rows.items())
    ]
    return {
        "phase_overlaps": phase_rows,
        "adjacent_unit_overlaps": unit_rows,
        "stream_mapping": stream_rows,
        "group_skew_summary": skew_rows,
        "idle_gap_summary": gap_rows,
        "launch_api_summary": api_summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.analysis).read_text())
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose a fresh schedule-summary directory")
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "source_analysis": str(args.analysis),
        "source_sqlite": data["source_sqlite"],
        **schedule_summary(data),
        "limits": [
            "Same NSYS report clock only; profile timings not steady-state throughput.",
            "Unit-adjacent metadata does not prove actual network progress "
            "or current-unit compute ownership.",
            "Early launch with later common completion supports peer-wait evidence, not SM proof.",
        ],
    }
    for key, values in result.items():
        if isinstance(values, list) and values and isinstance(values[0], dict):
            csv_write(output / f"{key}.csv", values)
    (output / "schedule_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_sqlite": data["source_sqlite"],
                "adjacent_pairs": len(result["adjacent_unit_overlaps"]),
            }
        )
    )


if __name__ == "__main__":
    main()
