#!/usr/bin/env python3
"""Audit only the announced Q10 allocator snapshots, without opening GPU traces.

Reuse the bounded primitive loader and source-pinned native allocator replay.
Reported peaks come from genuine per-rank step/summary JSON, not reconstructed
snapshot counters. Endpoint block maps are independently saved snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import analyze_memory as memory


def read_json(path: Path) -> tuple[Any, dict[str, Any]]:
    payload = path.read_bytes()
    return json.loads(payload), {
        "path": str(path),
        "bytes": len(payload),
        "sha256_local": hashlib.sha256(payload).hexdigest(),
    }


def history_guards(start: dict, end: dict, capacity: int) -> list[dict]:
    if len(start["device_traces"]) != len(end["device_traces"]):
        raise memory.SnapshotError("start/end trace device count differs")
    rows = []
    for device, (before, after) in enumerate(
        zip(start["device_traces"], end["device_traces"], strict=True)
    ):
        prefix = len(before) <= len(after) and before == after[: len(before)]
        reached = max(len(before), len(after)) >= capacity
        rows.append(
            {
                "device": device,
                "start_count": len(before),
                "end_count": len(after),
                "max_entries": capacity,
                "start_exact_prefix_of_end": prefix,
                "capacity_reached": reached,
                "replay_candidate": prefix and not reached,
            }
        )
    return rows


def supported_settings(before: dict, after: dict) -> bool:
    return (
        before == after
        and before.get("expandable_segments") is False
        and before.get("max_split_size") == -1
        and bool(before.get("roundup_power2_divisions"))
        and all(item == 0 for item in before["roundup_power2_divisions"].values())
        and "cudaMallocAsync" not in before.get("PYTORCH_CUDA_ALLOC_CONF", "")
    )


def reported_peak_checks(steps: list[dict], summary: dict, replay: memory.NativeReplay) -> list:
    measured = [row for row in steps if row.get("phase") == "measured"]
    if len(measured) != summary.get("measured_steps") or len(measured) != 2:
        raise memory.SnapshotError("Q10 profile must contain exactly two measured steps")
    result = []
    for summary_key, step_key, replay_key in (
        ("max_allocated_bytes", "peak_allocated_bytes", "allocated_bytes"),
        ("max_reserved_bytes", "peak_reserved_bytes", "reserved_bytes"),
    ):
        observed = memory.integer(summary[summary_key], summary_key)
        per_step = [memory.integer(row[step_key], step_key) for row in measured]
        computed = replay.peaks[replay_key]["bytes"]
        result.append(
            {
                "field": summary_key,
                "summary_reported": observed,
                "measured_step_reported": per_step,
                "maximum_measured_step": max(per_step),
                "replayed_whole_snapshot_window": computed,
                "summary_matches_steps": observed == max(per_step),
                "summary_matches_replay": observed == computed,
                "scope": "Two measured steps plus reporting/barrier inside snapshot window; "
                "per-step peak reset does not supply per-step allocator-event boundaries.",
            }
        )
    return result


def audit_rank(capture: Path, implementation: str, rank: int, output: Path) -> tuple[dict, list]:
    start, start_provenance = memory.load_snapshot(
        capture / f"memory_start.rank{rank}.pickle", 256 * 1024**2
    )
    end, end_provenance = memory.load_snapshot(
        capture / f"memory_end.rank{rank}.pickle", 256 * 1024**2
    )
    environment, environment_provenance = read_json(capture / f"environment.rank{rank}.json")
    summary, summary_provenance = read_json(capture / f"summary.rank{rank}.json")
    steps, steps_provenance = read_json(capture / f"steps.rank{rank}.json")
    if (
        environment.get("rank") != rank
        or environment.get("world_size") != 4
        or environment["config"].get("implementation") != implementation
        or not environment["config"].get("profile")
        or summary.get("implementation") != implementation
        or summary.get("rank") != rank
        or not summary.get("profile")
        or any(row.get("rank") != rank for row in steps)
    ):
        raise memory.SnapshotError("environment/summary/steps identity mismatch")
    start_audit, end_audit = memory.validate_snapshot(start), memory.validate_snapshot(end)
    capacity = memory.integer(
        environment["config"]["memory_history_max_entries"], "memory_history_max_entries", 1
    )
    guards = history_guards(start, end, capacity)
    if start_audit["devices_with_segments"] != [rank]:
        raise memory.SnapshotError("single-host rank/device metadata differs from snapshot")
    if (
        environment["torch_version"].split("+", 1)[0] != "2.9.1"
        or environment["torch_git_version"] != memory.ALLOCATOR_SOURCE["torch_git_version"]
        or not all(row["replay_candidate"] for row in guards)
        or not supported_settings(
            start_audit["allocator_settings"], end_audit["allocator_settings"]
        )
    ):
        raise memory.SnapshotError("source version, complete history, or allocator settings guard")
    events = end["device_traces"][rank][len(start["device_traces"][rank]) :]
    replay = memory.NativeReplay(start, rank)
    for index, event in enumerate(events, len(start["device_traces"][rank])):
        replay.consume(event, index)
    endpoint = replay.compare_end(end)
    peaks = reported_peak_checks(steps, summary, replay)
    passed = (
        endpoint["all_block_addresses_sizes_states_live_requests_match"]
        and endpoint["segment_addresses_sizes_streams_pools_match"]
        and endpoint["counter_match"]
        and all(row["summary_matches_steps"] and row["summary_matches_replay"] for row in peaks)
    )
    live = replay.live_rows()
    surviving = [row for row in live if row["allocation_origin"] == "START_HELD"]
    new_live = [row for row in live if row["allocation_origin"] == "WINDOW_ALLOC"]
    curves = memory.sample_points(replay)
    row = {
        "status": "VERIFIED" if passed else "UNVERIFIED",
        "implementation": implementation,
        "rank": rank,
        "capture_dir": str(capture),
        "raw_provenance": {"start": start_provenance, "end": end_provenance},
        "json_provenance": {
            "environment": environment_provenance,
            "summary": summary_provenance,
            "steps": steps_provenance,
        },
        "start": start_audit,
        "end": end_audit,
        "history": guards,
        "consumed_event_count": len(events),
        "action_counts": dict(Counter(event["action"] for event in events)),
        "endpoint": endpoint,
        "reported_peak_comparison": peaks,
        "peaks": replay.peaks,
        "max_awaiting_free": replay.max_awaiting,
        "oom_events": sum(event["action"] == "oom" for event in events),
        "start_to_end_counter_delta": {
            key: end_audit["counters"][key] - start_audit["counters"][key]
            for key in start_audit["counters"]
        },
        "endpoint_generation": {
            "start_held_surviving_blocks": len(surviving),
            "start_held_surviving_storage_bytes": sum(row["storage_bytes"] for row in surviving),
            "window_alloc_still_live_blocks": len(new_live),
            "window_alloc_still_live_storage_bytes": sum(row["storage_bytes"] for row in new_live),
        },
        "held_peak_summary": memory.held_summary(replay.peak_allocated_blocks),
        "held_endpoint_summary": memory.held_summary(live),
    }
    stem = f"{implementation}.rank{rank}"
    memory.write_csv(output / f"held.start.{stem}.csv", memory.block_rows(start, rank))
    memory.write_csv(output / f"held.end.{stem}.csv", memory.block_rows(end, rank))
    memory.write_csv(output / f"held.replayed_end.{stem}.csv", live)
    memory.write_csv(output / f"held.peak_allocated.{stem}.csv", replay.peak_allocated_blocks)
    memory.write_csv(output / f"allocator_curve.{stem}.csv", curves)
    return row, curves


def compact(row: dict) -> dict:
    if row["status"] not in ("VERIFIED", "UNVERIFIED"):
        return {key: row.get(key) for key in ("implementation", "rank", "status", "reason")}
    return {
        "implementation": row["implementation"],
        "rank": row["rank"],
        "status": row["status"],
        "history_events": row["consumed_event_count"],
        "peak_allocated_bytes": row["peaks"]["allocated_bytes"]["bytes"],
        "peak_active_bytes": row["peaks"]["active_bytes"]["bytes"],
        "peak_reserved_bytes": row["peaks"]["reserved_bytes"]["bytes"],
        "endpoint_allocated_bytes": row["end"]["counters"]["allocated_bytes"],
        "endpoint_active_bytes": row["end"]["counters"]["active_bytes"],
        "endpoint_reserved_bytes": row["end"]["counters"]["reserved_bytes"],
        "endpoint_inactive_cache_bytes": row["end"]["counters"]["inactive_cache_bytes"],
        "max_awaiting_free_bytes": row["max_awaiting_free"]["bytes"],
        "allocated_delta_bytes": row["start_to_end_counter_delta"]["allocated_bytes"],
        "reserved_delta_bytes": row["start_to_end_counter_delta"]["reserved_bytes"],
        "endpoint_awaiting_free_bytes": row["end"]["counters"]["awaiting_free_bytes"],
        "oom_events": row["oom_events"],
    }


def write_plot(output: Path, curves: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(12, 12), squeeze=False)
    for column, implementation in enumerate(("fsdp1", "fsdp2")):
        for rank in range(4):
            axis = axes[rank, column]
            points = curves.get((implementation, rank), [])
            for key, label, color, style in (
                ("reserved_bytes", "reserved incl. inactive cache", "#777777", "-"),
                ("active_bytes", "active incl. pending free", "#d1752a", "--"),
                ("allocated_bytes", "allocated", "#1766a8", "-"),
            ):
                axis.plot(
                    [row["event_index"] for row in points],
                    [row[key] / memory.GIB for row in points],
                    label=label,
                    color=color,
                    linestyle=style,
                    linewidth=0.9,
                )
            axis.set_title(f"{implementation.upper()} rank {rank}")
            axis.set_xlabel("Allocator event index (independent captures; not GPU time)")
            axis.set_ylabel("GiB")
            axis.set_ylim(bottom=0)
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.966))
    figure.suptitle("Q10: independently verified native allocator replay", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    for extension in ("png", "svg"):
        figure.savefig(output / f"q10_allocator_replay.{extension}", dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q10-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh output directory; raw and earlier derivatives are preserved")
    args.output_dir.mkdir(parents=True)
    rows, curves = [], {}
    for implementation in ("fsdp1", "fsdp2"):
        capture = args.q10_dir / f"q10_profile_r0_{implementation}"
        for rank in range(4):
            try:
                row, points = audit_rank(capture, implementation, rank, args.output_dir)
                curves[(implementation, rank)] = points
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row = {
                    "implementation": implementation,
                    "rank": rank,
                    "status": "NOT_RUN" if isinstance(exc, FileNotFoundError) else "FAILED",
                    "reason": str(exc),
                }
            rows.append(row)
    passed = all(row["status"] == "VERIFIED" for row in rows)
    audit = {
        "status": "PASS" if passed else "PARTIAL",
        "expected_snapshot_file_count": 16,
        "snapshot_file_count": sum(2 for row in rows if "raw_provenance" in row),
        "source_semantics": memory.ALLOCATOR_SOURCE,
        "analyzer_local_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "replay_module_local_sha256": hashlib.sha256(
            Path(memory.__file__).read_bytes()
        ).hexdigest(),
        "provenance_scope": "Local hashes; remote-to-local delivery audit is separate.",
        "scope_boundaries": [
            "Allocator histories are not GPU stream execution timelines.",
            "Reserved includes inactive cache; it is not active parameter storage.",
            "Native allocator snapshots exclude direct NCCL/CUDA allocations.",
            "Captured callsites do not establish tensor identity/ownership; owner remains UNKNOWN.",
            "One two-step window with stable endpoints does not prove long-term absence of leaks.",
        ],
        "ranks": rows,
    }
    memory.write_json(args.output_dir / "q10_memory_audit.json", audit)
    table = [compact(row) for row in rows]
    memory.write_csv(args.output_dir / "summary.csv", table)
    if passed:
        write_plot(args.output_dir, curves)
    lines = [
        "# Q10 allocator audit",
        "",
        f"Status: {audit['status']}; 16 snapshots / 8 rank pairs.",
        "",
        "| engine | rank | replay | events | peak allocated GiB | "
        "end allocated GiB | reserve GiB |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in table:
        if row["status"] == "VERIFIED":
            lines.append(
                f"| {row['implementation']} | {row['rank']} | VERIFIED | {row['history_events']} | "
                f"{row['peak_allocated_bytes'] / memory.GIB:.9f} | "
                f"{row['endpoint_allocated_bytes'] / memory.GIB:.9f} | "
                f"{row['endpoint_reserved_bytes'] / memory.GIB:.9f} |"
            )
        else:
            lines.append(
                f"| {row['implementation']} | {row['rank']} | {row['status']} | — | — | — | — |"
            )
    lines.extend(["", *audit["scope_boundaries"], "", audit["provenance_scope"]])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "summary": table}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
