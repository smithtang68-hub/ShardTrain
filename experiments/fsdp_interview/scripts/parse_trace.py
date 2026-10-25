"""Read-only NSYS SQLite parser with exact NVTX→launch API→kernel attribution.

All GPU/NSYS API timestamps use the single report clock. Runtime monotonic_ns
timestamps remain separate. NCCL roles require communication event metadata;
unmatched kernels are never assigned a role by stream number or nearest time.
Temporal overlap is kernel residency intersection, not saved training time.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

STEP_RE = re.compile(r"interview/step/(\d+)/rank(\d+)/([^/]+)")
Q10_STEP_RE = re.compile(r"q10/step\s+phase=measured\s+step=(\d+)")
RANK_RE = re.compile(r"(?:^|[/ ])rank[=]?(\d+)(?:[/ ]|$)")
COMM_PREFIXES = ("interview/comm/", "interview/collective/")


def merge(intervals):
    values = sorted((start, end) for start, end in intervals if end > start)
    result = []
    for start, end in values:
        if result and start <= result[-1][1]:
            result[-1] = result[-1][0], max(result[-1][1], end)
        else:
            result.append((start, end))
    return result


def intersection(first, second):
    first, second = merge(first), merge(second)
    result = []
    left = right = 0
    while left < len(first) and right < len(second):
        start = max(first[left][0], second[right][0])
        end = min(first[left][1], second[right][1])
        if end > start:
            result.append((start, end))
        if first[left][1] < second[right][1]:
            left += 1
        else:
            right += 1
    return result


def duration_ms(intervals):
    return sum(end - start for start, end in merge(intervals)) / 1e6


def clip_intervals(intervals, start, end):
    return [
        (max(left, start), min(right, end))
        for left, right in intervals
        if right > start and left < end
    ]


def complement(intervals, start, end):
    result = []
    previous = start
    for left, right in merge(clip_intervals(intervals, start, end)):
        if left > previous:
            result.append((previous, left))
        previous = max(previous, right)
    if previous < end:
        result.append((previous, end))
    return result


def read_events(paths):
    result = []
    for path in paths or []:
        with Path(path).open(encoding="utf-8") as source:
            result.extend(json.loads(line) for line in source if line.strip())
    return result


def _columns(database, table):
    return {row["name"] for row in database.execute(f'PRAGMA table_info("{table}")')}


def _namespace(value):
    return value >> 24 if value is not None else None


def _dict_rows(database, query):
    return [dict(row) for row in database.execute(query)]


def load_sqlite(path):
    path = Path(path).resolve()
    database = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA query_only=ON")
    tables = {
        row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    required = {"NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_KERNEL"}
    if not required <= tables:
        database.close()
        raise ValueError(f"Missing NSYS tables: {sorted(required - tables)}")
    strings = {
        row["id"]: row["value"] for row in database.execute("SELECT id,value FROM StringIds")
    }
    scopes = []
    for row in _dict_rows(database, "SELECT rowid AS scope_id,* FROM NVTX_EVENTS"):
        label = row.get("text") or strings.get(row.get("textId"), "")
        if row.get("globalTid") is None or row.get("end") is None or row["end"] < row["start"]:
            continue
        row.update(label=label, pid=_namespace(row["globalTid"]))
        scopes.append(row)
    kernels = _dict_rows(database, "SELECT rowid AS kernel_rowid,* FROM CUPTI_ACTIVITY_KIND_KERNEL")
    for row in kernels:
        name_id = row.get("demangledName", row.get("shortName", row.get("mangledName")))
        row.update(
            kernel_name=strings.get(name_id, f"nameId:{name_id}"),
            pid=_namespace(row.get("globalPid")),
        )
        row["is_nccl"] = "nccl" in row["kernel_name"].lower()
    apis = []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables:
            continue
        for row in _dict_rows(database, f'SELECT rowid AS api_rowid,* FROM "{table}"'):
            row.update(
                api_table=table,
                api_name=strings.get(row.get("nameId"), "unknown"),
                pid=_namespace(row.get("globalTid")),
            )
            apis.append(row)
    copies = []
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            for row in _dict_rows(database, f'SELECT rowid AS activity_rowid,* FROM "{table}"'):
                row.update(activity_table=table, pid=_namespace(row.get("globalPid")))
                copies.append(row)
    synchronization = []
    if "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION" in tables:
        for row in _dict_rows(
            database, "SELECT rowid AS activity_rowid,* FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
        ):
            row.update(pid=_namespace(row.get("globalPid", row.get("globalTid"))))
            synchronization.append(row)
    database.close()
    return {
        "path": str(path),
        "tables": sorted(tables),
        "strings": strings,
        "scopes": scopes,
        "kernels": kernels,
        "apis": apis,
        "copies": copies,
        "synchronization": synchronization,
    }


def _enclosing_scopes(scopes_by_tid, api, *, communication_only=False):
    scopes = scopes_by_tid.get(api.get("globalTid"), [])
    return [
        scope
        for scope in scopes
        if scope["start"] <= api["start"]
        and api["end"] <= scope["end"]
        and (not communication_only or scope["label"].startswith(COMM_PREFIXES))
    ]


def correlate(data, events):
    scopes_by_tid = defaultdict(list)
    event_by_label = defaultdict(list)
    for scope in data["scopes"]:
        scopes_by_tid[scope["globalTid"]].append(scope)
    for event in events:
        if event.get("nvtx_label"):
            event_by_label[event["nvtx_label"]].append(event)
    apis_by_correlation = defaultdict(list)
    for api in data["apis"]:
        if api.get("correlationId") is not None:
            apis_by_correlation[(api["pid"], api["correlationId"])].append(api)
    matched = []
    for kernel in data["kernels"]:
        launches = apis_by_correlation.get((kernel["pid"], kernel.get("correlationId")), [])
        candidates = {}
        enclosing_labels = set()
        for api in launches:
            for scope in _enclosing_scopes(scopes_by_tid, api):
                enclosing_labels.add(scope["label"])
                if scope["label"].startswith(COMM_PREFIXES):
                    for event in event_by_label.get(scope["label"], []):
                        key = event["rank"], event["runtime_op_id"]
                        candidates[key] = event
        row = {
            "kernel_rowid": kernel["kernel_rowid"],
            "pid": kernel["pid"],
            "kernel_name": kernel["kernel_name"],
            "is_nccl": kernel["is_nccl"],
            "gpu_start_ns": kernel["start"],
            "gpu_end_ns": kernel["end"],
            "duration_ms": (kernel["end"] - kernel["start"]) / 1e6,
            "stream_id": kernel.get("streamId"),
            "device_id": kernel.get("deviceId"),
            "correlation_id": kernel.get("correlationId"),
            "launch_apis": [
                {
                    key: api.get(key)
                    for key in ("api_table", "api_rowid", "api_name", "start", "end", "globalTid")
                }
                for api in launches
            ],
            "enclosing_nvtx_labels": sorted(enclosing_labels),
            "status": "UNMATCHED",
            "family": None,
            "rank": None,
            "runtime_op_id": None,
            "nvtx_label": None,
        }
        if kernel["is_nccl"] and len(candidates) == 1:
            event = next(iter(candidates.values()))
            # A correlation alone is insufficient: launch API must lie inside
            # exactly this event's NVTX on the *same thread*.
            row.update(
                status="MATCHED",
                family=event["family"],
                rank=event["rank"],
                runtime_op_id=event["runtime_op_id"],
                nvtx_label=event["nvtx_label"],
                unit_id=event["unit_id"],
                phase=event["phase"],
                channel=event["channel"],
                group_role=event.get("group_role"),
            )
        elif len(candidates) > 1:
            row["failure_reason"] = "Ambiguous enclosing per-op NVTX/event metadata"
        matched.append(row)
    return matched


def collective_rows(data, events, kernels):
    by_key = defaultdict(list)
    scopes_by_label = defaultdict(list)
    for scope in data["scopes"]:
        scopes_by_label[scope["label"]].append(scope)
    captured_labels = {
        scope["label"] for scope in data["scopes"] if scope["label"].startswith(COMM_PREFIXES)
    }
    for kernel in kernels:
        if kernel["status"] == "MATCHED":
            by_key[(kernel["rank"], kernel["runtime_op_id"])].append(kernel)
    result = []
    for event in events:
        label = event.get("nvtx_label")
        if label not in captured_labels:
            continue  # warmup/outside capture is not an unmatched captured op
        values = by_key.get((event["rank"], event["runtime_op_id"]), [])
        exact_apis = []
        for value in values:
            for api in value["launch_apis"]:
                scopes = [
                    scope
                    for scope in scopes_by_label[label]
                    if scope["globalTid"] == api["globalTid"]
                    and scope["start"] <= api["start"]
                    and api["end"] <= scope["end"]
                ]
                if scopes:
                    exact_apis.append(api)
        row = {
            **event,
            "trace_status": "MATCHED" if values else "UNMATCHED",
            "gpu_start_timestamp": min(value["gpu_start_ns"] for value in values)
            if values
            else None,
            "gpu_end_timestamp": max(value["gpu_end_ns"] for value in values) if values else None,
            "gpu_clock": "NSYS report nanoseconds; all captured ranks share one report clock",
            "cpu_nsys_submit_ns": min(api["start"] for api in exact_apis) if exact_apis else None,
            "gpu_kernel_count": len(values),
            "gpu_kernel_residency_union_ms": duration_ms(
                [(v["gpu_start_ns"], v["gpu_end_ns"]) for v in values]
            ),
            "stream_ids": sorted({value["stream_id"] for value in values}),
            "correlation_ids": sorted({value["correlation_id"] for value in values}),
            "kernel_rowids": [value["kernel_rowid"] for value in values],
            "source_sqlite": data["path"],
        }
        result.append(row)
    return result


def group_skew(collectives):
    by_collective = defaultdict(list)
    for row in collectives:
        key = row["process_group_label"], tuple(row["group_members"]), row["collective_sequence"]
        by_collective[key].append(row)
    result = []
    for (label, members, sequence), values in sorted(by_collective.items()):
        matched = (
            len(values) == len(members)
            and {row["rank"] for row in values} == set(members)
            and all(
                row["trace_status"] == "MATCHED" and row["cpu_nsys_submit_ns"] is not None
                for row in values
            )
        )
        result.append(
            {
                "process_group_label": label,
                "group_members": list(members),
                "collective_sequence": sequence,
                "kind": values[0]["collective_type"],
                "unit_id": values[0]["unit_id"],
                "phase": values[0]["phase"],
                "measured_step": values[0].get("measured_step"),
                "runtime_microstep": values[0]["runtime_microstep"],
                "status": "MATCHED" if matched else "UNMATCHED",
                "nsys_cpu_launch_submit_skew_ms": (
                    (
                        max(row["cpu_nsys_submit_ns"] for row in values)
                        - min(row["cpu_nsys_submit_ns"] for row in values)
                    )
                    / 1e6
                )
                if matched
                else None,
                "gpu_start_skew_ms": (
                    (
                        max(row["gpu_start_timestamp"] for row in values)
                        - min(row["gpu_start_timestamp"] for row in values)
                    )
                    / 1e6
                )
                if matched
                else None,
                "gpu_end_skew_ms": (
                    (
                        max(row["gpu_end_timestamp"] for row in values)
                        - min(row["gpu_end_timestamp"] for row in values)
                    )
                    / 1e6
                )
                if matched
                else None,
                "per_rank": {
                    str(row["rank"]): {
                        key: row.get(key)
                        for key in (
                            "cpu_nsys_submit_ns",
                            "gpu_start_timestamp",
                            "gpu_end_timestamp",
                            "gpu_kernel_residency_union_ms",
                            "stream_ids",
                            "rank_injection",
                        )
                    }
                    for row in values
                },
                "timebase": (
                    "one NSYS report; runtime cpu_collective_submit_ns remains a different timebase"
                ),
            }
        )
    return result


def step_windows(data, rank_pids=None):
    pid_to_rank = {int(pid): int(rank) for rank, pid in (rank_pids or {}).items()}
    for scope in data["scopes"]:
        match = RANK_RE.search(scope["label"])
        if match:
            pid_to_rank[scope["pid"]] = int(match.group(1))
    result = []
    for scope in data["scopes"]:
        match = STEP_RE.search(scope["label"])
        q10 = Q10_STEP_RE.search(scope["label"])
        if match:
            step, rank, implementation = int(match.group(1)), int(match.group(2)), match.group(3)
        elif q10:
            step, rank, implementation = (
                int(q10.group(1)),
                pid_to_rank.get(scope["pid"]),
                "q10_fsdp_only",
            )
        else:
            continue
        result.append(
            {
                "scope_id": scope["scope_id"],
                "pid": scope["pid"],
                "rank": rank,
                "step": step,
                "implementation": implementation,
                "start_ns": scope["start"],
                "end_ns": scope["end"],
                "global_tid": scope["globalTid"],
                "label": scope["label"],
            }
        )
    return sorted(result, key=lambda row: (row["start_ns"], row["pid"]))


def step_analysis(data, kernels, windows):
    results, gaps = [], []
    by_pid = defaultdict(list)
    copy_by_pid = defaultdict(list)
    for kernel in kernels:
        by_pid[kernel["pid"]].append(kernel)
    for activity in data["copies"]:
        copy_by_pid[activity["pid"]].append((activity["start"], activity["end"]))
    for window in windows:
        start, end, pid = window["start_ns"], window["end_ns"], window["pid"]
        own = [
            row for row in by_pid[pid] if row["gpu_end_ns"] > start and row["gpu_start_ns"] < end
        ]

        def intervals(rows, start=start, end=end):
            return clip_intervals(
                [(row["gpu_start_ns"], row["gpu_end_ns"]) for row in rows], start, end
            )

        compute = intervals([row for row in own if not row["is_nccl"]])
        nccl = intervals([row for row in own if row["is_nccl"]])
        ag = intervals([row for row in own if row["family"] == "fsdp_parameter_ag"])
        rs = intervals([row for row in own if row["family"] == "fsdp_gradient_rs"])
        sp_ag = intervals([row for row in own if row["family"] == "sp_activation_ag"])
        sp_all = intervals(
            [
                row
                for row in own
                if row["family"] in {"sp_activation_ag", "sp_activation_rs", "tp_norm_allreduce"}
            ]
        )
        norm = intervals(
            [
                row
                for row in own
                if row["family"] == "tp_norm_allreduce"
                and row.get("group_role") == "norm_gradient_bucket"
            ]
        )
        replicated_gradient = intervals(
            [
                row
                for row in own
                if row["family"] == "tp_norm_allreduce"
                and row.get("group_role") in {"tp_replicated_gradient", "replicated_gradient"}
            ]
        )
        all_activity = intervals(own) + clip_intervals(copy_by_pid[pid], start, end)
        # concurrency must use individual kernel intervals, not their union.
        nccl_concurrency = _concurrency_histogram(
            [
                (max(row["gpu_start_ns"], start), min(row["gpu_end_ns"], end))
                for row in own
                if row["is_nccl"]
            ],
            start,
            end,
        )
        row = {
            **window,
            "window_ms": (end - start) / 1e6,
            "all_gpu_active_union_ms": duration_ms(all_activity),
            "gpu_idle_ms": duration_ms(complement(all_activity, start, end)),
            "compute_kernel_union_ms": duration_ms(compute),
            "nccl_kernel_union_ms": duration_ms(nccl),
            "compute_idle_ms": duration_ms(complement(compute, start, end)),
            "compute_nccl_temporal_overlap_ms": duration_ms(intersection(compute, nccl)),
            "nccl_without_non_nccl_kernel_ms": (
                duration_ms(nccl) - duration_ms(intersection(compute, nccl))
            ),
            "fsdp_ag_sp_ag_temporal_overlap_ms": duration_ms(intersection(ag, sp_ag)),
            "fsdp_ag_fsdp_rs_temporal_overlap_ms": duration_ms(intersection(ag, rs)),
            "fsdp_all_sp_or_norm_temporal_overlap_ms": duration_ms(intersection(ag + rs, sp_all)),
            "fsdp_ag_rs_compute_temporal_overlap_ms": duration_ms(
                intersection(intersection(ag, rs), compute)
            ),
            "fsdp_ag_rs_norm_temporal_overlap_ms": duration_ms(
                intersection(intersection(ag, rs), norm)
            ),
            "tp_norm_gradient_allreduce_union_ms": duration_ms(norm),
            "tp_replicated_gradient_allreduce_union_ms": duration_ms(replicated_gradient),
            "nccl_concurrency_ms": nccl_concurrency,
            "matched_nccl_kernels": sum(
                value["is_nccl"] and value["status"] == "MATCHED" for value in own
            ),
            "unmatched_nccl_kernels": sum(
                value["is_nccl"] and value["status"] != "MATCHED" for value in own
            ),
            "scope": (
                "all GPU activities clipped to CPU NVTX step window; concurrent residency only"
            ),
        }
        results.append(row)
        future = sorted(
            [value for value in by_pid[pid] if value["gpu_start_ns"] >= start],
            key=lambda value: value["gpu_start_ns"],
        )
        future_starts = [value["gpu_start_ns"] for value in future]
        for left, right in complement(all_activity, start, end):
            if right - left < 100_000:
                continue
            index = bisect_right(future_starts, right - 1)
            next_kernel = future[index] if index < len(future) else None
            next_apis = next_kernel["launch_apis"] if next_kernel else []
            api_start = min(api["start"] for api in next_apis) if next_apis else None
            gap = {
                "pid": pid,
                "rank": window["rank"],
                "step": window["step"],
                "start_ns": left,
                "end_ns": right,
                "gap_ms": (right - left) / 1e6,
                "next_kernel": next_kernel["kernel_name"] if next_kernel else None,
                "next_launch_api_start_ns": api_start,
                "host_before_next_launch_api_ms": min(right - left, max(0, api_start - left)) / 1e6
                if api_start is not None
                else None,
                "interpretation": "NO_ATTRIBUTION",
            }
            if api_start is not None:
                gap["interpretation"] = (
                    "HOST_BEFORE_NEXT_API"
                    if api_start > left
                    else "API_ALREADY_ENTERED_OR_PENDING_GPU_WORK"
                )
            gaps.append(gap)
    return results, sorted(gaps, key=lambda row: -row["gap_ms"])


def _concurrency_histogram(intervals, start, end):
    points = defaultdict(int)
    for left, right in intervals:
        if right > left:
            points[left] += 1
            points[right] -= 1
    active, previous = 0, start
    totals = defaultdict(int)
    for timestamp, delta in sorted(points.items()):
        totals[active] += timestamp - previous
        active += delta
        previous = timestamp
    totals[active] += end - previous
    return {str(key): value / 1e6 for key, value in sorted(totals.items())}


def stream_distribution(kernels):
    by_stream = defaultdict(list)
    for kernel in kernels:
        by_stream[(kernel["pid"], kernel["device_id"], kernel["stream_id"])].append(kernel)
    result = []
    for (pid, device, stream), rows in sorted(by_stream.items(), key=lambda item: str(item[0])):
        result.append(
            {
                "pid": pid,
                "device_id": device,
                "stream_id": stream,
                "kernel_count": len(rows),
                "kernel_residency_sum_ms": sum(row["duration_ms"] for row in rows),
                "family_counts": dict(
                    Counter(
                        row["family"] or ("unattributed_nccl" if row["is_nccl"] else "non_nccl")
                        for row in rows
                    )
                ),
                "channels": sorted({row.get("channel") for row in rows if row.get("channel")}),
                "note": "NCCL role labels only arise from exact per-op metadata correlation",
            }
        )
    return result


def pinned_timelines(data, kernels):
    """Confirm P4 transfer/GEMM overlap by exact API correlation, not event brackets."""
    by_key = defaultdict(list)
    scopes_by_tid = defaultdict(list)
    for api in data["apis"]:
        if api.get("correlationId") is not None:
            by_key[(api["pid"], api["correlationId"])].append(api)
    for scope in data["scopes"]:
        scopes_by_tid[scope["globalTid"]].append(scope)
    copy_labels = defaultdict(list)
    for activity in data["copies"]:
        labels = set()
        for api in by_key.get((activity["pid"], activity.get("correlationId")), []):
            labels.update(scope["label"] for scope in _enclosing_scopes(scopes_by_tid, api))
        for label in labels:
            copy_labels[(activity["pid"], label)].append(activity)
    result = []
    for scope in data["scopes"]:
        label = scope["label"]
        if not label.startswith("q01q03/") or "/bytes=" not in label or "/iteration=" not in label:
            continue
        copies = copy_labels.get((scope["pid"], label), [])
        gemms = [
            row
            for row in kernels
            if row["pid"] == scope["pid"]
            and label in row["enclosing_nvtx_labels"]
            and "q01q03/independent_gemm_submit" in row["enclosing_nvtx_labels"]
        ]
        transfer_intervals = [(row["start"], row["end"]) for row in copies]
        compute_intervals = [(row["gpu_start_ns"], row["gpu_end_ns"]) for row in gemms]
        result.append(
            {
                "pid": scope["pid"],
                "nvtx_label": label,
                "status": "MATCHED" if copies else "UNMATCHED",
                "cpu_nvtx_start_ns": scope["start"],
                "cpu_nvtx_end_ns": scope["end"],
                "cpu_nvtx_duration_ms": (scope["end"] - scope["start"]) / 1e6,
                "matched_copy_count": len(copies),
                "matched_compute_kernel_count": len(gemms),
                "copy_union_ms": duration_ms(transfer_intervals),
                "gemm_union_ms": duration_ms(compute_intervals),
                "transfer_gemm_temporal_overlap_ms": duration_ms(
                    intersection(transfer_intervals, compute_intervals)
                ),
                "copy_stream_ids": sorted({row.get("streamId") for row in copies}),
                "compute_stream_ids": sorted({row["stream_id"] for row in gemms}),
                "copy_intervals_ns": transfer_intervals,
                "compute_intervals_ns": compute_intervals,
                "evidence": "same PID/correlation API inside exact outer and GEMM-submit NVTX",
            }
        )
    return result


def host_allocation_apis(data):
    """Actual CUDA host allocator API calls with exact same-thread NVTX scopes."""
    names = ("HostAlloc", "HostRegister", "FreeHost", "HostUnregister")
    scopes_by_tid = defaultdict(list)
    for scope in data["scopes"]:
        scopes_by_tid[scope["globalTid"]].append(scope)
    result = []
    for api in data["apis"]:
        if not any(name in api["api_name"] for name in names):
            continue
        scopes = _enclosing_scopes(scopes_by_tid, api)
        result.append(
            {
                "api_table": api["api_table"],
                "rowid": api["api_rowid"],
                "name": api["api_name"],
                "start": api["start"],
                "end": api["end"],
                "globalTid": api["globalTid"],
                "pid": api["pid"],
                "cpu_api_duration_ms": (api["end"] - api["start"]) / 1e6,
                "enclosing_nvtx": [
                    {
                        "scope_id": scope["scope_id"],
                        "label": scope["label"],
                        "start": scope["start"],
                        "end": scope["end"],
                    }
                    for scope in sorted(scopes, key=lambda item: item["end"] - item["start"])
                ],
                "scope_status": "MATCHED" if scopes else "UNMATCHED",
                "timebase": "same NSYS CPU API/NVTX report nanoseconds",
            }
        )
    return sorted(result, key=lambda row: (row["start"], row["pid"]))


def analyze(path, event_paths=None, rank_pids=None):
    data = load_sqlite(path)
    events = read_events(event_paths)
    kernels = correlate(data, events)
    collectives = collective_rows(data, events, kernels)
    windows = step_windows(data, rank_pids)
    steps, gaps = step_analysis(data, kernels, windows)
    return {
        "source_sqlite": data["path"],
        "source_size_bytes": Path(path).stat().st_size,
        "source_tables": data["tables"],
        "timebase": "NSYS report nanoseconds; runtime CPU monotonic clock is kept separate",
        "step_windows": windows,
        "step_metrics": steps,
        "collectives": collectives,
        "group_skew": group_skew(collectives),
        "stream_distribution": stream_distribution(kernels),
        "gpu_idle_gaps": gaps,
        "kernels": kernels,
        "auxiliary_nvtx_ranges": [
            {key: row[key] for key in ("scope_id", "pid", "globalTid", "start", "end", "label")}
            for row in data["scopes"]
            if row["label"].startswith(("q01", "q03", "q10", "interview/rank_gpu_injection"))
        ],
        "copy_activities": data["copies"],
        "pinned_transfer_timelines": pinned_timelines(data, kernels),
        "host_allocation_apis": host_allocation_apis(data),
        "gpu_synchronization_activities": data["synchronization"],
        "cpu_synchronization_apis": [
            {
                key: api.get(key)
                for key in (
                    "api_table",
                    "api_rowid",
                    "api_name",
                    "start",
                    "end",
                    "globalTid",
                    "pid",
                )
            }
            for api in data["apis"]
            if any(
                name in api["api_name"]
                for name in (
                    "StreamWaitEvent",
                    "EventSynchronize",
                    "StreamSynchronize",
                    "DeviceSynchronize",
                )
            )
        ],
        "scope_limits": [
            "One or two profiled steps do not establish steady-state throughput",
            "NCCL duration includes waits/residency; no SM utilization or physical wire-byte proof",
            "Overlap intersections are not saved time or exact critical-path reconstruction",
            "Unattributed official FSDP kernels retain their enclosing source NVTX labels "
            "but no inferred PG role",
            "GPU idle gaps show submission/API timing evidence, "
            "not GC/GIL/I/O cause without sampling",
        ],
    }


def _csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target, fieldnames=sorted({key for row in rows for key in row}) or ["status"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--events", nargs="*")
    parser.add_argument("--rank-pids", help="JSON object mapping rank to PID")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rank_pids = (
        json.loads(Path(args.rank_pids).read_text(encoding="utf-8")) if args.rank_pids else None
    )
    result = analyze(args.sqlite, args.events, rank_pids)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "step_metrics",
        "collectives",
        "group_skew",
        "stream_distribution",
        "gpu_idle_gaps",
        "pinned_transfer_timelines",
        "host_allocation_apis",
    ):
        _csv(destination / f"{name}.csv", result[name])
    (destination / "trace_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_sqlite": result["source_sqlite"],
                "steps": len(result["step_windows"]),
                "matched_collectives": sum(
                    row["trace_status"] == "MATCHED" for row in result["collectives"]
                ),
                "unmatched_collectives": sum(
                    row["trace_status"] != "MATCHED" for row in result["collectives"]
                ),
                "kernel_count": len(result["kernels"]),
            }
        )
    )


if __name__ == "__main__":
    main()
