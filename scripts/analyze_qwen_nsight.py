"""Read-only Nsight SQLite audit of observed Qwen communication/consumer GPU boundaries.

No Torch, CUDA execution, SSH, or pickle loading. Timestamps are Nsight nanoseconds.
NVTX marks identify CPU submission windows; completion comes only from GPU activities.
"""

import argparse
import bisect
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

COMM_KINDS = {"fsdp_ag", "fsdp_rs", "sp_ag", "sp_rs", "tp_norm_allreduce"}
STEP_COMPLETE = re.compile(r"shardtrain/step_complete/(-?\d+)/rank(\d+)$")
LAYER_RANGE = re.compile(r"shardtrain/qwen/layer(\d+)$")
PID_MASK = (1 << 24) - 1
INVALID_EVENT = (1 << 32) - 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in merged(intervals))


def intersection(left, right):
    left, right = merged(left), merged(right)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append([start, end])
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def marker_text(row):
    ticket = row.get("op_id")
    if ticket is None:
        ticket = row.get("consumer_ticket")
    if ticket is None:
        ticket = row.get("producer_ticket")
    return (
        f"shardtrain/{row['event_type']}/U{row.get('unit_id')}/{row.get('phase')}"
        f"/step{row.get('step_id')}/op{ticket}/pg{row.get('pg_id')}"
        f"/{row.get('dependency_type')}"
    )


def read_events(directory):
    result = {}
    for path in sorted(directory.glob("runtime_events.rank*.jsonl")):
        rank = int(re.fullmatch(r"runtime_events.rank(\d+)\.jsonl", path.name)[1])
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        require(rows and all(row["rank"] == rank for row in rows), f"event rank mismatch: {path}")
        result[rank] = rows
    require(result, f"no runtime_events.rankN.jsonl in {directory}")
    return result


def align_events(rows, marks):
    aligned = []
    index = 0
    for mark in marks:
        text = mark["text"]
        if STEP_COMPLETE.fullmatch(text) or text.startswith("shardtrain/qwen/"):
            continue
        while index < len(rows) and marker_text(rows[index]) != text:
            index += 1
        require(index < len(rows), f"NVTX/runtime event mismatch at {mark['start']}: {text}")
        aligned.append({**rows[index], "trace_ns": mark["start"], "globalTid": mark["globalTid"]})
        index += 1
    require(aligned, "no captured runtime markers aligned")
    return aligned


def columns(db, table):
    return {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}


def load_trace(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("StringIds", "CUPTI_ACTIVITY_KIND_KERNEL", "NVTX_EVENTS"):
            require(table in tables, f"missing required Nsight table: {table}")
        names = dict(db.execute("SELECT id,value FROM StringIds"))
        apis = {}
        duplicate_apis = 0
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables:
                continue
            for supplied in db.execute(f'SELECT * FROM "{table}" ORDER BY start'):
                row = dict(supplied)
                if row.get("globalTid") is None or row.get("correlationId") is None:
                    continue
                row["pid_key"] = row["globalTid"] >> 24
                row["name"] = names.get(row.get("nameId"), "")
                key = (row["pid_key"], row["correlationId"])
                if key in apis:
                    # Runtime APIs take precedence over a mirrored Driver entry.
                    duplicate_apis += 1
                    continue
                apis[key] = row
        require(apis, "Nsight export has no correlated CUDA API records")
        gpu = []
        for table, category in (
            ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
            ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
            ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
        ):
            if table not in tables:
                continue
            for supplied in db.execute(f'SELECT * FROM "{table}" ORDER BY start'):
                row = dict(supplied)
                if row.get("globalPid") is None:
                    continue
                row["pid_key"] = row["globalPid"] >> 24
                row["category"] = category
                row["name"] = names.get(row.get("shortName", row.get("demangledName")), category)
                row["nccl"] = category == "kernel" and "nccl" in row["name"].lower()
                row["gemm"] = category == "kernel" and any(
                    word in row["name"].lower() for word in ("gemm", "gemv", "cublas")
                )
                api = apis.get((row["pid_key"], row.get("correlationId")))
                row["api"] = api
                gpu.append(row)
        marks = defaultdict(list)
        for supplied in db.execute("SELECT rowid,* FROM NVTX_EVENTS ORDER BY start,rowid"):
            row = dict(supplied)
            row["text"] = row.get("text") or names.get(row.get("textId"), "")
            if row.get("globalTid") is not None and row["text"].startswith("shardtrain/"):
                marks[row["globalTid"] >> 24].append(row)
        event_records = []
        syncs = []
        for table, target in (
            ("CUPTI_ACTIVITY_KIND_CUDA_EVENT", event_records),
            ("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", syncs),
        ):
            if table in tables:
                target.extend(dict(row) for row in db.execute(f'SELECT * FROM "{table}"'))
        processes = {}
        if "PROCESSES" in tables:
            processes = {
                row["globalPid"] >> 24: {"pid": row["pid"], "name": row["name"]}
                for row in db.execute("SELECT * FROM PROCESSES")
                if row["globalPid"] is not None
            }
        return {
            "gpu": gpu,
            "apis": apis,
            "marks": marks,
            "event_records": event_records,
            "syncs": syncs,
            "processes": processes,
            "schema": {table: sorted(columns(db, table)) for table in sorted(tables)},
            "duplicate_api_keys_runtime_precedence": duplicate_apis,
        }


def rank_mapping(trace, events, directory, explicit):
    candidates = defaultdict(set)
    for pid, marks in trace["marks"].items():
        for mark in marks:
            match = STEP_COMPLETE.fullmatch(mark["text"])
            if match:
                candidates[int(match[2])].add(pid)
    mapping = {}
    for rank in events:
        source = "step_complete/rankN NVTX"
        possible = candidates.get(rank, set())
        if rank in explicit:
            possible = {pid for pid in trace["marks"] if (pid & PID_MASK) == explicit[rank]}
            source = "explicit --rank-pid"
        elif not possible:
            metadata = directory / f"environment.rank{rank}.json"
            data = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}
            if data.get("pid") is not None:
                possible = {pid for pid in trace["marks"] if (pid & PID_MASK) == data["pid"]}
                source = "environment.rankN.json pid"
            elif len(events) == 1 and len(trace["marks"]) == 1:
                possible = set(trace["marks"])
                source = "unique single-rank process with ShardTrain NVTX"
        require(len(possible) == 1, f"cannot unambiguously map rank {rank} to PID; use --rank-pid")
        mapping[rank] = {"pid_key": next(iter(possible)), "source": source}
    require(
        len({row["pid_key"] for row in mapping.values()}) == len(events), "multiple ranks share PID"
    )
    return mapping


def activity_detail(row):
    return {
        "category": row["category"],
        "name": row["name"],
        "device": row["deviceId"],
        "stream": row["streamId"],
        "gpu_start_ns": row["start"],
        "gpu_end_ns": row["end"],
        "cuda_correlation_id": row.get("correlationId"),
        "api_start_ns": row["api"]["start"] if row["api"] else None,
        "api_end_ns": row["api"]["end"] if row["api"] else None,
    }


def annotate_activities(gpu, aligned, ranges):
    issues = [row for row in aligned if row["event_type"] == "collective_issue"]
    issue_times = [row["trace_ns"] for row in issues]
    work_end = {row["op_id"]: row for row in aligned if row["event_type"] == "work_issued"}
    activations = [
        row for row in aligned if row["event_type"] in {"unit_forward", "unit_pre_backward"}
    ]
    activation_times = [row["trace_ns"] for row in activations]
    by_op = defaultdict(list)
    for activity in gpu:
        api = activity["api"]
        activity["kind"] = "unattributed_nccl" if activity["nccl"] else activity["category"]
        activity["op_id"] = None
        activity["compute_unit"] = None
        activity["compute_phase"] = None
        activity["compute_step"] = None
        activity["compute_generation"] = None
        activity["compute_activation_trace_ns"] = None
        if api is None:
            continue
        if activity["nccl"]:
            index = bisect.bisect_right(issue_times, api["start"]) - 1
            if index >= 0:
                issue = issues[index]
                end = work_end.get(issue["op_id"])
                if end and api["end"] <= end["trace_ns"]:
                    activity.update(
                        kind=issue.get("kind", issue["op_kind"]),
                        op_id=issue["op_id"],
                        unit_id=issue["unit_id"],
                        phase=issue["phase"],
                        step=issue["step_id"],
                        generation=issue.get("generation"),
                        materialization_generation=issue.get("materialization_generation"),
                        pg_id=issue["pg_id"],
                        issue_channel=issue["issue_stream"],
                        launch_tid_matches_issue=api["globalTid"] == issue["globalTid"],
                    )
                    by_op[issue["op_id"]].append(activity)
        else:
            index = bisect.bisect_right(activation_times, api["start"]) - 1
            if index >= 0:
                row = activations[index]
                activity["compute_unit"] = row["unit_id"]
                activity["compute_phase"] = row["phase"]
                activity["compute_step"] = row["step_id"]
                activity["compute_generation"] = row.get("generation")
                activity["compute_activation_trace_ns"] = row["trace_ns"]
            # Named Qwen layer ranges provide a stronger forward-unit label.
            if activity["category"] == "kernel":
                for nvtx in ranges:
                    match = LAYER_RANGE.fullmatch(nvtx["text"])
                    if (
                        match
                        and nvtx["globalTid"] == api["globalTid"]
                        and nvtx["start"] <= api["start"] <= nvtx["end"]
                    ):
                        activity["compute_unit"] = int(match[1]) + 1
                        activity["compute_phase"] = "FWD"
                        break
    return issues, by_op


def physical_boundaries(gpu, aligned, issues, by_op):
    ordered_gpu = sorted((row for row in gpu if row["api"]), key=lambda row: row["api"]["start"])
    launch_times = [row["api"]["start"] for row in ordered_gpu]
    copyout, consumers, same_pg, order_edges = [], [], [], []
    missing = Counter()
    for index, row in enumerate(aligned):
        if row["event_type"] == "copy_out_start":
            ready = next(
                (
                    candidate
                    for candidate in aligned[index + 1 :]
                    if candidate["event_type"] == "copy_out_ready"
                    and all(
                        candidate.get(key) == row.get(key)
                        for key in ("unit_id", "phase", "step_id", "generation")
                    )
                ),
                None,
            )
            producer = by_op.get(row.get("producer_ticket"), [])
            if row.get("producer_ticket") is None:
                missing["copyout_no_collective_degree1_or_retained"] += 1
                continue
            if ready is None or not producer:
                missing["copyout_missing_ready_or_AG_kernel"] += 1
                continue
            lo = bisect.bisect_left(launch_times, row["trace_ns"])
            hi = bisect.bisect_right(launch_times, ready["trace_ns"])
            copies = [
                candidate
                for candidate in ordered_gpu[lo:hi]
                if not candidate["nccl"]
                and candidate["api"]["globalTid"] == row["globalTid"]
                and row["trace_ns"] <= candidate["api"]["start"]
                and candidate["api"]["end"] <= ready["trace_ns"]
            ]
            if not copies:
                missing["copyout_no_correlated_GPU_activity"] += 1
                continue
            first = min(copies, key=lambda candidate: candidate["start"])
            ag_end = max(kernel["end"] for kernel in producer)
            copyout.append(
                {
                    "step": row["step_id"],
                    "unit": row["unit_id"],
                    "phase": row["phase"],
                    "op_id": row["producer_ticket"],
                    "generation": row.get("generation"),
                    "producer_generation": producer[0].get("generation"),
                    "AG_gpu_end_ns": ag_end,
                    "first_copyout_activity": activity_detail(first),
                    "copyout_activity_count": len(copies),
                    "gap_ns": first["start"] - ag_end,
                    "evidence": "GPU activity correlated inside exact copy_out_start/ready marks",
                }
            )
        if row["event_type"] != "collective_consumer_ready" or row["dependency_type"] != "DATA_DEP":
            continue
        producer = by_op.get(row["op_id"], [])
        if not producer or producer[0]["kind"] not in {"sp_ag", "sp_rs", "fsdp_rs"}:
            continue
        # This is a bounded observation of the next eligible launch, not a complete
        # tensor RAW analysis. The exact copy-out check above is stronger evidence.
        boundary = next(
            (
                candidate["trace_ns"]
                for candidate in aligned[index + 1 :]
                if candidate["event_type"] in {"unit_forward", "unit_pre_backward", "step_end"}
            ),
            2**63 - 1,
        )
        lo = bisect.bisect_left(launch_times, row["trace_ns"])
        hi = bisect.bisect_left(launch_times, boundary)
        candidates = [
            candidate
            for candidate in ordered_gpu[lo:hi]
            if candidate["category"] == "kernel"
            and not candidate["nccl"]
            and candidate["api"]["globalTid"] == row["globalTid"]
            and row["trace_ns"] <= candidate["api"]["start"] < boundary
            and (producer[0]["kind"] != "sp_ag" or candidate["gemm"])
        ]
        if not candidates:
            missing[f"{producer[0]['kind']}_no_eligible_consumer_boundary"] += 1
            continue
        consumer = min(candidates, key=lambda candidate: candidate["api"]["start"])
        end = max(kernel["end"] for kernel in producer)
        consumers.append(
            {
                "step": row["step_id"],
                "op_id": row["op_id"],
                "kind": producer[0]["kind"],
                "producer_gpu_end_ns": end,
                "consumer": activity_detail(consumer),
                "gap_ns": consumer["start"] - end,
                "evidence": "next same-thread launch before next unit/step; observational boundary",
            }
        )

    def order_check(previous, current, scope):
        left, right = by_op.get(previous["op_id"], []), by_op.get(current["op_id"], [])
        if not left or not right:
            missing[f"{scope}_unmapped_endpoint"] += 1
            return None
        return {
            "previous_op": previous["op_id"],
            "current_op": current["op_id"],
            "previous_pg": previous["pg_id"],
            "current_pg": current["pg_id"],
            "previous_kind": previous.get("kind", previous["op_kind"]),
            "current_kind": current.get("kind", current["op_kind"]),
            "previous_channel": previous["issue_stream"],
            "current_channel": current["issue_stream"],
            "previous_gpu_streams": sorted({row["streamId"] for row in left}),
            "current_gpu_streams": sorted({row["streamId"] for row in right}),
            "previous_gpu_end_ns": max(row["end"] for row in left),
            "current_gpu_start_ns": min(row["start"] for row in right),
            "gap_ns": min(row["start"] for row in right) - max(row["end"] for row in left),
            "scope": scope,
        }

    previous_pg = {}
    issue_by_id = {row["op_id"]: row for row in issues}
    for current in issues:
        pg = current["pg_id"]
        if pg in previous_pg:
            check = order_check(previous_pg[pg], current, "same_pg")
            if check:
                same_pg.append(check)
        previous_pg[pg] = current
    seen = set()
    for edge in aligned:
        if edge["event_type"] != "dependency_edge" or edge["dependency_type"] != "PG_ORDER":
            continue
        key = (edge.get("producer_ticket"), edge.get("consumer_ticket"))
        if key in seen or any(ticket not in issue_by_id for ticket in key):
            continue
        seen.add(key)
        check = order_check(issue_by_id[key[0]], issue_by_id[key[1]], "recorded_PG_ORDER")
        if check:
            check["recorded_wait_channel"] = edge.get("wait_stream")
            order_edges.append(check)
    return {
        "copyout": copyout,
        "consumer_boundaries": consumers,
        "same_pg_order": same_pg,
        "same_pg_cross_channel": [
            row for row in same_pg if row["previous_channel"] != row["current_channel"]
        ],
        "recorded_PG_ORDER": order_edges,
        "unmatched": dict(missing),
        "violations": [
            {"category": kind, **row}
            for kind, checks in (
                ("copyout_before_AG_end", copyout),
                ("consumer_boundary_before_collective_end", consumers),
                ("same_pg_GPU_order", same_pg),
                ("recorded_PG_ORDER_GPU_order", order_edges),
            )
            for row in checks
            if row["gap_ns"] < 0
        ],
    }


def event_record_wait_audit(trace, pid, gpu, aligned, example_limit):
    apis = trace["apis"]
    records, waits = [], []
    for supplied in trace["event_records"]:
        if supplied.get("globalPid") is None or supplied["globalPid"] >> 24 != pid:
            continue
        row = dict(supplied)
        api = apis.get((pid, row.get("correlationId")))
        if api is not None and "EventRecord" in api["name"]:
            row["api"] = api
            records.append(row)
    for supplied in trace["syncs"]:
        if supplied.get("globalPid") is None or supplied["globalPid"] >> 24 != pid:
            continue
        row = dict(supplied)
        api = apis.get((pid, row.get("correlationId")))
        if api is not None and "StreamWaitEvent" in api["name"]:
            row["api"] = api
            waits.append(row)
    # Propagate observed GPU completion lower bounds through matched record/wait
    # calls. The event table itself has no GPU completion timestamp. A missing
    # record cannot establish a complete bound and is explicitly retained.
    operations = []
    for row in gpu:
        if row["api"]:
            operations.append((row["api"]["end"], 0, "activity", row))
    operations.extend((row["api"]["start"], 1, "record", row) for row in records)
    operations.extend((row["api"]["end"], 2, "wait", row) for row in waits)
    stream_bounds = {}
    record_by_event = {}
    record_by_cycle = {}
    for _, _, kind, row in sorted(operations, key=lambda operation: operation[:2]):
        stream = (row.get("deviceId"), row.get("contextId"), row.get("streamId"))
        key = (row.get("contextId"), row.get("eventId"))
        cycle = row.get("eventSyncId")
        cycle_key = (*key, cycle)
        if kind == "activity":
            stream_bounds[stream] = max(stream_bounds.get(stream, 0), row["end"])
        elif kind == "record":
            row["observed_source_completion_lower_bound_ns"] = stream_bounds.get(stream)
            record_by_event[key] = row
            if cycle not in (None, INVALID_EVENT):
                record_by_cycle[cycle_key] = row
        else:
            if cycle not in (None, INVALID_EVENT):
                record = record_by_cycle.get(cycle_key)
                row["record_match_quality"] = "eventSyncId exact cycle"
            else:
                record = record_by_event.get(key)
                row["record_match_quality"] = (
                    "latest preceding same context/eventId; reuse not proven"
                )
            row["record"] = record
            bound = record.get("observed_source_completion_lower_bound_ns") if record else None
            row["observed_source_completion_lower_bound_ns"] = bound
            if bound is not None:
                stream_bounds[stream] = max(stream_bounds.get(stream, 0), bound)
    by_tid = defaultdict(list)
    for row in aligned:
        by_tid[row["globalTid"]].append(row)
    by_tid_times = {tid: [row["trace_ns"] for row in rows] for tid, rows in by_tid.items()}
    by_stream = defaultdict(list)
    for activity in sorted(gpu, key=lambda row: row["api"]["start"] if row["api"] else -1):
        if activity["api"]:
            by_stream[
                (activity.get("deviceId"), activity.get("contextId"), activity.get("streamId"))
            ].append(activity)
    by_stream_times = {
        key: [row["api"]["start"] for row in rows] for key, rows in by_stream.items()
    }
    matched = []
    counts = Counter()
    missing_by_type = Counter()
    missing_by_reason = Counter()
    missing_examples = []
    used_dependencies = set()
    for wait in sorted(waits, key=lambda row: row["api"]["start"]):
        api = wait["api"]
        timeline = by_tid[api["globalTid"]]
        index = bisect.bisect_right(by_tid_times.get(api["globalTid"], []), api["start"]) - 1
        if index < 0:
            continue
        edge = timeline[index]
        edge_key = (edge["globalTid"], edge["trace_ns"])
        if (
            edge["event_type"] != "dependency_edge"
            or edge.get("work_wait_called")
            or edge_key in used_dependencies
        ):
            continue
        used_dependencies.add(edge_key)
        record = wait.get("record")
        if record is None:
            counts["dependency_wait_without_record"] += 1
            missing_by_type[edge["dependency_type"]] += 1
            reason = (
                "invalid_event_id_in_export"
                if wait.get("eventId") == INVALID_EVENT
                else "no_preceding_record_for_exported_event"
            )
            missing_by_reason[reason] += 1
            if len(missing_examples) < example_limit:
                missing_examples.append(
                    {
                        "step": edge["step_id"],
                        "dependency_type": edge["dependency_type"],
                        "consumer_label": edge.get("consumer"),
                        "event_id": wait.get("eventId"),
                        "wait_gpu_stream": wait.get("streamId"),
                        "wait_api_start_ns": api["start"],
                        "reason": reason,
                    }
                )
            continue
        counts["matched_dependency_record_wait"] += 1
        counts[wait["record_match_quality"]] += 1
        stream_key = (wait.get("deviceId"), wait.get("contextId"), wait.get("streamId"))
        following = by_stream.get(stream_key, [])
        consumer_index = bisect.bisect_left(by_stream_times.get(stream_key, []), api["end"])
        consumer = following[consumer_index] if consumer_index < len(following) else None
        bound = wait["observed_source_completion_lower_bound_ns"]
        gap = consumer["start"] - bound if consumer and bound is not None else None
        if gap is not None:
            counts["observed_lower_bound_consumer_checks"] += 1
        matched.append(
            {
                "step": edge["step_id"],
                "dependency_type": edge["dependency_type"],
                "consumer_label": edge.get("consumer"),
                "producer_ticket": edge.get("producer_ticket"),
                "consumer_ticket": edge.get("consumer_ticket"),
                "recorded_wait_channel": edge.get("wait_stream"),
                "event_id": wait["eventId"],
                "event_sync_id": wait.get("eventSyncId"),
                "record_match_quality": wait["record_match_quality"],
                "record_source_gpu_stream": record["streamId"],
                "wait_gpu_stream": wait["streamId"],
                "wait_api_start_ns": api["start"],
                "source_completion_lower_bound_ns": bound,
                "first_later_activity_same_stream": activity_detail(consumer) if consumer else None,
                "gap_ns": gap,
            }
        )
    violations = [row for row in matched if row["gap_ns"] is not None and row["gap_ns"] < 0]
    counts["lower_bound_violations"] = len(violations)
    return {
        "cuda_event_record_count": len(records),
        "cuda_stream_wait_activity_count": len(waits),
        "counts": dict(counts),
        "missing_record_dependency_types": dict(missing_by_type),
        "missing_record_reasons": dict(missing_by_reason),
        "missing_record_examples": missing_examples,
        "examples": matched[:example_limit],
        "violations": violations,
        "gpu_event_completion_timestamp_present": False,
        "every_python_event_cycle_verified": False,
        "note": "Observed source activity end is a lower bound. The first later activity and "
        "record/wait linkage do not prove every tensor dependency; legacy eventId reuse is weaker.",
    }


def training_activities(compute, named_training_ranges):
    """Select PID-local submissions, including PyTorch autograd worker threads.

    The caller has selected one PID. Backward's NVTX range surrounds the main
    thread's loss.backward call, while its CUDA launches may occur on a worker.
    Requiring identical TIDs drops those kernels. Ranges identify submissions,
    never GPU completion.
    """
    return [
        row
        for row in compute
        if not named_training_ranges
        or any(
            nvtx["start"] <= row["api"]["start"] <= nvtx["end"] for nvtx in named_training_ranges
        )
    ]


def overlap_steps(gpu, aligned, by_op, ranges, example_limit):
    windows = []
    for row in aligned:
        if row["event_type"] == "step_end":
            step = row["step_id"]
            start = min(
                candidate["trace_ns"] for candidate in aligned if candidate["step_id"] == step
            )
            windows.append((step, start, row["trace_ns"]))
    steps = []
    examples = []
    prefetch_checks = []
    named_training_ranges = [
        row
        for row in ranges
        if row["text"]
        in {
            "shardtrain/qwen/model_forward",
            "shardtrain/qwen/loss",
            "shardtrain/qwen/backward",
            "shardtrain/qwen/optimizer",
        }
    ]
    for step, start, end in windows:
        activities = [row for row in gpu if row["api"] and start <= row["api"]["start"] <= end]
        kernels = [row for row in activities if row["category"] == "kernel"]
        nccl = [row for row in kernels if row["nccl"]]
        compute = [row for row in kernels if not row["nccl"]]
        # Always retain the complete NVTX step window. If training role ranges
        # exist, additionally exclude oracle/reconstruction outside those ranges.
        training = training_activities(compute, named_training_ranges)
        compute_iv = [(row["start"], row["end"]) for row in training]
        all_compute_iv = [(row["start"], row["end"]) for row in compute]
        gemm_iv = [(row["start"], row["end"]) for row in training if row["gemm"]]
        nccl_iv = [(row["start"], row["end"]) for row in nccl]
        by_kind = {}
        for kind in sorted({row["kind"] for row in nccl}):
            selected = [row for row in nccl if row["kind"] == kind]
            iv = [(row["start"], row["end"]) for row in selected]
            by_kind[kind] = {
                "kernel_count": len(selected),
                "gpu_union_ns": duration(iv),
                "compute_overlap_union_ns": duration(intersection(iv, compute_iv)),
                "gemm_overlap_union_ns": duration(intersection(iv, gemm_iv)),
                "gpu_streams": sorted({row["streamId"] for row in selected}),
                "issue_channels": sorted({row.get("issue_channel", "unknown") for row in selected}),
            }
        for communication in nccl:
            for computation in training:
                overlap = min(communication["end"], computation["end"]) - max(
                    communication["start"], computation["start"]
                )
                if overlap > 0:
                    examples.append(
                        {
                            "step": step,
                            "kind": communication["kind"],
                            "op_id": communication["op_id"],
                            "communication_unit": communication.get("unit_id"),
                            "communication_phase": communication.get("phase"),
                            "communication_generation": communication.get("generation"),
                            "communication": activity_detail(communication),
                            "computation_unit": computation.get("compute_unit"),
                            "computation_phase": computation.get("compute_phase"),
                            "computation_generation": computation.get("compute_generation"),
                            "computation": activity_detail(computation),
                            "overlap_ns": overlap,
                        }
                    )
        steps.append(
            {
                "step": step,
                "nvtx_CPU_submission_window_ns": [start, end],
                "associated_GPU_activity_span_ns": [
                    min(row["start"] for row in activities),
                    max(row["end"] for row in activities),
                ]
                if activities
                else None,
                "kernel_count": len(kernels),
                "training_kernel_count": len(training),
                "training_submission_thread_ids": sorted(
                    {row["api"]["globalTid"] & PID_MASK for row in training}
                ),
                "nccl_union_ns": duration(nccl_iv),
                "training_compute_union_ns": duration(compute_iv),
                "all_step_non_nccl_union_ns": duration(all_compute_iv),
                "training_gemm_union_ns": duration(gemm_iv),
                "nccl_compute_overlap_union_ns": duration(intersection(nccl_iv, compute_iv)),
                "nccl_gemm_overlap_union_ns": duration(intersection(nccl_iv, gemm_iv)),
                "compute_gpu_streams": sorted({row["streamId"] for row in training}),
                "by_kind": by_kind,
            }
        )
    # Exact deferred-SP boundary provides current and next unit IDs. Conservative
    # prefetch is inferred only inside an adjacent current unit's active interval,
    # never from a later on-demand AG after that unit has retired/reduced.
    targets = []
    seen_prefetch_ops = set()
    boundary_records = []
    boundary_occurrences = Counter()
    for index, edge in enumerate(aligned):
        if edge["event_type"] != "prefetch_boundary":
            continue
        signature = (
            edge["step_id"],
            edge["phase"],
            edge["unit_id"],
            edge.get("param_id"),
            edge.get("target_unit"),
        )
        boundary_occurrences[signature] += 1
        # A skipped prefetch must never steal an AG from a later use of the same
        # tied slot (e.g. embedding and distant LM head in one forward).
        issue = None
        resolution = "no immediate collective observed"
        for candidate in aligned[index + 1 :]:
            if candidate["event_type"] == "collective_issue":
                if (
                    candidate.get("kind", candidate.get("op_kind")) == "fsdp_ag"
                    and candidate["unit_id"] == edge.get("target_unit")
                    and candidate["phase"] == edge["phase"]
                    and candidate["step_id"] == edge["step_id"]
                ):
                    issue = candidate
                    resolution = "immediate AG ticket"
                break
            if candidate["event_type"] in {
                "prefetch_skipped_retained",
                "prefetch_boundary",
                "unit_forward",
                "unit_pre_backward",
                "step_end",
            }:
                resolution = candidate["event_type"]
                break
        boundary_records.append(
            {
                "step": edge["step_id"],
                "phase": edge["phase"],
                "source_unit": edge["unit_id"],
                "source_generation": edge.get("generation"),
                "source_parameter": edge.get("param_id"),
                "target_unit": edge.get("target_unit"),
                "source_use_boundary_occurrence": boundary_occurrences[signature],
                "boundary_trace_ns": edge["trace_ns"],
                "resolution": resolution,
                "op_id": issue["op_id"] if issue else None,
            }
        )
        if issue is None or not by_op.get(issue["op_id"]):
            continue
        edge = {**edge, "source_use_boundary_occurrence": boundary_occurrences[signature]}
        targets.append((edge, issue, "explicit prefetch_boundary target"))
        seen_prefetch_ops.add(issue["op_id"])
    active = {}
    for row in aligned:
        if row["event_type"] in {"unit_forward", "unit_pre_backward"}:
            active[row["phase"]] = row
        if row["event_type"] in {"RESHARDED", "FWD_RETAINED", "unit_grads_ready"}:
            current = active.get(row["phase"])
            if current and current["unit_id"] == row["unit_id"]:
                active.pop(row["phase"], None)
        if row["event_type"] != "collective_issue" or row["op_id"] in seen_prefetch_ops:
            continue
        current = active.get(row["phase"])
        if (
            current
            and row.get("kind", row.get("op_kind")) == "fsdp_ag"
            and row["step_id"] == current["step_id"]
            and row["unit_id"] == current["unit_id"] + (1 if row["phase"] == "FWD" else -1)
            and by_op.get(row["op_id"])
        ):
            targets.append(
                (
                    {**current, "target_unit": row["unit_id"]},
                    row,
                    "adjacent AG launch inside current unit active interval",
                )
            )
    for edge, issue, evidence in targets:
        comm = by_op[issue["op_id"]]
        activation = next(
            (
                candidate
                for candidate in reversed(aligned)
                if candidate["trace_ns"] <= edge["trace_ns"]
                and candidate["event_type"] in {"unit_forward", "unit_pre_backward"}
            ),
            None,
        )
        source_matches = activation is not None and all(
            activation.get(key) == edge.get(key)
            for key in ("step_id", "phase", "unit_id", "generation")
        )
        current = [
            row
            for row in gpu
            if row["api"]
            and row["category"] == "kernel"
            and not row["nccl"]
            and row.get("compute_unit") == edge["unit_id"]
            and row.get("compute_phase") == edge["phase"]
            and row.get("compute_step") == edge["step_id"]
            and row.get("compute_generation") == edge.get("generation")
            and source_matches
            and row.get("compute_activation_trace_ns") == activation["trace_ns"]
            and row["api"]["start"] >= edge["trace_ns"]
            and any(
                step == edge["step_id"] and start <= row["api"]["start"] <= end
                for step, start, end in windows
            )
        ]
        comm_iv = [(row["start"], row["end"]) for row in comm]
        prefetch_checks.append(
            {
                "step": edge["step_id"],
                "phase": edge["phase"],
                "current_unit": edge["unit_id"],
                "target_unit": edge["target_unit"],
                "op_id": issue["op_id"],
                "source_parameter": edge.get("param_id"),
                "source_generation": edge.get("generation"),
                "target_generation": issue.get("generation"),
                "source_use_boundary_occurrence": edge.get("source_use_boundary_occurrence"),
                "source_boundary_trace_ns": edge["trace_ns"],
                "source_matches_latest_unit_activation": source_matches,
                "latest_unit_activation": {
                    key: activation.get(key)
                    for key in ("unit_id", "phase", "step_id", "generation", "trace_ns")
                }
                if activation
                else None,
                "source_use_scope": "current unit activation"
                if source_matches
                else "source slot reused outside latest unit activation; consumer scope unresolved",
                "AG_gpu_span_ns": [
                    min(row["start"] for row in comm),
                    max(row["end"] for row in comm),
                ],
                "current_unit_compute_overlap_union_ns": duration(
                    intersection(comm_iv, [(row["start"], row["end"]) for row in current])
                )
                if source_matches
                else None,
                "current_unit_gemm_overlap_union_ns": duration(
                    intersection(
                        comm_iv, [(row["start"], row["end"]) for row in current if row["gemm"]]
                    )
                )
                if source_matches
                else None,
                "evidence": evidence + "; compute unit from layer/activation marker",
            }
        )
    # Keep the largest intersections as reproducible Nsight navigation examples.
    examples.sort(key=lambda row: row["overlap_ns"], reverse=True)
    return {
        "steps": steps,
        "largest_overlap_examples": examples[:example_limit],
        "prefetch_checks": prefetch_checks,
        "prefetch_boundaries": boundary_records,
        "prefetch_attribution_note": "A tied slot may have multiple uses in one phase/generation. "
        "Each boundary/ticket is separate; unresolved reuse receives null current-unit overlap.",
        "compute_selection": "PID-local Qwen training role NVTX submission ranges, "
        "including autograd worker threads"
        if named_training_ranges
        else "complete runtime-marker NVTX step; may include validation/packing/optimizer kernels",
        "note": "GPU intervals are unioned per rank/step; ranks are never summed as elapsed time. "
        "NCCL residency can include peer waiting. Temporal overlap is not throughput improvement.",
    }


def analyze_rank(trace, rank, mapping, rows, example_limit):
    pid = mapping["pid_key"]
    marks = trace["marks"][pid]
    aligned = align_events(rows, marks)
    gpu = [row for row in trace["gpu"] if row["pid_key"] == pid]
    require(gpu, f"no GPU activities for rank {rank}")
    ranges = [row for row in marks if row.get("end") is not None]
    issues, by_op = annotate_activities(gpu, aligned, ranges)
    boundaries = physical_boundaries(gpu, aligned, issues, by_op)
    overlap = overlap_steps(gpu, aligned, by_op, ranges, example_limit)
    require(overlap["steps"], f"no complete captured runtime step for rank {rank}")
    event_audit = event_record_wait_audit(trace, pid, gpu, aligned, example_limit)
    captured_steps = {row["step"] for row in overlap["steps"]}
    training_issues = [
        row
        for row in issues
        if row["step_id"] in captured_steps and row.get("kind", row.get("op_kind")) in COMM_KINDS
    ]
    all_captured_step_issues = [row for row in issues if row["step_id"] in captured_steps]
    unmapped = [row["op_id"] for row in training_issues if not by_op.get(row["op_id"])]
    aggregate = Counter()
    kind_aggregate = defaultdict(Counter)
    for step in overlap["steps"]:
        for key in (
            "nccl_union_ns",
            "training_compute_union_ns",
            "training_gemm_union_ns",
            "nccl_compute_overlap_union_ns",
            "nccl_gemm_overlap_union_ns",
        ):
            aggregate[key] += step[key]
        for kind, metrics in step["by_kind"].items():
            for key in (
                "kernel_count",
                "gpu_union_ns",
                "compute_overlap_union_ns",
                "gemm_overlap_union_ns",
            ):
                kind_aggregate[kind][key] += metrics[key]
    return {
        "rank": rank,
        "pid": pid & PID_MASK,
        "pid_namespace_key": pid,
        "rank_mapping_source": mapping["source"],
        "process_metadata": trace["processes"].get(pid),
        "devices": sorted({row["deviceId"] for row in gpu}),
        "gpu_kernel_streams": sorted(
            {row["streamId"] for row in gpu if row["category"] == "kernel"}
        ),
        "aligned_runtime_markers": len(aligned),
        "GPU_activities_without_correlated_API": sum(row["api"] is None for row in gpu),
        "captured_training_collectives": len(training_issues),
        "mapped_training_collectives": len(training_issues) - len(unmapped),
        "unmapped_training_op_ids": unmapped,
        "mapped_nccl_kernel_count": sum(len(items) for items in by_op.values()),
        "mapped_nccl_kernels_by_kind": dict(
            Counter(activity["kind"] for items in by_op.values() for activity in items)
        ),
        "count_scopes": {
            "captured_steps": sorted(captured_steps),
            "math_collective_operations_over_all_captured_steps": len(training_issues),
            "math_collective_kinds": sorted(COMM_KINDS),
            "all_collective_operations_with_captured_step_ids": len(all_captured_step_issues),
            "mapped_nccl_kernels_scope": "all collective kinds aligned in this PID's captured "
            "NVTX/runtime markers, including scalar loss/scaler/timing metadata",
            "GPU_boundary_and_event_counts_scope": "one rank over the captured NVTX markers; "
            "not per step or a sum of rank elapsed times",
        },
        "unattributed_nccl_kernel_count": sum(row["nccl"] and row["op_id"] is None for row in gpu),
        "physical_boundaries": boundaries,
        "cuda_record_wait_observations": event_audit,
        "overlap": overlap,
        "aggregate_over_captured_steps_ns": dict(aggregate),
        "per_step_mean_ms": {
            key: value / len(overlap["steps"]) / 1e6 for key, value in aggregate.items()
        },
        "by_kind_aggregate_ns": {kind: dict(metrics) for kind, metrics in kind_aggregate.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank-pid", action="append", default=[], metavar="RANK=PID")
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--require-nccl", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Reject unmapped captured collectives/copyouts",
    )
    args = parser.parse_args(argv)
    args.sqlite, args.events_dir, args.output = (
        path.resolve() for path in (args.sqlite, args.events_dir, args.output)
    )
    print(f"ANALYZING TRACE: {args.sqlite}", flush=True)
    native = args.sqlite.with_suffix(".nsys-rep")
    print(f"NATIVE TRACE: {native}", flush=True)
    result = {
        "sqlite": str(args.sqlite),
        "native_trace": str(native),
        "runtime_events_directory": str(args.events_dir),
        "time_unit": "nanoseconds",
        "physical_event_graph_fully_verified": False,
        "speedup_verified": False,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        explicit = {}
        for item in args.rank_pid:
            rank, pid = (int(value) for value in item.split("="))
            require(rank not in explicit, "duplicate --rank-pid")
            explicit[rank] = pid
        events = read_events(args.events_dir)
        trace = load_trace(args.sqlite)
        mapping = rank_mapping(trace, events, args.events_dir, explicit)
        result["ranks"] = [
            analyze_rank(trace, rank, mapping[rank], rows, args.examples)
            for rank, rows in sorted(events.items())
        ]
        violations = sum(
            len(row["physical_boundaries"]["violations"])
            + len(row["cuda_record_wait_observations"]["violations"])
            for row in result["ranks"]
        )
        incomplete = any(
            row["unmapped_training_op_ids"]
            or any(
                value
                for key, value in row["physical_boundaries"]["unmatched"].items()
                if key.startswith("copyout_") and "degree1" not in key
            )
            for row in result["ranks"]
        )
        nccl = sum(row["mapped_nccl_kernel_count"] for row in result["ranks"])
        require(
            not args.require_nccl or len(events) > 1 and nccl > 0,
            "actual multi-rank NCCL GPU kernels required",
        )
        result.update(
            status="failed"
            if violations or args.require_complete and incomplete
            else "observed_boundaries_passed",
            observed_GPU_boundary_violations=violations,
            complete_collective_copyout_coverage=not incomplete,
            mapped_nccl_kernels=nccl,
            schema={
                key: trace["schema"][key]
                for key in (
                    "CUPTI_ACTIVITY_KIND_KERNEL",
                    "CUPTI_ACTIVITY_KIND_RUNTIME",
                    "NVTX_EVENTS",
                    "CUPTI_ACTIVITY_KIND_CUDA_EVENT",
                    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
                )
                if key in trace["schema"]
            },
            limitations=[
                "NVTX marks delimit CPU submission, never GPU completion.",
                "Correlation IDs are matched within a PID namespace, never across processes.",
                "Exact AG-to-copyout launch association is stronger than next-consumer heuristics.",
                "CUDA event tables lack GPU completion timestamps; activity lower bounds checked.",
                "No statement of every event/reuse-cycle/tensor dependency or performance speedup.",
                "NCCL kernel residency can include peer waiting as well as payload transfer.",
            ],
        )
    except (ValueError, KeyError, OSError, sqlite3.Error) as error:
        result.update(status="failed", error=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key not in {"ranks", "schema"}}
    if "ranks" in result:
        summary["ranks"] = [
            {
                "rank": row["rank"],
                "pid": row["pid"],
                "steps": len(row["overlap"]["steps"]),
                "collectives_mapped": row["mapped_training_collectives"],
                "collectives_unmapped": len(row["unmapped_training_op_ids"]),
                "copyout_checks": len(row["physical_boundaries"]["copyout"]),
                "same_pg_cross_channel_checks": len(
                    row["physical_boundaries"]["same_pg_cross_channel"]
                ),
                "prefetch_checks": len(row["overlap"]["prefetch_checks"]),
                "per_step_mean_ms": row["per_step_mean_ms"],
            }
            for row in result["ranks"]
        ]
    print(json.dumps(summary, indent=2))
    print(f"OUTPUT: {args.output}")
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
