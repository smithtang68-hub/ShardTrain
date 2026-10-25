"""Audit one explicitly announced official Q10 trace; preserve NSYS namespaces.

Only the supplied profile's SQLite is read. OS PID to rank is taken from its
environment metadata and PROCESSES, never process order. Indexed base parser
keeps full namespace keys for API/kernel correlation. Standard NCCL kernel names
identify operation families, but unknown process-group/target-unit roles stay
UNKNOWN. CPU microstep attribution uses unique containing same-process launch
API ranges; it is not a GPU range-duration or nearest-kernel heuristic.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.fsdp_interview.scripts import parse_trace  # noqa: E402
from experiments.fsdp_interview.scripts.analyze_gemm_trace import ScopeIndex  # noqa: E402
from experiments.fsdp_interview.scripts.parse_trace_indexed import analyze  # noqa: E402

MICRO_RE = re.compile(r"q10/microstep_(\d+)/(forward|backward)$")
PID_DOC = "https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#serialized-process-and-thread-identifiers"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def csv_write(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def operation_family(name: str) -> str:
    if "nccl" not in name.lower():
        return "NON_NCCL_KERNEL"
    for family in ("AllGather", "ReduceScatter", "AllReduce", "Broadcast", "SendRecv"):
        if family in name:
            return family
    return "NCCL_UNKNOWN"


def containing(scopes: list[dict], pid: int, start: int, end: int) -> list[dict]:
    return [
        row for row in scopes if row["pid"] == pid and row["start"] <= start and end <= row["end"]
    ]


def known_record_predecessor(records: list[dict], wait_start: int):
    """Only accept a serialized latest observed record of the same event key."""
    position = bisect.bisect_right([row["api"]["start"] for row in records], wait_start) - 1
    if position < 0:
        return None, "UNKNOWN_RECORD_BEFORE_CAPTURE_OR_MISSING"
    selected = records[position]
    if selected["api"]["end"] > wait_start or any(
        row["api"]["end"] > selected["api"]["start"] for row in records[:position]
    ):
        return None, "AMBIGUOUS_CONCURRENT_RECORD"
    return selected, "MATCHED_RECORDED_EVENT_GENERATION"


def collect_operations(data: dict, namespace_to_rank: dict[int, int]) -> list[dict]:
    micro_scopes = [
        row for row in data["auxiliary_nvtx_ranges"] if MICRO_RE.fullmatch(row["label"])
    ]
    steps = [dict(row, start=row["start_ns"], end=row["end_ns"]) for row in data["step_windows"]]
    result = []
    for kernel in data["kernels"]:
        stage_matches, step_matches = {}, {}
        for api in kernel["launch_apis"]:
            for scope in containing(micro_scopes, kernel["pid"], api["start"], api["end"]):
                stage_matches[scope["scope_id"]] = scope
            for scope in containing(steps, kernel["pid"], api["start"], api["end"]):
                step_matches[scope["scope_id"]] = scope
        stage = next(iter(stage_matches.values())) if len(stage_matches) == 1 else None
        step = next(iter(step_matches.values())) if len(step_matches) == 1 else None
        micro = MICRO_RE.fullmatch(stage["label"]) if stage else None
        result.append(
            {
                "rank": namespace_to_rank.get(kernel["pid"]),
                "process_namespace": kernel["pid"],
                "step": step["step"] if step else None,
                "microstep": int(micro.group(1)) if micro else None,
                "phase": micro.group(2) if micro else "UNKNOWN",
                "attribution": "UNIQUE_SAME_PROCESS_CPU_API_CONTAINMENT" if stage else "UNKNOWN",
                "operation_family": operation_family(kernel["kernel_name"]),
                "pg_role": "UNKNOWN",
                "target_parameter_unit": "UNKNOWN",
                "kernel_rowid": kernel["kernel_rowid"],
                "kernel_name": kernel["kernel_name"],
                "stream_id": kernel["stream_id"],
                "device_id": kernel["device_id"],
                "correlation_id": kernel["correlation_id"],
                "gpu_start_ns": kernel["gpu_start_ns"],
                "gpu_end_ns": kernel["gpu_end_ns"],
                "duration_ms": kernel["duration_ms"],
                "launch_apis": kernel["launch_apis"],
                "source_nvtx_labels": kernel["enclosing_nvtx_labels"],
                "section": "OPTIMIZER"
                if "q10/optimizer_step" in kernel["enclosing_nvtx_labels"]
                else "MICROSTEP"
                if micro
                else "OTHER_OR_UNKNOWN",
            }
        )
    return result


def synchronization_audit(database, scopes: list[dict], namespace_to_rank: dict[int, int]):
    strings = {
        row["id"]: row["value"] for row in database.execute("SELECT id,value FROM StringIds")
    }
    apis = [
        dict(row)
        for row in database.execute("SELECT rowid AS api_rowid,* FROM CUPTI_ACTIVITY_KIND_RUNTIME")
    ]
    api_by_correlation = defaultdict(list)
    scopes_by_thread = defaultdict(list)
    for scope in scopes:
        scopes_by_thread[scope["globalTid"]].append(scope)
    indices = {tid: ScopeIndex(values) for tid, values in scopes_by_thread.items()}
    for api in apis:
        api["process_namespace"] = api["globalTid"] >> 24
        api["api_name"] = strings.get(api["nameId"], "UNKNOWN")
        api["source_nvtx_labels"] = [
            row["label"]
            for row in indices.get(api["globalTid"], ScopeIndex([])).containing(
                api["start"], api["end"]
            )
        ]
        api_by_correlation[(api["process_namespace"], api["correlationId"])].append(api)
    records = defaultdict(list)
    event_record_api_counts = Counter(
        row["api_name"] for row in apis if "EventRecord" in row["api_name"]
    )
    event_record_rows = 0
    for event in database.execute(
        "SELECT rowid AS event_rowid,* FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT"
    ):
        event = dict(event)
        namespace = event["globalPid"] >> 24
        candidates = [
            row
            for row in api_by_correlation[(namespace, event["correlationId"])]
            if "EventRecord" in row["api_name"]
        ]
        if len(candidates) == 1:
            record = {**event, "api": candidates[0]}
            records[(namespace, event["deviceId"], event["contextId"], event["eventId"])].append(
                record
            )
            event_record_rows += 1
    for values in records.values():
        values.sort(key=lambda row: row["api"]["start"])
    record_overlap_count = sum(
        values[index - 1]["api"]["end"] > values[index]["api"]["start"]
        for values in records.values()
        for index in range(1, len(values))
    )
    wait_rows = []
    for activity in database.execute(
        "SELECT rowid AS activity_rowid,* FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE syncType=2"
    ):
        activity = dict(activity)
        namespace = activity["globalPid"] >> 24
        candidates = [
            row
            for row in api_by_correlation[(namespace, activity["correlationId"])]
            if "StreamWaitEvent" in row["api_name"]
        ]
        row = {
            "rank": namespace_to_rank.get(namespace),
            "process_namespace": namespace,
            "device_id": activity["deviceId"],
            "context_id": activity["contextId"],
            "event_id": activity["eventId"],
            "wait_correlation_id": activity["correlationId"],
            "target_stream": activity["streamId"],
            "source_stream": None,
            "gpu_wait_start_ns": activity["start"],
            "gpu_wait_end_ns": activity["end"],
            "gpu_wait_duration_ms": (activity["end"] - activity["start"]) / 1e6,
            "status": "UNKNOWN",
            "source_nvtx_labels": [],
            "wait_nvtx_labels": [],
            "pg_role": "UNKNOWN",
            "dependent_parameter_unit": "UNKNOWN",
        }
        if len(candidates) == 1:
            wait_api = candidates[0]
            row.update(
                wait_api_start_ns=wait_api["start"],
                wait_api_end_ns=wait_api["end"],
                wait_nvtx_labels=wait_api["source_nvtx_labels"],
            )
            own = records.get(
                (namespace, activity["deviceId"], activity["contextId"], activity["eventId"]), []
            )
            record, row["status"] = known_record_predecessor(own, wait_api["start"])
            if record is not None:
                row.update(
                    status="MATCHED_RECORDED_EVENT_GENERATION",
                    source_stream=record["streamId"],
                    source_record_correlation_id=record["correlationId"],
                    source_record_api_start_ns=record["api"]["start"],
                    source_record_api_end_ns=record["api"]["end"],
                    source_nvtx_labels=record["api"]["source_nvtx_labels"],
                )
        wait_rows.append(row)
    sync_apis = [
        {
            "rank": namespace_to_rank.get(row["process_namespace"]),
            "process_namespace": row["process_namespace"],
            "api_rowid": row["api_rowid"],
            "api_name": row["api_name"],
            "start_ns": row["start"],
            "end_ns": row["end"],
            "duration_ms": (row["end"] - row["start"]) / 1e6,
            "source_nvtx_labels": row["source_nvtx_labels"],
        }
        for row in apis
        if any(
            word in row["api_name"]
            for word in ("EventSynchronize", "DeviceSynchronize", "StreamSynchronize")
        )
    ]
    return (
        wait_rows,
        sync_apis,
        {
            "record_api_counts": dict(event_record_api_counts),
            "cuda_event_rows_with_unique_record_api": event_record_rows,
            "overlapping_known_same_event_record_api_intervals": record_overlap_count,
            "event_wait_status_counts": dict(Counter(row["status"] for row in wait_rows)),
            "generation_guard": "Latest observed same namespace/device/context/eventId "
            "record must finish before wait API; "
            "overlapping known record API intervals are ambiguous.",
            "capture_limit": "Missing CUDA_EVENT eventId metadata (including APIs without "
            "a matching activity row) prevents complete producer recovery; UNKNOWN is preserved.",
        },
    )


def summarize(operations: list[dict], waits: list[dict], expected_groups: int = 9) -> dict:
    grouped = defaultdict(list)
    for row in operations:
        if row["step"] is not None:
            grouped[(row["rank"], row["step"])].append(row)
    steps = []
    for (rank, step), rows in sorted(grouped.items()):
        families = defaultdict(list)
        for row in rows:
            families[row["operation_family"]].append((row["gpu_start_ns"], row["gpu_end_ns"]))
        steps.append(
            {
                "rank": rank,
                "step": step,
                "kernel_counts": dict(Counter(row["operation_family"] for row in rows)),
                "microstep_counts": [
                    {
                        "microstep": micro,
                        "phase": phase,
                        "counts": dict(
                            Counter(
                                row["operation_family"]
                                for row in rows
                                if row["microstep"] == micro and row["phase"] == phase
                            )
                        ),
                    }
                    for micro in (0, 1)
                    for phase in ("forward", "backward")
                ],
                "streams_by_family": {
                    family: sorted(
                        {row["stream_id"] for row in rows if row["operation_family"] == family}
                    )
                    for family in families
                },
                "ag_rs_overlap_ms": parse_trace.duration_ms(
                    parse_trace.intersection(families["AllGather"], families["ReduceScatter"])
                ),
                "ag_non_nccl_overlap_ms": parse_trace.duration_ms(
                    parse_trace.intersection(families["AllGather"], families["NON_NCCL_KERNEL"])
                ),
                "rs_non_nccl_overlap_ms": parse_trace.duration_ms(
                    parse_trace.intersection(families["ReduceScatter"], families["NON_NCCL_KERNEL"])
                ),
                "unknown_micro_nccl_kernels": sum(
                    row["operation_family"] != "NON_NCCL_KERNEL" and row["microstep"] is None
                    for row in rows
                ),
                "optimizer_non_nccl_kernel_count": sum(
                    row["section"] == "OPTIMIZER" and row["operation_family"] == "NON_NCCL_KERNEL"
                    for row in rows
                ),
                "accumulation_trace_check": {
                    "first_microstep_rs": sum(
                        row["operation_family"] == "ReduceScatter" and row["microstep"] == 0
                        for row in rows
                    ),
                    "final_microstep_rs": sum(
                        row["operation_family"] == "ReduceScatter" and row["microstep"] == 1
                        for row in rows
                    ),
                    "expected_parameter_groups": expected_groups,
                    "pass": sum(
                        row["operation_family"] == "ReduceScatter" and row["microstep"] == 0
                        for row in rows
                    )
                    == 0
                    and sum(
                        row["operation_family"] == "ReduceScatter" and row["microstep"] == 1
                        for row in rows
                    )
                    == expected_groups,
                    "scope": "Q10 child groups + root; two microsteps; "
                    "count evidence, not gradient math",
                },
            }
        )
    edges = defaultdict(list)
    for row in waits:
        edges[(row["rank"], row["source_stream"], row["target_stream"], row["status"])].append(row)
    return {
        "steps": steps,
        "event_edges": [
            {
                "rank": key[0],
                "source_stream": key[1],
                "target_stream": key[2],
                "status": key[3],
                "count": len(rows),
                "gpu_wait_union_ms": parse_trace.duration_ms(
                    [(row["gpu_wait_start_ns"], row["gpu_wait_end_ns"]) for row in rows]
                ),
            }
            for key, rows in edges.items()
        ],
        "unknown_role_policy": "PG roles and parameter target units remain UNKNOWN "
        "without exact metadata.",
        "overlap_scope": "GPU kernel residency intersections; not useful concurrent "
        "progress, time saved, or critical path.",
    }


def temporal_plot(output: Path, operations: list[dict], data: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rank, step = 0, 0
    selected = [row for row in operations if row["rank"] == rank and row["step"] == step]
    window = next(
        row for row in data["step_windows"] if row["rank"] == rank and row["step"] == step
    )
    start = window["start_ns"]
    streams = sorted({row["stream_id"] for row in selected})
    colors = {"AllGather": "#2470a2", "ReduceScatter": "#d17724", "NON_NCCL_KERNEL": "#777777"}
    figure, axis = plt.subplots(figsize=(13, 4.5))
    for position, stream in enumerate(streams):
        for family, color in colors.items():
            intervals = [
                ((row["gpu_start_ns"] - start) / 1e6, row["duration_ms"])
                for row in selected
                if row["stream_id"] == stream and row["operation_family"] == family
            ]
            axis.broken_barh(intervals, (position - 0.25, 0.5), facecolors=color)
    cpu_position = len(streams)
    for scope in data["auxiliary_nvtx_ranges"]:
        if scope["pid"] != window["pid"] or not MICRO_RE.fullmatch(scope["label"]):
            continue
        if scope["start"] < window["start_ns"] or scope["end"] > window["end_ns"]:
            continue
        micro = MICRO_RE.fullmatch(scope["label"])
        left, width = (scope["start"] - start) / 1e6, (scope["end"] - scope["start"]) / 1e6
        axis.broken_barh(
            [(left, width)],
            (cpu_position - 0.25, 0.5),
            facecolors="#d5e5ed" if micro.group(2) == "forward" else "#eadacb",
        )
        axis.text(
            left + width / 2,
            cpu_position,
            f"m{micro.group(1)} {micro.group(2)}",
            ha="center",
            va="center",
            fontsize=9,
        )
    axis.set_yticks(
        list(range(len(streams))) + [cpu_position],
        [f"CUDA stream {stream}" for stream in streams] + ["CPU microstep scopes"],
    )
    axis.set_xlabel("Same NSYS report time, ms from rank 0 step 0 CPU start")
    axis.set_title(
        "Official Q10: observed kernel residency and CPU scopes (profile, not timing run)"
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(
        handles=[Patch(color=color, label=family) for family, color in colors.items()],
        loc="lower right",
    )
    figure.tight_layout()
    figure.savefig(output / "mechanism_temporal.rank0.step0.png", dpi=160)
    figure.savefig(output / "mechanism_temporal.rank0.step0.svg")
    plt.close(figure)
    return {
        "rank": rank,
        "step": step,
        "scope": "GPU kernel residency; CPU NVTX scopes are separate; no PG/unit role inferred",
        "files": ["mechanism_temporal.rank0.step0.png", "mechanism_temporal.rank0.step0.svg"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("choose a fresh Q10 analysis output; earlier analyses are preserved")
    database_path = args.profile_dir / "timeline.sqlite"
    database = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA query_only=ON")
    process_rows = [
        dict(row) for row in database.execute("SELECT globalPid,pid,name FROM PROCESSES")
    ]
    mapping_rows, rank_namespaces, provenance = [], {}, {}
    for rank in range(4):
        path = args.profile_dir / f"environment.rank{rank}.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata["rank"] != rank or not metadata["config"]["profile"]:
            raise ValueError("profile environment identity mismatch")
        if metadata["config"]["accumulation_steps"] != 2:
            raise ValueError("Q10 microstep audit currently requires accumulation_steps=2")
        matches = [row for row in process_rows if row["pid"] == metadata["pid"]]
        if len(matches) != 1:
            raise ValueError("OS PID has missing/ambiguous PROCESSES namespace")
        process = matches[0]
        global_pid = process["globalPid"]
        if (global_pid >> 24) & 0xFFFFFF != metadata["pid"]:
            raise ValueError("PROCESSES row differs from documented Nsight PID encoding")
        namespace = global_pid >> 24
        rank_namespaces[str(rank)] = namespace
        mapping_rows.append(
            {
                "rank": rank,
                "os_pid": metadata["pid"],
                "global_pid": global_pid,
                "process_namespace": namespace,
                "vm_id": (global_pid >> 48) & 0xFF,
                "hardware_id": (global_pid >> 56) & 0xFF,
                "process_name": process["name"],
                "source": str(path),
                "schema_source": PID_DOC,
            }
        )
        provenance[str(path)] = {"sha256_local": sha(path), "bytes": path.stat().st_size}
    namespace_to_rank = {int(namespace): int(rank) for rank, namespace in rank_namespaces.items()}
    if len(namespace_to_rank) != 4:
        raise ValueError("duplicate rank process namespace")
    args.output_dir.mkdir(parents=True)
    dump(args.output_dir / "rank_namespaces.json", rank_namespaces)
    csv_write(args.output_dir / "rank_process_mapping.csv", mapping_rows)
    data = analyze(database_path, rank_pids=rank_namespaces)
    dump(args.output_dir / "indexed_trace_analysis.json", data)
    # Load NVTX scopes once for same-thread CPU synchronization attribution.
    source_data = parse_trace.load_sqlite(database_path)
    operations = collect_operations(data, namespace_to_rank)
    waits, sync_apis, event_guard = synchronization_audit(
        database, source_data["scopes"], namespace_to_rank
    )
    database.close()
    result = summarize(operations, waits, metadata["config"]["layers"] + 1)
    result["event_generation_guard"] = event_guard
    result["cpu_isfinite_scopes"] = [
        {
            "rank": window["rank"],
            "step": window["step"],
            "count": sum(
                scope["label"].startswith("aten::isfinite")
                and scope["pid"] == window["pid"]
                and window["start_ns"] <= scope["start"]
                and scope["end"] <= window["end_ns"]
                for scope in source_data["scopes"]
            ),
            "evidence": "complete CPU NVTX scopes, including operations without GPU kernels",
        }
        for window in data["step_windows"]
    ]
    result["temporal_plot"] = temporal_plot(args.output_dir, operations, data)
    result.update(source_sqlite=str(database_path.resolve()), rank_process_mapping=mapping_rows)
    dump(args.output_dir / "q10_trace_audit.json", result)
    csv_write(args.output_dir / "operations.csv", operations)
    csv_write(args.output_dir / "event_dependencies.csv", waits)
    csv_write(args.output_dir / "cpu_synchronization_apis.csv", sync_apis)
    csv_write(args.output_dir / "step_metrics.csv", data["step_metrics"])
    provenance[str(database_path)] = {
        "sha256_local": sha(database_path),
        "bytes": database_path.stat().st_size,
    }
    dump(
        args.output_dir / "analysis_manifest.json",
        {
            "argv": sys.argv,
            "script_sha256": sha(Path(__file__)),
            "inputs": provenance,
            "remote_hash_match": "NOT_VERIFIED_BY_THIS_ANALYSIS",
            "parser_provenance": data["parser_provenance"],
            "scope": "one explicitly supplied announced Q10 profile; no other trace read",
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "steps": len(result["steps"]),
                "event_waits": len(waits),
                "matched_event_waits": sum(
                    row["status"] == "MATCHED_RECORDED_EVENT_GENERATION" for row in waits
                ),
                "rank_process_mapping": mapping_rows,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
