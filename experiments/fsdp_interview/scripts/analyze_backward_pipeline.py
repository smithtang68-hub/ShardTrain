"""Exact host-unit ownership -> launch API -> GPU kernel BWD pipeline audit.

Runtime's actual unit_pre_backward/unit_grads_ready marks delimit an ownership
context on the same CPU thread. An API must be fully inside that context; its
correlation already links the GPU kernel in the frozen parser. No clock-nearest
matching or kernel-name layer guess is used. Root/tied nested contexts select
the smallest complete containing ownership interval.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.fsdp_interview.scripts.analyze_communication import csv_write  # noqa: E402
from experiments.fsdp_interview.scripts.analyze_gemm_trace import ScopeIndex  # noqa: E402
from experiments.fsdp_interview.scripts.parse_trace import duration_ms, intersection  # noqa: E402

MARK_RE = re.compile(r"shardtrain/(unit_pre_backward|unit_grads_ready)/U(\d+)/BWD/step(\d+)/")


def ownership_markers(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    query = """SELECT n.rowid AS nvtx_rowid,n.start,n.globalTid,
      COALESCE(n.text,s.value) AS label FROM NVTX_EVENTS n
      LEFT JOIN StringIds s ON n.textId=s.id
      WHERE COALESCE(n.text,s.value) LIKE 'shardtrain/unit_pre_backward/%'
      OR COALESCE(n.text,s.value) LIKE 'shardtrain/unit_grads_ready/%'
      ORDER BY n.start"""
    rows = [dict(row) for row in connection.execute(query)]
    connection.close()
    contexts, failures, pending = [], [], {}
    for row in rows:
        match = MARK_RE.match(row["label"])
        if not match:
            continue
        kind, unit, micro = match.group(1), int(match.group(2)), int(match.group(3))
        key = row["globalTid"], unit, micro
        if kind == "unit_pre_backward":
            if key in pending:
                failures.append({**row, "reason": "duplicate ownership start"})
            else:
                pending[key] = row
        elif key not in pending:
            failures.append({**row, "reason": "no same-thread ownership start"})
        else:
            first = pending.pop(key)
            contexts.append(
                {
                    "start": first["start"],
                    "end": row["start"],
                    "globalTid": row["globalTid"],
                    "pid": row["globalTid"] >> 24,
                    "unit_id": unit,
                    "runtime_microstep": micro,
                    "start_nvtx_rowid": first["nvtx_rowid"],
                    "end_nvtx_rowid": row["nvtx_rowid"],
                }
            )
    failures.extend({**row, "reason": "no same-thread ownership end"} for row in pending.values())
    return contexts, failures


def associate_current_compute(data, contexts):
    by_tid = defaultdict(list)
    for row in contexts:
        by_tid[row["globalTid"]].append(row)
    indices = {tid: ScopeIndex(rows) for tid, rows in by_tid.items()}
    # This mapping comes from exact collective correlation, not stream-number convention.
    compute_streams = defaultdict(set)
    for row in data["collectives"]:
        if row["trace_status"] == "MATCHED" and row["channel"] == "compute":
            pid = next(
                window["pid"] for window in data["step_windows"] if window["rank"] == row["rank"]
            )
            compute_streams[pid].update(row["stream_ids"])
    result = []
    for kernel in data["kernels"]:
        if kernel["is_nccl"] or kernel["stream_id"] not in compute_streams[kernel["pid"]]:
            continue
        candidates = {}
        for api in kernel["launch_apis"]:
            index = indices.get(api["globalTid"])
            if index is None:
                continue
            containing = index.containing(api["start"], api["end"])
            if not containing:
                continue
            minimum = min(row["end"] - row["start"] for row in containing)
            for row in containing:
                if row["end"] - row["start"] == minimum:
                    candidates[(row["pid"], row["unit_id"], row["runtime_microstep"])] = row
        if len(candidates) == 1:
            context = next(iter(candidates.values()))
            result.append(
                {
                    **kernel,
                    "unit_id": context["unit_id"],
                    "runtime_microstep": context["runtime_microstep"],
                    "ownership_start_nvtx_rowid": context["start_nvtx_rowid"],
                    "ownership_end_nvtx_rowid": context["end_nvtx_rowid"],
                    "ownership_source": "same-thread complete API "
                    "inside actual BWD ownership marks",
                }
            )
    return result


def pipeline_rows(data, compute):
    by_key = defaultdict(list)
    for row in data["collectives"]:
        if row["trace_status"] == "MATCHED" and row["phase"] == "BWD":
            by_key[(row["rank"], row["runtime_microstep"])].append(row)
    result = []

    def intervals(rows):
        return [(row["gpu_start_timestamp"], row["gpu_end_timestamp"]) for row in rows]

    for (rank, micro), rows in sorted(by_key.items()):
        pid = next(window["pid"] for window in data["step_windows"] if window["rank"] == rank)
        for ag in [row for row in rows if row["family"] == "fsdp_parameter_ag"]:
            target = ag["unit_id"]
            current, owner = target + 1, target + 2
            rs = [
                row
                for row in rows
                if row["unit_id"] == owner and row["family"] == "fsdp_gradient_rs"
            ]
            norm = [
                row
                for row in rows
                if row["unit_id"] == owner
                and row["family"] == "tp_norm_allreduce"
                and row["group_role"] == "norm_gradient_bucket"
            ]
            kernels = [
                row
                for row in compute
                if row["pid"] == pid
                and row["unit_id"] == current
                and row["runtime_microstep"] == micro
            ]
            if not rs or not norm or not kernels:
                continue
            a, r, n = intervals([ag]), intervals(rs), intervals(norm)
            c = [(row["gpu_start_ns"], row["gpu_end_ns"]) for row in kernels]
            result.append(
                {
                    "rank": rank,
                    "runtime_microstep": micro,
                    "gradient_owner_unit": owner,
                    "current_bwd_unit": current,
                    "target_prefetch_unit": target,
                    "ag_opid": ag["runtime_op_id"],
                    "rs_opids": [row["runtime_op_id"] for row in rs],
                    "norm_opids": [row["runtime_op_id"] for row in norm],
                    "current_compute_kernel_rowids": [row["kernel_rowid"] for row in kernels],
                    "ag_rs_norm_owner_ms": duration_ms(intersection(intersection(a, r), n)),
                    "ag_rs_compute_current_ms": duration_ms(intersection(intersection(a, r), c)),
                    "ag_norm_then_compute_ms": duration_ms(intersection(a, n + c)),
                    "ag_rs_norm_or_compute_ms": duration_ms(
                        intersection(intersection(a, r), n + c)
                    ),
                    "norm_owner_compute_current_ms": duration_ms(intersection(n, c)),
                    "ag_intervals_ns": a,
                    "rs_intervals_ns": r,
                    "norm_owner_intervals_ns": n,
                    "compute_current_intervals_ns": c,
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose a fresh BWD ownership analysis directory")
    data = json.loads(Path(args.analysis).read_text())
    contexts, failures = ownership_markers(data["source_sqlite"])
    compute = associate_current_compute(data, contexts)
    pipelines = pipeline_rows(data, compute)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "source_sqlite": data["source_sqlite"],
        "source_analysis": str(args.analysis),
        "current_bwd_compute_attribution": (
            "UNKNOWN_NO_OWNERSHIP_MARKERS"
            if not contexts
            else "MATCHED"
            if compute
            else "UNMATCHED_NO_EXACT_COMPUTE"
        ),
        "ownership_contexts": contexts,
        "ownership_failures": failures,
        "current_compute_kernels": compute,
        "pipelines": pipelines,
        "limits": [
            "Ownership uses actual CPU runtime marks plus exact correlated launch APIs, "
            "not nearest GPU timing.",
            "Non-NCCL compute-stream kernels can include elementwise/copies, not only GEMMs.",
            "Residency intersection is not measured saved time or SM/network progress.",
            "Norm owner U, current backward U-1, target prefetched parameter U-2.",
            "Absent ownership markers mean UNKNOWN current-unit compute, not zero compute overlap.",
        ],
    }
    for name in ("ownership_contexts", "ownership_failures", "pipelines"):
        csv_write(output / f"{name}.csv", report[name])
    (output / "backward_pipeline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_sqlite": data["source_sqlite"],
                "contexts": len(contexts),
                "failures": len(failures),
                "matched_compute_kernels": len(compute),
                "pipelines": len(pipelines),
            }
        )
    )


if __name__ == "__main__":
    main()
