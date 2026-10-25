"""Read-only local ATen shape -> NVTX -> launch API -> CUDA kernel association.

Only observed plain-local mm/addmm/bmm metadata is a trusted matrix shape.
Logical DTensor and FakeTensor planning rows never enter executed FLOP counts.
Each invocation retains every exactly associated kernel, including auxiliary
kernels. Invocation residency is not claimed to be pure GEMM hardware time.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.fsdp_interview.scripts.parse_trace import duration_ms, load_sqlite

TRUSTED_SOURCES = {"PLAIN_LOCAL_ATEN", "DTENSOR_DISPATCH_PLAIN_LOCAL_ATEN"}


def trusted_local(row):
    return (
        row.get("classification") == "LOCAL"
        and row.get("shape_source") in TRUSTED_SOURCES
        and all(
            row.get(side, {}).get("kind") in {"Tensor", "Parameter"} for side in ("left", "right")
        )
    )


class ScopeIndex:
    """An interval tree for exact containing ranges; arbitrary overlaps are legal."""

    def __init__(self, rows):
        self.center = None
        if not rows:
            return
        points = sorted((row["start"] + row["end"]) // 2 for row in rows)
        self.center = points[len(points) // 2]
        left, right, crossing = [], [], []
        for row in rows:
            target = (
                left
                if row["end"] < self.center
                else right
                if row["start"] > self.center
                else crossing
            )
            target.append(row)
        self.by_start = sorted(crossing, key=lambda row: row["start"])
        self.by_end = sorted(crossing, key=lambda row: row["end"], reverse=True)
        self.left, self.right = ScopeIndex(left), ScopeIndex(right)

    def containing(self, start, end):
        if self.center is None:
            return []
        result = []
        if start < self.center:
            for row in self.by_start:
                if row["start"] > start:
                    break
                if end <= row["end"]:
                    result.append(row)
            result.extend(self.left.containing(start, end))
        else:
            for row in self.by_end:
                if row["end"] < start:
                    break
                if end <= row["end"]:
                    result.append(row)
            result.extend(self.right.containing(start, end))
        return result


def read_shapes(paths):
    result = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    result.append(
                        {
                            **json.loads(line),
                            "source_shapes": str(path),
                            "source_line": line_number,
                        }
                    )
    return result


def correlate_shapes(data, shapes):
    """All associations require the actual same-thread enclosing GEMM NVTX."""
    scopes_by_label = defaultdict(list)
    scopes_by_tid = defaultdict(list)
    apis_by_tid = defaultdict(list)
    kernels_by_correlation = defaultdict(list)
    for scope in data["scopes"]:
        scopes_by_label[scope["label"]].append(scope)
        scopes_by_tid[scope["globalTid"]].append(scope)
    for api in data["apis"]:
        apis_by_tid[api["globalTid"]].append(api)
    api_starts = {}
    for tid, apis in apis_by_tid.items():
        apis.sort(key=lambda api: api["start"])
        api_starts[tid] = [api["start"] for api in apis]
    scope_indices = {tid: ScopeIndex(scopes) for tid, scopes in scopes_by_tid.items()}
    for kernel in data["kernels"]:
        kernels_by_correlation[(kernel["pid"], kernel.get("correlationId"))].append(kernel)
    trusted_labels = {
        row["nvtx_label"] for row in shapes if trusted_local(row) and row.get("nvtx_emitted")
    }
    label_counts = Counter(row.get("nvtx_label") for row in shapes)
    duplicate_labels = {label for label in trusted_labels if label_counts[label] != 1}
    result, kernel_rows = [], []
    for metadata in shapes:
        row = {
            **metadata,
            "source_sqlite": data["path"],
            "gpu_trace_status": "UNMATCHED",
            "kernel_rowids": [],
            "gpu_kernel_count": 0,
            "gpu_start_ns": None,
            "gpu_end_ns": None,
            "gpu_kernel_residency_union_ms": None,
            "operation_residency_TFLOP_s": None,
            "mathematical_gemm_flops": None,
            "stream_ids": [],
            "enclosing_source_nvtx": [],
            "reason_trace": None,
        }
        if not trusted_local(row):
            row.update(gpu_trace_status="EXCLUDED", reason_trace="not observed plain-local GEMM")
            result.append(row)
            continue
        if row.get("operation_status") != "RETURNED":
            row.update(gpu_trace_status="EXCLUDED", reason_trace="operation did not return")
            result.append(row)
            continue
        label = row.get("nvtx_label")
        if not label or not row.get("nvtx_emitted"):
            row["reason_trace"] = "no emitted CUDA invocation NVTX"
            result.append(row)
            continue
        if label in duplicate_labels:
            row["reason_trace"] = "duplicate per-invocation label metadata"
            result.append(row)
            continue
        scope_candidates = scopes_by_label.get(label, [])
        native_tid = row.get("cpu_native_thread_id")
        scope_candidates = [
            scope
            for scope in scope_candidates
            if native_tid is not None and scope["globalTid"] & ((1 << 24) - 1) == native_tid
        ]
        if len(scope_candidates) != 1:
            row["reason_trace"] = "missing/ambiguous exact NVTX on metadata native thread"
            result.append(row)
            continue
        scope = scope_candidates[0]
        kernels, launch_apis = {}, {}
        tid = scope["globalTid"]
        starts = api_starts.get(tid, [])
        first = bisect_left(starts, scope["start"])
        last = bisect_right(starts, scope["end"])
        for api in apis_by_tid[tid][first:last]:
            if not (scope["start"] <= api["start"] and api["end"] <= scope["end"]):
                continue
            # A kernel inside a nested trusted GEMM NVTX belongs to that nested
            # invocation only. This prevents double counting a composite call.
            inner = [
                candidate
                for candidate in scope_indices[tid].containing(api["start"], api["end"])
                if candidate["label"] in trusted_labels
                and candidate["label"] != label
                and scope["start"] <= candidate["start"]
                and candidate["end"] <= scope["end"]
                and candidate["start"] <= api["start"]
                and api["end"] <= candidate["end"]
            ]
            if inner:
                continue
            associated = kernels_by_correlation.get((api["pid"], api.get("correlationId")), [])
            for kernel in associated:
                if kernel.get("is_nccl"):
                    continue
                kernels[kernel["kernel_rowid"]] = kernel
                launch_apis[(api["api_table"], api["api_rowid"])] = api
        if not kernels:
            row["reason_trace"] = "no exact launch API/correlation non-NCCL kernel association"
            result.append(row)
            continue
        values = sorted(kernels.values(), key=lambda value: value["start"])
        residency = duration_ms([(value["start"], value["end"]) for value in values])
        dimensions = [row.get(key) for key in ("M", "N", "K", "batch_count")]
        flops = 2
        if any(not isinstance(value, int) or value <= 0 for value in dimensions):
            flops = None
        else:
            for value in dimensions:
                flops *= value
        ancestors = [
            ancestor
            for ancestor in scope_indices[tid].containing(scope["start"], scope["end"])
            if ancestor["label"] != label
            and ancestor["start"] <= scope["start"]
            and scope["end"] <= ancestor["end"]
        ]
        row.update(
            gpu_trace_status="MATCHED",
            kernel_rowids=[value["kernel_rowid"] for value in values],
            gpu_kernel_count=len(values),
            gpu_start_ns=min(value["start"] for value in values),
            gpu_end_ns=max(value["end"] for value in values),
            gpu_kernel_residency_union_ms=residency,
            mathematical_gemm_flops=flops,
            operation_residency_TFLOP_s=flops / residency / 1e9 if flops and residency else None,
            stream_ids=sorted({value.get("streamId") for value in values}),
            enclosing_source_nvtx=[
                {key: ancestor[key] for key in ("scope_id", "label", "start", "end")}
                for ancestor in sorted(ancestors, key=lambda value: value["end"] - value["start"])
            ],
            launch_apis=[
                {
                    key: api.get(key)
                    for key in (
                        "api_table",
                        "api_rowid",
                        "api_name",
                        "globalTid",
                        "start",
                        "end",
                        "correlationId",
                    )
                }
                for api in launch_apis.values()
            ],
        )
        for value in values:
            kernel_rows.append(
                {
                    "rank": row["rank"],
                    "cpu_ordinal": row["cpu_ordinal"],
                    "nvtx_label": label,
                    "operator": row["operator"],
                    "M": row["M"],
                    "N": row["N"],
                    "K": row["K"],
                    "batch_count": row["batch_count"],
                    "left_dtype": row.get("left", {}).get("dtype"),
                    "right_dtype": row.get("right", {}).get("dtype"),
                    "output_dtype": row.get("output", {}).get("dtype"),
                    "kernel_rowid": value["kernel_rowid"],
                    "kernel_name": value["kernel_name"],
                    "gpu_start_ns": value["start"],
                    "gpu_end_ns": value["end"],
                    "duration_ms": (value["end"] - value["start"]) / 1e6,
                    "stream_id": value.get("streamId"),
                    "device_id": value.get("deviceId"),
                    "correlation_id": value.get("correlationId"),
                    "source_sqlite": data["path"],
                    "scope": "kernel launched by observed local ATen call; may be auxiliary",
                }
            )
        result.append(row)
    return result, kernel_rows


def shape_summary(invocations):
    groups = defaultdict(list)
    for row in invocations:
        if row["gpu_trace_status"] == "MATCHED":
            key = (
                row["rank"],
                row["operator"],
                row["M"],
                row["N"],
                row["K"],
                row["batch_count"],
                row.get("left", {}).get("dtype"),
                row.get("right", {}).get("dtype"),
                row.get("output", {}).get("dtype"),
                row.get("phase"),
            )
            groups[key].append(row)
    result = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        rank, operator, m, n, k, batch, left, right, output, phase = key
        durations = [row["gpu_kernel_residency_union_ms"] for row in values]
        result.append(
            {
                "rank": rank,
                "operator": operator,
                "M": m,
                "N": n,
                "K": k,
                "batch_count": batch,
                "left_dtype": left,
                "right_dtype": right,
                "output_dtype": output,
                "caller_phase": phase,
                "invocations": len(values),
                "sum_operation_kernel_residency_ms": sum(durations),
                "mean_operation_kernel_residency_ms": statistics.mean(durations),
                "minimum_operation_kernel_residency_ms": min(durations),
                "maximum_operation_kernel_residency_ms": max(durations),
                "mathematical_gemm_flops": sum(
                    row["mathematical_gemm_flops"] or 0 for row in values
                ),
                "scope": "local observed dimensions; residency may include auxiliary kernels",
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
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--shapes", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = load_sqlite(args.sqlite)
    shapes = read_shapes(args.shapes)
    invocations, kernels = correlate_shapes(data, shapes)
    summary = shape_summary(invocations)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve previous GEMM analysis; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "source_sqlite": data["path"],
        "source_shapes": [str(Path(path)) for path in args.shapes],
        "invocations": invocations,
        "kernels": kernels,
        "shape_summary": summary,
        "counts": {
            status: sum(row["gpu_trace_status"] == status for row in invocations)
            for status in ("MATCHED", "UNMATCHED", "EXCLUDED")
        },
        "timebase": "NSYS API/NVTX/GPU report nanoseconds; CPU metadata monotonic time unused",
        "limits": [
            "Shape-diagnostic CPU instrumentation changes submission timing; not baseline MFU.",
            "M/N/K/B come from executed plain-local ATen operands, not CUDA grid or tile names.",
            "Kernel association covers all local operation kernels, including auxiliary kernels.",
            "Operation residency TFLOP/s is a mathematical estimate, "
            "not hardware counter throughput.",
            "Caller phase is preserved; FWD/BWD requires actual enclosing source NVTX evidence.",
            "No tensor references or GPU synchronization are introduced by this offline script.",
        ],
    }
    for name, rows in (
        ("invocations", invocations),
        ("kernels", kernels),
        ("shape_summary", summary),
    ):
        csv_write(output / f"gemm_{name}.csv", rows)
    (output / "gemm_trace_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"source_sqlite": data["path"], **report["counts"]}))


if __name__ == "__main__":
    main()
