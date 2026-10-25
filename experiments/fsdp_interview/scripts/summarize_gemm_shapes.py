"""Summarize only explicitly supplied exact GEMM trace associations.

FWD/BWD uses actual enclosing source NVTX. Shape, ordinal and caller phase
never determine that classification. Diagnostic times are not benchmark MFU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for value in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(value)
    return digest.hexdigest()


def source_phase(row):
    labels = [scope["label"] for scope in row.get("enclosing_source_nvtx", [])]
    forward = [label for label in labels if label == "shardtrain/qwen/model_forward"]
    backward = [
        label
        for label in labels
        if label.startswith("autograd::engine::evaluate_function:")
        and re.search(r"\b[A-Za-z_][A-Za-z_0-9]*Backward\d*\b", label)
    ]
    if forward and not backward:
        return "FWD", forward
    if backward and not forward:
        return "BWD", backward
    return "UNKNOWN", forward + backward


def kernel_category(name):
    # This is explicitly a name-based category, not a hardware counter claim.
    if "splitKreduce_kernel" in name:
        return "AUXILIARY_SPLITK_REDUCE_NAMED"
    if "gemm" in name.lower():
        return "GEMM_NAMED"
    return "UNKNOWN"


def union_ms(rows):
    intervals = sorted((row["gpu_start_ns"], row["gpu_end_ns"]) for row in rows)
    total, left, right = 0, None, None
    for start, end in intervals:
        if left is None:
            left, right = start, end
        elif start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return (total + (right - left if left is not None else 0)) / 1e6


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
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
    parser.add_argument("--analysis", nargs="+", required=True, metavar="LABEL=EXACT_REPORT.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("preserve old analysis; use a fresh directory")
    result, shapes, phases, kernel_names, records = [], [], [], [], []
    labels = set()
    for specification in args.analysis:
        label, value = specification.split("=", 1)
        if label in labels:
            raise ValueError("analysis labels must be unique")
        labels.add(label)
        path = Path(value)
        report = json.loads(path.read_text(encoding="utf-8"))
        matched = [row for row in report["invocations"] if row["gpu_trace_status"] == "MATCHED"]
        if not matched:
            raise ValueError("no actual exactly matched operations; do not create plots or PASS")
        kernels_by_op = defaultdict(list)
        for row in report["kernels"]:
            kernels_by_op[(row["rank"], row["cpu_ordinal"])].append(row)
        grouped = defaultdict(list)
        phase_groups = defaultdict(list)
        for row in matched:
            phase, evidence = source_phase(row)
            kernels = kernels_by_op[(row["rank"], row["cpu_ordinal"])]
            if len(kernels) != row["gpu_kernel_count"] or sorted(
                k["kernel_rowid"] for k in kernels
            ) != sorted(row["kernel_rowids"]):
                raise ValueError("kernel invocation association counts disagree")
            record = {
                "trace_label": label,
                "rank": row["rank"],
                "cpu_ordinal": row["cpu_ordinal"],
                "operator": row["operator"],
                "M": row["M"],
                "N": row["N"],
                "K": row["K"],
                "batch_count": row["batch_count"],
                "left_dtype": row["left"]["dtype"],
                "right_dtype": row["right"]["dtype"],
                "output_dtype": row["output"]["dtype"],
                "verified_source_phase": phase,
                "phase_nvtx_evidence": evidence,
                "invocation_nvtx": row["nvtx_label"],
                "operation_kernel_union_ms": row["gpu_kernel_residency_union_ms"],
                "kernel_count": len(kernels),
                "stream_ids": row["stream_ids"],
                "gemm_named_union_ms": union_ms(
                    [k for k in kernels if kernel_category(k["kernel_name"]) == "GEMM_NAMED"]
                ),
                "splitk_reduce_named_union_ms": union_ms(
                    [
                        k
                        for k in kernels
                        if kernel_category(k["kernel_name"]) == "AUXILIARY_SPLITK_REDUCE_NAMED"
                    ]
                ),
                "unknown_named_union_ms": union_ms(
                    [k for k in kernels if kernel_category(k["kernel_name"]) == "UNKNOWN"]
                ),
            }
            records.append(record)
            key = tuple(
                record[name]
                for name in (
                    "rank",
                    "verified_source_phase",
                    "operator",
                    "M",
                    "N",
                    "K",
                    "batch_count",
                    "left_dtype",
                    "right_dtype",
                    "output_dtype",
                )
            )
            grouped[key].append(record)
            phase_groups[(row["rank"], phase, row["left"]["dtype"])].append(record)
        for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
            row = dict(
                zip(
                    (
                        "rank",
                        "verified_source_phase",
                        "operator",
                        "M",
                        "N",
                        "K",
                        "batch_count",
                        "left_dtype",
                        "right_dtype",
                        "output_dtype",
                    ),
                    key,
                    strict=True,
                )
            )
            durations = [value["operation_kernel_union_ms"] for value in values]
            shapes.append(
                {
                    "trace_label": label,
                    **row,
                    "invocations": len(values),
                    "operation_kernel_union_p50_ms": percentile(durations, 0.5),
                    "operation_kernel_union_p95_ms": percentile(durations, 0.95),
                    "minimum_ms": min(durations),
                    "maximum_ms": max(durations),
                    "sum_operation_kernel_union_ms": sum(durations),
                    "gemm_named_sum_ms": sum(v["gemm_named_union_ms"] for v in values),
                    "splitk_reduce_named_sum_ms": sum(
                        v["splitk_reduce_named_union_ms"] for v in values
                    ),
                    "unknown_named_sum_ms": sum(v["unknown_named_union_ms"] for v in values),
                }
            )
        for (rank, phase, dtype), values in sorted(phase_groups.items()):
            phases.append(
                {
                    "trace_label": label,
                    "rank": rank,
                    "verified_source_phase": phase,
                    "dtype": dtype,
                    "invocations": len(values),
                    "sum_operation_kernel_union_ms": sum(
                        v["operation_kernel_union_ms"] for v in values
                    ),
                }
            )
        names = defaultdict(list)
        for row in report["kernels"]:
            names[(row["rank"], kernel_category(row["kernel_name"]), row["kernel_name"])].append(
                row
            )
        for (rank, category, name), values in sorted(names.items()):
            kernel_names.append(
                {
                    "trace_label": label,
                    "rank": rank,
                    "name_category": category,
                    "kernel_name": name,
                    "count": len(values),
                    "duration_sum_ms": sum(v["duration_ms"] for v in values),
                    "duration_p50_ms": percentile([v["duration_ms"] for v in values], 0.5),
                    "duration_p95_ms": percentile([v["duration_ms"] for v in values], 0.95),
                }
            )
        result.append(
            {
                "trace_label": label,
                "analysis_path": str(path),
                "analysis_sha256": sha256(path),
                "source_sqlite": report["source_sqlite"],
                "counts": report["counts"],
                "exact_kernel_count": len(report["kernels"]),
                "dtype_counts": dict(Counter(row["left"]["dtype"] for row in matched)),
                "verified_phase_counts": dict(Counter(source_phase(row)[0] for row in matched)),
                "performance_claim_allowed": False,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("shapes_by_verified_phase", shapes),
        ("phase_dtype_summary", phases),
        ("kernel_names", kernel_names),
        ("invocations_with_verified_phase", records),
    ):
        write_csv(args.output_dir / f"{name}.csv", rows)
    payload = {
        "traces": result,
        "shape_groups": shapes,
        "phase_dtype_summary": phases,
        "kernel_names": kernel_names,
        "limits": [
            "Only actual source NVTX determines FWD/BWD; ambiguous or missing remains UNKNOWN.",
            "Names distinguish visible GEMM and splitK reduction kernels, "
            "not hardware tensor-core counters.",
            "Operation residency may include auxiliary kernels; "
            "never count one mathematical GEMM once per kernel.",
            "Instrumented traces are diagnostics, never steady benchmark MFU or speed ranking.",
        ],
        "analyzer_sha256": sha256(Path(__file__)),
    }
    (args.output_dir / "gemm_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 实际本地 GEMM 形状与 kernel 关联",
        "",
        "本报告仅分析明确传入的诊断结果；不用于正式 MFU 或框架速度排名。",
        "",
    ]
    for trace in result:
        lines.extend(
            [
                f"## {trace['trace_label']}",
                "",
                f"原始 SQLite：`{trace['source_sqlite']}`。",
                f"精确对应状态：`{trace['counts']}`；kernel 数 {trace['exact_kernel_count']}。",
                f"实际操作 dtype：`{trace['dtype_counts']}`；"
                f"由真实 enclosing NVTX 判断阶段：`{trace['verified_phase_counts']}`。",
                "",
                "| Rank | 阶段 | ATen | M/N/K | B | Dtype | N | "
                "Operation union P50/P95 ms | SplitK reduce sum ms |",
                "|---:|---|---|---|---:|---|---:|---:|---:|",
            ]
        )
        for row in shapes:
            if row["trace_label"] == trace["trace_label"]:
                lines.append(
                    f"| {row['rank']} | {row['verified_source_phase']} | {row['operator']} | "
                    f"{row['M']}/{row['N']}/{row['K']} | {row['batch_count']} | "
                    f"{row['left_dtype']} | {row['invocations']} | "
                    f"{row['operation_kernel_union_p50_ms']:.6f}/"
                    f"{row['operation_kernel_union_p95_ms']:.6f} | "
                    f"{row['splitk_reduce_named_sum_ms']:.6f} |"
                )
        lines.append("")
    lines.extend(
        [
            "## 判读边界",
            "",
            *[f"- {limit}" for limit in payload["limits"]],
            "",
            "`GEMM_NAMED` 和 `AUXILIARY_SPLITK_REDUCE_NAMED` "
            "是保留原kernel名称的可审计分类；其他名字全部UNKNOWN。"
            "完整invocation/API/kernel rowid关系仍在输入报告；精度、形状不根据CUDA grid/tile猜测。",
        ]
    )
    (args.output_dir / "gemm_local_shapes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
