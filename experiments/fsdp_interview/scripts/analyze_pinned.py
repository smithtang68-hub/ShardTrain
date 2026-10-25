"""Analyze supplied Q1/Q3 evidence without turning missing runs into results.

The result directory must contain the new pinned benchmark's config, summary,
samples and environment. Trace arguments are LABEL=trace_analysis.json paths
from parse_trace.py. CUDA event brackets and exact NSYS activities stay separate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

CASE_NAMES = ("P1", "P2", "P3", "P4_serial", "P4", "P5", "P6", "P7")
LABEL_RE = re.compile(
    r"^q01q03/(?P<case>P\d(?:_serial)?)/bytes=(?P<bytes>\d+)"
    r"/repeat=(?P<repeat>\d+)/iteration=(?P<iteration>\d+)/warmup=(?P<warmup>[01])$"
)
SOURCE_COMMIT = "5811a8d7da873dd699ff6687092c225caffcf1bb"
SOURCE_BASE = f"https://github.com/pytorch/pytorch/blob/{SOURCE_COMMIT}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    identities = sorted({(row["experiment_id"], row["buffer_bytes"]) for row in rows})
    for case, size in identities:
        samples = [
            row
            for row in rows
            if row["experiment_id"] == case
            and row["buffer_bytes"] == size
            and row.get("warmup") is False
        ]
        if not samples:
            continue
        result = {
            "case": case,
            "buffer_bytes": size,
            "buffer_mib": size / 2**20,
            "samples": len(samples),
            "repetitions": sorted({row["repetition"] for row in samples}),
            "all_data_equal": all(row.get("data_equal") is True for row in samples),
            "all_sample_status_pass": all(row.get("status") == "PASS" for row in samples),
        }
        for field in (
            "cpu_pin_ms",
            "cpu_copy_submit_ms",
            "cpu_end_to_end_ms",
            "cuda_h2d_event_ms",
            "cuda_d2h_event_ms",
            "cuda_compute_event_ms",
            "cuda_total_event_ms",
            "temporal_overlap_ms",
            "h2d_event_gbps",
            "d2h_event_gbps",
            "end_to_end_gbps",
        ):
            values = [float(row[field]) for row in samples if numeric(row.get(field))]
            if values:
                result[f"{field}_p50"] = statistics.median(values)
                result[f"{field}_p95"] = percentile(values, 0.95)
                result[f"{field}_min"] = min(values)
                result[f"{field}_max"] = max(values)
        groups.append(result)
    return groups


def compare_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["case"], row["buffer_bytes"]): row for row in groups}
    comparisons = []
    for size in sorted({row["buffer_bytes"] for row in groups}):
        for baseline, candidate, question in (
            ("P4_serial", "P4", "same GEMMs plus H2D: completed serial vs independent streams"),
            ("P2", "P3", "same pinned H2D: blocking vs async submission"),
            ("P3", "P6", "same H2D: pre-pinned vs per-copy pin and CPU copy"),
        ):
            before, after = by_key.get((baseline, size)), by_key.get((candidate, size))
            if not before or not after:
                continue
            base_ms = before["cpu_end_to_end_ms_p50"]
            candidate_ms = after["cpu_end_to_end_ms_p50"]
            comparisons.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "buffer_bytes": size,
                    "question": question,
                    "baseline_completed_p50_ms": base_ms,
                    "candidate_completed_p50_ms": candidate_ms,
                    "candidate_minus_baseline_ms": candidate_ms - base_ms,
                    "baseline_over_candidate_completed_p50": base_ms / candidate_ms
                    if candidate_ms > 0
                    else None,
                    "baseline_copy_cpu_p50_ms": before["cpu_copy_submit_ms_p50"],
                    "candidate_copy_cpu_p50_ms": after["cpu_copy_submit_ms_p50"],
                    "candidate_cpu_pin_p50_ms": after["cpu_pin_ms_p50"],
                    "evidence": "non-profile CPU wall time after CUDA completion event",
                }
            )
    return comparisons


def analyze_benchmark(directory: Path) -> dict[str, Any]:
    required = ("summary.json", "samples.json", "environment.json", "config.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        return {
            "status": "NOT_RUN",
            "result_dir": str(directory),
            "missing_evidence": missing,
            "groups": [],
            "comparisons": [],
        }
    config = read_json(directory / "config.json")
    summary = read_json(directory / "summary.json")
    rows = read_json(directory / "samples.json")
    environment = read_json(directory / "environment.json")
    issues = []
    if config.get("profile") is not False:
        issues.append("Formal performance requires config.profile=false")
    if summary.get("status") != "PASS":
        issues.append("Benchmark summary is not PASS")
    if not isinstance(rows, list) or not rows:
        issues.append("No executed samples")
        rows = []
    groups = summarize_samples(rows)
    expected = {
        (case, size * 2**20)
        for case in config.get("cases", [])
        for size in config.get("sizes_mib", [])
    }
    present = {(row["case"], row["buffer_bytes"]) for row in groups}
    if present != expected or not expected:
        issues.append("Measured case/size coverage differs from benchmark configuration")
    count = config.get("iterations", 0) * config.get("repetitions", 0)
    for row in groups:
        if row["samples"] != count:
            issues.append(f"Incomplete sample count for {row['case']}/{row['buffer_bytes']}")
        if not row["all_data_equal"] or not row["all_sample_status_pass"]:
            issues.append(f"Failed data equality or sample status: {row['case']}")
    allocations_path, probe_path = directory / "allocations.json", directory / "token_probe.json"
    allocations = read_json(allocations_path) if allocations_path.is_file() else []
    token_probe = read_json(probe_path) if probe_path.is_file() else []
    if allocations and not all(
        row.get("already_pinned_same_pointer") is True for row in allocations
    ):
        issues.append("Already pinned identity check failed")
    if config.get("tokens_npy") and (
        not token_probe or not all(row.get("data_equal") is True for row in token_probe)
    ):
        issues.append("Configured real-token H2D probe missing or failed")
    if environment.get("torch_git_version") != SOURCE_COMMIT:
        issues.append("Installed Torch commit differs from the audited 2.9.1 commit")
    cache_observations = []
    for allocation in allocations:
        previous = allocation.get("host_stats_after_setup", {})
        for entry in allocation.get("case_host_stats", []):
            current = entry.get("host_stats", {})
            current_count, previous_count = (
                current.get("num_host_alloc"),
                previous.get("num_host_alloc"),
            )
            cache_observations.append(
                {
                    "buffer_bytes": allocation["buffer_bytes"],
                    "case": entry["case"],
                    "repetition": entry["repetition"],
                    "slow_path_host_alloc_count_delta": current_count - previous_count
                    if numeric(current_count) and numeric(previous_count)
                    else None,
                    "host_reserved_bytes": current.get("reserved_bytes.current"),
                    "host_allocated_bytes": current.get("allocated_bytes.current"),
                    "rss_bytes": entry.get("rss_bytes"),
                    "evidence": "allocator counters differenced at untimed group boundaries",
                }
            )
            previous = current
    return {
        "status": "PASS" if rows and groups and not issues else "PARTIAL",
        "result_dir": str(directory),
        "issues": issues,
        "config": config,
        "environment": environment,
        "groups": groups,
        "comparisons": compare_groups(groups),
        "allocations": allocations,
        "cache_observations": cache_observations,
        "token_probe": token_probe,
        "source_files": [
            {"path": str(directory / name), "sha256": sha256(directory / name)} for name in required
        ],
        "scope": "group repeats in one process; not three independent process launches",
    }


def _api_enclosing_labels(api: dict[str, Any], scopes: list[dict[str, Any]]) -> list[str]:
    labels = set(api.get("enclosing_nvtx_labels", []))
    labels.update(row["label"] for row in api.get("enclosing_nvtx", []))
    start, end = api.get("start"), api.get("end")
    if numeric(start) and numeric(end):
        for scope in scopes:
            if (
                scope.get("globalTid") == api.get("globalTid")
                and scope["start"] <= start
                and end <= scope["end"]
            ):
                labels.add(scope["label"])
    return sorted(labels)


def analyze_trace(label: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "label": label,
            "path": str(path),
            "status": "NOT_RUN",
            "reason": "No parser report",
        }
    parsed = read_json(path)
    print(
        json.dumps(
            {
                "analyzing_parser_report": str(path),
                "source_sqlite": parsed.get("source_sqlite"),
                "label": label,
            }
        ),
        flush=True,
    )
    rows = []
    for original in parsed.get("pinned_transfer_timelines", []):
        match = LABEL_RE.fullmatch(original.get("nvtx_label", ""))
        if not match:
            continue
        row = dict(original)
        row.update(
            case=match["case"],
            buffer_bytes=int(match["bytes"]),
            repetition=int(match["repeat"]),
            iteration=int(match["iteration"]),
            warmup=match["warmup"] == "1",
        )
        rows.append(row)
    measured = [row for row in rows if not row["warmup"]]
    p4 = [row for row in measured if row["case"] == "P4"]
    serial = [row for row in measured if row["case"] == "P4_serial"]
    matched = [
        row
        for row in p4 + serial
        if row.get("status") == "MATCHED"
        and row.get("matched_copy_count", 0) > 0
        and row.get("matched_compute_kernel_count", 0) > 0
    ]
    allocation_apis = []
    scopes = parsed.get("auxiliary_nvtx_ranges", [])
    # A report lacking host_allocation_apis proves neither allocation path.
    for api in parsed.get("host_allocation_apis", []):
        row = dict(api)
        row["api_name"] = api.get("api_name", api.get("name", ""))
        row["enclosing_nvtx_labels"] = _api_enclosing_labels(row, scopes)
        row["pin_setup_exactly_enclosed"] = any(
            item.startswith("q01q03/pinned_preallocate/bytes=")
            for item in row["enclosing_nvtx_labels"]
        )
        allocation_apis.append(row)
    names = ("cudaHostAlloc", "cudaHostRegister", "cudaFreeHost", "cudaHostUnregister")
    api_counts = {
        name: sum(name in row.get("api_name", "") for row in allocation_apis) for name in names
    }
    setup_counts = {
        name: sum(
            name in row.get("api_name", "") and row["pin_setup_exactly_enclosed"]
            for row in allocation_apis
        )
        for name in ("cudaHostAlloc", "cudaHostRegister")
    }
    environment_path = Path(parsed.get("source_sqlite", "")).parent / "environment.json"
    trace_allocator = (
        read_json(environment_path).get("allocator", {}) if environment_path.is_file() else {}
    )
    requested_register = trace_allocator.get("configured_or_source_default", {}).get(
        "pinned_use_cuda_host_register"
    )
    requested_path_status = "UNSPECIFIED"
    if requested_register is True:
        requested_path_status = (
            "REQUESTED_REGISTER_OBSERVED"
            if setup_counts["cudaHostRegister"]
            else "REQUESTED_REGISTER_BUT_HOSTALLOC_OBSERVED"
            if setup_counts["cudaHostAlloc"]
            else "REQUESTED_REGISTER_UNMATCHED"
        )
    timeline_summary = []
    for case in ("P4_serial", "P4"):
        for size in sorted({row["buffer_bytes"] for row in matched if row["case"] == case}):
            selected = [
                row for row in matched if row["case"] == case and row["buffer_bytes"] == size
            ]
            overlap = [float(row["transfer_gemm_temporal_overlap_ms"]) for row in selected]
            timeline_summary.append(
                {
                    "case": case,
                    "buffer_bytes": size,
                    "samples": len(selected),
                    "exact_transfer_gemm_overlap_p50_ms": statistics.median(overlap),
                    "exact_transfer_gemm_overlap_p95_ms": percentile(overlap, 0.95),
                    "copy_union_p50_ms": statistics.median(
                        row["copy_union_ms"] for row in selected
                    ),
                    "gemm_union_p50_ms": statistics.median(
                        row["gemm_union_ms"] for row in selected
                    ),
                    "evidence": "same PID/correlation API enclosed by exact per-iteration NVTX",
                }
            )
    issues = []
    if not measured:
        issues.append("No measured pinned iterations in supplied parser report")
    if len(matched) != len(p4 + serial) or not p4 or not serial:
        issues.append("P4/P4_serial exact copy+GEMM attribution incomplete")
    if "host_allocation_apis" not in parsed:
        issues.append("Parser report lacks allocation API evidence")
    return {
        "label": label,
        "path": str(path),
        "source_sqlite": parsed.get("source_sqlite"),
        "report_sha256": sha256(path),
        "status": "PASS" if measured and not issues else "PARTIAL",
        "issues": issues,
        "measured_iterations": len(measured),
        "matched_p4_iterations": len(matched),
        "pinned_timelines": measured,
        "timeline_summary": timeline_summary,
        "host_allocation_apis": allocation_apis,
        "allocation_api_counts": api_counts,
        "exact_pin_setup_api_counts": setup_counts,
        "allocation_path_status": "OBSERVED" if any(setup_counts.values()) else "UNMATCHED",
        "trace_allocator_environment": trace_allocator,
        "requested_path_status": requested_path_status,
        "timebase": parsed.get("timebase"),
    }


def trace_argument(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        if not label or not path:
            raise ValueError("trace arguments must be LABEL=parser_report.json")
        return label, Path(path)
    path = Path(value)
    return path.parent.name, path


def make_plots(
    output: Path, benchmark: dict[str, Any], traces: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    groups = benchmark.get("groups", [])
    # Only non-profile, successful, measured data can become performance plots.
    can_plot_benchmark = (
        benchmark.get("status") == "PASS"
        and benchmark.get("config", {}).get("profile") is False
        and bool(groups)
    )
    timeline_rows = [
        (trace, row)
        for trace in traces
        for row in trace.get("pinned_timelines", [])
        if row.get("status") == "MATCHED"
        and row.get("case") in {"P4", "P4_serial"}
        and row.get("matched_copy_count", 0) > 0
        and row.get("matched_compute_kernel_count", 0) > 0
    ]
    if not can_plot_benchmark and not timeline_rows:
        return [
            {"status": "NOT_RUN", "reason": "No valid measured benchmark or exact trace intervals"}
        ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [{"status": "NOT_RUN", "reason": "matplotlib unavailable"}]
    artifacts = []

    def save(figure, name: str, evidence: str):
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            figure.savefig(path, dpi=180)
            artifacts.append({"status": "GENERATED", "path": str(path), "evidence": evidence})
        plt.close(figure)

    if can_plot_benchmark:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for case in CASE_NAMES:
            selected = sorted(
                [row for row in groups if row["case"] == case], key=lambda row: row["buffer_bytes"]
            )
            if not selected:
                continue
            sizes = [row["buffer_mib"] for row in selected]
            axes[0].plot(
                sizes, [row["cpu_end_to_end_ms_p50"] for row in selected], marker="o", label=case
            )
            axes[0].fill_between(
                sizes,
                [row["cpu_end_to_end_ms_p50"] for row in selected],
                [row["cpu_end_to_end_ms_p95"] for row in selected],
                alpha=0.09,
            )
            direction = "d2h" if case == "P5" else "h2d"
            key = f"{direction}_event_gbps_p50"
            if all(key in row for row in selected):
                axes[1].plot(sizes, [row[key] for row in selected], marker="o", label=case)
            axes[2].plot(
                sizes, [row["cpu_copy_submit_ms_p50"] for row in selected], marker="o", label=case
            )
        for axis in axes:
            axis.set_xlabel("Transfer buffer size (MiB)")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Completed CPU wall latency (ms), P50/P95 shade")
        axes[1].set_ylabel("Event-bracket DMA bandwidth (GB/s), P50")
        axes[2].set_ylabel("CPU copy-call time (ms), P50")
        axes[0].legend(fontsize=8)
        figure.suptitle("Unprofiled measurements: P4 adds GEMMs; P7 is H2D+D2H roundtrip")
        save(figure, "pinned_transfer_vs_size", "non-profile measured samples.csv/json")
        serial = [
            row
            for row in benchmark["comparisons"]
            if row["baseline"] == "P4_serial" and row["candidate"] == "P4"
        ]
        if serial:
            figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            sizes = [row["buffer_bytes"] / 2**20 for row in serial]
            axis.plot(
                sizes, [row["baseline_completed_p50_ms"] for row in serial], "o-", label="P4_serial"
            )
            axis.plot(
                sizes, [row["candidate_completed_p50_ms"] for row in serial], "o-", label="P4"
            )
            axis.set_xlabel("H2D buffer size (MiB)")
            axis.set_ylabel("Completed CPU wall latency P50 (ms)")
            axis.set_title("Same independent GEMMs + H2D; end-to-end overlap comparison")
            axis.legend()
            axis.grid(alpha=0.25)
            save(figure, "pinned_serial_vs_overlap", "same-workload non-profile P4/P4_serial")
    representatives = []
    for trace in traces:
        source = [row for owner, row in timeline_rows if owner is trace]
        if not source:
            continue
        maximum_size = max(row["buffer_bytes"] for row in source)
        selected = []
        for case in ("P4_serial", "P4"):
            case_rows = [
                row for row in source if row["case"] == case and row["buffer_bytes"] == maximum_size
            ]
            if not case_rows:
                continue
            middle = statistics.median(row["copy_union_ms"] for row in case_rows)
            representative = min(case_rows, key=lambda row: abs(row["copy_union_ms"] - middle))
            selected.append(representative)
            representatives.append(
                {
                    "trace_label": trace["label"],
                    "source_sqlite": trace["source_sqlite"],
                    **representative,
                }
            )
        figure, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        tick_positions, tick_labels = [], []
        for index, row in enumerate(selected):
            intervals = row["copy_intervals_ns"] + row["compute_intervals_ns"]
            origin = min(start for start, _ in intervals)
            for offset, key, color, role in (
                (0, "copy_intervals_ns", "tab:blue", "memcpy"),
                (1, "compute_intervals_ns", "tab:orange", "GEMM"),
            ):
                y = index * 3 + offset
                values = [((start - origin) / 1e6, (end - start) / 1e6) for start, end in row[key]]
                axis.broken_barh(values, (y, 0.8), facecolors=color)
                tick_positions.append(y + 0.4)
                tick_labels.append(f"{row['case']} {role}; NSYS process {row['pid']}")
        axis.set_yticks(tick_positions, tick_labels)
        axis.set_xlabel("Actual activity interval relative to first GPU activity (ms)")
        axis.set_title(f"{trace['label']}: exact API correlation, {maximum_size / 2**20:g} MiB")
        axis.grid(axis="x", alpha=0.25)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace["label"])
        save(figure, f"pinned_nsys_timeline_{safe_label}", str(trace["source_sqlite"]))
    if representatives:
        write_json(output / "timeline_representatives.json", representatives)
    return artifacts


def number(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if numeric(value) else "未测"


def markdown_report(result: dict[str, Any]) -> str:
    benchmark, traces = result["benchmark"], result["traces"]
    lines = [
        "# Q1/Q3 Pinned Memory：本次实验分析",
        "",
        f"Q1 状态：**{result['q1_status']}**；Q3 状态：**{result['q3_status']}**。",
        "",
        f"正式结果目录：`{benchmark['result_dir']}`。仅使用明确传入的新结果和 trace parser 报告。",
        "",
        "## 实验技术与正确性",
        "",
        "实现使用 Tensor.pin_memory、empty(pin_memory=True)、Tensor.copy_、CUDA Stream/Event、",
        "record_stream、独立 FP16 torch.mm，以及有限的 host pool。未修改主 runtime。",
        "P4_serial/P4 的计算量相同；P7 是两方向往返，不能直接拿其端到端时间与单次 H2D 排名。",
        "",
    ]
    if benchmark.get("issues") or benchmark.get("missing_evidence"):
        lines.extend(["证据限制：", ""])
        for issue in benchmark.get("issues", []) + benchmark.get("missing_evidence", []):
            lines.append(f"- {issue}")
        lines.append("")
    if benchmark.get("environment"):
        env = benchmark["environment"]
        lines.extend(
            [
                f"实际 Torch：`{env.get('torch')}`，commit `{env.get('torch_git_version')}`。",
                f"CUDA build：`{env.get('cuda_build')}`；设备：`{env.get('device_properties')}`。",
                f"输入 token 文件 SHA256：`{env.get('tokens_sha256')}`。",
                "token byte sweep 重复原始字节达到指定容量；实际 token batch 另测。",
                "",
            ]
        )
    groups = benchmark.get("groups", [])
    if groups:
        lines.extend(
            [
                "## 非 profile 性能观察",
                "",
                "CPU copy-call 是提交耗时；完成时间包含 Event 等待。",
                "CUDA Event 是拷贝区间括号；精确 DMA 时长以 NSYS 活动为准。",
                "P50/P95 来自全部已执行 measured samples，预热排除。",
                "",
                "| Case | MiB | N | 完成 P50/P95 ms | Copy CPU ms | Pin CPU ms | GB/s H2D/D2H |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in groups:
            lines.append(
                f"| {row['case']} | {row['buffer_mib']:g} | {row['samples']} | "
                f"{number(row.get('cpu_end_to_end_ms_p50'))}/"
                f"{number(row.get('cpu_end_to_end_ms_p95'))} | "
                f"{number(row.get('cpu_copy_submit_ms_p50'))} | "
                f"{number(row.get('cpu_pin_ms_p50'))} | "
                f"{number(row.get('h2d_event_gbps_p50'))}/{number(row.get('d2h_event_gbps_p50'))} |"
            )
        lines.extend(["", "同工作量完成时间比较：", ""])
        for row in benchmark.get("comparisons", []):
            lines.append(
                f"- {row['buffer_bytes'] / 2**20:g} MiB，"
                f"{row['candidate']} 相对 {row['baseline']}："
                f"完成 P50 {number(row['candidate_completed_p50_ms'])} vs "
                f"{number(row['baseline_completed_p50_ms'])} ms；"
                f"候选减基线 {number(row['candidate_minus_baseline_ms'])} ms。"
            )
        lines.append("")
        if benchmark.get("allocations"):
            lines.extend(
                [
                    "独立计量的 setup 分配/CPU fill 与 RAM（不进入拷贝计时）：",
                    "",
                    "| MiB | Pinned allocation ms | CPU fill ms | Pinned pool MiB | RSS MiB |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for row in benchmark["allocations"]:
                rss = row.get("rss_bytes_after_setup")
                lines.append(
                    f"| {row['buffer_bytes'] / 2**20:g} | "
                    f"{number(row.get('pinned_preallocation_cpu_ms'))} | "
                    f"{number(row.get('pinned_initial_fill_cpu_ms'))} | "
                    f"{number(row.get('explicit_pool_pinned_bytes', 0) / 2**20)} | "
                    f"{number(rss / 2**20 if numeric(rss) else None)} |"
                )
            lines.append("")
        p6_cache = [row for row in benchmark.get("cache_observations", []) if row["case"] == "P6"]
        for row in p6_cache:
            lines.append(
                f"- P6 {row['buffer_bytes'] / 2**20:g} MiB，repeat {row['repetition']}："
                f"本组 warmup+measured 的 allocator slow-path host allocation 增量 "
                f"{number(row['slow_path_host_alloc_count_delta'], 0)} 次。"
            )
        if p6_cache:
            lines.extend(
                [
                    "",
                    "该计数来自 allocator stats，结合本次分配 API trace 判断 cache reuse。",
                    "第一分配不是独立 OS 冷启动测量；RSS 包含 pageable 和其他 CPU 内存。",
                    "",
                ]
            )
    else:
        lines.extend(["没有实际 benchmark 数据；不生成性能曲线、不填写速度或 PASS。", ""])
    lines.extend(["## Trace：实际区间与分配 API", ""])
    if not traces:
        lines.extend(["未提供 trace parser 报告；尚不能确认 DMA/GEMM overlap 或实际分配 API。", ""])
    for trace in traces:
        lines.extend(
            [
                f"### {trace['label']}",
                "",
                f"parser：`{trace['path']}`；SQLite：`{trace.get('source_sqlite', '未提供')}`。",
                f"状态：`{trace['status']}`；"
                f"实际 allocator 路径：`{trace.get('allocation_path_status', '未测')}`。",
                f"意图配置核对：`{trace.get('requested_path_status', 'UNSPECIFIED')}`。"
                "目录名称与环境变量不代替实际 CUDA API 证据。",
                "",
            ]
        )
        for issue in trace.get("issues", []):
            lines.append(f"- {issue}")
        for row in trace.get("timeline_summary", []):
            lines.append(
                f"- {row['case']}，{row['buffer_bytes'] / 2**20:g} MiB，N={row['samples']}："
                f"精确 memcpy/GEMM temporal overlap P50/P95 "
                f"{number(row['exact_transfer_gemm_overlap_p50_ms'])}/"
                f"{number(row['exact_transfer_gemm_overlap_p95_ms'])} ms；"
                f"copy union P50 {number(row['copy_union_p50_ms'])} ms，"
                f"GEMM union P50 {number(row['gemm_union_p50_ms'])} ms。"
            )
        for name, count in trace.get("allocation_api_counts", {}).items():
            enclosed = trace.get("exact_pin_setup_api_counts", {}).get(name)
            lines.append(
                f"- {name}：capture 中 {count} 次；"
                f"明确包含于 pinned_preallocate NVTX："
                f"{enclosed if enclosed is not None else '不适用'}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 原因证据与缓存条件",
            "",
            "pin_memory 对普通 CPU Tensor 创建 pinned storage 后复制。",
            "already-pinned 返回自身。",
            f"[Memory.cpp]({SOURCE_BASE}/aten/src/ATen/native/Memory.cpp#L39-L66)。",
            "",
            "allocator 先处理 pending events，再把请求向上取整为 2 的幂、查对应 free list。",
            "缓存命中需要该大小桶存在可用块；仍有未完成 async stream 使用的块不能立即复用。",
            "record_event 登记流使用，释放时记录完成 Event。",
            "query 就绪且关联 event_count 清零后才回 free list。",
            f"[CachingHostAllocator.h]({SOURCE_BASE}/aten/src/ATen/core/CachingHostAllocator.h#L180-L298)。",
            "",
            "缓存未命中默认 cudaHostAlloc。",
            "register 配置启用时 malloc、页面预触碰、cudaHostRegister。",
            "P6 多次调用 pin_memory，可能只首次触发昂贵分配，随后是 CPU copy 和缓存处理。",
            "不能将每次 pin_memory 等同为每次 cudaHostRegister/unregister。",
            f"[CachingHostAllocator.cpp]({SOURCE_BASE}/aten/src/ATen/cuda/CachingHostAllocator.cpp#L69-L102)。",
            "",
            "CUDA allocator 的 per-device EventPool 复用 disable-timing Event。",
            "本实验 enable-timing Event 是另一类。",
            f"[EventPool]({SOURCE_BASE}/aten/src/ATen/cuda/CachingHostAllocator.cpp#L14-L62)。",
            "",
            "脚本 capture start 位于首个 pinned_preallocate 之前；缓存未命中的 allocation API",
            "有机会进入本次 capture；分配 API 缺失时只报 UNMATCHED，不把配置值当成 API 实证。",
            "HostAllocator stats、指针复用和 API 次数共同支持缓存解释。",
            "单靠缺少 API 不能证明全部来源。",
            "",
            "## 局限性与面试表述",
            "",
            "temporal overlap 是实际区间交集，不能直接当作节省时间。",
            "端到端收益看 P4_serial/P4 的非 profile 控制。",
            "P4 GEMM 不使用此次传入的数据，模拟前一批计算与下一批 H2D 的独立关系。",
            "DataLoader token probe 使用 num_workers=0，不衡量多 worker/后台 pin-thread 的吞吐。",
            "GPU snapshot 不包含 host pinned storage；GPU内存、HostAllocator stats、RSS 分开报告。",
            "原始 samples、command/config/environment、NSYS/SQLite、snapshots 一起保留。",
            "",
            "可准确讲述：异步 H2D 减少 CPU 提交等待、独立流使 DMA/GEMM 可以重叠；是否更快必须引用",
            "本次同工作量完成时间与实际 trace。",
            "CPU读取 D2H 数据、源 pinned buffer 改写/复用必须等待完成。",
            "不根据源码预设本机速度，不根据 Event bracket 宣称精确硬件 overlap。",
            "",
            "## 复现与产物",
            "",
            f"分析命令：`{result['analysis_command']}`。",
            "",
        ]
    )
    for artifact in result.get("plots", []):
        if artifact.get("path"):
            lines.append(f"- `{artifact['path']}`：{artifact['evidence']}")
        else:
            lines.append(f"- 绘图 `{artifact['status']}`：{artifact.get('reason')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path, required=True, help="Unprofiled pinned benchmark directory"
    )
    parser.add_argument("--trace-analysis", nargs="*", default=[], metavar="LABEL=REPORT.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "analysis.json").exists():
        raise FileExistsError(
            "choose a fresh analysis output directory to preserve previous evidence"
        )
    benchmark = analyze_benchmark(args.result_dir)
    pairs = [trace_argument(value) for value in args.trace_analysis]
    if len({label for label, _ in pairs}) != len(pairs):
        raise ValueError("trace labels must be unique")
    traces = [analyze_trace(label, path) for label, path in pairs]
    confirmed_overlap = any(
        trace.get("status") == "PASS"
        and any(row["case"] == "P4" for row in trace.get("timeline_summary", []))
        for trace in traces
    )
    observed_alloc = any(
        trace.get("exact_pin_setup_api_counts", {}).get("cudaHostAlloc", 0) > 0 for trace in traces
    )
    observed_register = any(
        trace.get("exact_pin_setup_api_counts", {}).get("cudaHostRegister", 0) > 0
        for trace in traces
    )
    q1_status = (
        "PASS"
        if benchmark["status"] == "PASS" and confirmed_overlap
        else "PARTIAL"
        if benchmark.get("groups")
        else "NOT_RUN"
    )
    q3_status = (
        "PASS"
        if benchmark["status"] == "PASS" and observed_alloc and observed_register
        else "PARTIAL"
        if benchmark.get("groups") or traces
        else "NOT_RUN"
    )
    command = ["python", str(Path(__file__)), "--result-dir", str(args.result_dir)]
    if args.trace_analysis:
        command += ["--trace-analysis", *args.trace_analysis]
    command += ["--output-dir", str(destination)]
    result = {
        "q1_status": q1_status,
        "q3_status": q3_status,
        "benchmark": benchmark,
        "traces": traces,
        "analysis_command": " ".join(command),
        "analysis_argv": command,
        "analyzer_sha256": sha256(Path(__file__)),
        "source_audit_commit": SOURCE_COMMIT,
        "missing_data_policy": "No fabricated values or performance plots without valid samples",
    }
    result["plots"] = make_plots(destination, benchmark, traces)
    write_json(destination / "analysis.json", result)
    csv_write(destination / "measured_groups.csv", benchmark.get("groups", []))
    csv_write(destination / "controlled_comparisons.csv", benchmark.get("comparisons", []))
    csv_write(destination / "host_cache_observations.csv", benchmark.get("cache_observations", []))
    csv_write(
        destination / "exact_nsys_timeline_summary.csv",
        [
            {"trace_label": trace["label"], **row}
            for trace in traces
            for row in trace.get("timeline_summary", [])
        ],
    )
    (destination / "Q1_Q3_pinned_memory.md").write_text(markdown_report(result), encoding="utf-8")
    print(
        json.dumps({"q1_status": q1_status, "q3_status": q3_status, "output_dir": str(destination)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
