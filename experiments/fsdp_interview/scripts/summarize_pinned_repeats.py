"""Aggregate actual fresh Q1/Q3 processes and explicitly supplied new traces.

Preserves per-process statistics, verifies raw returned hashes, and refuses
profiled timing as a performance baseline. No GPU or SSH operation is performed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.fsdp_interview.scripts.analyze_pinned import (  # noqa: E402
    analyze_benchmark,
    analyze_trace,
    compare_groups,
    csv_write,
    make_plots,
    markdown_report,
    sha256,
    summarize_samples,
    trace_argument,
    write_json,
)


def normalized_config(config):
    return {key: value for key, value in config.items() if key != "output_dir"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--trace-analysis", nargs="*", default=[])
    parser.add_argument("--return-manifest", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("preserve previous analysis; choose a fresh output directory")
    runs = [analyze_benchmark(path) for path in args.result_dirs]
    if not runs or any(run["status"] != "PASS" for run in runs):
        raise ValueError("all fresh benchmark directories must contain successful non-profile data")
    if any(
        normalized_config(run["config"]) != normalized_config(runs[0]["config"]) for run in runs
    ):
        raise ValueError("fresh process configurations differ")
    for field in ("tokens_sha256", "source_file_sha256", "torch_git_version", "device_properties"):
        if len({run["environment"][field] for run in runs}) != 1:
            raise ValueError(f"fresh process environment differs: {field}")
    pairs = [trace_argument(value) for value in args.trace_analysis]
    if len({label for label, _ in pairs}) != len(pairs):
        raise ValueError("trace labels must be unique")
    traces = [analyze_trace(label, path) for label, path in pairs]
    if any(trace["status"] != "PASS" for trace in traces):
        raise ValueError(
            "all supplied trace reports must contain exact measured copy/GEMM evidence"
        )
    # Verify only the explicitly requested current timing/trace directories.
    allowed = {path.resolve() for path in args.result_dirs}
    for trace in traces:
        allowed.add(Path(trace["source_sqlite"]).parent.resolve())
    verification, verified_paths, covered = [], set(), set()
    for manifest_path in args.return_manifest:
        manifest = json.loads(manifest_path.read_text())
        raw_root = manifest_path.parent.resolve()
        for entry in manifest["files"]:
            path = (raw_root / Path(entry["path"])).resolve()
            owner = next((root for root in allowed if root in path.parents), None)
            if owner is None or path in verified_paths:
                continue
            valid = path.is_file() and path.stat().st_size == entry["bytes"]
            actual = sha256(path) if valid else None
            valid = valid and actual == entry["sha256"]
            verification.append(
                {
                    **entry,
                    "source_manifest": str(manifest_path),
                    "local_path": str(path),
                    "actual_sha256": actual,
                    "valid": valid,
                }
            )
            verified_paths.add(path)
            covered.add(owner)
    if covered != allowed or not verification or not all(row["valid"] for row in verification):
        raise ValueError("raw returned payload manifest verification failed")
    rows, process_groups, cache, allocations, probes = [], [], [], [], []
    for index, (path, run) in enumerate(zip(args.result_dirs, runs, strict=True)):
        samples = json.loads((path / "samples.json").read_text())
        rows.extend({**sample, "fresh_process": index} for sample in samples)
        process_groups.extend({"fresh_process": index, **row} for row in run["groups"])
        cache.extend({"fresh_process": index, **row} for row in run["cache_observations"])
        allocations.extend({"fresh_process": index, **row} for row in run["allocations"])
        probes.extend({"fresh_process": index, **row} for row in run["token_probe"])
    groups = summarize_samples(rows)
    for row in groups:
        selected = [
            value
            for value in process_groups
            if value["case"] == row["case"] and value["buffer_bytes"] == row["buffer_bytes"]
        ]
        medians = [value["cpu_end_to_end_ms_p50"] for value in selected]
        row.update(
            fresh_processes=len(selected),
            per_process_completed_p50_ms=medians,
            fresh_process_p50_median_ms=statistics.median(medians),
            fresh_process_p50_min_ms=min(medians),
            fresh_process_p50_max_ms=max(medians),
        )
    observed_alloc = any(
        trace["exact_pin_setup_api_counts"].get("cudaHostAlloc", 0) > 0 for trace in traces
    )
    observed_register = any(
        trace["exact_pin_setup_api_counts"].get("cudaHostRegister", 0) > 0 for trace in traces
    )
    anomalies = [
        row
        for row in cache
        if isinstance(row.get("host_allocated_bytes"), int)
        and isinstance(row.get("host_reserved_bytes"), int)
        and row["host_allocated_bytes"] > row["host_reserved_bytes"]
    ]
    benchmark = {
        "status": "PASS",
        "result_dir": "; ".join(str(path) for path in args.result_dirs),
        "config": {**runs[0]["config"], "fresh_processes": len(runs)},
        "environment": runs[0]["environment"],
        "groups": groups,
        "comparisons": compare_groups(groups),
        "allocations": allocations,
        "cache_observations": cache,
        "token_probe": probes,
        "scope": f"{len(runs)} fresh processes; each has three within-process group repeats",
    }
    argv = [
        "python",
        str(Path(__file__)),
        "--result-dirs",
        *map(str, args.result_dirs),
        "--trace-analysis",
        *args.trace_analysis,
        "--return-manifest",
        *map(str, args.return_manifest),
        "--output-dir",
        str(args.output_dir),
    ]
    result = {
        "q1_status": "PASS" if traces else "PARTIAL",
        "q3_status": "PASS" if observed_alloc and observed_register else "PARTIAL",
        "benchmark": benchmark,
        "traces": traces,
        "fresh_runs": [
            {key: run[key] for key in ("result_dir", "config", "environment", "status")}
            for run in runs
        ],
        "per_process_groups": process_groups,
        "raw_manifest_verification": verification,
        "raw_manifest_verification_kind": "CLIENT_PULL_MANIFEST_SELF_CONSISTENCY",
        "remote_original_sha_verified": False,
        "analysis_argv": argv,
        "analysis_command": " ".join(argv),
        "analyzer_sha256": sha256(Path(__file__)),
        "count_validation": {
            "measured_byte_samples": sum(not row["warmup"] for row in rows),
            "all_byte_samples_data_equal": all(row["data_equal"] for row in rows),
            "token_probe_samples": len(probes),
            "all_token_probes_data_equal": all(row["data_equal"] for row in probes),
            "token_shapes": sorted({tuple(row["shape"]) for row in probes}),
            "manifest_files_verified": len(verification),
        },
        "host_counter_warning": {
            "status": "INVALID_AS_RESIDENT_COUNTER" if anomalies else "NO_CONTRADICTION_OBSERVED",
            "contradictory_group_endpoints": len(anomalies),
            "examples": anomalies[:5],
            "maximum_reported_allocated_bytes": max(
                (row["host_allocated_bytes"] for row in cache), default=None
            ),
            "maximum_host_reserved_bytes": max(
                (row["host_reserved_bytes"] for row in cache), default=None
            ),
            "maximum_case_rss_bytes": max((row["rss_bytes"] for row in cache), default=None),
            "interpretation": "Invalid residency counter; no leak or proven bug claim",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["plots"] = make_plots(args.output_dir, benchmark, traces)
    write_json(args.output_dir / "analysis.json", result)
    csv_write(args.output_dir / "measured_groups.csv", groups)
    csv_write(args.output_dir / "per_process_groups.csv", process_groups)
    csv_write(args.output_dir / "controlled_comparisons.csv", benchmark["comparisons"])
    csv_write(args.output_dir / "host_cache_observations.csv", cache)
    csv_write(args.output_dir / "token_probe_samples.csv", probes)
    csv_write(
        args.output_dir / "exact_nsys_timeline_summary.csv",
        [
            {"trace_label": trace["label"], **row}
            for trace in traces
            for row in trace["timeline_summary"]
        ],
    )
    report = markdown_report(result)
    report += (
        "\n## 三次独立进程的重复性与 Host counter 限制\n\n"
        f"已聚合 {len(runs)} 次 fresh process，各 3 个组内重复，每组每次 20 measured/5 warmup；"
        "每个 case/size 共180 measured。P50/P95 是 pooled measured samples，"
        "另表保存每进程中位数区间。\n\n"
        "host allocated_bytes.current 出现大于 host_reserved 的观测，不能作为实际 pinned驻留计数。"
        "保留原始值，只用 reserved、分配/释放API、slow-path count 和 RSS支持有限驻留判断。"
        "源码存在process_events_for_specific_size(-1)减量路径疑点，但不是本实验已经证明的bug。\n\n"
        "原始载荷哈希已按 Windows客户端下载manifest逐文件自一致验证；"
        "远端原始SHA跨端核验由最终归档清单另行完成。GPU snapshots不涵盖host pinned内存。\n"
        "图中 PID 为只读parser保留的 NSYS编码process namespace；"
        "不将其当作未经解码的操作系统PID。\n"
    )
    (args.output_dir / "Q1_Q3_pinned_memory.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {"q1": result["q1_status"], "q3": result["q3_status"], **result["count_validation"]}
        )
    )


if __name__ == "__main__":
    main()
