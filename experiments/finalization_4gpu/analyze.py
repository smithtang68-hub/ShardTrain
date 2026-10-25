"""Read-only raw-result analysis; profile samples never enter formal throughput."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from .cli import MATRIX, load, save, sha

OLD = Path(__file__).resolve().parents[2] / "experiments/fsdp_interview/scripts"
sys.path.insert(0, str(OLD))
from analyze_communication import timing_aggregation, training_run  # noqa: E402


def _csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            for row in rows
        )


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=False)
    summaries, steps, checks, raw_index, initial = [], [], [], [], {}
    expected_inputs, source_hash, workload = {}, None, None
    for case, (engine, mapping) in MATRIX.items():
        for repeat in range(3):
            directory = root / "benchmark" / f"benchmark_{case}_r{repeat}"
            if not directory.is_dir():
                checks.append(
                    {
                        "case": case,
                        "repeat": repeat,
                        "status": "NOT_RUN",
                        "reason": "Formal run directory absent",
                    }
                )
                continue
            summary, records = training_run(directory)
            if (
                summary["profile"]
                or summary["steps"] != 30
                or summary["status"] != "PASS"
                or not summary["performance_claim_allowed"]
            ):
                raise ValueError("Profiled, incomplete or failed run cannot become a formal result")
            payloads = [load(directory / f"benchmark.rank{rank}.json") for rank in range(4)]
            for rank, payload in enumerate(payloads):
                if (
                    payload["warmup"] != 10
                    or payload["implementation"] != engine
                    or payload["policy"] != "C0"
                    or payload["guarded_baseline"]
                    or payload["event_recorder"]
                ):
                    raise ValueError("Formal C0_fast/native-matched measurement contract differs")
                if (
                    payload["seq_len"] != 512
                    or payload["microbatch"] != 4
                    or payload["accumulation_steps"] != 2
                ):
                    raise ValueError("Formal workload shape differs")
                fingerprint = [
                    (
                        row["optimizer_step"],
                        row["input_block_start"],
                        row["input_batch_sha256"],
                        row["valid_tokens"],
                        row["unique_input_tokens"],
                    )
                    for row in payload["warmup_steps"] + payload["steps"]
                ]
                if any(row[3:] != (8176, 8192) for row in fingerprint):
                    raise ValueError("Actual mask-derived valid label/input counts differ")
                if rank in expected_inputs and expected_inputs[rank] != fingerprint:
                    raise ValueError(
                        "Logical-rank input order/hash differs across mapping or engine"
                    )
                expected_inputs[rank] = fingerprint
                audit_path = directory / f"placement.rank{rank}.json"
                audit = load(audit_path)
                if (
                    audit["status"] != "PASS"
                    or audit["mapping_id"] != mapping
                    or not audit["initial_local_state"]
                ):
                    raise ValueError("Placement or initial-state evidence is missing")
                key = (engine, rank)
                state = audit["initial_local_state"]["sha256"]
                if key in initial and initial[key] != state:
                    raise ValueError("Same engine/logical-rank initial shard fingerprint changed")
                initial[key] = state
                current_source = payload.get("source_manifest_sha256")
                if not current_source or source_hash is not None and current_source != source_hash:
                    raise ValueError("Tested snapshots differ or source provenance is missing")
                source_hash = current_source
                if workload is not None and payload["workload_hash"] != workload:
                    raise ValueError("Matched workload fingerprint changed")
                workload = payload["workload_hash"]
                for path in (directory / f"benchmark.rank{rank}.json", audit_path):
                    raw_index.append(
                        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
                    )
            summary.update(series=case, mapping_id=mapping, repeat=repeat)
            for row in records:
                row["series"] = case
            summaries.append(summary)
            steps.extend(records)
            checks.append({"case": case, "repeat": repeat, "status": "PASS"})
    aggregate = timing_aggregation(summaries, steps)
    by_case = {row["series"]: row for row in aggregate}
    comparisons = []
    for left, right, meaning in [
        ("S0", "S1", "self placement"),
        ("N0", "N1", "native placement"),
        ("N0", "S0", "engine M0"),
        ("N1", "S1", "engine M1"),
    ]:
        if left not in by_case or right not in by_case:
            continue
        baseline, candidate = by_case[left], by_case[right]
        comparisons.append(
            {
                "comparison": meaning,
                "baseline": left,
                "candidate": right,
                "throughput_increase_fraction": candidate["training_tokens_per_s"]
                / baseline["training_tokens_per_s"]
                - 1,
                "mean_time_decrease_fraction": 1
                - candidate["mean_step_ms"] / baseline["mean_step_ms"],
                "caution": "Three independent runs; 90 steps are not 90 independent experiments",
            }
        )
    profile_overheads = []
    for case in ("S0", "S1"):
        directory = root / "profiles" / f"profile_{case}_r0"
        if directory.is_dir() and case in by_case:
            summary, _ = training_run(directory)
            profile_overheads.append(
                {
                    "case": case,
                    "profile_mean_step_ms": summary["mean_step_ms"],
                    "unprofiled_mean_step_ms": by_case[case]["mean_step_ms"],
                    "ratio": summary["mean_step_ms"] / by_case[case]["mean_step_ms"],
                    "scope": "Mechanism evidence only; no allocator history or all-ATen NVTX; "
                    "snapshots states-only",
                }
            )
    nccl = []
    for path in sorted((root / "stages").glob("nccl_reference_*.json")):
        value = load(path)
        for row in value.get("metrics", []):
            nccl.append(
                {
                    "stage": value["name"],
                    **{
                        key: item
                        for key, item in value["stage"].items()
                        if key not in {"argv", "binary"}
                    },
                    **row,
                    "status": value["status"],
                }
            )
    complete = len(summaries) == 12 and all(row["status"] == "PASS" for row in checks)
    payload = {
        "status": "PASS" if complete else "NOT_RUN",
        "formal_runs": len(summaries),
        "formal_global_steps": len(steps),
        "source_manifest_sha256": source_hash,
        "input_assignment_check": "PASS" if summaries else "NOT_RUN",
        "same_engine_initial_shard_check": "PASS" if summaries else "NOT_RUN",
        "checks": checks,
        "runs": summaries,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "profile_overheads": profile_overheads,
        "raw_index": raw_index,
        "analysis_source_sha256": sha(__file__),
        "limitations": [
            "NCCL residency is not pure transfer latency",
            "Temporal overlap is not saved time",
            "No cross-mapping engine advantage claim",
        ],
    }
    save(output / "summary.json", payload)
    for filename, rows in [
        ("runs.csv", summaries),
        ("aggregate.csv", aggregate),
        ("comparisons.csv", comparisons),
        ("steps.csv", steps),
        ("nccl_reference.csv", nccl),
        ("profile_overhead.csv", profile_overheads),
    ]:
        _csv(output / filename, rows)
    if aggregate:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = [row["series"] for row in aggregate]
            figure, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].bar(
                labels,
                [row["mean_step_ms"] for row in aggregate],
                color=["#225ea8" if label.startswith("S") else "#f28e2b" for label in labels],
            )
            for index, label in enumerate(labels):
                points = [row["mean_step_ms"] for row in summaries if row["series"] == label]
                axes[0].scatter([index] * len(points), points, color="black", s=20, zorder=3)
            axes[0].set_ylabel("Mean optimizer step (ms); dots = independent runs")
            for index, key in enumerate(("peak_allocated_GiB", "peak_reserved_GiB")):
                axes[1].bar(
                    [i + index * 0.35 for i in range(len(labels))],
                    [row[key] for row in aggregate],
                    width=0.35,
                    label=key,
                )
            axes[1].set_xticks([i + 0.175 for i in range(len(labels))], labels)
            axes[1].set_ylabel("Worst-rank GiB")
            axes[1].legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(output / "training_and_memory.png", dpi=160)
            plt.close(figure)
        except ImportError as error:
            save(
                output / "plots_unavailable.json",
                {"status": "BLOCKED_ENVIRONMENT", "reason": repr(error)},
            )
    return {
        "status": payload["status"],
        "formal_runs": len(summaries),
        "summary": str(output / "summary.json"),
    }
