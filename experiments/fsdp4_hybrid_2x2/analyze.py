"""Reuse existing per-rank timing validator and statistics; four separate contrasts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .cli import audit_training, check_correctness
from .contracts import CASES, save


def analyze(root, output, data):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fsdp_interview/scripts"))
    from analyze_communication import timing_aggregation, training_run

    from experiments.finalization_4gpu.analyze import _csv

    output.mkdir(parents=True, exist_ok=False)
    contract = audit_training(root, data, "benchmark", repeats=3)
    check_correctness(root, contract["source_manifest_sha256"])
    capacity = audit_training(root, data, "capacity", repeats=1)
    if capacity["source_manifest_sha256"] != contract["source_manifest_sha256"]:
        raise ValueError("Capacity and benchmark source differs")
    runs, steps = [], []
    for case in CASES:
        for repeat in range(3):
            directory = root / "benchmark" / f"p1_benchmark_{case}_train_r{repeat}"
            run, rows = training_run(directory)
            if run["status"] != "PASS" or run["profile"] or run["steps"] != 30:
                raise ValueError("Incomplete/nonformal sample")
            run.update(series=case, repeat=repeat)
            for row in rows:
                row["series"] = case
            runs.append(run)
            steps.extend(rows)
    aggregate = timing_aggregation(runs, steps)
    by_case = {r["series"]: r for r in aggregate}
    comparisons = []
    for candidate, baseline, kind in [
        ("S-F", "N-F", "Runtime/FSDP-only"),
        ("S-H", "N-H", "Runtime/Hybrid"),
        ("S-F", "S-H", "Self strategy"),
        ("N-F", "N-H", "Native strategy"),
    ]:
        a, b = by_case[candidate], by_case[baseline]
        comparisons.append(
            dict(
                candidate=candidate,
                baseline=baseline,
                contrast=kind,
                throughput_advantage=a["training_tokens_per_s"] / b["training_tokens_per_s"] - 1,
                time_reduction=1 - a["mean_step_ms"] / b["mean_step_ms"],
            )
        )
    result = dict(
        status="BENCHMARK_2X2_PASS",
        formal_runs=12,
        measured_global_steps=360,
        contract_audit=contract,
        aggregate=aggregate,
        comparisons=comparisons,
        runs=runs,
        limitations=[
            "Three independent runs per group;90steps are correlated",
            "Strategy comparison changes F/T/SP/local GEMM/data replicas together",
            "Native matched is not native optimum; FP16 independent numerical limit remains",
        ],
    )
    save(output / "summary.json", result)
    for name, values in [
        ("runs", runs),
        ("steps", steps),
        ("aggregate", aggregate),
        ("comparisons", comparisons),
    ]:
        _csv(output / (name + ".csv"), values)
    (output / "summary.md").write_text(
        "# P1 matched 2×2\n\n" + json.dumps(result["aggregate"], indent=2) + "\n", encoding="utf-8"
    )
