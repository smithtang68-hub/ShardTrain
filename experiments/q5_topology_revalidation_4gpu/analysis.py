"""Read-only current M0 Q5 aggregation; profiles never enter formal timing."""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

from .contracts import POLICIES, load, save, sha

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fsdp_interview/scripts"))
from analyze_communication import timing_aggregation, training_run  # noqa: E402


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trend(times):
    center = (len(times) - 1) / 2
    slope = (
        sum((index - center) * value for index, value in enumerate(times))
        / sum((index - center) ** 2 for index in range(len(times)))
        if len(times) > 1
        else None
    )
    first, last = statistics.mean(times[:5]), statistics.mean(times[-5:])
    return dict(
        linear_slope_ms_per_step=slope,
        first_five_mean_ms=first,
        last_five_mean_ms=last,
        last_first_ratio=last / first,
        descriptive_drift_flag=abs(last / first - 1) > 0.05,
        scope="5% is a preregistered descriptive flag, not a statistical PASS threshold",
    )


def telemetry(path):
    values = {}
    if not path.exists():
        return dict(status="NOT_AVAILABLE")
    for line in path.read_text(encoding="utf-8").splitlines():
        for row in __import__("json").loads(line).get("devices", []):
            for name in ("free_bytes", "temperature_c", "sm_clock_mhz", "memory_clock_mhz"):
                if row.get(name) is not None:
                    values.setdefault(name, []).append(row[name])
    return {
        field + suffix: operation(numbers)
        for field, numbers in values.items()
        for suffix, operation in (("_min", min), ("_max", max))
    }


def analyze(run_root, output=None):
    root = Path(run_root)
    target = Path(output) if output else root / "p2/analysis"
    target.mkdir(parents=True, exist_ok=False)
    gate = load(root / "p2/contracts/small_oracle_gate.json")
    contract = load(root / "p2/contracts/collective_contract_audit.json")
    if gate["status"] != "PASS" or contract["status"] != "PASS":
        raise ValueError("Cannot aggregate missing required policy gates")
    summaries, steps, ranks, trends, missing = [], [], [], [], []
    initial, inputs, sources = {}, {}, set()
    for policy in POLICIES:
        for repeat in range(3):
            directory = root / "p2/benchmark" / policy / ("r" + str(repeat))
            if not all((directory / f"benchmark.rank{rank}.json").is_file() for rank in range(4)):
                missing.append(dict(policy=policy, repeat=repeat, status="NOT_RUN_OR_INCOMPLETE"))
                continue
            summary, rows = training_run(directory)
            if (
                summary["status"] != "PASS"
                or summary["profile"]
                or summary["steps"] != 30
                or summary["seq_len"] != 512
                or summary["microbatch"] != 4
                or summary["accumulation_steps"] != 2
                or summary["unique_input_tokens_per_step"] != 8192
                or summary["unique_training_tokens_per_step"] != 8176
            ):
                raise ValueError("Ineligible formal policy sample")
            for rank in range(4):
                value = load(directory / f"benchmark.rank{rank}.json")
                observer = value["instrumentation"]
                completion = load(directory / f"training_completion.rank{rank}.json")
                placement = load(directory / f"placement.rank{rank}.json")
                if (
                    value["event_recorder"]
                    or value["profile"]
                    or value["warmup"] != 10
                    or observer["record_events"]
                    or observer["gpu_timing"]
                    or observer["status"] != "RECORDED"
                    or observer["guarded_baseline"] != (policy == "C0_guarded")
                    or value["policy"] != ("C0" if policy == "C0_guarded" else policy)
                    or completion["status"] != "PASS"
                    or completion["adam_leaf_step_min"] != 40
                    or completion["adam_leaf_step_max"] != 40
                    or placement["status"] != "PASS"
                ):
                    raise ValueError("Observer costs, actual updates or policy identity differ")
                actual_input = [
                    (
                        row["optimizer_step"],
                        row["input_block_start"],
                        row["input_batch_sha256"],
                        row["valid_tokens"],
                    )
                    for row in value["warmup_steps"] + value["steps"]
                ]
                actual_initial = (
                    placement["initial_local_state"],
                    placement["actual_uuid"],
                    placement["actual_pci_bus_id"],
                    placement["actual_pg_members"],
                )
                if rank in initial and initial[rank] != actual_initial:
                    raise ValueError("Formal policies changed initial state or UUID/PG assignment")
                if rank in inputs and inputs[rank] != actual_input:
                    raise ValueError("Formal policies changed logical-rank input assignment")
                initial[rank], inputs[rank] = actual_initial, actual_input
                sources.add(value["source_manifest_sha256"])
                if (
                    placement["placement_manifest_sha256"] != contract["placement_manifest_sha256"]
                    or completion["source_manifest_sha256"] != contract["source_manifest_sha256"]
                ):
                    raise ValueError("Formal run differs from capacity source/placement")
                ranks += [
                    dict(policy=policy, repeat=repeat, rank=rank, **row) for row in value["steps"]
                ]
            summary.update(
                policy_label=policy,
                repeat=repeat,
                telemetry=telemetry(
                    root / "logs" / f"p2_benchmark_{policy}_r{repeat}" / "nvml_memory.jsonl"
                ),
            )
            summaries.append(summary)
            steps += rows
            trends.append(
                dict(policy=policy, repeat=repeat, **trend([row["step_time_ms"] for row in rows]))
            )
    if sources and sources != {contract["source_manifest_sha256"]}:
        raise ValueError("Formal samples mix frozen execution sources")
    aggregate = timing_aggregation(summaries, steps)
    comparisons = []
    by_policy = {row["series"].removeprefix("shardtrain/"): row for row in aggregate}
    control = by_policy.get("C0_guarded")
    if control:
        left_range = [
            row["mean_step_ms"] for row in summaries if row["policy_label"] == "C0_guarded"
        ]
        for policy in ("C1", "C2"):
            other = by_policy.get(policy)
            if not other:
                continue
            right_range = [
                row["mean_step_ms"] for row in summaries if row["policy_label"] == policy
            ]
            overlap = max(min(left_range), min(right_range)) <= min(
                max(left_range), max(right_range)
            )
            comparisons.append(
                dict(
                    control="C0_guarded",
                    constrained=policy,
                    c0_mean_ms=control["mean_step_ms"],
                    constrained_mean_ms=other["mean_step_ms"],
                    constrained_minus_c0_ms=other["mean_step_ms"] - control["mean_step_ms"],
                    c0_time_reduction_fraction=1 - control["mean_step_ms"] / other["mean_step_ms"],
                    c0_throughput_advantage_fraction=control["training_tokens_per_s"]
                    / other["training_tokens_per_s"]
                    - 1,
                    run_mean_range_overlap=overlap,
                    interpretation="INCONCLUSIVE_RUN_MEAN_RANGES_OVERLAP"
                    if overlap
                    else "OBSERVED_DIRECTION_SEPARATED_IN_THREE_RUNS",
                    limits="Intervention includes event-wait scheduling costs; "
                    "not ideal wire/overlap savings",
                )
            )
    profile_comparisons = []
    for policy in ("C0_guarded", "C1"):
        directory = root / "p2/profiles" / policy
        if not all((directory / f"benchmark.rank{rank}.json").is_file() for rank in range(4)):
            continue
        profile, capture_steps = training_run(directory)
        baseline = by_policy.get(policy)
        if (
            not profile["profile"]
            or profile["steps"] != 1
            or profile["source_manifest_sha256"] != contract["source_manifest_sha256"]
            or not baseline
        ):
            raise ValueError("Profile comparison does not bind the current formal policy")
        same_step = [
            row["step_time_ms"]
            for row in steps
            if row["series"] == profile["series"]
            and row["optimizer_step"] == capture_steps[0]["optimizer_step"]
        ]
        profile_comparisons.append(
            dict(
                policy=policy,
                profile_step_ms=profile["mean_step_ms"],
                formal_mean_ms=baseline["mean_step_ms"],
                observed_ratio=profile["mean_step_ms"] / baseline["mean_step_ms"],
                same_optimizer_step=capture_steps[0]["optimizer_step"],
                unprofiled_same_step_samples_ms=same_step,
                unprofiled_same_step_median_ms=statistics.median(same_step),
                unprofiled_same_step_min_ms=min(same_step),
                unprofiled_same_step_max_ms=max(same_step),
                scope="One capture versus independent formal processes; "
                "not causal profiler overhead",
            )
        )
    mechanism_path = root / "p2/mechanism/summary.json"
    mechanism = load(mechanism_path) if mechanism_path.exists() else dict(status="NOT_RUN")
    if (
        mechanism.get("status") == "PASS"
        and mechanism.get("source_manifest_sha256") != contract["source_manifest_sha256"]
    ):
        raise ValueError("Mechanism evidence differs from current formal source snapshot")
    stage_records = [
        dict(path=str(path), sha256=sha(path), **load(path))
        for path in sorted((root / "stages").glob("p2_*.json"))
    ]
    status = (
        "Q5_M0_THREE_POLICY_PASS"
        if (
            len(summaries) == 9
            and not missing
            and len(profile_comparisons) == 2
            and mechanism["status"] == "PASS"
        )
        else "PARTIAL_Q5_M0"
    )
    result = dict(
        status=status,
        independent_runs=len(summaries),
        formal_measured_steps=len(steps),
        source_manifest_sha256=contract["source_manifest_sha256"],
        gate=gate,
        contract=contract,
        runs=summaries,
        aggregate=aggregate,
        comparisons=comparisons,
        trends=trends,
        missing=missing,
        profile_comparisons=profile_comparisons,
        mechanism=mechanism,
        stages=stage_records,
        retained_failed_stages=[row for row in stage_records if row.get("status") != "PASS"],
        p0_continuation_review=load(root / "p0_continuation_review.json"),
        p0_continuation_review_sha256=sha(root / "p0_continuation_review.json"),
        C3="NOT_RUN_OPTIONAL",
        Q12="NOT_RUN_OPTIONAL",
        source_script_sha256=sha(__file__),
        limits=[
            "P0 strict independent FAIL retained separately; "
            "completion is not all-correctness PASS",
            "90 steps per policy are correlated; three independent process means are reported",
            "Profile steps and P1 C0_fast excluded; no cross-server causal speedup inference",
        ],
    )
    save(target / "summary.json", result)
    for name, values in (
        ("runs", summaries),
        ("steps", steps),
        ("rank_steps", ranks),
        ("aggregate", aggregate),
        ("comparisons", comparisons),
        ("trends", trends),
    ):
        write_csv(target / (name + ".csv"), values)
    plots(target, summaries, aggregate, steps)
    return result


def plots(output, summaries, aggregate, steps):
    if not aggregate:
        return
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for index, policy in enumerate(POLICIES):
        own = [row for row in summaries if row["policy_label"] == policy]
        if not own:
            continue
        axes[0].bar(policy, statistics.mean(row["mean_step_ms"] for row in own))
        axes[0].scatter(
            [policy] * len(own), [row["mean_step_ms"] for row in own], color="black", zorder=3
        )
        axes[1].bar(
            index - 0.18,
            max(row["peak_allocated_bytes"] for row in own) / 2**30,
            width=0.36,
            label="Allocated" if index == 0 else None,
        )
        axes[1].bar(
            index + 0.18,
            max(row["peak_reserved_bytes"] for row in own) / 2**30,
            width=0.36,
            label="Reserved" if index == 0 else None,
        )
    axes[0].set_ylabel("Mean step time (ms)")
    axes[0].set_title("Dots: independent process means")
    axes[1].set_xticks(range(3), POLICIES)
    axes[1].set_ylabel("Worst-rank peak GiB")
    axes[1].legend()
    figure.savefig(output / "q5_performance_memory.png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
    for axis, policy in zip(axes, POLICIES, strict=True):
        for row in summaries:
            if row["policy_label"] == policy:
                own = [step for step in steps if step["directory"] == row["directory"]]
                axis.plot(
                    [step["measured_step"] for step in own],
                    [step["step_time_ms"] for step in own],
                    label="run" + str(row["repeat"]),
                )
        axis.set_ylabel(policy + " ms")
        axis.legend()
    axes[-1].set_xlabel("Measured optimizer step (correlated samples)")
    figure.savefig(output / "q5_run_trends.png", dpi=170, bbox_inches="tight")
    plt.close(figure)
