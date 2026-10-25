"""Plot Q12 repeated unprofiled timing separately from single-trace responses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.fsdp_interview.scripts.analyze_communication import percentile  # noqa: E402


def condition(mode, delay):
    return f"CPU {delay:g} ms" if mode == "cpu" else "GPU 2048 x 64"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-summary", required=True)
    parser.add_argument("--skew-summary")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Fresh output directory required: {output}")
    timing = json.loads(Path(args.timing_summary).read_text(encoding="utf-8"))
    conditions = [("cpu", 0), ("cpu", 2), ("cpu", 10), ("cpu", 50), ("gpu", None)]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    output.mkdir(parents=True)
    paths = []
    figure, axis = plt.subplots(figsize=(10, 5))
    for x, (mode, delay) in enumerate(conditions):
        runs = [
            row
            for row in timing["training_runs"]
            if row["injection"]["mode"] == mode
            and (mode == "gpu" or row["injection"]["delay_ms"] == delay)
        ]
        means = [row["mean_step_ms"] for row in runs]
        steps = [
            row["step_time_ms"]
            for row in timing["training_steps"]
            if row["series"] in {run["series"] for run in runs}
        ]
        for r, mean in enumerate(means):
            axis.scatter(
                x + (r - 1) * 0.07,
                mean,
                marker="o",
                color="tab:blue",
                label="independent run mean" if x == 0 and r == 0 else None,
            )
        axis.scatter(
            x,
            percentile(steps, 0.5),
            marker="_",
            s=320,
            color="black",
            label="pooled 90-step P50" if x == 0 else None,
        )
        axis.scatter(
            x,
            percentile(steps, 0.95),
            marker="^",
            s=55,
            color="tab:orange",
            label="pooled 90-step P95" if x == 0 else None,
        )
    axis.set_xticks(range(len(conditions)), [condition(*value) for value in conditions])
    axis.set_ylabel("Optimizer step time (ms)")
    axis.set_title("Q12: 3 independent runs x 30 measured steps; 10 warmup per run")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.text(
        0.5,
        0.02,
        "One injection per optimizer step, rank0 / unit1 / microstep0 / first FWD AG; "
        "delay does not show monotonic step cost",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    target = output / "q12_repeated_timing.png"
    figure.savefig(target, dpi=170)
    plt.close(figure)
    paths.append(target.name)
    if args.skew_summary:
        summaries = json.loads(Path(args.skew_summary).read_text(encoding="utf-8"))
        mapped = {}
        for summary in summaries:
            for row in summary["targets"]:
                if row["status"] != "MATCHED" or len(row["signed_comparisons"]) != 1:
                    continue
                mode = row["injection"]["mode"]
                key = mode, row["injection"].get("cpu_delay_ms") if mode == "cpu" else None
                if key in mapped:
                    raise ValueError(f"Multiple diagnostic samples for {key}: plot explicitly")
                mapped[key] = row
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.7))
        for x, key in enumerate(conditions):
            row = mapped.get(key)
            if row is None:
                continue
            comparison = row["signed_comparisons"][0]
            axes[0].scatter(
                x - 0.08,
                row["injection"]["cpu_injection_wall_ms"],
                label="CPU wall injection / enqueue" if x == 0 else None,
                color="tab:orange",
            )
            axes[0].scatter(
                x + 0.08,
                comparison["target_minus_peer_api_ms"],
                marker="s",
                label="NSYS API target minus peer" if x == 0 else None,
                color="tab:blue",
            )
            for offset, name, title, marker in (
                (-0.08, "target_minus_peer_gpu_start_ms", "GPU start target minus peer", "o"),
                (0.08, "target_minus_peer_gpu_end_ms", "GPU end target minus peer", "s"),
            ):
                axes[1].scatter(
                    x + offset,
                    comparison[name],
                    marker=marker,
                    color="tab:blue" if marker == "o" else "tab:orange",
                    label=title if x == 0 else None,
                )
            axes[2].bar(
                x - 0.16,
                comparison["target_kernel_residency_ms"],
                0.3,
                label="rank0 target" if x == 0 else None,
                color="tab:blue",
            )
            axes[2].bar(
                x + 0.16,
                comparison["peer_kernel_residency_ms"],
                0.3,
                label=f"rank{comparison['peer_rank']} same PG peer" if x == 0 else None,
                color="tab:orange",
            )
        for axis, title in zip(
            axes,
            (
                "CPU injection and actual launch skew",
                "Actual GPU skew",
                "Selected NCCL kernel residency",
            ),
            strict=True,
        ):
            axis.set_xticks(
                range(len(conditions)),
                [condition(*value) for value in conditions],
                rotation=25,
                ha="right",
            )
            axis.set_ylabel("ms")
            axis.set_title(title)
            axis.axhline(0, color="gray", linewidth=0.7)
            axis.grid(axis="y", alpha=0.2)
            axis.legend(fontsize=8)
        figure.suptitle("Q12 diagnostic: one exact PG(0,2) / unit1 first FWD AG per trace")
        figure.text(
            0.5,
            0.01,
            "Same-report signed skew: positive means rank0 later. "
            "Different trace clocks are never subtracted. Kernel residency includes waiting.",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.05, 1, 0.95))
        target = output / "q12_exact_target_response.png"
        figure.savefig(target, dpi=170)
        plt.close(figure)
        paths.append(target.name)
    (output / "plot_provenance.json").write_text(
        json.dumps(
            {
                "timing_summary": str(Path(args.timing_summary).resolve()),
                "skew_summary": str(Path(args.skew_summary).resolve())
                if args.skew_summary
                else None,
                "paths": paths,
                "scope": "unprofiled repeats separated from single-step mechanism profiles",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"plots": paths, "output_dir": str(output)}))


if __name__ == "__main__":
    main()
