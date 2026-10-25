"""Plot a real, explicitly selected owner/current/target GPU residency window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.fsdp_interview.scripts.parse_trace import merge  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.pipeline_json).read_text())
    rows = data["pipelines"]
    if not rows:
        raise ValueError("No real matched BWD pipeline windows supplied")
    selected = max(rows, key=lambda row: row["ag_rs_compute_current_ms"])
    matching = [
        row
        for row in rows
        if row["runtime_microstep"] == selected["runtime_microstep"]
        and row["target_prefetch_unit"] == selected["target_prefetch_unit"]
    ]
    matching.sort(key=lambda row: row["rank"])
    groups = [
        ("ag_intervals_ns", "Parameter AG target", "#16856a"),
        ("rs_intervals_ns", "Gradient RS owner", "#286ca8"),
        ("norm_owner_intervals_ns", "Norm AR owner", "#df8d25"),
        ("compute_current_intervals_ns", "BWD compute current", "#7957a4"),
    ]
    left = min(start for row in matching for key, _, _ in groups for start, _ in row[key])
    right = max(end for row in matching for key, _, _ in groups for _, end in row[key])
    padding = max(500_000, (right - left) * 0.02)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(matching), 1, figsize=(13, 2.15 * len(matching)), sharex=True)
    if len(matching) == 1:
        axes = [axes]
    for axis, row in zip(axes, matching, strict=True):
        for lane, (key, _label, color) in enumerate(groups):
            intervals = merge(row[key])
            axis.broken_barh(
                [(start / 1e9, (end - start) / 1e9) for start, end in intervals],
                (lane - 0.3, 0.6),
                facecolors=color,
            )
        axis.set_yticks(range(4), [label for _, label, _ in groups], fontsize=8)
        axis.set_ylim(3.55, -0.55)
        axis.set_xlim((left - padding) / 1e9, (right + padding) / 1e9)
        axis.grid(axis="x", alpha=0.2)
        axis.set_title(
            f"Rank {row['rank']}: owner U{row['gradient_owner_unit']} / "
            f"current U{row['current_bwd_unit']} / target U{row['target_prefetch_unit']} | "
            f"AG∩RS∩current compute {row['ag_rs_compute_current_ms']:.3f} ms",
            fontsize=9,
            loc="left",
        )
    axes[-1].set_xlabel("NSYS report time (s), same report clock for all ranks")
    figure.suptitle(
        f"{Path(data['source_sqlite']).parent.name}: "
        f"BWD microstep {selected['runtime_microstep']}\n"
        "Norm gradient belongs to owner; norm_owner then compute_current "
        "are serial on compute stream",
        fontsize=11,
    )
    figure.text(
        0.02,
        0.008,
        "Selected largest observed AG∩RS∩current-compute window; one diagnostic step. "
        "Kernel residency includes waits; not saved time.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.925))
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose fresh timeline output")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "backward_owner_current_target.png", dpi=170)
    plt.close(figure)
    (output / "plot_provenance.json").write_text(
        json.dumps(
            {
                "source_pipeline": str(args.pipeline_json),
                "source_sqlite": data["source_sqlite"],
                "selection": "largest observed AG-RS-current compute residency intersection, "
                "explicitly diagnostic",
                "selected_rank": selected["rank"],
                "runtime_microstep": selected["runtime_microstep"],
                "target_unit": selected["target_prefetch_unit"],
                "common_window_ns": [left - padding, right + padding],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
