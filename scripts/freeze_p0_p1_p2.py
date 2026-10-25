"""Freeze the bounded numerics, 2x2 and Q5 experiments with the existing freezer.

This includes all three new experiment scopes without changing the historical
four-GPU freezer's source coverage or any previous manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import freeze_finalization as freezer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    freezer.FOLDERS = (
        *freezer.FOLDERS,
        "experiments/fp16_numerics",
        "experiments/fsdp4_hybrid_2x2",
        "experiments/q5_topology_revalidation_4gpu",
    )
    root = Path(__file__).resolve().parents[1]
    receipt = freezer.freeze(
        root,
        Path(args.output_dir).resolve(),
        args.version,
        snapshot_namespace="p0_p1_p2_4gpu",
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
