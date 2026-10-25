"""Wrap unchanged four-layer independent oracle helpers in the original Q5 hook."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments/finalization_4gpu"))
sys.path.insert(0, str(PROJECT / "experiments/fsdp_interview/cases"))

from q05_nccl_contention import instrumentation  # noqa: E402

from experiments.q5_topology_revalidation_4gpu.contracts import load, save, sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("fp32", "same_start", "overflow"), required=True)
    parser.add_argument(
        "--policy", choices=("C0_recorded", "C0_guarded", "C1", "C2"), required=True
    )
    parser.add_argument("--contract")
    parser.add_argument("--placement-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    import amp_worker as amp
    import mapping_worker as mapping

    source = os.environ["SHARDTRAIN_SOURCE_MANIFEST"]
    contract = load(args.contract) if args.contract else None
    if args.policy != "C0_recorded" and (
        not contract
        or contract.get("source_manifest_sha256") != sha(source)
        or contract.get("placement_manifest_sha256") != sha(args.placement_manifest)
    ):
        raise ValueError("Tiny contract does not bind current source/placement")
    output = Path(args.output)
    rank = int(os.environ["RANK"])
    fixture_args = SimpleNamespace(
        placement_manifest=args.placement_manifest,
        mapping_id="M0",
        output=str(output),
        mode=args.mode,
        steps=10,
        effective_updates=10,
    )
    observer = instrumentation(
        policy="C0" if args.policy.startswith("C0_") else args.policy,
        guarded_baseline=args.policy == "C0_guarded",
        contract=contract,
        run_id=args.run_id,
        experiment_id="p2_" + args.mode + "_" + args.policy,
        output_dir=output / "observer",
        record_events=True,
        gpu_timing=False,
        patch_trainer=True,
    )
    original = mapping.RuntimeObserver.registered

    def registered(instance, runtime):
        result = original(instance, runtime)
        observer._capture_layout(runtime)
        return result

    mapping.RuntimeObserver.registered = registered
    try:
        with observer:
            if args.mode == "fp32":
                mapping.run(fixture_args)
            else:
                amp.run(fixture_args)
    finally:
        mapping.RuntimeObserver.registered = original
        save(
            output / f"q5_observer.rank{rank}.json",
            dict(
                policy=args.policy,
                mode=args.mode,
                source_manifest_sha256=sha(source),
                placement_manifest_sha256=sha(args.placement_manifest),
                instrumentation=observer.report(),
                fixture_sources={
                    str(path.relative_to(PROJECT)): sha(path)
                    for path in (
                        PROJECT / "experiments/finalization_4gpu/mapping_worker.py",
                        PROJECT / "experiments/finalization_4gpu/amp_worker.py",
                        PROJECT / "tests/distributed/qwen2_worker.py",
                        PROJECT / "tests/distributed/qwen2_amp_worker.py",
                    )
                },
                scope="4L FP32 independent/SGD or AMP same-start/AdamW/replay/overflow; "
                "not independent FP16 rollout PASS, not performance",
            ),
        )


if __name__ == "__main__":
    main()
