"""Run the existing independent HF AMP gate with reversible scheduling hooks."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments/fsdp_interview/cases"))

from q05_nccl_contention import instrumentation  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["C0", "C1", "C2", "C3"], default="C0")
    p.add_argument("--contract")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layers", type=int, default=4)
    args = p.parse_args()
    source = PROJECT / "tests/distributed/qwen2_amp_worker.py"
    spec = importlib.util.spec_from_file_location("_interview_original_amp_gate", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = json.loads(Path(args.contract).read_text()) if args.contract else None
    with instrumentation(
        policy=args.policy,
        output_dir=Path(args.output_dir) / "instrumentation",
        run_id="fsdp_interview_20261014",
        experiment_id=f"correctness_{args.policy}",
        contract=contract,
        warmup_steps=0,
        record_events=True,
        gpu_timing=False,
        patch_trainer=True,
    ):
        module.main(
            [
                "--f",
                "2",
                "--t",
                "2",
                "--sp",
                "--layers",
                str(args.layers),
                "--optimizer",
                "adamw",
                "--backend",
                "nccl",
                "--device",
                "cuda",
                "--policy",
                "stream_ordered",
                "--flags",
                "111",
                "--profile-mode",
                "off",
                "--output-dir",
                args.output_dir,
            ]
        )
    Path(args.output_dir, "fixture_provenance.json").write_text(
        json.dumps(
            {
                "source": str(source.relative_to(PROJECT)),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "modified": False,
                "scope": "tiny architecture HF dense loss/gradient/optimizer replay "
                "and AMP overflow",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
