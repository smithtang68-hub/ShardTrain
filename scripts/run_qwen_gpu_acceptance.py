"""Serial, fail-closed V100 Qwen HF/AMP acceptance with reproducible commands."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from run_matrix import execute

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--skip-initial-endpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    summary_file = destination / "extended_acceptance_summary.json"
    if summary_file.exists():
        parser.error("Choose a fresh output root to preserve previous results")
    cases = []
    for policy in ("conservative", "stream_ordered"):
        for value in range(8):
            flags = f"{value:03b}"
            if args.skip_initial_endpoints and (policy, flags) in {
                ("conservative", "000"),
                ("stream_ordered", "111"),
            }:
                continue
            cases.append((f"tiny4_{policy}_{flags}", 2, 2, True, 4, policy, flags))
        flags = "000" if policy == "conservative" else "111"
        for layers in (12, 28):
            cases.append((f"tiny{layers}_{policy}_{flags}", 2, 2, True, layers, policy, flags))
        for f, t, sp in ((2, 1, False), (1, 2, True), (1, 2, False), (4, 1, False)):
            name = f"axis_F{f}T{t}_SP{int(sp)}_{policy}_{flags}"
            cases.append((name, f, t, sp, 4, policy, flags))
    plan = []
    for name, f, t, sp, layers, policy, flags in cases:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={f * t}",
            "tests/distributed/qwen2_worker.py",
            "--backend",
            "nccl",
            "--device",
            "cuda",
            "--f",
            str(f),
            "--t",
            str(t),
            "--layers",
            str(layers),
            "--steps",
            "3",
            "--policy",
            policy,
            "--flags",
            flags,
            "--profile-mode",
            "torch",
            "--debug.output_dir",
            str(destination / name),
        ]
        if sp:
            command.append("--sp")
        plan.append({"name": name, "kind": "hf_fp32", "command": command})
    for policy, flags in (("conservative", "000"), ("stream_ordered", "111")):
        for optimizer in ("sgd", "adamw"):
            name = f"amp_{optimizer}_{policy}_{flags}"
            plan.append(
                {
                    "name": name,
                    "kind": "hf_fp16_amp",
                    "command": [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc_per_node=4",
                        "tests/distributed/qwen2_amp_worker.py",
                        "--f",
                        "2",
                        "--t",
                        "2",
                        "--sp",
                        "--backend",
                        "nccl",
                        "--optimizer",
                        optimizer,
                        "--policy",
                        policy,
                        "--flags",
                        flags,
                        "--profile-mode",
                        "torch",
                        "--output-dir",
                        str(destination / name),
                    ],
                }
            )
    (destination / "extended_acceptance_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    source = {}
    for path in sorted(ROOT.glob("src/**/*.py")):
        source[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [
        Path(__file__),
        ROOT / "tests/distributed/qwen2_worker.py",
        ROOT / "tests/distributed/qwen2_amp_worker.py",
    ]:
        source[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "extended_source_sha256.json").write_text(
        json.dumps(source, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps({"cases": len(plan), "gpu_jobs_started": False}))
        return 0
    rows = []
    for item in plan:
        directory = destination / item["name"]
        if directory.exists():
            raise RuntimeError(f"Experiment directory already exists: {directory}")
        directory.mkdir()
        (directory / "command.json").write_text(
            json.dumps(item["command"], indent=2) + "\n", encoding="utf-8"
        )
        print(f"Starting {item['name']}", flush=True)
        started = time.monotonic()
        status = execute(item["command"], directory, args.timeout)
        rows.append(
            {
                "name": item["name"],
                "kind": item["kind"],
                "returncode": status,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        summary_file.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"Finished {item['name']}: {status}", flush=True)
        if status:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
