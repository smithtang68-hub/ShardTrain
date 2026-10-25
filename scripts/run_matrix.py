"""Bounded experiment runner with correctness gates, dry run and per-run failure archive."""

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
MATRIX = [(1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (4, 2), (2, 4), (8, 1)]


def execute(command, directory, timeout):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=sys.platform != "win32",
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        status = process.returncode
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=20,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate(timeout=20)
        status = -1
    (directory / "launcher.log").write_text(output)
    return status


def compare_sequences(directory, f, t):
    events = {}
    for rank in range(f * t):
        file = directory / f"runtime_events.rank{rank}.jsonl"
        events[rank] = [
            json.loads(line)
            for line in file.read_text().splitlines()
            if '"collective_issue"' in line
        ]
    for axis, groups in (
        ("world", [list(range(f * t))]),
        ("tp", [[coord * t + i for i in range(t)] for coord in range(f)]),
        ("fsdp", [[i * t + coord for i in range(f)] for coord in range(t)]),
    ):
        for ranks in groups:
            signatures = [
                [
                    (e["pg_seq"], e["kind"], e["numel"], e["dtype"], e["unit_id"], e["phase"])
                    for e in events[rank]
                    if e["pg_id"] == axis
                ]
                for rank in ranks
            ]
            if any(value != signatures[0] for value in signatures):
                raise RuntimeError(f"PG order divergence: {axis} {ranks}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "nccl"], default="cpu")
    parser.add_argument("--stages", default="E0,E1,E2,E3,E5")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="experiments")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-ranks", type=int, default=8)
    parser.add_argument("--config", default="configs/runtime_cpu.yaml")
    parser.add_argument(
        "--communication-policy", choices=["conservative", "stream_ordered"], default="conservative"
    )
    args = parser.parse_args()
    stages = args.stages.split(",")
    if len(set(stages)) != len(stages) or any(
        stage not in [f"E{i}" for i in range(6)] for stage in stages
    ):
        parser.error("stages must be unique E0..E5 identifiers")
    stages.sort()
    if "E4" in stages and (args.backend != "nccl" or "E1" not in stages):
        parser.error("E4 requires NCCL and a successful E1 in this invocation")
    if args.max_ranks not in {1, 2, 4, 8} or args.timeout <= 0:
        parser.error("max-ranks must be 1/2/4/8 and timeout positive")
    if args.backend == "nccl" and not args.dry_run:
        if not torch.cuda.is_available() or not torch.distributed.is_nccl_available():
            parser.error("Actual NCCL backend unavailable; use a Linux CUDA host")
        if torch.cuda.device_count() < args.max_ranks:
            parser.error("Insufficient GPUs for requested max-ranks")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = Path(args.output).resolve() / f"run_{stamp}"
    run.mkdir(parents=True)
    revision = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
        or None
    )
    (run / "git_revision.txt").write_text(str(revision) + "\n")
    (run / "environment.json").write_text(
        json.dumps(
            dict(
                torch=torch.__version__,
                cuda_build=torch.version.cuda,
                nccl_available=torch.distributed.is_nccl_available(),
                cuda_device_count=torch.cuda.device_count(),
                python=sys.version,
                executable=sys.executable,
                git_commit=revision,
                dry_run=args.dry_run,
                communication_policy=args.communication_policy,
            ),
            indent=2,
        )
    )
    base = yaml.safe_load((ROOT / args.config).read_text())
    plan = []
    for stage in stages:
        matrix = [pair for pair in MATRIX if pair[0] * pair[1] <= args.max_ranks]
        variants = [(False, False, False)]
        if stage == "E2":
            matrix = [(2, 2)] if args.max_ranks >= 4 else [(1, 1)]
            variants = list(itertools.product((False, True), repeat=3))
        elif stage in {"E3", "E4", "E5"}:
            matrix = (
                [(4, 2)] if args.max_ranks == 8 else [(2, 2)] if args.max_ranks >= 4 else [(1, 1)]
            )
            variants = [(False, False, False), (True, True, True)]
        for f, t in matrix:
            for flags in variants:
                directory = run / f"{stage}_F{f}T{t}_flags{''.join(str(int(x)) for x in flags)}"
                directory.mkdir()
                config = json.loads(json.dumps(base))
                config["parallel"] = dict(
                    fsdp_degree=f,
                    tp_degree=t,
                    sp_enabled=t > 1,
                    device="cuda" if args.backend == "nccl" else "cpu",
                    backend="nccl" if args.backend == "nccl" else "gloo",
                )
                config["runtime"].update(
                    forward_prefetch=flags[0],
                    backward_prefetch=flags[1],
                    async_grad_rs=flags[2],
                    reshard_after_forward=True,
                    communication_policy=args.communication_policy,
                )
                config["debug"]["output_dir"] = str(directory)
                if stage == "E0":
                    # E0 is a raw PG/API smoke, before runtime policy validation.
                    config["runtime"].update(
                        communication_policy="conservative",
                        param_grouping="submodule",
                        reshard_after_forward=False,
                    )
                if stage == "E3" and not any(flags):
                    config["runtime"]["reshard_after_forward"] = False
                if stage == "E4":
                    config["model"].update(
                        hidden_size=512, num_heads=8, mlp_hidden_size=2048, seq_len=128
                    )
                    config["training"]["steps"] = 10
                if stage == "E5":
                    config["training"].update(seed=8712, steps=10)
                    config["model"].update(
                        hidden_size=24, num_heads=4, mlp_hidden_size=40, seq_len=8
                    )
                config_file = directory / "config.yaml"
                config_file.write_text(yaml.safe_dump(config, sort_keys=False))
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--nnodes=1",
                    f"--nproc_per_node={f * t}",
                    "--rdzv_backend=c10d",
                    "--rdzv_endpoint=127.0.0.1:0",
                    "--max_restarts=0",
                ]
                if stage == "E0" and args.backend == "nccl":
                    command += [
                        "scripts/nccl_smoke.py",
                        "--fsdp",
                        str(f),
                        "--tp",
                        str(t),
                        "--output",
                        str(directory),
                    ]
                elif stage == "E0":
                    command += [
                        "-m",
                        "sharded_runtime.train",
                        "--smoke-groups",
                        "--config",
                        str(config_file),
                        "--runtime.reshard_after_forward",
                        "false",
                    ]
                elif stage == "E4":
                    command += ["scripts/benchmark_runtime.py", "--config", str(config_file)]
                else:
                    command += [
                        "tests/distributed/runtime_worker.py",
                        "--input-grad",
                        "--config",
                        str(config_file),
                    ]
                    if stage == "E3":
                        command.append("--profile")
                (directory / "launch_command.txt").write_text(
                    json.dumps(command, ensure_ascii=False) + "\n"
                )
                plan.append(
                    dict(
                        stage=stage,
                        F=f,
                        T=t,
                        flags=flags,
                        directory=str(directory),
                        command=command,
                    )
                )
    (run / "plan.json").write_text(json.dumps(plan, indent=2))
    if args.dry_run:
        print(f"Planned {len(plan)} runs: {run}")
        return
    summaries = []
    for item in plan:
        directory = Path(item["directory"])
        print(f"Running {directory.name}", flush=True)
        code = execute(item["command"], directory, args.timeout)
        status = "passed" if code == 0 else "failed"
        if code == 0 and item["stage"] not in {"E0", "E4"}:
            correctness = [
                json.loads((directory / f"correctness.rank{rank}.json").read_text())
                for rank in range(item["F"] * item["T"])
            ]
            if not all(result["status"] == "passed" for result in correctness):
                status = "failed"
            try:
                compare_sequences(directory, item["F"], item["T"])
            except RuntimeError as exc:
                (directory / "sequence_failure.txt").write_text(str(exc))
                status = "failed"
        summaries.append(
            dict(
                stage=item["stage"],
                F=item["F"],
                T=item["T"],
                status=status,
                returncode=code,
                directory=str(directory),
            )
        )
        (run / "summary.json").write_text(
            json.dumps(
                dict(
                    backend=args.backend,
                    runs=summaries,
                    actual_NCCL=args.backend == "nccl",
                    communication_policy=args.communication_policy,
                    performance_interpretation="Trace review required",
                ),
                indent=2,
            )
        )
        if status != "passed":
            raise SystemExit(f"Failed gate: {directory}; later stages stopped")
    print(f"Passed {len(summaries)} runs: {run}")


if __name__ == "__main__":
    main()
