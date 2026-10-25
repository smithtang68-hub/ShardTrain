"""Gated GPU matrix, separate Nsight replays, and profiler-free timing runs."""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import yaml
from run_matrix import execute

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--nsys", required=True)
    parser.add_argument("--max-ranks", type=int, choices=[4, 8], default=4)
    parser.add_argument("--budget-seconds", type=int, default=4800)
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.budget_seconds
    rows = []

    def run(command, directory, kind):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        print(f"Running {kind}: {directory}", flush=True)
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise SystemExit("Time budget reached; completed artifacts are preserved")
        status = execute(command, directory, remaining)
        rows.append(dict(kind=kind, directory=str(directory), returncode=status))
        (destination / "acceptance_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        if status:
            raise SystemExit(f"Failed {kind}: {directory}")

    for policy in ("conservative", "stream_ordered"):
        directory = destination / policy
        run(
            [
                sys.executable,
                "scripts/run_matrix.py",
                "--backend",
                "nccl",
                "--max-ranks",
                str(args.max_ranks),
                "--stages",
                "E0,E1,E2,E3,E4,E5",
                "--communication-policy",
                policy,
                "--output",
                str(directory / "matrix"),
                "--timeout",
                "600",
            ],
            directory,
            "gated_matrix",
        )
        matrix_runs = sorted((directory / "matrix").glob("run_*"))
        matrix = matrix_runs[-1]
        summary = json.loads((matrix / "summary.json").read_text())
        if not summary["actual_NCCL"] or any(r["status"] != "passed" for r in summary["runs"]):
            raise SystemExit("NCCL correctness gate did not pass")
        plan = json.loads((matrix / "plan.json").read_text())
        for item in plan:
            if item["stage"] not in {"E3", "E4"}:
                continue
            case = Path(item["directory"])
            replay = directory / "nsight" / case.name
            replay.mkdir(parents=True, exist_ok=True)
            config = yaml.safe_load((case / "config.yaml").read_text())
            config["debug"]["output_dir"] = str(replay)
            config_file = replay / "config.yaml"
            config_file.write_text(yaml.safe_dump(config, sort_keys=False))
            command = item["command"].copy()
            command[command.index("--config") + 1] = str(config_file)
            command.extend(["--profile-mode", "nsight"])
            run(
                [
                    args.nsys,
                    "profile",
                    "--trace=cuda,nvtx,osrt",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--cuda-memory-usage=true",
                    "--force-overwrite=true",
                    "--output",
                    str(replay / "timeline"),
                    *command,
                ],
                replay,
                "nsight_profile",
            )
            report = replay / "timeline.nsys-rep"
            if not report.is_file():
                raise SystemExit(f"Missing Nsight report: {report}")
            run(
                [
                    args.nsys,
                    "export",
                    "--type=sqlite",
                    "--force-overwrite=true",
                    "--output",
                    str(replay / "timeline.sqlite"),
                    str(report),
                ],
                replay / "export",
                "nsight_export",
            )
            with sqlite3.connect(replay / "timeline.sqlite") as db:
                kernels = db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[
                    0
                ]
                nvtx = db.execute("SELECT COUNT(*) FROM NVTX_EVENTS").fetchone()[0]
            if not kernels or not nvtx:
                raise SystemExit(f"Nsight trace has no CUDA kernels or NVTX ranges: {replay}")
            (replay / "trace_validation.json").write_text(
                json.dumps(dict(cuda_kernels=kernels, nvtx_events=nvtx), indent=2) + "\n"
            )
        for item in plan:
            if item["stage"] != "E4":
                continue
            case = Path(item["directory"])
            timing = directory / "timing" / case.name
            timing.mkdir(parents=True, exist_ok=True)
            config = yaml.safe_load((case / "config.yaml").read_text())
            config["debug"]["output_dir"] = str(timing)
            config["training"]["steps"] = 30
            config_file = timing / "config.yaml"
            config_file.write_text(yaml.safe_dump(config, sort_keys=False))
            command = item["command"].copy()
            command[command.index("--config") + 1] = str(config_file)
            command.extend(["--warmup", "5", "--profile-mode", "off"])
            run(command, timing, "profiler_free_timing")
    print(f"Acceptance complete: {destination}", flush=True)


if __name__ == "__main__":
    main()
