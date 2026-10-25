"""Reproducible staged matrix; stop on unsafe memory, failed correctness or cleanup."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from owned_stage import BASE, PROJECT, run_stage, write_json  # noqa: E402

PYTHON = sys.executable
NSYS = (
    "/root/shardtrain_work/20261011/nsight_cli_2024_6/opt/nvidia/"
    "nsight-systems-cli/2024.6.1/bin/nsys"
)
WORKER = "experiments/fsdp_interview/scripts/train_workload.py"
RUN_ID = "fsdp_interview_20261014"


def execute(name, argv, env=None, timeout=900):
    if shutil.disk_usage(BASE).free < 3 * 2**30:
        raise RuntimeError(
            "Less than 3GiB disk margin; download current artifacts before continuing"
        )
    result = run_stage(name, argv, timeout=timeout, extra_env=env)
    if result["status"] != "passed":
        raise RuntimeError(f"Stage failed; preserve logs and diagnose before continuing: {name}")
    return result


def train_command(
    data,
    output,
    name,
    engine="shardtrain",
    *,
    policy="C0",
    warmup=10,
    steps=30,
    record=False,
    profile=False,
    injection=None,
    guarded=False,
):
    cmd = [
        PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        WORKER,
        "--implementation",
        engine,
        "--data",
        str(data),
        "--output-dir",
        str(output),
        "--run-id",
        RUN_ID,
        "--experiment-id",
        name,
        "--warmup",
        str(warmup),
        "--steps",
        str(steps),
        "--policy",
        policy,
    ]
    if policy != "C0" or guarded:
        cmd += ["--contract", str(BASE / "results/capacity/contract_28L.json")]
    if guarded:
        cmd.append("--guarded-baseline")
    if record:
        cmd.append("--record-events")
    if injection:
        cmd += ["--injection-json", json.dumps(injection)]
    if profile:
        cmd.append("--profile")
        cmd = [
            NSYS,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-memory-usage=true",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=false",
            "--output",
            str(output / "timeline"),
        ] + cmd
    return cmd


def check_training(directory, steps, *, safety=False):
    rows = [json.loads((directory / f"benchmark.rank{rank}.json").read_text()) for rank in range(4)]
    for row in rows:
        if row["status"] != "PASS" or row["steady_steps_completed"] != steps:
            raise RuntimeError(f"Missing/failed rank result {directory}: {row['rank']}")
        if not all(s["finite_gradients"] for s in row["steps"]):
            raise RuntimeError("A measured gradient was nonfinite")
        if (
            safety
            and max(row["peak_reserved_bytes"], row["warmup_peak_reserved_bytes"]) > 26 * 2**30
        ):
            raise RuntimeError("Workload violates the conservative 26GiB reserved budget")
        if safety and min(s["physical_free_bytes_endpoint"] for s in row["steps"]) < 4 * 2**30:
            raise RuntimeError("Workload violates the 4GiB physical-free margin")
    return rows


def qualify(directory, output):
    sys.path.insert(0, str(PROJECT / "experiments/fsdp_interview/cases"))
    from q05_nccl_contention import validate_schedule

    write_json(
        output,
        validate_schedule([directory / f"communication_events.rank{r}.jsonl" for r in range(4)]),
    )


def capacity(data):
    outputs = []
    for engine in ["native", "shardtrain"]:
        name = f"capacity_28L_{engine}_S512_MB4_A2"
        out = BASE / "results/capacity" / engine
        execute(
            name, train_command(data, out, name, engine, steps=10, record=engine == "shardtrain")
        )
        results = check_training(out, 10, safety=True)
        outputs.append(
            {
                "engine": engine,
                "directory": str(out),
                "max_reserved_bytes": max(
                    max(x["peak_reserved_bytes"], x["warmup_peak_reserved_bytes"]) for x in results
                ),
                "minimum_endpoint_free_bytes": min(
                    s["physical_free_bytes_endpoint"] for x in results for s in x["steps"]
                ),
            }
        )
    qualify(BASE / "results/capacity/shardtrain", BASE / "results/capacity/contract_28L.json")
    write_json(
        BASE / "results/capacity/workload_selection.json",
        {
            "status": "PASS",
            "seq_len": 512,
            "microbatch": 4,
            "accumulation": 2,
            "fsdp": 2,
            "tp": 2,
            "sp": True,
            "policy": "matched FP16 linear, FP32 AG/RS/master/state",
            "dataset": str(data),
            "not_memory_limit_search": True,
            "implementations": outputs,
        },
    )


def main_timing(data):
    # Rotate/reverse policies across independent repetitions to reduce ordering bias.
    for repeat, order in enumerate(
        [
            [
                ("native", "C0", False),
                ("shardtrain", "C0", False),
                ("shardtrain", "C0", True),
                ("shardtrain", "C1", False),
                ("shardtrain", "C2", False),
                ("shardtrain", "C3", False),
            ],
            [
                ("shardtrain", "C3", False),
                ("shardtrain", "C2", False),
                ("shardtrain", "C1", False),
                ("shardtrain", "C0", True),
                ("shardtrain", "C0", False),
                ("native", "C0", False),
            ],
            [
                ("shardtrain", "C2", False),
                ("native", "C0", False),
                ("shardtrain", "C0", True),
                ("shardtrain", "C3", False),
                ("shardtrain", "C0", False),
                ("shardtrain", "C1", False),
            ],
        ]
    ):
        for engine, policy, guarded in order:
            label = "C0_guarded" if guarded else policy
            name = f"timing_r{repeat}_{engine}_{label}"
            out = BASE / "results/timing" / name
            execute(name, train_command(data, out, name, engine, policy=policy, guarded=guarded))
            check_training(out, 30)


def export_capture(name, out):
    rep, sqlite = out / "timeline.nsys-rep", out / "timeline.sqlite"
    if not rep.is_file() or rep.stat().st_size == 0:
        raise RuntimeError(f"Missing Nsight report for {name}")
    execute(
        name + "_export",
        [
            NSYS,
            "export",
            "--type=sqlite",
            "--force-overwrite=false",
            "--output",
            str(sqlite),
            str(rep),
        ],
        timeout=600,
    )
    if not sqlite.is_file() or sqlite.stat().st_size == 0:
        raise RuntimeError("SQLite export missing")
    for rank in range(4):
        start_exists = any(
            (out / f"{stem}.rank{rank}.pickle").is_file()
            for stem in ["memory.start", "memory_start"]
        )
        end_exists = any(
            (out / f"{stem}.rank{rank}.pickle").is_file() for stem in ["memory.end", "memory_end"]
        )
        if not start_exists or not end_exists:
            raise RuntimeError("A rank allocator snapshot is missing")


def profiles(data):
    cases = [
        ("shardtrain", "C0", True, None),
        ("shardtrain", "C1", False, None),
        ("shardtrain", "C2", False, None),
        ("shardtrain", "C3", False, None),
        ("native", "C0", False, None),
    ]
    for engine, policy, guarded, _injection in cases:
        label = "C0_guarded" if guarded else policy
        name = f"profile_{engine}_{label}"
        out = BASE / "results/profiles" / name
        out.mkdir(parents=True, exist_ok=False)
        execute(
            name,
            train_command(
                data,
                out,
                name,
                engine,
                policy=policy,
                guarded=guarded,
                profile=True,
                record=True,
                steps=1,
            ),
            timeout=900,
        )
        check_training(out, 1)
        export_capture(name, out)


def rank_skew(data, profile=False):
    settings = [("cpu", 0), ("cpu", 2), ("cpu", 10), ("cpu", 50), ("gpu", 0)]
    for repeat in range(1 if profile else 3):
        for mode, delay in settings if repeat % 2 == 0 else list(reversed(settings)):
            injection = {
                "rank": 0,
                "unit_id": 1,
                "phase": "FWD",
                "kind": "fsdp_ag",
                "mode": mode,
                "delay_ms": delay,
                "microstep_index": 0,
                "measured_only": True,
                "gemm_size": 2048,
                "gemm_repeats": 64,
            }
            name = f"{'profile' if profile else 'timing'}_skew_r{repeat}_{mode}_{delay}ms"
            out = BASE / "results/rank_skew" / name
            if profile:
                out.mkdir(parents=True, exist_ok=False)
            execute(
                name,
                train_command(
                    data,
                    out,
                    name,
                    steps=1 if profile else 30,
                    record=profile,
                    profile=profile,
                    injection=injection,
                ),
            )
            check_training(out, 1 if profile else 30)
            if profile:
                export_capture(name, out)


def q10(profile=False):
    for repeat in range(1 if profile else 3):
        for impl in ["fsdp1", "fsdp2"] if repeat % 2 == 0 else ["fsdp2", "fsdp1"]:
            name = f"q10_{'profile' if profile else 'timing'}_r{repeat}_{impl}"
            out = BASE / "results/q10" / name
            if profile:
                out.mkdir(parents=True, exist_ok=False)
            cmd = [
                PYTHON,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=4",
                "experiments/fsdp_interview/cases/q10_fsdp_only_benchmark.py",
                "--implementation",
                impl,
                "--output-dir",
                str(out),
                "--warmup",
                "10",
                "--steps",
                "30",
            ]
            if profile:
                cmd += [
                    "--profile",
                    "--profile-steps",
                    "2",
                    "--trace-path",
                    str(out / "timeline.nsys-rep"),
                ]
                cmd = [
                    NSYS,
                    "profile",
                    "--trace=cuda,nvtx,osrt",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--cuda-memory-usage=true",
                    "--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop",
                    "--force-overwrite=false",
                    "--output",
                    str(out / "timeline"),
                ] + cmd
            execute(name, cmd)
            if profile:
                export_capture(name, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=["capacity", "timing", "profiles", "skew", "skew_profiles", "q10", "q10_profiles"],
        required=True,
    )
    p.add_argument("--data", default=str(BASE / "data/wikitext103_rows_s512/tokens.npy"))
    args = p.parse_args()
    source = Path(args.data)
    if not args.phase.startswith("q10") and not source.is_file():
        raise FileNotFoundError("Frozen data must be complete before model experiments")
    phases = {
        "capacity": lambda: capacity(source),
        "timing": lambda: main_timing(source),
        "profiles": lambda: profiles(source),
        "skew": lambda: rank_skew(source),
        "skew_profiles": lambda: rank_skew(source, True),
        "q10": lambda: q10(),
        "q10_profiles": lambda: q10(True),
    }
    phases[args.phase]()
    print(json.dumps({"phase": args.phase, "status": "PASS"}), flush=True)


if __name__ == "__main__":
    main()
