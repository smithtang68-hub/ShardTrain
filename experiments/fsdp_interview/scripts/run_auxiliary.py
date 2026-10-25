"""Owned auxiliary experiments; every output directory is fresh and bounded."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_matrix import (  # noqa: E402
    BASE,
    NSYS,
    PYTHON,
    check_training,
    execute,
    export_capture,
    train_command,
)


def capture_command(out, command):
    return [
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
    ] + command


def export(name, out, *, distributed=False):
    rep = out / "timeline.nsys-rep"
    if not rep.is_file() or rep.stat().st_size == 0:
        raise RuntimeError(f"Missing Nsight report: {rep}")
    execute(
        name + "_export",
        [
            NSYS,
            "export",
            "--type=sqlite",
            "--force-overwrite=false",
            "--output",
            str(out / "timeline.sqlite"),
            str(rep),
        ],
        timeout=600,
    )
    if not (out / "timeline.sqlite").is_file():
        raise RuntimeError("Missing SQLite export")
    if not distributed:
        for stem in ["memory_start", "memory_end"]:
            if not (out / f"{stem}.pickle").is_file():
                raise RuntimeError(f"Missing allocator snapshot {stem}")


def pinned(data):
    script = "experiments/fsdp_interview/cases/q01_q03_pinned_memory.py"
    for repeat in range(3):
        name = f"pinned_timing_r{repeat}"
        out = BASE / "results/pinned" / name
        out.mkdir(parents=True, exist_ok=False)
        execute(
            name,
            [
                PYTHON,
                script,
                "--output-dir",
                str(out),
                "--tokens-npy",
                str(data),
                "--no-record-memory",
            ],
            env={
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTORCH_ALLOC_CONF": "",
                "PYTORCH_CUDA_ALLOC_CONF": "",
            },
            timeout=900,
        )
        if json.loads((out / "summary.json").read_text())["status"] != "PASS":
            raise RuntimeError("Pinned correctness failed")
    for label, allocator in [
        ("default", None),
        ("hostregister", "pinned_use_cuda_host_register:True,pinned_num_register_threads:4"),
    ]:
        name = f"pinned_profile_{label}"
        out = BASE / "results/pinned" / name
        out.mkdir(parents=True, exist_ok=False)
        command = [
            PYTHON,
            script,
            "--output-dir",
            str(out),
            "--tokens-npy",
            str(data),
            "--profile",
            "--iterations",
            "3",
            "--warmup",
            "2",
            "--repetitions",
            "1",
            "--sizes-mib",
            "128",
        ]
        env = {"CUDA_VISIBLE_DEVICES": "0", "PYTORCH_ALLOC_CONF": "", "PYTORCH_CUDA_ALLOC_CONF": ""}
        if allocator:
            env["PYTORCH_ALLOC_CONF"] = allocator
            # This installed 2.9.1 host allocator still reads the legacy key.
            # Both keys agree; only actual traced CUDA APIs qualify this path.
            env["PYTORCH_CUDA_ALLOC_CONF"] = allocator
        execute(name, capture_command(out, command), env=env, timeout=900)
        if json.loads((out / "summary.json").read_text())["status"] != "PASS":
            raise RuntimeError("Pinned profile correctness failed")
        export(name, out)


def q9_roundtrip():
    script = "experiments/fsdp_interview/cases/q09_param_representation.py"
    for source_world, target_world in [(4, 2), (2, 4)]:
        checkpoint = BASE / f"results/q09/roundtrip_{source_world}_to_{target_world}/checkpoint"
        for action, world in [("save", source_world), ("load", target_world)]:
            name = f"q09_roundtrip_{source_world}_to_{target_world}_{action}_w{world}"
            out = BASE / "results/q09" / name
            out.mkdir(parents=True, exist_ok=False)
            execute(
                name,
                [
                    PYTHON,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={world}",
                    script,
                    "--implementation",
                    "all",
                    "--output-dir",
                    str(out),
                    "--roundtrip-action",
                    action,
                    "--checkpoint-dir",
                    str(checkpoint),
                ],
                env={"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(world))},
                timeout=600,
            )


def gemm_shapes(data):
    # Re-enabling local DTensor dispatch is first qualified on real Linux/GPU.
    cpu = BASE / "results/gemm_shapes/cpu_linux"
    execute(
        "gemm_shapes_cpu_linux",
        [
            PYTHON,
            "experiments/fsdp_interview/cases/gemm_shapes.py",
            "--cpu-selfcheck",
            "--output-dir",
            str(cpu),
        ],
        timeout=180,
    )
    cpu_result = json.loads((cpu / "gemm_shapes_cpu_selfcheck.json").read_text())
    if cpu_result["dtensor_world1"]["status"] != "PASS":
        raise RuntimeError("Real Linux CPU DTensor shape qualification failed")
    smoke_results = {}
    for engine in ["shardtrain", "native"]:
        name = f"gemm_shapes_smoke_4L_{engine}"
        out = BASE / "results/gemm_shapes" / name
        cmd = train_command(data, out, name, engine, warmup=1, steps=2)
        cmd += ["--layers", "4", "--gemm-shapes"]
        execute(name, cmd, timeout=600)
        smoke_results[engine] = check_training(out, 2)
        for rank in range(4):
            manifest = json.loads((out / f"gemm_shapes_manifest.rank{rank}.json").read_text())
            if not manifest["restored"] or manifest["dropped_records"]:
                raise RuntimeError("GEMM mode did not restore or exceeded its bound")
            rows = [
                json.loads(line)
                for line in (out / f"gemm_shapes.rank{rank}.jsonl").read_text().splitlines()
            ]
            if not any(row["classification"] == "LOCAL" for row in rows):
                raise RuntimeError("No measured local matrix call")
            if engine == "native" and not any(
                row["shape_source"] == "DTENSOR_DISPATCH_PLAIN_LOCAL_ATEN"
                and row["classification"] == "LOCAL"
                for row in rows
            ):
                raise RuntimeError("Native final local matrix call was not observed")
    for custom, native in zip(
        smoke_results["shardtrain"][0]["steps"], smoke_results["native"][0]["steps"], strict=True
    ):
        if abs(custom["loss"] - native["loss"]) > 1e-3:
            raise RuntimeError("Shape smoke matched loss differs beyond 1e-3")
    for engine, policy in [("shardtrain", "C0"), ("shardtrain", "C1"), ("native", "C0")]:
        name = f"gemm_shapes_profile_{engine}_{policy}"
        out = BASE / "results/gemm_shapes" / name
        out.mkdir(parents=True, exist_ok=False)
        cmd = train_command(
            data, out, name, engine, policy=policy, steps=1, record=True, profile=True
        )
        cmd.append("--gemm-shapes")
        execute(name, cmd, timeout=900)
        check_training(out, 1)
        export_capture(name, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pinned", "q9_roundtrip", "gemm_shapes"], required=True)
    parser.add_argument("--data", default=str(BASE / "data/wikitext103_rows_s512/tokens.npy"))
    args = parser.parse_args()
    if args.phase == "pinned":
        pinned(Path(args.data))
    elif args.phase == "q9_roundtrip":
        q9_roundtrip()
    else:
        gemm_shapes(Path(args.data))
    print(json.dumps({"phase": args.phase, "status": "PASS"}), flush=True)


if __name__ == "__main__":
    main()
