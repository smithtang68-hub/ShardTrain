"""Read-only environment, exact APIs, clock/power and topology inventory."""

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    commands = {
        "nvidia_smi.txt": ["nvidia-smi"],
        "topology.txt": ["nvidia-smi", "topo", "-m"],
        "p2p_read.txt": ["nvidia-smi", "topo", "-p2p", "r"],
        "p2p_write.txt": ["nvidia-smi", "topo", "-p2p", "w"],
        "gpu_clocks_power.csv": [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,pstate,clocks.sm,clocks.mem,clocks.max.sm,power.draw,power.limit,temperature.gpu,memory.total,memory.free,utilization.gpu",
            "--format=csv",
        ],
        "cpu.txt": ["lscpu"],
        "processes.txt": ["ps", "-eo", "pid,ppid,comm,args"],
        "disk.txt": ["df", "-h"],
        "host_ram.txt": ["free", "-h"],
    }
    codes = {}
    for name, cmd in commands.items():
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            (out / name).write_text(r.stdout + r.stderr, encoding="utf-8")
            codes[name] = r.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            codes[name] = repr(error)
    from torch.distributed.distributed_c10d import ProcessGroupNCCL
    from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecisionPolicy, fully_shard

    packages = {}
    for name in ["torch", "transformers", "safetensors", "pyarrow", "huggingface-hub"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "timestamp_unix": time.time(),
        "python": platform.python_version(),
        "python_executable": os.sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "packages": packages,
        "commands": codes,
        "fully_shard_signature": str(inspect.signature(fully_shard)),
        "mixed_precision_policy_signature": str(inspect.signature(MixedPrecisionPolicy)),
        "fsdp1_signature": str(inspect.signature(FullyShardedDataParallel)),
        "process_group_nccl_options_doc": ProcessGroupNCCL.Options.__doc__,
        "process_group_nccl_fields": [
            x for x in dir(ProcessGroupNCCL.Options()) if not x.startswith("_")
        ],
        "executables": {x: shutil.which(x) for x in ["nsys", "nvcc", "git", "make", "g++"]},
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("NCCL_", "TORCH_NCCL_", "PYTORCH_CUDA_ALLOC_CONF"))
        },
        "devices": [
            {
                "rank": i,
                "name": torch.cuda.get_device_name(i),
                "capability": list(torch.cuda.get_device_capability(i)),
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
                "native_bf16": torch.cuda.is_bf16_supported(including_emulation=False),
            }
            for i in range(torch.cuda.device_count())
        ],
    }
    (out / "environment.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
