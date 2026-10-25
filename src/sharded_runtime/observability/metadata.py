import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from sharded_runtime.config import Config
from sharded_runtime.distributed.context import WorkerEnvironment


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def write_json(filename: Path, data: Any) -> None:
    filename.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_metadata(
    directory: Path, config: Config, worker: WorkerEnvironment, device: torch.device
) -> None:
    rank = worker.rank
    (directory / f"resolved_config.rank{rank}.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    write_json(
        directory / f"environment.rank{rank}.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "packages": {
                package: importlib.metadata.version(package) for package in ("torch", "PyYAML")
            },
            "cuda_build": torch.version.cuda,
            "nccl_available": torch.distributed.is_nccl_available(),
            "gloo_available": torch.distributed.is_gloo_available(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "rank": rank,
            "local_rank": worker.local_rank,
            "world_size": worker.world_size,
            "git_commit": git_revision(),
            "milestone": "v0.7 M1-M3",
            "scope": "reference, TP/SP, shard runtime or group smoke; see resolved config",
            "nccl_correctness_verified": False,
        },
    )
