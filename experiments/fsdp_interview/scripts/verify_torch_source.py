"""Compare installed FSDP Python modules with the exact reported official commit."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import requests
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    modules = [
        "torch.distributed.fsdp._runtime_utils",
        "torch.distributed.fsdp._fully_shard._fsdp_param_group",
        "torch.distributed.fsdp._fully_shard._fsdp_collectives",
        "torch.distributed.fsdp._fully_shard._fsdp_state",
    ]
    result = {"torch": torch.__version__, "git_commit": torch.version.git_version, "files": []}
    for module in modules:
        installed = Path(importlib.import_module(module).__file__)
        relative = module.replace(".", "/") + ".py"
        url = f"https://raw.githubusercontent.com/pytorch/pytorch/{torch.version.git_version}/{relative}"
        response = requests.get(url, timeout=(15, 60))
        response.raise_for_status()
        expected = response.content
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
        local_hash = hashlib.sha256(installed.read_bytes()).hexdigest()
        official_hash = hashlib.sha256(expected).hexdigest()
        result["files"].append(
            {
                "module": module,
                "installed_path": str(installed),
                "official_url": url,
                "installed_sha256": local_hash,
                "official_sha256": official_hash,
                "status": "PASS" if local_hash == official_hash else "DIFF",
            }
        )
    result["status"] = "PASS" if all(x["status"] == "PASS" for x in result["files"]) else "DIFF"
    (out / "source_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
