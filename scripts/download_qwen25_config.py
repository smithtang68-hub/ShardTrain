"""Download only the pinned official Qwen2.5-1.5B config, never model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import yaml

MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/qwen25_1p5b_config.yaml"))
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".yaml", ".yml", ".json"}:
        parser.error("--output must use .yaml, .yml or .json")
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/config.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise RuntimeError("Refusing config response larger than 64 KiB")
    config = json.loads(raw)
    if config.get("model_type") != "qwen2":
        raise RuntimeError("Downloaded configuration is not model_type=qwen2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".json":
        args.output.write_bytes(raw)
    else:
        args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    provenance = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "source_url": url,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "downloaded_weights": False,
    }
    metadata = args.output.with_suffix(".provenance.json")
    metadata.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
