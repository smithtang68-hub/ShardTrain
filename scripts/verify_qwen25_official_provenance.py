"""Certify fixed-revision Qwen2.5-1.5B local bytes and bind optional oracle results.

No download or model loading occurs. Expected weight SHA-256 is the official HF
LFS object ID at the pinned revision, not a digest supplied by the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
API_URL = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
EXPECTED_FILES = {
    "config.json": {
        "bytes": 684,
        "sha256": "0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77",
        "authority": "SHA-256 of the config fetched directly from the pinned official resolve URL",
    },
    "model.safetensors": {
        "bytes": 3087467144,
        "sha256": "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796",
        "authority": "Official HF API sibling model.safetensors.lfs.sha256 at pinned revision",
    },
}


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_oracle(directory, checkpoint):
    files = sorted(directory.glob("checkpoint_oracle.rank*.json"))
    if not files:
        raise ValueError(f"No per-rank checkpoint oracle records in {directory}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    world = records[0]["world_size"]
    if len(records) != world or {row["rank"] for row in records} != set(range(world)):
        raise ValueError("Oracle rank files do not cover exactly the declared world")
    accepted_paths = {checkpoint.resolve(), (checkpoint / "model.safetensors").resolve()}
    for file, row in zip(files, records, strict=True):
        if file.name != f"checkpoint_oracle.rank{row['rank']}.json":
            raise ValueError("Oracle file name does not match its actual rank")
        if row["status"] != "passed" or row["errors"] or row["world_size"] != world:
            raise ValueError(f"Oracle was not accepted: {file.name}")
        if Path(row["source"]["checkpoint_path"]).resolve() not in accepted_paths:
            raise ValueError(f"Oracle used a different checkpoint path: {file.name}")
        if row["source"]["hf_config_sha256"] != EXPECTED_FILES["config.json"]["sha256"]:
            raise ValueError(f"Oracle configuration is not the pinned official bytes: {file.name}")
        parallel = row["parallel"]
        if parallel["F"] * parallel["T"] != world or row["coordinate"] != {
            "fsdp": row["rank"] // parallel["T"],
            "tp": row["rank"] % parallel["T"],
        }:
            raise ValueError(f"Oracle topology is inconsistent: {file.name}")
        if row["dtype"] != "fp32" or row["reference_device"] != "cpu":
            raise ValueError("Expected a FP32 runtime versus HF CPU checkpoint oracle")
        if row["optimizer_updates"] != 0 or not row["ownership"]["weights_unchanged"]:
            raise ValueError("Oracle updated checkpoint weights")
        if not row["ownership"]["stable_shard_identity"] or any(
            row["ownership"][key] != 0 for key in ["live_buffers", "live_fullparam_leases"]
        ):
            raise ValueError("Oracle did not retire its runtime lifecycle")
        comparisons = row["comparisons"]
        if {item["name"] for item in comparisons} != {"logits", "global_loss"} or not all(
            item["passed"]
            and math.isfinite(item["max_abs_error"])
            and item["actual_nonfinite_count"] == item["reference_nonfinite_count"] == 0
            for item in comparisons
        ):
            raise ValueError("Oracle numerical comparison was not accepted")
    return {
        "status": "passed",
        "directory": str(directory.resolve()),
        "world_size": world,
        "actual_multi_process_NCCL": world > 1
        and all(row["actual_NCCL"] and row["backend"] == "nccl" for row in records),
        "parallel": records[0]["parallel"],
        "max_abs_logits_error": max(
            item["max_abs_error"]
            for row in records
            for item in row["comparisons"]
            if item["name"] == "logits"
        ),
        "max_abs_global_loss_error": max(
            item["max_abs_error"]
            for row in records
            for item in row["comparisons"]
            if item["name"] == "global_loss"
        ),
        "records": [{"path": str(path.resolve()), "sha256": digest_file(path)} for path in files],
        "scope": "Fixed-input logits/loss checkpoint oracle; cleanup backward only, "
        "not gradient comparison, training benchmark or overlap certification",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--oracle-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output record already exists; choose a new --output to preserve prior data")
    payload = {
        "status": "failed",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "official_metadata_api_url": API_URL,
        "checkpoint_directory": str(args.checkpoint.resolve()),
        "official_checkpoint_bytes_verified": False,
        "official_checkpoint_numerical_reference_verified": False,
        "weights_downloaded_by_script": False,
        "files": [],
        "errors": [],
    }
    failed = None
    try:
        if not args.checkpoint.is_dir():
            raise ValueError("checkpoint must be the directory holding the official single file")
        if (args.checkpoint / "model.safetensors.index.json").exists():
            raise ValueError("Pinned official revision has one model.safetensors, not an index")
        for name, expected in EXPECTED_FILES.items():
            path = args.checkpoint / name
            row = {
                "filename": name,
                "source_url": f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}",
                "expected": expected,
            }
            payload["files"].append(row)
            if not path.is_file() or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"Missing/size-mismatched official file: {path}")
            row["actual_bytes"] = path.stat().st_size
            row["actual_sha256"] = digest_file(path)
            if row["actual_sha256"] != expected["sha256"]:
                raise ValueError(f"Official SHA-256 mismatch: {path}")
        payload["official_checkpoint_bytes_verified"] = True
        if args.oracle_dir is not None:
            payload["oracle"] = verify_oracle(args.oracle_dir, args.checkpoint)
            payload["official_checkpoint_numerical_reference_verified"] = True
        project = Path(__file__).resolve().parents[1]
        payload["project_source_sha256"] = {
            filename: digest_file(project / filename)
            for filename in [
                "scripts/verify_qwen25_official_provenance.py",
                "scripts/verify_qwen2_checkpoint.py",
                "src/sharded_runtime/models/qwen2_checkpoint.py",
                "src/sharded_runtime/runtime/qwen2_adapter.py",
                "src/sharded_runtime/runtime/qwen2_operators.py",
            ]
        }
        payload["status"] = "passed"
    except Exception as exc:
        failed = exc
        payload["errors"].append(repr(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    if failed is not None:
        parser.error(f"Official checkpoint provenance failed: {failed}; see {args.output}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
