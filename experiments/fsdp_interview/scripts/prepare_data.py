"""Freeze a bounded real WikiText-103 prefix into deterministic token blocks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--blocks", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    if args.seq_len <= 0 or args.blocks <= 0:
        raise ValueError("Sequence length and block count must be positive")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "tokens.npy").exists():
        raise FileExistsError("Keep previous input evidence; choose a fresh output directory")
    started = time.time()
    meta = {
        "seed": args.seed,
        "seq_len": args.seq_len,
        "blocks": args.blocks,
        "created_unix": started,
        "source": "synthetic" if args.synthetic else "WikiText-103 train",
        "packing": "Continuous nonempty text rows, EOS separator; "
        "truncate prefix to exact block count",
        "model_repository": "Qwen/Qwen2.5-1.5B",
    }
    if args.synthetic:
        tokens = np.random.default_rng(args.seed).integers(
            0, 151936, (args.blocks, args.seq_len), dtype=np.int64
        )
        meta["fallback_reason"] = "Explicit synthetic invocation; not real text"
    else:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
        from transformers import AutoTokenizer

        api = HfApi()
        dataset_commit = api.dataset_info("Salesforce/wikitext").sha
        tokenizer_commit = api.model_info("Qwen/Qwen2.5-1.5B").sha
        cache = out / "download_cache"
        filename = "wikitext-103-raw-v1/train-00000-of-00002.parquet"
        parquet = hf_hub_download(
            "Salesforce/wikitext",
            filename,
            repo_type="dataset",
            revision=dataset_commit,
            cache_dir=cache,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B", revision=tokenizer_commit, cache_dir=cache, trust_remote_code=False
        )
        needed = args.blocks * args.seq_len
        packed, consumed, text_hash = [], 0, hashlib.sha256()
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=128, columns=["text"]):
            texts = [x for x in batch.column(0).to_pylist() if x.strip()]
            for text, ids in zip(
                texts, tokenizer(texts, add_special_tokens=False)["input_ids"], strict=True
            ):
                consumed += 1
                text_hash.update(text.encode("utf-8"))
                text_hash.update(b"\0")
                packed.extend(ids)
                packed.append(tokenizer.eos_token_id)
                if len(packed) >= needed:
                    break
            if len(packed) >= needed:
                break
        if len(packed) < needed:
            raise ValueError("Selected train shard does not contain enough tokens")
        tokens = np.asarray(packed[:needed], dtype=np.int64).reshape(args.blocks, args.seq_len)
        meta.update(
            dataset_commit=dataset_commit,
            tokenizer_commit=tokenizer_commit,
            dataset_filename=filename,
            parquet_sha256=sha256(parquet),
            consumed_nonempty_rows=consumed,
            text_prefix_sha256=text_hash.hexdigest(),
            tokenizer_eos_token_id=tokenizer.eos_token_id,
            versions={
                x: importlib.metadata.version(x)
                for x in ["pyarrow", "transformers", "huggingface-hub"]
            },
        )
    if tokens.min() < 0 or tokens.max() >= 151936:
        raise ValueError("Token IDs outside the configured Qwen vocabulary")
    np.save(out / "tokens.npy", tokens, allow_pickle=False)
    meta.update(
        token_count=int(tokens.size),
        min_id=int(tokens.min()),
        max_id=int(tokens.max()),
        tokens_sha256=sha256(out / "tokens.npy"),
        elapsed_seconds=time.time() - started,
    )
    (out / "dataset.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
