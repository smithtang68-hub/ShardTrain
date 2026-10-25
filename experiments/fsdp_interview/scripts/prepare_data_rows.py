"""Bounded origin API fallback when WikiText CDN parquet transfers time out.

The exact JSON responses and tokenizer revision are frozen. The rows service is
not revision-addressed: do not claim its responses match the dataset git commit.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests
from huggingface_hub import HfApi
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blocks", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.blocks <= 0 or args.seq_len <= 0:
        raise ValueError("Positive blocks and sequence length required")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=args.resume)
    if (out / "tokens.npy").exists():
        raise FileExistsError("Do not replace a completed frozen input")
    api = HfApi()
    revision = api.model_info("Qwen/Qwen2.5-1.5B").sha
    dataset_head = api.dataset_info("Salesforce/wikitext").sha
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        revision=revision,
        cache_dir=out / "tokenizer_cache",
        trust_remote_code=False,
    )
    needed, packed, raw_rows, offset = args.seq_len * args.blocks, [], [], 0
    while len(packed) < needed and offset < 20000:
        query = urlencode(
            dict(
                dataset="Salesforce/wikitext",
                config="wikitext-103-raw-v1",
                split="train",
                offset=offset,
                length=100,
            )
        )
        url = "https://datasets-server.huggingface.co/rows?" + query
        cached = out / f"rows_{offset:06d}.json"
        if cached.exists():
            raw = cached.read_bytes()
            record = json.loads(raw)
        else:
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=(20, 60))
                    response.raise_for_status()
                    raw = response.content
                    record = response.json()
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            cached.write_bytes(raw)
        raw_rows.append(
            {
                "offset": offset,
                "url": url,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
        texts = [row["row"]["text"] for row in record["rows"] if row["row"]["text"].strip()]
        for ids in tok(texts, add_special_tokens=False)["input_ids"]:
            packed.extend(ids)
            packed.append(tok.eos_token_id)
        offset += len(record["rows"])
        if not record["rows"]:
            break
        print(json.dumps({"rows": offset, "tokens": len(packed), "needed": needed}), flush=True)
    if len(packed) < needed:
        raise RuntimeError("Origin rows service did not provide the requested token count")
    tokens = np.asarray(packed[:needed], dtype=np.int64).reshape(args.blocks, args.seq_len)
    if tokens.min() < 0 or tokens.max() >= 151936:
        raise ValueError("Tokens outside Qwen vocabulary")
    np.save(out / "tokens.npy", tokens, allow_pickle=False)
    meta = {
        "source": "WikiText-103 train",
        "source_transport": "Hugging Face datasets-server rows API",
        "dataset_repository": "Salesforce/wikitext",
        "dataset_config": "wikitext-103-raw-v1",
        "dataset_head_observed": dataset_head,
        "revision_limit": "rows service is not revision-addressed; "
        "exact cached response hashes define input",
        "tokenizer_repository": "Qwen/Qwen2.5-1.5B",
        "tokenizer_commit": revision,
        "seq_len": args.seq_len,
        "blocks": args.blocks,
        "token_count": int(tokens.size),
        "tokens_sha256": hashlib.sha256((out / "tokens.npy").read_bytes()).hexdigest(),
        "min_id": int(tokens.min()),
        "max_id": int(tokens.max()),
        "created_unix": time.time(),
        "raw_responses": raw_rows,
        "packing": "nonempty train prefix text with EOS separators; fixed exact blocks",
    }
    (out / "dataset.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "dataset": str(out / "dataset.json")}), flush=True)


if __name__ == "__main__":
    main()
