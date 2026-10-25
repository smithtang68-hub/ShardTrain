"""CPU-only comparison and data-assignment contracts; no CUDA imported."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

TOKEN_SHA = "9bc98fb7e97dc261eabf597e550bb9f403a9ab71d8d97956bb31ffa7def9f5ea"
CASES = {
    "S-H": dict(engine="shardtrain", F=2, T=2, SP=True, microbatch=4),
    "S-F": dict(engine="shardtrain", F=4, T=1, SP=False, microbatch=2),
    "N-H": dict(engine="native", F=2, T=2, SP=True, microbatch=4),
    "N-F": dict(engine="native", F=4, T=1, SP=False, microbatch=2),
}
ORDERS = [["S-F", "N-H", "N-F", "S-H"], ["S-H", "N-F", "N-H", "S-F"], ["N-F", "S-F", "S-H", "N-H"]]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write("\n")


def groups(case):
    c = CASES[case]
    return {
        "fsdp": [[f * c["T"] + t for f in range(c["F"])] for t in range(c["T"])],
        "tp": [[f * c["T"] + t for t in range(c["T"])] for f in range(c["F"])],
    }


def input_blocks(case, rank, step, block_count=1024):
    c = CASES[case]
    if not 0 <= rank < 4 or block_count < 640 or block_count % 16:
        raise ValueError("Fixed four-rank input contract requires >=640 blocks, divisible by16")
    local = c["microbatch"] * 2
    start = step * 16 % block_count + rank // c["T"] * local
    return list(range(start, start + local))


def comparison_contract():
    microsteps = {}
    for case, c in CASES.items():
        microsteps[case] = [
            sorted(
                block
                for f in range(c["F"])
                for block in input_blocks(case, f * c["T"], 0)[
                    m * c["microbatch"] : (m + 1) * c["microbatch"]
                ]
            )
            for m in range(2)
        ]
    return {
        "schema_version": 1,
        "status": "SOURCE_AUDITED_NOT_GPU_VALIDATED",
        "cases": {k: {**v, "accumulation": 2, "groups": groups(k)} for k, v in CASES.items()},
        "model": "complete random Qwen2.5-1.5B;28L;unique_parameters1543714304",
        "seq_len": 512,
        "global_batch": 16,
        "input_positions": 8192,
        "valid_labels": 8176,
        "tokens_sha256": TOKEN_SHA,
        "seed": 1234,
        "optimizer": "AdamW lr1e-5 foreach=False",
        "precision": "FP16 linear/MLP/head; FP32 math-SDPA/master/AG/RS/Adam state",
        "sync": "Every microbatch FSDP gradient synchronization; no native-only no_sync",
        "timing": "Existing train_workload per-step max-rank whole-device wall time;"
        "precache/hash outside",
        "source": "Every actual global parameter reconstructed and hashed before timed warmup",
        "microstep_global_block_ids_step0": microsteps,
        "data_difference": "Full optimizer-step16-block set identical; "
        "H/F microstep8-block subsets "
        "differ under existing replica-contiguous input path. "
        "Same-layout self/native subsets identical.",
        "native_F4": "Experiment-only TP1 no-op plan; existing official fully_shard unchanged",
        "comparisons": [["S-F", "N-F"], ["S-H", "N-H"], ["S-F", "S-H"], ["N-F", "N-H"]],
        "limits": [
            "Strategy changes F/T/SP/local GEMM/data replicas together, "
            "not single-variable causality",
            "Strict FP16 independent FAIL remains; P0 gate is separate",
            "No native optimum search",
        ],
    }
