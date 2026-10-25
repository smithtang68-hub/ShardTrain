"""Independent native HF arithmetic controls, no custom model or distributed execution."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import torch  # noqa: E402

from experiments.finalization_4gpu.legacy_fp16_partition_control import (  # noqa: E402
    MODEL_NAMES,
    fixture_contract,
    patch_native_linears,
)
from experiments.fp16_numerics.common import (  # noqa: E402
    BIAS,
    OPTIMIZER,
    cpu,
    distribution,
    immutable_json,
    named_hash,
    pack,
    sha,
)
from experiments.fp16_numerics.replay_adamw import replay_one  # noqa: E402


def run(args):
    if not 1 <= args.updates <= 10:
        raise ValueError("Bounded control supports one through ten updates")
    if importlib.metadata.version("transformers") != "4.57.3":
        raise RuntimeError("Versioned HF oracle requires transformers==4.57.3")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Fresh control directory required")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    config, fixture = fixture_contract()
    torch.manual_seed(313)
    hfconfig = Qwen2Config(**config)
    hfconfig._attn_implementation = "sdpa"
    base = Qwen2ForCausalLM(hfconfig).train()
    models = {name: copy.deepcopy(base).to(device) for name in MODEL_NAMES}
    del base
    for name in MODEL_NAMES[1:]:
        patch_native_linears(models[name])
    params = {name: dict(model.named_parameters()) for name, model in models.items()}
    hashes = {name: named_hash(values) for name, values in params.items()}
    if len(set(hashes.values())) != 1:
        raise AssertionError("Control models did not start with identical global weights")
    optimizers = {
        name: torch.optim.AdamW(values.values(), **OPTIMIZER) for name, values in params.items()
    }
    scalers = {
        name: torch.amp.GradScaler(device.type, init_scale=128.0, growth_interval=2)
        for name in MODEL_NAMES
    }
    result = {
        "schema_version": 1,
        "capture_status": "FAILED",
        "strict_status": "NOT_RUN",
        "mode": "pure_hf_independent",
        "actual_custom_runtime_executed": False,
        "actual_distributed_collectives": False,
        "actual_SP_execution": False,
        "reference_synced_each_step": False,
        "performance_claim_allowed": False,
        "gpu_test_pass": False,
        "device": str(device),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "source_manifest_sha256": __import__("os").environ.get("SHARDTRAIN_SOURCE_MANIFEST_SHA256"),
        "config": config,
        "optimizer_options": OPTIMIZER,
        "seed": 313,
        "initial_scale": 128.0,
        "growth_interval": 2,
        "initial_parameter_hashes": hashes,
        "requested_effective_updates": args.updates,
        "completed_effective_updates": 0,
        "steps": [],
        "errors": [],
        "limitations": [
            "Native HF full-head SDPA remains; no per-TP attention/GQA split.",
            "Native full-sequence loss; no actual SP CE. Row partial sums emulate FP16 rounding.",
            (
                "TP2/F2 scales native means by local_count*2/global_count, then "
                "FP32 averages gradients."
            ),
            (
                "No NCCL, custom autograd, shard leaf, parameter materialization, "
                "stream scheduling or actual SP."
            ),
            "Pure control FAIL does not exclude a ShardTrain bug; CPU output is not V100 evidence.",
        ],
    }
    if device.type == "cuda":
        result["environment"] = {
            "gpu": torch.cuda.get_device_name(device),
            "uuid": str(torch.cuda.get_device_properties(device).uuid),
            "capability": list(torch.cuda.get_device_capability(device)),
            "cuda": torch.version.cuda,
        }
    began = time.perf_counter()
    try:
        for step in range(args.updates):
            batches = [
                [value.to(device) for value in fixture["token_batch"](replica, step)]
                for replica in range(2)
            ]
            full = [torch.cat([batch[column] for batch in batches]) for column in range(4)]
            counts = [int(batch[1][:, 1:].ne(-100).sum()) for batch in batches]
            snapshots, logits = {}, {}
            for name in MODEL_NAMES:
                model, optimizer, scaler = models[name], optimizers[name], scalers[name]
                values = params[name]
                is_f2 = name == MODEL_NAMES[2]
                before = {key: cpu(value) for key, value in values.items()}
                pre_forward_hash = named_hash(values)
                # Native HF grads are independent for each arithmetic replica.
                accumulated = {key: torch.zeros_like(value) for key, value in values.items()}
                contributions, parts, loss = [], [], 0.0
                for batch in batches if is_f2 else [full]:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, dtype=torch.float16, cache_enabled=False):
                        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                            prediction = model(
                                input_ids=batch[0],
                                labels=batch[1],
                                attention_mask=batch[2],
                                position_ids=batch[3],
                                use_cache=False,
                            )
                    count = int(batch[1][:, 1:].ne(-100).sum())
                    objective = (
                        prediction.loss * count * 2 / sum(counts) if is_f2 else prediction.loss
                    )
                    loss += float(prediction.loss.detach()) * count / sum(counts)
                    parts.append(cpu(prediction.logits))
                    scaler.scale(objective).backward()
                    contributions.append(cpu(values[BIAS].grad))
                    for key, value in values.items():
                        accumulated[key].add_(value.grad)
                for key, value in values.items():
                    value.grad = accumulated[key] / (2 if is_f2 else 1)
                scaled = cpu(values[BIAS].grad)
                scale_before = scaler.get_scale()
                scaler.unscale_(optimizer)
                if not all(torch.isfinite(value.grad).all() for value in values.values()):
                    raise FloatingPointError("Pure HF control encountered nonfinite gradient")
                pre = pack(values[BIAS], optimizer)
                gradients = {key: cpu(value.grad) for key, value in values.items()}
                scaler.step(optimizer)
                scaler.update()
                post = pack(values[BIAS], optimizer)
                replay = replay_one(pre, post, device=str(device))
                if not all(row["bitwise_equal"] for row in replay["residuals"].values()):
                    raise AssertionError("Same-device pure HF AdamW replay was not bit-exact")
                snapshots[name] = {
                    "pre": pre,
                    "post": post,
                    "before": before,
                    "gradients": gradients,
                    "after": {key: cpu(value) for key, value in values.items()},
                    "loss": loss,
                    "scale_before": scale_before,
                    "scale_after": scaler.get_scale(),
                    "pre_forward_hash": pre_forward_hash,
                    "gradient_contributions_scaled": contributions,
                    "gradient_reduced_scaled": scaled,
                    "replay": replay,
                }
                logits[name] = torch.cat(parts)
            row = {
                "step": step,
                "effective_update": step + 1,
                "valid_tokens": sum(counts),
                "local_valid_tokens": counts,
                "models": {
                    name: {
                        key: data[key]
                        for key in (
                            "loss",
                            "scale_before",
                            "scale_after",
                            "pre_forward_hash",
                            "replay",
                        )
                    }
                    for name, data in snapshots.items()
                },
                "comparisons": {},
            }
            for name in MODEL_NAMES[1:]:
                actual, expected = snapshots[name], snapshots[MODEL_NAMES[0]]
                pair = {
                    "loss_abs": abs(actual["loss"] - expected["loss"]),
                    "logits": distribution(logits[name], logits[MODEL_NAMES[0]]),
                    "parameter_gradients": {},
                    "parameters_after": {},
                }
                for key in actual["gradients"]:
                    pair["parameter_gradients"][key] = distribution(
                        actual["gradients"][key], expected["gradients"][key]
                    )
                    pair["parameters_after"][key] = distribution(
                        actual["after"][key], expected["after"][key]
                    )
                pair["all_parameters"] = distribution(
                    torch.cat([v.reshape(-1) for v in actual["after"].values()]),
                    torch.cat([v.reshape(-1) for v in expected["after"].values()]),
                )
                pair["all_gradients"] = distribution(
                    torch.cat([v.reshape(-1) for v in actual["gradients"].values()]),
                    torch.cat([v.reshape(-1) for v in expected["gradients"].values()]),
                )
                pair["strict_parameter_failures"] = [
                    key for key, stats in pair["parameters_after"].items() if not stats["passed"]
                ]
                row["comparisons"][name] = pair
                capture = {
                    "schema_version": 1,
                    "step": step,
                    "effective_update": step + 1,
                    "mode": "pure_hf_independent",
                    "reference_synced_from_runtime": False,
                    "logical_parameter": BIAS,
                    "global_indices": list(range(16)),
                    "runtime": {"pre": actual["pre"], "post": actual["post"]},
                    "reference": {"pre": expected["pre"], "post": expected["post"]},
                    "actual_model": name,
                    "reference_model": MODEL_NAMES[0],
                    "actual_custom_runtime_executed": False,
                    "source_manifest_sha256": result["source_manifest_sha256"],
                    "arithmetic_contributions_scaled": actual["gradient_contributions_scaled"],
                    "actual_reduced_scaled": actual["gradient_reduced_scaled"],
                }
                tensor_path = output / f"{name}.step{step:02d}.pt"
                if tensor_path.exists():
                    raise FileExistsError(tensor_path)
                torch.save(capture, tensor_path)
                pair.update(tensor_path=tensor_path.name, tensor_sha256=sha(tensor_path))
            immutable_json(output / f"step{step:02d}.json", row)
            result["steps"].append(row)
            result["completed_effective_updates"] += 1
        result["strict_status"] = (
            "FAIL"
            if any(
                pair["strict_parameter_failures"]
                for row in result["steps"]
                for pair in row["comparisons"].values()
            )
            else "PASS"
        )
        result["capture_status"] = "COMPLETED"
    except BaseException as error:
        result["errors"].append(repr(error))
        raise
    finally:
        result["diagnostic_wall_seconds"] = time.perf_counter() - began
        immutable_json(output / "control.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
