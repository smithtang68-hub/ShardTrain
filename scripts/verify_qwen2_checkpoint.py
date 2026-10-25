"""Finite-input FP32 checkpoint oracle: HF CPU versus the real ShardTrain runtime.

This performs one backward solely to retire runtime units. No optimizer is
created and no checkpoint weights are updated. It downloads nothing.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from sharded_runtime.config import (
    Config,
    ConfigError,
    DebugConfig,
    ModelConfig,
    ParallelConfig,
    RuntimeConfig,
    TrainingConfig,
)
from sharded_runtime.distributed.context import WorkerEnvironment
from sharded_runtime.launch.bootstrap import WorkerSession
from sharded_runtime.models.qwen2 import Qwen2ModelConfig
from sharded_runtime.models.qwen2_factory import build_qwen_model
from sharded_runtime.observability.metadata import write_json
from sharded_runtime.observability.profiling import ProfileCapture
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.training.qwen2_trainer import training_range


def _compare(actual, expected, name, atol, rtol):
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    error = (actual - expected).abs()
    return {
        "name": name,
        "shape": list(actual.shape),
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(expected.dtype),
        "max_abs_error": error.max().item(),
        "max_rel_error": (error / expected.abs().clamp_min(1e-12)).max().item(),
        "atol": atol,
        "rtol": rtol,
        "passed": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "actual_nonfinite_count": int((~torch.isfinite(actual)).sum()),
        "reference_nonfinite_count": int((~torch.isfinite(expected)).sum()),
    }


def _shard_digest(runtime):
    digest = hashlib.sha256()
    for name, handle in sorted(runtime.registry.items()):
        digest.update(name.encode())
        value = handle.local_shard.detach().cpu().contiguous().view(torch.uint8)
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _token_batch(args, vocabulary, replica):
    generator = torch.Generator().manual_seed(args.seed + replica)
    tokens = torch.randint(vocabulary, (args.micro_batch, args.seq_len), generator=generator)
    return (
        tokens,
        tokens.clone(),
        torch.ones_like(tokens),
        torch.arange(args.seq_len).expand(args.micro_batch, -1),
    )


def _hf_reference(args, qwen, checkpoint_directory, replica):
    # A genuine independent HF loader, rather than passing our reader's tensors
    # into the reference model. Each rank uses CPU, never a dense GPU fallback.
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**qwen.to_dict())
    config.use_cache = False
    reference, loading = Qwen2ForCausalLM.from_pretrained(
        str(checkpoint_directory),
        config=config,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    allowed_missing = {"lm_head.weight"} if qwen.tie_word_embeddings else set()
    if (
        set(loading.get("missing_keys", [])) - allowed_missing
        or loading.get("unexpected_keys")
        or loading.get("mismatched_keys")
        or loading.get("error_msgs")
    ):
        raise ValueError(f"HF checkpoint loading was not strict: {loading}")
    if (
        qwen.tie_word_embeddings
        and reference.lm_head.weight is not reference.model.embed_tokens.weight
    ):
        raise ValueError("HF tied embeddings are not an actual Parameter alias")
    reference.eval()
    batches = [_token_batch(args, qwen.vocab_size, index) for index in range(args.f)]
    combined = [torch.cat([batch[index] for batch in batches]) for index in range(4)]
    with torch.no_grad():
        expected = reference(
            input_ids=combined[0],
            labels=combined[1],
            attention_mask=combined[2],
            position_ids=combined[3],
            use_cache=False,
        )
    start = replica * args.micro_batch
    logits = expected.logits[start : start + args.micro_batch].clone()
    loss = expected.loss.clone()
    del reference, expected
    gc.collect()
    return batches[replica], logits, loss, loading


def run(args):
    worker = WorkerEnvironment.read()
    directory = Path(args.directory)
    if not args.directory.strip():
        raise ConfigError("output directory must not be empty")
    existing = [
        directory / f"{prefix}.rank{worker.rank}.json"
        for prefix in ["environment", "checkpoint_oracle", "profiling"]
        if (directory / f"{prefix}.rank{worker.rank}.json").exists()
    ]
    if existing:
        raise ConfigError(
            "Output directory already contains experiment records; choose a new "
            f"--output-dir to preserve prior data: {', '.join(path.name for path in existing)}"
        )
    payload = {
        "status": "failed",
        "rank": worker.rank,
        "world_size": worker.world_size,
        "dtype": "fp32",
        "model_mode": "eval",
        "use_cache": False,
        "reference_device": "cpu",
        "reference_attention": "HF eager",
        "runtime_attention": "ShardTrain PyTorch SDPA math",
        "scope": "Fixed synthetic token input; checkpoint logits and global shifted-label loss. "
        "One backward retires units; gradients are not compared and no optimizer is created.",
        "optimizer_updates": 0,
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "profile_mode": args.profile_mode,
        "capture_scope": "Runtime forward and lifecycle-cleanup backward; "
        "HF reference and initialization excluded",
        "source": {
            "hf_config_path": str(Path(args.hf_config).resolve()),
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "weights_downloaded_by_script": False,
            "official_weight_origin_verified": False,
            "checkpoint_origin": "User-provided local safetensors; source origin is not certified",
        },
        "comparisons": [],
        "errors": [],
    }
    capture = runtime = None
    failure = None
    try:
        if args.seq_len < 2 or args.micro_batch < 1 or args.cpu_threads < 1:
            raise ValueError("seq-len >= 2, micro-batch >= 1 and cpu-threads >= 1 are required")
        if not all(math.isfinite(value) and value >= 0 for value in [args.atol, args.rtol]):
            raise ValueError("atol and rtol must be finite nonnegative numbers")
        if args.profile_mode == "nsight" and args.device != "cuda":
            raise ValueError("Nsight capture requires CUDA")
        torch.set_num_threads(args.cpu_threads)
        version = importlib.metadata.version("transformers")
        if version != "4.57.3":
            raise ValueError(f"Checkpoint oracle requires transformers==4.57.3; found {version}")
        payload["versions"] = {
            "torch": torch.__version__,
            "transformers": version,
            "safetensors": importlib.metadata.version("safetensors"),
            "cuda_build": torch.version.cuda,
        }
        qwen = Qwen2ModelConfig.from_file(args.hf_config)
        payload["source"]["validated_config"] = qwen.to_dict()
        config = Config(
            model=ModelConfig(
                model_type="qwen2",
                hf_config_path=str(Path(args.hf_config).resolve()),
                checkpoint_path=str(Path(args.checkpoint).resolve()),
                vocab_size=qwen.vocab_size,
                layers=qwen.num_hidden_layers,
                hidden_size=qwen.hidden_size,
                num_heads=qwen.num_attention_heads,
                num_key_value_heads=qwen.num_key_value_heads,
                mlp_hidden_size=qwen.intermediate_size,
                seq_len=args.seq_len,
                rms_norm_eps=qwen.rms_norm_eps,
                rope_theta=qwen.rope_theta,
                tie_word_embeddings=qwen.tie_word_embeddings,
            ),
            parallel=ParallelConfig(
                fsdp_degree=args.f,
                tp_degree=args.t,
                sp_enabled=args.sp,
                device=args.device,
                backend=args.backend,
            ),
            runtime=RuntimeConfig(reshard_after_forward=True, param_grouping="block"),
            training=TrainingConfig(
                dtype="fp32", seed=args.seed, local_batch_size=args.micro_batch, steps=1
            ),
            debug=DebugConfig(output_dir=args.directory, distributed_timeout_seconds=args.timeout),
        )
        config.validate(worker.world_size, mode="runtime")
        payload["source"]["hf_config_sha256"] = hashlib.sha256(
            Path(args.hf_config).read_bytes()
        ).hexdigest()
        provenance = Path(args.hf_config).with_suffix(".provenance.json")
        if provenance.is_file():
            payload["source"]["config_provenance"] = json.loads(
                provenance.read_text(encoding="utf-8")
            )
        with WorkerSession(config, "runtime") as session:
            mesh = session.context
            torch.manual_seed(args.seed + worker.rank)
            payload["coordinate"] = asdict(mesh.coordinate)
            payload["device"] = str(mesh.device)
            payload["backend"] = dist.get_backend() if dist.is_initialized() else "none"
            payload["actual_NCCL"] = worker.world_size > 1 and payload["backend"] == "nccl"
            payload["parallel"] = {"F": args.f, "T": args.t, "SP": args.sp}
            model = build_qwen_model(config.model, args.seed)
            reader = model._qwen_weight_source
            aliases = (
                {"lm_head.weight": "model.embed_tokens.weight"} if qwen.tie_word_embeddings else {}
            )
            reader.validate(
                {
                    name: tuple(parameter.shape)
                    for name, parameter in model.named_parameters(remove_duplicate=False)
                },
                aliases,
            )
            checkpoint_directory = reader.path if reader.path.is_dir() else reader.path.parent
            standard_sources = [
                checkpoint_directory / name
                for name in ["model.safetensors", "model.safetensors.index.json"]
                if (checkpoint_directory / name).is_file()
            ]
            if len(standard_sources) != 1 or (
                reader.path.is_file() and reader.path != standard_sources[0]
            ):
                raise ValueError(
                    "HF oracle requires exactly one standard local model.safetensors or "
                    "model.safetensors.index.json; the requested path must select that checkpoint"
                )
            payload["source"]["checkpoint_tensor_count"] = len(reader.keys)
            payload["source"]["checkpoint_dtypes"] = sorted(set(reader.dtypes.values()))
            payload["source"]["tied_aliases"] = aliases
            batch, expected_logits, expected_loss, loading = _hf_reference(
                args, qwen, checkpoint_directory, mesh.coordinate.fsdp
            )
            payload["hf_loading_info"] = loading
            runtime = ShardedRuntime(
                device=mesh.device,
                reshard_after_forward=True,
                communication_policy="conservative",
                directory=session.directory,
            ).register(model, mesh, sp=args.sp)
            model.eval()
            before_digest = _shard_digest(runtime)
            identities = {
                name: (id(handle.local_shard), handle.local_shard.data_ptr())
                for name, handle in runtime.registry.items()
            }
            capture = ProfileCapture(
                session.directory, worker.rank, mesh.device, mode=args.profile_mode
            )
            capture.start()
            with runtime.compute_context():
                runtime.begin_step(0)
                tokens, labels, mask, positions = [tensor.to(mesh.device) for tensor in batch]
                with training_range("checkpoint_oracle_forward", mesh.device):
                    actual = model(
                        input_ids=tokens,
                        labels=labels,
                        attention_mask=mask,
                        position_ids=positions,
                        use_cache=False,
                    )
                logits = actual.logits.detach().cpu()
                with training_range("checkpoint_oracle_cleanup_backward", mesh.device):
                    actual.loss.backward()
                    runtime.finalize_backward()
                reported_loss = runtime.qwen_reported_loss().detach().cpu()
                runtime.end_step()
            expected_logits = runtime.input_for_rank(expected_logits)
            payload["comparisons"] = [
                _compare(logits, expected_logits, "logits", args.atol, args.rtol),
                _compare(reported_loss, expected_loss, "global_loss", args.atol, args.rtol),
            ]
            payload["hf_global_loss"] = expected_loss.item()
            payload["runtime_global_loss"] = reported_loss.item()
            payload["global_valid_tokens"] = args.f * args.micro_batch * (args.seq_len - 1)
            payload["input_shape_per_fsdp_replica"] = list(tokens.shape)
            payload["input_token_ids"] = batch[0].tolist()
            payload["ownership"] = {
                "persistent_shard_count": len(runtime.registry),
                "persistent_shard_numel": sum(
                    handle.local_shard.numel() for handle in runtime.registry.values()
                ),
                "live_buffers": len(runtime.comm.live_buffers),
                "live_fullparam_leases": sum(unit.lease is not None for unit in runtime.units),
                "stable_shard_identity": identities
                == {
                    name: (id(handle.local_shard), handle.local_shard.data_ptr())
                    for name, handle in runtime.registry.items()
                },
                "weights_unchanged": before_digest == _shard_digest(runtime),
                "parameter_sha256": before_digest,
            }
            own = payload["ownership"]
            accepted = (
                all(row["passed"] for row in payload["comparisons"])
                and own["stable_shard_identity"]
                and own["weights_unchanged"]
                and own["live_buffers"] == own["live_fullparam_leases"] == 0
            )
            if worker.world_size > 1:
                flag = torch.tensor(int(accepted), dtype=torch.int32, device=mesh.device)
                runtime.comm.unit_id, runtime.comm.phase = None, "ORACLE"
                runtime.comm.issue(
                    "world",
                    "checkpoint_oracle_acceptance",
                    dist.group.WORLD,
                    flag,
                    flag,
                    lambda async_: dist.all_reduce(flag, op=dist.ReduceOp.MIN, async_op=async_),
                    channel="fsdp_rs",
                    asynchronous=False,
                    group_role="oracle_metadata",
                )
                accepted = bool(flag.item())
            capture.step(0)
            if not accepted:
                raise AssertionError(
                    "Checkpoint logits/loss or lifecycle verification failed on at least one "
                    "rank; inspect per-rank comparisons"
                )
            payload["status"] = "passed"
    except Exception as exc:
        failure = exc
        payload["errors"].append(repr(exc))
    finally:
        if capture is not None:
            try:
                capture.close()
            except Exception as exc:
                payload["errors"].append("profiling: " + repr(exc))
                if failure is None:
                    failure = exc
                    payload["status"] = "failed"
        write_json(directory / f"checkpoint_oracle.rank{worker.rank}.json", payload)
    if failure is not None:
        raise ConfigError(f"Checkpoint oracle failed: {failure}") from failure
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--hf-config", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local standard HF safetensors directory/file/index; no downloads",
    )
    parser.add_argument("--output-dir", "--debug.output_dir", dest="directory", required=True)
    parser.add_argument("--f", type=int, default=1)
    parser.add_argument("--t", type=int, default=1)
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=3e-5)
    parser.add_argument("--rtol", type=float, default=3e-5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--profile-mode", choices=["off", "torch", "nsight"], default="off")
    args = parser.parse_args(argv)
    try:
        run(args)
    except ConfigError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
