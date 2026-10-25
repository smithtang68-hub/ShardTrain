"""Reproducible Qwen runtime workload; profiling and timing are separate modes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from sharded_runtime.config import Config, ConfigError, load_config
from sharded_runtime.distributed.context import WorkerEnvironment
from sharded_runtime.launch.bootstrap import WorkerSession
from sharded_runtime.models.qwen2_factory import build_qwen_model
from sharded_runtime.observability.metadata import write_json
from sharded_runtime.observability.profiling import ProfileCapture
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.training.performance import qwen_performance_summary
from sharded_runtime.training.qwen2_precision import resolve_qwen_compute_dtype
from sharded_runtime.training.qwen2_trainer import QwenTrainer, training_range


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _versions() -> dict:
    packages = {}
    for name in ["transformers", "safetensors"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "nccl": torch.cuda.nccl.version() if dist.is_nccl_available() else None,
        "packages": packages,
        "platform": platform.platform(),
    }


def _model_source(config: Config, model) -> dict:
    source = {
        "initialization": "local_safetensors" if config.model.checkpoint_path else "random",
        "checkpoint_path": str(Path(config.model.checkpoint_path).resolve())
        if config.model.checkpoint_path
        else None,
        "seed": config.training.seed,
        "memory_strategy": "meta construction; one CPU source parameter at a time; "
        "direct copy into persistent FP32 TP/FSDP local shards",
        "tied_aliases": {"lm_head.weight": "model.embed_tokens.weight"}
        if model.lm_head.weight is model.model.embed_tokens.weight
        else {},
        "config": model.config.to_dict(),
        "global_unique_parameter_numel": sum(parameter.numel() for parameter in model.parameters()),
        "actual_checkpoint_logits_reference_verified": False,
    }
    if config.model.hf_config_path:
        path = Path(config.model.hf_config_path).resolve()
        source["hf_config_path"] = str(path)
        source["hf_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance = path.with_suffix(".provenance.json")
        if provenance.is_file():
            import json

            source["config_provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    return source


def _topology(session) -> None:
    if session.worker.rank != 0 or session.context.device.type != "cuda":
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        text = result.stdout + result.stderr
        status = {"returncode": result.returncode, "command": "nvidia-smi topo -m"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        text, status = str(exc), {"error": repr(exc), "command": "nvidia-smi topo -m"}
    (session.directory / "gpu_topology.txt").write_text(text, encoding="utf-8")
    write_json(session.directory / "gpu_topology_status.json", status)


def _max_rank_seconds(runtime, seconds: float, world_size: int) -> float:
    if world_size == 1:
        return seconds
    value = torch.tensor(seconds, dtype=torch.float64, device=runtime.device)
    runtime.comm.unit_id, runtime.comm.phase = None, "BENCHMARK"
    with training_range("benchmark_max_rank_time", runtime.device):
        runtime.comm.issue(
            "world",
            "benchmark_max_rank_seconds",
            dist.group.WORLD,
            value,
            value,
            lambda async_: dist.all_reduce(
                value, op=dist.ReduceOp.MAX, group=dist.group.WORLD, async_op=async_
            ),
            channel="fsdp_rs",
            asynchronous=False,
            group_role="benchmark_metadata",
        )
    return float(value.item())


def _memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {
            "max_allocated": None,
            "max_reserved": None,
            "allocated_end": None,
            "reserved_end": None,
        }
    return {
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "max_reserved": torch.cuda.max_memory_reserved(device),
        "allocated_end": torch.cuda.memory_allocated(device),
        "reserved_end": torch.cuda.memory_reserved(device),
    }


def _ownership(runtime, trainer) -> dict:
    parameters = runtime.sharded_parameters()
    return {
        "unit_count": len(runtime.units),
        "persistent_shard_count": len(parameters),
        "persistent_shard_numel_including_padding": sum(p.numel() for p in parameters),
        "persistent_shard_valid_numel": sum(
            handle.spec.valid_len(runtime.materializer.rank) for handle in runtime.registry.values()
        ),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "gradient_bytes": sum(
            p.grad.numel() * p.grad.element_size() for p in parameters if p.grad is not None
        ),
        "optimizer_tensor_bytes": sum(
            value.numel() * value.element_size()
            for state in trainer.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        ),
        "optimizer_state_parameter_count": len(trainer.optimizer.state),
        "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
        "master_weight_strategy": "Persistent FP32 local shards are optimizer master weights",
        "live_buffers": len(runtime.comm.live_buffers),
        "live_fullparam_leases": sum(unit.lease is not None for unit in runtime.units),
        "units_reduced_once_last_microbatch": all(unit.reductions == 1 for unit in runtime.units),
        "stable_shard_identity": all(
            (id(handle.local_shard), handle.local_shard.data_ptr()) == runtime.identities[name]
            for name, handle in runtime.registry.items()
        ),
    }


def run(
    config: Config,
    *,
    warmup: int = 0,
    profile_mode: str = "off",
    profile_detail: str = "low_overhead",
    runtime_events: bool | None = None,
    peak_tflops_per_gpu: float | None = None,
) -> dict:
    """Train the configured Qwen model and save one JSON result per rank.

    Timing synchronizes at optimizer-step boundaries. Max-rank aggregation and
    the next-step barrier are outside the measured interval. CUDA events measure
    the elapsed training timeline, not a sum of GPU kernel busy intervals.
    """
    if config.model.model_type != "qwen2":
        raise ConfigError("Qwen benchmark requires model.model_type=qwen2")
    if type(warmup) is not int or warmup < 0:
        raise ConfigError("warmup must be a nonnegative integer")
    if profile_mode not in {"off", "torch", "nsight"}:
        raise ConfigError("profile_mode must be off, torch, or nsight")
    if profile_detail not in {"low_overhead", "full"}:
        raise ConfigError("profile_detail must be low_overhead or full")
    if profile_mode == "nsight" and config.parallel.device != "cuda":
        raise ConfigError("Nsight capture requires CUDA; use torch or off for CPU")
    if peak_tflops_per_gpu is not None and (
        not math.isfinite(peak_tflops_per_gpu) or peak_tflops_per_gpu <= 0
    ):
        raise ConfigError("peak-tflops-per-gpu must be finite and positive")
    runtime_events = profile_mode != "off" if runtime_events is None else runtime_events
    rank = WorkerEnvironment.read().rank
    directory = Path(config.debug.output_dir)
    previous = [
        directory / f"{prefix}.rank{rank}.json"
        for prefix in ["environment", "benchmark", "profiling"]
        if (directory / f"{prefix}.rank{rank}.json").exists()
    ]
    if previous:
        raise ConfigError(
            f"Output directory already contains rank {rank} experiment records: "
            f"{', '.join(path.name for path in previous)}. "
            "Choose a new --debug.output_dir to preserve prior traces and snapshots."
        )
    with WorkerSession(config, "runtime") as session:
        mesh = session.context
        # Dropout uses global Torch RNG; parameter sources use their own stable seed.
        torch.manual_seed(config.training.seed + session.worker.rank)
        rows, warmup_rows = [], []
        runtime = trainer = profiler = None
        failure = None
        payload = {
            "engine": "shardtrain",
            "status": "initializing",
            "rank": session.worker.rank,
            "world_size": session.worker.world_size,
            "coordinate": asdict(mesh.coordinate),
            "versions": _versions(),
            "device": str(mesh.device),
            "device_name": torch.cuda.get_device_name(mesh.device)
            if mesh.device.type == "cuda"
            else "CPU",
            "device_capability": torch.cuda.get_device_capability(mesh.device)
            if mesh.device.type == "cuda"
            else None,
            "attention_backend": "PyTorch SDPA math; explicit grouped KV expansion",
            "dtype": config.training.dtype,
            "optimizer": config.training.optimizer,
            "micro_batch": config.training.local_batch_size,
            "global_batch": config.training.local_batch_size
            * config.training.accumulation_steps
            * config.parallel.fsdp_degree,
            "sequence_length": config.model.seq_len,
            "accumulation_steps": config.training.accumulation_steps,
            "warmup": warmup,
            "steady_steps_requested": config.training.steps,
            "profile_mode": profile_mode,
            "profile_detail": profile_detail,
            "communication_policy": config.runtime.communication_policy,
            "runtime_switches": asdict(config.runtime),
            "runtime_events_enabled": runtime_events,
            "precision_policy": {
                "compute": config.training.dtype,
                "linear_backward": config.training.dtype,
                "parameters": "fp32",
                "parameter_all_gather": "fp32",
                "gradient_reduce": "fp32",
                "optimizer_state": "fp32",
                "norm_accumulation": "fp32",
                "loss": "fp32",
                "sp_forward_input_all_gather": config.training.dtype,
                "sp_forward_output_reduce_scatter": config.training.dtype,
                "sp_backward_output_grad_all_gather": config.training.dtype,
                "sp_backward_input_grad_reduce_scatter": "fp32",
            },
            "oom": False,
            "nonfinite_loss": False,
            "nonfinite_detected": False,
            "errors": [],
        }
        try:
            try:
                resolve_qwen_compute_dtype(config.training.dtype, mesh.device)
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc
            _topology(session)
            try:
                model = build_qwen_model(config.model, config.training.seed)
            except (OSError, ValueError, TypeError) as exc:
                raise ConfigError(f"Cannot construct Qwen model/checkpoint: {exc}") from exc
            payload["model_source"] = _model_source(config, model)
            runtime = ShardedRuntime(
                device=mesh.device,
                reshard_after_forward=config.runtime.reshard_after_forward,
                forward_prefetch=config.runtime.forward_prefetch,
                backward_prefetch=config.runtime.backward_prefetch,
                async_grad_rs=config.runtime.async_grad_rs,
                communication_policy=config.runtime.communication_policy,
                directory=session.directory,
                diagnostics_enabled=runtime_events,
            ).register(model, mesh, sp=config.parallel.sp_enabled)
            trainer = QwenTrainer(
                model,
                runtime,
                config.training.lr,
                dtype=config.training.dtype,
                optimizer=config.training.optimizer,
                accumulation_steps=config.training.accumulation_steps,
            )
            payload["optimizer_hyperparameters"] = trainer.optimizer.defaults
            generator = torch.Generator().manual_seed(config.training.seed + mesh.coordinate.fsdp)
            ids = torch.randint(
                model.config.vocab_size,
                (
                    config.training.local_batch_size * config.training.accumulation_steps,
                    config.model.seq_len,
                ),
                generator=generator,
            ).to(mesh.device)
            labels, mask = ids.clone(), torch.ones_like(ids)
            payload["input_batch_digest"] = hashlib.sha256(
                json.dumps(ids.cpu().tolist(), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            positions = torch.arange(config.model.seq_len, device=mesh.device).unsqueeze(0)
            for _ in range(warmup):
                result = trainer.train_step(
                    ids, labels, attention_mask=mask, position_ids=positions
                )
                warmup_row = asdict(result)
                payload["nonfinite_detected"] |= result.overflow
                if not math.isfinite(result.loss):
                    payload["nonfinite_loss"] = True
                    warmup_row["loss"] = None
                    warmup_rows.append(warmup_row)
                    raise FloatingPointError("Nonfinite Qwen warmup loss")
                warmup_rows.append(warmup_row)
            payload["initialization_and_warmup_memory"] = _memory(mesh.device)
            # Persist full diagnostic rows before entering the measured window.
            runtime.comm.flush_diagnostics(release_events=True)
            if mesh.device.type == "cuda":
                torch.cuda.synchronize(mesh.device)
                torch.cuda.reset_peak_memory_stats(mesh.device)
            profiler = ProfileCapture(
                session.directory,
                session.worker.rank,
                mesh.device,
                mode=profile_mode,
                detail=profile_detail,
            )
            profiler.start()
            payload["status"] = "running"
            for step in range(config.training.steps):
                if dist.is_initialized():
                    dist.barrier()
                start_event = end_event = None
                if mesh.device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                start = time.perf_counter()
                result = trainer.train_step(
                    ids, labels, attention_mask=mask, position_ids=positions
                )
                if end_event is not None:
                    end_event.record()
                    torch.cuda.synchronize(mesh.device)
                seconds = time.perf_counter() - start
                gpu_ms = start_event.elapsed_time(end_event) if start_event is not None else None
                if not math.isfinite(result.loss):
                    payload["nonfinite_loss"] = True
                    raise FloatingPointError("Nonfinite Qwen measured loss")
                payload["nonfinite_detected"] |= result.overflow
                maximum = _max_rank_seconds(runtime, seconds, session.worker.world_size)
                rows.append(
                    {
                        **asdict(result),
                        "measured_step": step,
                        "seconds": seconds,
                        "max_rank_seconds": maximum,
                        "cuda_elapsed_ms": gpu_ms,
                        "tokens_per_second": result.valid_tokens / maximum,
                        **_memory(mesh.device),
                    }
                )
                profiler.step(result.step)
                # The step timer and step_complete marker both precede serialization.
                runtime.comm.flush_diagnostics(release_events=True)
                session.logger.info(
                    "qwen step=%d loss=%.8f tokens=%d max_rank_seconds=%.6f overflow=%s",
                    result.step,
                    result.loss,
                    result.valid_tokens,
                    maximum,
                    result.overflow,
                )
            payload["status"] = "completed"
        except BaseException as exc:
            failure = exc
            payload["status"] = "failed"
            payload["oom"] = (
                isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
            )
            payload["errors"].append(repr(exc))
            payload["nonfinite_detected"] |= (
                payload["nonfinite_loss"] or "nonfinite" in str(exc).lower()
            )
        finally:
            if profiler is not None:
                try:
                    profiler.close()
                except Exception as exc:
                    payload["errors"].append("profiling: " + repr(exc))
                    if failure is None:
                        failure, payload["status"] = exc, "failed"
            durations = [row["max_rank_seconds"] for row in rows]
            payload.update(
                {
                    "steps": rows,
                    "warmup_steps": warmup_rows,
                    "steady_steps_completed": len(rows),
                    "step_seconds_p50": _percentile(durations, 0.50),
                    "step_seconds_p95": _percentile(durations, 0.95),
                    "tokens_per_second": sum(row["valid_tokens"] for row in rows) / sum(durations)
                    if durations
                    else None,
                    "overflow_steps": sum(row["overflow"] for row in rows),
                    "warmup_overflow_steps": sum(row["overflow"] for row in warmup_rows),
                    "optimizer_updates": sum(not row["overflow"] for row in rows),
                    **_memory(mesh.device),
                    "timing_method": "Per-step barrier before timer; CPU wall including "
                    "training and "
                    "end-of-step CUDA synchronization; max-rank reduction outside timer. "
                    "CUDA event duration includes the whole training timeline. "
                    "Valid next-token labels are global over FSDP/SP, "
                    "excluding ignored/last tokens.",
                    "profiling_note": "Profiled timing is instrumented; use profile-mode off for "
                    "performance comparisons. Allocator snapshots exclude "
                    "direct CUDA/NCCL allocations.",
                    "nccl_correctness_verified": False,
                    "benchmark_scope": "Synthetic repeated token batch; "
                    "no speedup claim without reference "
                    "and timeline inspection; CPU does not validate NCCL/V100 behavior.",
                }
            )
            if runtime is not None and trainer is not None:
                payload["ownership"] = _ownership(runtime, trainer)
                if mesh.device.type == "cuda" and failure is None:
                    payload["pg_order_gpu_waits"] = runtime.comm.gpu_wait_timings()
            if runtime is not None:
                try:
                    runtime.comm.close_diagnostics()
                except Exception as exc:
                    payload["errors"].append("runtime diagnostics: " + repr(exc))
                    if failure is None:
                        failure, payload["status"] = exc, "failed"
            if "model_source" in payload:
                source = payload["model_source"]
                payload["performance"] = qwen_performance_summary(
                    parameter_numel=source["global_unique_parameter_numel"],
                    layers=source["config"]["num_hidden_layers"],
                    hidden_size=source["config"]["hidden_size"],
                    sequence_length=payload["sequence_length"],
                    global_batch=payload["global_batch"],
                    steps=rows,
                    world_size=session.worker.world_size,
                    peak_tflops_per_gpu=peak_tflops_per_gpu,
                    profile_mode=profile_mode,
                )
            write_json(session.directory / f"benchmark.rank{session.worker.rank}.json", payload)
        if failure is not None:
            raise failure
        return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default="configs/qwen25_tiny.yaml")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-mode", choices=["off", "torch", "nsight"], default="off")
    parser.add_argument(
        "--profile-detail", choices=["low_overhead", "full"], default="low_overhead"
    )
    parser.add_argument("--runtime-events", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--peak-tflops-per-gpu", type=float, default=None)
    args, overrides = parser.parse_known_args(argv)
    try:
        run(
            load_config(args.config, overrides),
            warmup=args.warmup,
            profile_mode=args.profile_mode,
            profile_detail=args.profile_detail,
            runtime_events={"auto": None, "on": True, "off": False}[args.runtime_events],
            peak_tflops_per_gpu=args.peak_tflops_per_gpu,
        )
    except ConfigError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
