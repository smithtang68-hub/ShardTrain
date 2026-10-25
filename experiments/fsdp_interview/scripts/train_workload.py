"""Independent real-token workload; production runtime is not modified.

Timing and trace processes are separate. All measured steps consume the same
deterministic data order, and TP/SP never duplicate the global token count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

# ruff: noqa: E402

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments/fsdp_interview/cases"))

from sharded_runtime.config import load_config
from sharded_runtime.launch.bootstrap import WorkerSession
from sharded_runtime.models.qwen2_factory import build_qwen_model
from sharded_runtime.observability.profiling import ProfileCapture
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.training.performance import qwen_performance_summary
from sharded_runtime.training.qwen2_native_baseline import NativeQwenTrainer, build_native_qwen
from sharded_runtime.training.qwen2_trainer import QwenTrainer


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def percentile(values, q):
    if not values:
        return None
    a = sorted(values)
    position = (len(a) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return a[low] + (a[high] - a[low]) * (position - low)


def memory(device):
    free, total = torch.cuda.mem_get_info(device)
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "physical_free_bytes_endpoint": free,
        "physical_total_bytes": total,
    }


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--implementation", choices=["shardtrain", "native"], required=True)
    p.add_argument("--config", default="experiments/fsdp_interview/configs/primary.yaml")
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-id", default="fsdp_interview_20261014")
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--microbatch", type=int, default=None)
    p.add_argument("--accumulation", type=int, default=None)
    p.add_argument("--policy", choices=["C0", "C1", "C2", "C3"], default="C0")
    p.add_argument("--contract")
    p.add_argument("--record-events", action="store_true")
    p.add_argument("--guarded-baseline", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-detail", choices=["low_overhead", "minimal"], default="low_overhead")
    p.add_argument(
        "--gemm-shapes",
        action="store_true",
        help="separate local-ATen shape diagnostic; excludes this run from performance claims",
    )
    p.add_argument("--injection-json", default=None)
    p.add_argument("--memory-history-max-entries", type=int, default=80000)
    return p.parse_args()


def run(args):
    if args.warmup < 0 or args.steps <= 0:
        raise ValueError("Warmup must be nonnegative and measured steps must be positive")
    config = load_config(args.config)
    if args.layers is not None:
        # Explicit reduced-depth smoke; full model config is only used for 28L.
        config = replace(config, model=replace(config.model, layers=args.layers, hf_config_path=""))
    config = replace(
        config,
        debug=replace(config.debug, output_dir=args.output_dir),
        training=replace(
            config.training,
            steps=args.steps,
            local_batch_size=args.microbatch or config.training.local_batch_size,
            accumulation_steps=args.accumulation or config.training.accumulation_steps,
        ),
    )
    rank, local_rank, world = (int(os.environ[x]) for x in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
    config.validate(world, mode="runtime")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"benchmark.rank{rank}.json").exists():
        raise FileExistsError("Keep prior measurements; choose a fresh output directory")
    config_hash = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()
    work_config = config.to_dict()
    work_config.pop("debug")
    work_config["training"].pop("steps")
    workload_hash = hashlib.sha256(json.dumps(work_config, sort_keys=True).encode()).hexdigest()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(config.training.seed + rank)
    injection = None
    if args.injection_json:
        from q12_rank_skew import RankDelay

        injection = RankDelay(**json.loads(args.injection_json))
    if args.implementation == "native" and (args.policy != "C0" or injection):
        raise ValueError("Controlled policies and injection apply only to ShardTrain")
    rows, warmup_rows, runtime, capture = [], [], None, None
    observer = nullcontext()
    dataset = json.loads(Path(args.data).with_name("dataset.json").read_text())
    cpu_blocks = np.load(args.data, allow_pickle=False)
    if (
        cpu_blocks.dtype != np.int64
        or cpu_blocks.ndim != 2
        or cpu_blocks.shape[1] != config.model.seq_len
        or cpu_blocks.shape[0] == 0
    ):
        raise ValueError("Token blocks do not match the workload shape")
    if hashlib.sha256(Path(args.data).read_bytes()).hexdigest() != dataset["tokens_sha256"]:
        raise ValueError("Token data hash changed")
    dp_rank = rank // config.parallel.tp_degree
    local_total = config.training.local_batch_size * config.training.accumulation_steps
    global_total = local_total * config.parallel.fsdp_degree
    if len(cpu_blocks) % global_total:
        raise ValueError("Block count must divide global optimizer-step batch")
    # Token pool is only a few MiB. All loading/H2D is outside step timing.
    pool = torch.from_numpy(cpu_blocks).to(device)
    mask = torch.ones((local_total, config.model.seq_len), dtype=torch.long, device=device)
    positions = torch.arange(config.model.seq_len, device=device).unsqueeze(0)
    payload = {
        "run_id": args.run_id,
        "experiment_id": args.experiment_id,
        "implementation": args.implementation,
        "config_hash": config_hash,
        "workload_hash": workload_hash,
        "config": config.to_dict(),
        "rank": rank,
        "world_size": world,
        "fsdp_group_size": config.parallel.fsdp_degree,
        "tp_group_size": config.parallel.tp_degree,
        "sp_group_size": config.parallel.tp_degree if config.parallel.sp_enabled else 1,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "nccl": torch.cuda.nccl.version(),
        "cuda": torch.version.cuda,
        "model_id": "Qwen2.5-1.5B" if config.model.layers == 28 else "Qwen reduced-depth smoke",
        "num_layers": config.model.layers,
        "seq_len": config.model.seq_len,
        "microbatch": config.training.local_batch_size,
        "accumulation_steps": config.training.accumulation_steps,
        "unique_input_tokens_per_step": global_total * config.model.seq_len,
        "precision": "FP16 linear/autocast; FP32 master/parameter AG/gradient RS/state; "
        "math SDPA FP32 BMM",
        "optimizer": config.training.optimizer,
        "seed": config.training.seed,
        "dataset": dataset,
        "policy": args.policy,
        "profile": args.profile,
        "event_recorder": args.record_events or args.profile,
        "guarded_baseline": args.guarded_baseline,
        "injection": json.loads(args.injection_json) if args.injection_json else None,
        "status": "INITIALIZING",
        "errors": [],
        "warmup": args.warmup,
        "steps_requested": args.steps,
        "git_commit": os.environ.get("SHARDTRAIN_SOURCE_GIT_COMMIT", "recorded_in_source_manifest"),
        "source_manifest": os.environ.get("SHARDTRAIN_SOURCE_MANIFEST"),
        "source_manifest_sha256": os.environ.get("SHARDTRAIN_SOURCE_MANIFEST_SHA256"),
        "timing": "start barrier outside timer; CUDA events on compute stream; "
        "wall timer includes whole-device sync; rank maximum reduction outside timer",
        "trace_path": str(out / "timeline.nsys-rep") if args.profile else None,
        "profile_detail": args.profile_detail if args.profile else None,
    }
    session = None
    primary_error = None
    observer_entered = False
    shape_capture = None
    try:
        if args.implementation == "shardtrain":
            session = WorkerSession(config, "runtime")
            session.__enter__()
            context = session.context
            model = build_qwen_model(config.model, config.training.seed)
            unique_numel = sum(p.numel() for p in model.parameters())
            runtime = ShardedRuntime(
                device=device,
                reshard_after_forward=config.runtime.reshard_after_forward,
                forward_prefetch=config.runtime.forward_prefetch,
                backward_prefetch=config.runtime.backward_prefetch,
                async_grad_rs=config.runtime.async_grad_rs,
                communication_policy=config.runtime.communication_policy,
                directory=out,
                diagnostics_enabled=args.record_events or args.profile,
            ).register(model, context, sp=config.parallel.sp_enabled)
            trainer = QwenTrainer(
                model,
                runtime,
                config.training.lr,
                dtype=config.training.dtype,
                optimizer=config.training.optimizer,
                accumulation_steps=config.training.accumulation_steps,
            )
            from q05_nccl_contention import instrumentation, runtime_layout

            payload["runtime_layout"] = runtime_layout(runtime)
            contract = json.loads(Path(args.contract).read_text()) if args.contract else None
            observer = instrumentation(
                policy=args.policy,
                output_dir=out,
                run_id=args.run_id,
                experiment_id=args.experiment_id,
                warmup_steps=args.warmup,
                contract=contract,
                gpu_timing=False,
                injection=injection,
                patch_trainer=False,
                record_events=args.record_events or args.profile,
                guarded_baseline=args.guarded_baseline,
            )
            observer.__enter__()
            observer_entered = True
            compute_context = runtime.compute_context
            payload["ownership_before"] = {
                "parameter_shard_bytes": sum(
                    h.local_shard.numel() * h.local_shard.element_size()
                    for h in runtime.registry.values()
                ),
                "tied_identity_preserved": model.lm_head.weight is model.model.embed_tokens.weight,
            }
        else:
            from torch.distributed.device_mesh import init_device_mesh

            from sharded_runtime.distributed.nccl_ordering import prepare_nccl_launch_order

            prepare_nccl_launch_order()
            dist.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=device)
            mesh = init_device_mesh(
                "cuda",
                (config.parallel.fsdp_degree, config.parallel.tp_degree),
                mesh_dim_names=("fsdp", "tp"),
            )
            model, native = build_native_qwen(config, mesh, device, policy="matched")
            payload["native_policy"] = native
            unique_numel = native["global_unique_parameter_numel"]
            trainer = NativeQwenTrainer(model, mesh, config)
            compute_context = nullcontext
        payload["model_unique_parameter_numel"] = unique_numel
        payload["optimizer_hyperparameters"] = trainer.optimizer.defaults
        payload["initialization_memory"] = memory(device)

        def batch_for_step(step):
            start = (step * global_total) % len(pool) + dp_rank * local_total
            return pool[start : start + local_total], start

        for step in range(args.warmup + args.steps):
            measured = step >= args.warmup
            ids, data_start = batch_for_step(step)
            if runtime is not None:
                observer.begin_step(step, measured=measured, runtime=runtime)
            if measured and step == args.warmup:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                dist.barrier()
                if args.profile:
                    if args.profile_detail == "minimal":
                        from minimal_profile_capture import MinimalProfileCapture

                        capture = MinimalProfileCapture(
                            out, rank, device, max_entries=args.memory_history_max_entries
                        )
                    else:
                        capture = ProfileCapture(
                            out,
                            rank,
                            device,
                            mode="nsight",
                            max_entries=args.memory_history_max_entries,
                            detail="low_overhead",
                        )
                    capture.start()
                    torch.cuda.profiler.start()
                if args.gemm_shapes:
                    from gemm_shapes import GemmShapeCapture

                    shape_capture = GemmShapeCapture(
                        out,
                        rank=rank,
                        implementation=args.implementation,
                        run_id=args.run_id,
                        experiment_id=args.experiment_id,
                    )
                    shape_capture.__enter__()
            if shape_capture is not None:
                shape_capture.set_step(
                    optimizer_step=step, measured_step=step - args.warmup, phase="train_step"
                )
            dist.barrier()
            start_ev, end_ev = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            with compute_context():
                start_ev.record()
            started = time.perf_counter()
            if args.profile and measured:
                torch.cuda.nvtx.range_push(
                    f"interview/step/{step}/rank{rank}/{args.implementation}"
                )
            result = trainer.train_step(ids, attention_mask=mask, position_ids=positions)
            with compute_context():
                end_ev.record()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if args.profile and measured:
                torch.cuda.nvtx.range_pop()
            max_time = torch.tensor(elapsed, device=device, dtype=torch.float64)
            dist.all_reduce(max_time, op=dist.ReduceOp.MAX)
            maximum = float(max_time.item())
            row = {
                **asdict(result),
                "optimizer_step": step,
                "measured_step": step - args.warmup,
                "step_time_ms": maximum * 1000,
                "rank_step_time_ms": elapsed * 1000,
                "cuda_event_ms": start_ev.elapsed_time(end_ev),
                "valid_tokens": result.valid_tokens,
                "unique_input_tokens": global_total * config.model.seq_len,
                "unique_training_tokens": result.valid_tokens,
                "tokens_per_s": global_total * config.model.seq_len / maximum,
                "training_tokens_per_s": result.valid_tokens / maximum,
                "finite_gradients": not result.overflow,
                "input_block_start": data_start,
                "input_batch_sha256": hashlib.sha256(
                    cpu_blocks[data_start : data_start + local_total].tobytes()
                ).hexdigest(),
                **memory(device),
            }
            if not math.isfinite(result.loss) or result.overflow:
                raise FloatingPointError(f"Nonfinite/AMP overflow at optimizer step {step}")
            (rows if measured else warmup_rows).append(row)
            if capture is not None:
                capture.step(step)
            if runtime is not None:
                observer.flush(release_events=True)
                runtime.comm.flush_diagnostics(release_events=True)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "experiment": args.experiment_id,
                            "step": step,
                            "measured": measured,
                            "loss": result.loss,
                            "seconds": maximum,
                            "reserved_GiB": row["reserved_bytes"] / 2**30,
                        }
                    ),
                    flush=True,
                )
        if args.profile:
            # All ranks finish their final kernels before the first profiler stop.
            dist.barrier()
            torch.cuda.synchronize(device)
        payload["status"] = "PASS"
        if runtime is not None:
            payload["instrumentation"] = observer.report()
            payload["ownership_after"] = {
                "parameter_shard_bytes": sum(
                    h.local_shard.numel() * h.local_shard.element_size()
                    for h in runtime.registry.values()
                ),
                "gradient_shard_bytes": sum(
                    h.local_shard.grad.numel() * h.local_shard.grad.element_size()
                    for h in runtime.registry.values()
                    if h.local_shard.grad is not None
                ),
            }
        else:
            from sharded_runtime.training.qwen2_native_baseline import _ownership

            payload["ownership_after"] = _ownership(model, trainer)
    except BaseException as error:
        primary_error = error
        payload["status"] = "FAIL"
        payload["errors"].append(repr(error))
        raise
    finally:
        cleanup_errors = []

        def attempt_cleanup(label, operation):
            try:
                operation()
            except BaseException as error:
                cleanup_errors.append(f"{label}: {error!r}")

        if shape_capture is not None:
            attempt_cleanup(
                "GEMM shape instrumentation",
                lambda: shape_capture.__exit__(
                    type(primary_error) if primary_error else None,
                    primary_error,
                    primary_error.__traceback__ if primary_error else None,
                ),
            )
            payload["gemm_shape_diagnostic"] = True
            payload["performance_claim_allowed"] = False
        if capture is not None:
            attempt_cleanup("profile artifacts", capture.close)
            attempt_cleanup("profiler stop", torch.cuda.profiler.stop)
        if runtime is not None:
            if observer_entered:
                attempt_cleanup(
                    "instrumentation",
                    lambda: observer.__exit__(
                        type(primary_error) if primary_error else None,
                        primary_error,
                        primary_error.__traceback__ if primary_error else None,
                    ),
                )
                payload["instrumentation"] = observer.report()
            attempt_cleanup("runtime diagnostics", runtime.comm.close_diagnostics)
        durations = [row["step_time_ms"] for row in rows]
        payload.update(
            steps=rows,
            warmup_steps=warmup_rows,
            steady_steps_completed=len(rows),
            p50_step_ms=percentile(durations, 0.5),
            p95_step_ms=percentile(durations, 0.95),
            mean_step_ms=sum(durations) / len(durations) if durations else None,
            peak_allocated_bytes=max((r["peak_allocated_bytes"] for r in rows), default=0),
            peak_reserved_bytes=max((r["peak_reserved_bytes"] for r in rows), default=0),
            warmup_peak_reserved_bytes=max(
                (r["peak_reserved_bytes"] for r in warmup_rows), default=0
            ),
        )
        if rows:
            payload["tokens_per_s"] = sum(r["unique_input_tokens"] for r in rows) / (
                sum(durations) / 1000
            )
            payload["unique_training_tokens_per_step"] = rows[0]["valid_tokens"]
            payload["training_tokens_per_s"] = sum(r["valid_tokens"] for r in rows) / (
                sum(durations) / 1000
            )
            payload["performance"] = qwen_performance_summary(
                parameter_numel=unique_numel,
                layers=model.config.num_hidden_layers,
                hidden_size=model.config.hidden_size,
                sequence_length=config.model.seq_len,
                global_batch=global_total,
                steps=[
                    {
                        "max_rank_seconds": r["step_time_ms"] / 1000,
                        "valid_tokens": r["valid_tokens"],
                    }
                    for r in rows
                ],
                world_size=world,
                peak_tflops_per_gpu=125.0,
                profile_mode="nsight" if args.profile else "off",
            )
        if session is not None:
            attempt_cleanup(
                "worker session",
                lambda: session.__exit__(
                    type(primary_error) if primary_error else None,
                    primary_error,
                    primary_error.__traceback__ if primary_error else None,
                ),
            )
        elif dist.is_initialized():
            attempt_cleanup("native process group", dist.destroy_process_group)
        if cleanup_errors:
            payload["status"] = "FAIL"
            payload["errors"].extend(cleanup_errors)
        write_json(out / f"benchmark.rank{rank}.json", payload)
        if cleanup_errors and primary_error is None:
            raise RuntimeError("; ".join(cleanup_errors))
    return payload


if __name__ == "__main__":
    run(parse())
