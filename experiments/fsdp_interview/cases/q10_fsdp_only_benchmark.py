"""Q10: matched FSDP-only mechanism benchmark, never a TP/SP speed ranking.

Both engines use FP32 parameters/compute/reductions/AdamW state, math SDPA,
identical middle-sized 8-block Transformer, and equal rank-dependent DP batches.
FSDP1 uses original parameters, BACKWARD_PRE and limit_all_gathers. FSDP2 uses
public fully_shard with per-block groups and public adjacent prefetch methods.
Each optimizer step accumulates two microsteps, suppressing gradient reduction
until the last microstep through each engine's public API.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import os
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from q09_param_representation import (
    error_row,
    full_gradients,
    git_commit,
    source_record,
    write_json,
)
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecisionPolicy,
    ShardingStrategy,
    fully_shard,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel


class Block(nn.Module):
    def __init__(self, index: int, hidden: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.index = index
        self.hidden = hidden
        self.heads = heads
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, ffn)
        self.down = nn.Linear(ffn, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.nvtx.range(f"q10/layer_{self.index:03d}/forward_compute"):
            batch, sequence, _ = x.shape
            normalized = self.norm1(x)
            qkv = self.qkv(normalized).reshape(
                batch, sequence, 3, self.heads, self.hidden // self.heads
            )
            q, k, v = qkv.unbind(2)
            q, k, v = (value.transpose(1, 2) for value in (q, k, v))
            with sdpa_kernel(SDPBackend.MATH):
                attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attention = attention.transpose(1, 2).reshape(batch, sequence, self.hidden)
            x = x + self.proj(attention)
            return x + self.down(F.gelu(self.up(self.norm2(x))))


class MediumTransformer(nn.Module):
    def __init__(self, layers: int, hidden: int, heads: int, ffn: int, vocab: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList([Block(i, hidden, heads, ffn) for i in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--implementation", choices=("fsdp1", "fsdp2"), required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--run-id", default="q10_fsdp_only")
    value.add_argument("--seq-len", type=int, default=512)
    value.add_argument("--microbatch", type=int, default=4)
    value.add_argument("--accumulation-steps", type=int, default=2)
    value.add_argument("--layers", type=int, default=8)
    value.add_argument("--hidden", type=int, default=512)
    value.add_argument("--heads", type=int, default=8)
    value.add_argument("--ffn", type=int, default=2048)
    value.add_argument("--vocab", type=int, default=4096)
    value.add_argument("--warmup", type=int, default=10)
    value.add_argument("--steps", type=int, default=30)
    value.add_argument("--seed", type=int, default=1234)
    value.add_argument("--timeout-seconds", type=int, default=180)
    value.add_argument("--correctness", action="store_true")
    value.add_argument("--profile", action="store_true")
    value.add_argument("--emit-autograd-nvtx", action=argparse.BooleanOptionalAction, default=True)
    value.add_argument("--profile-steps", type=int, default=2)
    value.add_argument("--trace-path", default=None)
    value.add_argument("--memory-history-max-entries", type=int, default=80000)
    value.add_argument("--atol", type=float, default=1e-5)
    value.add_argument("--rtol", type=float, default=1e-4)
    value.add_argument("--inspect-only", action="store_true")
    return value


def create_model(args: argparse.Namespace, device: torch.device) -> MediumTransformer:
    torch.manual_seed(args.seed)
    return MediumTransformer(args.layers, args.hidden, args.heads, args.ffn, args.vocab).to(
        device=device, dtype=torch.float32
    )


def backward_start(module, gradients):
    torch.cuda.nvtx.range_push(f"q10/layer_{module.index:03d}/backward_compute_hooks")


def backward_end(module, input_gradients, output_gradients):
    torch.cuda.nvtx.range_pop()


def wrap_model(model: MediumTransformer, args: argparse.Namespace, device: torch.device):
    for block in model.blocks:
        block.register_full_backward_pre_hook(backward_start)
        block.register_full_backward_hook(backward_end)
    if args.implementation == "fsdp1":
        options = {
            "use_orig_params": True,
            "device_id": device,
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
            "forward_prefetch": True,
            "limit_all_gathers": True,
        }
        for index in range(len(model.blocks)):
            model.blocks[index] = FSDP(model.blocks[index], **options)
        return FSDP(model, **options)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("fsdp",))
    for block in model.blocks:
        fully_shard(
            block,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32),
        )
    fully_shard(
        model,
        mesh=mesh,
        # FSDP1 FULL_SHARD's root _post_forward_reshard keeps its full
        # root flat storage; match that policy for embedding/norm/head here.
        reshard_after_forward=False,
        mp_policy=MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32),
    )
    model.set_modules_to_forward_prefetch([model.blocks[0]])
    for current, following in zip(model.blocks[:-1], model.blocks[1:], strict=True):
        current.set_modules_to_forward_prefetch([following])
    for index in range(len(model.blocks) - 1, 0, -1):
        model.blocks[index].set_modules_to_backward_prefetch([model.blocks[index - 1]])
    return model


def make_inputs(args: argparse.Namespace, device: torch.device) -> list[tuple]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1000 + dist.get_rank())
    result = []
    for _ in range(args.accumulation_steps):
        tokens = torch.randint(0, args.vocab, (args.microbatch, args.seq_len), generator=generator)
        # Shifted LM labels from S+1 independently generated tokens would be equally
        # valid; fixed legal targets here keep the controlled shapes explicit.
        labels = torch.randint(0, args.vocab, (args.microbatch, args.seq_len), generator=generator)
        result.append((tokens.to(device), labels.to(device)))
    return result


def optimizer_for(model: nn.Module):
    return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, foreach=False)


def local_gradient(parameter: torch.Tensor) -> torch.Tensor | None:
    if parameter.grad is None:
        return None
    return parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad


def finite_gradients(model: nn.Module) -> bool:
    flags = [
        torch.isfinite(grad).all()
        for param in model.parameters()
        if (grad := local_gradient(param)) is not None
    ]
    return bool(torch.stack(flags).all().item()) if flags else False


def backward_microsteps(model, args, inputs, collect_output=False):
    losses = []
    outputs = []
    for index, (tokens, labels) in enumerate(inputs):
        sync = index == len(inputs) - 1
        if args.implementation == "fsdp2":
            model.set_requires_gradient_sync(sync)
            context = nullcontext()
        else:
            context = nullcontext() if sync else model.no_sync()
        with context:
            with torch.cuda.nvtx.range(f"q10/microstep_{index}/forward"):
                logits = model(tokens)
                loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            if collect_output:
                outputs.append(logits.detach().clone())
            with torch.cuda.nvtx.range(f"q10/microstep_{index}/backward"):
                (loss / len(inputs)).backward()
            losses.append(loss.detach())
    return torch.stack(losses).mean(), outputs


def correctness(model, dense, optimizer, args, inputs):
    """One complete FP32 accumulation step against DP dense reference."""
    optimizer.zero_grad(set_to_none=True)
    loss, outputs = backward_microsteps(model, args, inputs, collect_output=True)
    observed_gradients = full_gradients(
        model, "fsdp1_orig" if args.implementation == "fsdp1" else "fsdp2"
    )
    reference_optimizer = optimizer_for(dense)
    reference_losses = []
    checks = []
    for index, (tokens, labels) in enumerate(inputs):
        logits = dense(tokens)
        reference_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        (reference_loss / len(inputs)).backward()
        reference_losses.append(reference_loss.detach())
        checks.append(
            error_row(
                f"output/microstep_{index}", outputs[index], logits.detach(), args.atol, args.rtol
            )
        )
    checks.append(
        error_row("loss", loss, torch.stack(reference_losses).mean(), args.atol, args.rtol)
    )
    for name, parameter in dense.named_parameters():
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(dist.get_world_size())
        checks.append(
            error_row(
                f"gradient/{name}",
                observed_gradients.get(name),
                parameter.grad,
                args.atol,
                args.rtol,
            )
        )
    optimizer.step()
    reference_optimizer.step()
    observed = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=False)
    )
    for name, parameter in dense.named_parameters():
        checks.append(
            error_row(f"update/{name}", observed[name], parameter.detach(), args.atol, args.rtol)
        )
    optimizer.zero_grad(set_to_none=True)
    return {"status": "PASS" if all(row["pass"] for row in checks) else "FAIL", "checks": checks}


def train_step(model, optimizer, args, inputs, device, step, phase):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    with torch.cuda.nvtx.range(f"q10/step phase={phase} step={step}"):
        cuda_start.record()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = backward_microsteps(model, args, inputs)
        finite = finite_gradients(model)
        if not finite:
            raise FloatingPointError(f"Non-finite gradient rank={dist.get_rank()} step={step}")
        with torch.cuda.nvtx.range("q10/optimizer_step"):
            optimizer.step()
        cuda_end.record()
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    with torch.cuda.nvtx.range("q10/metrics_outside_timer"):
        loss_value = float(loss.item())
    return {
        "run_id": args.run_id,
        "experiment_id": f"Q10_{args.implementation}_fsdp_only",
        "implementation": args.implementation,
        "config_hash": args.config_hash,
        "git_commit": args.git_commit,
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "fsdp_group_size": dist.get_world_size(),
        "tp_group_size": 1,
        "sp_group_size": 1,
        "model_id": "Q10_MediumTransformer",
        "num_layers": args.layers,
        "seq_len": args.seq_len,
        "microbatch": args.microbatch,
        "accumulation_steps": args.accumulation_steps,
        "precision": "fp32_all",
        "optimizer": "AdamW(foreach=False)",
        "attention_backend": "SDPBackend.MATH",
        "seed": args.seed,
        "step": step,
        "phase": phase,
        "step_time_ms": elapsed_ms,
        "cuda_event_ms": cuda_start.elapsed_time(cuda_end),
        "unique_training_tokens_per_step": (
            dist.get_world_size() * args.microbatch * args.seq_len * args.accumulation_steps
        ),
        "valid_tokens_per_rank": args.microbatch * args.seq_len * args.accumulation_steps,
        "loss": loss_value,
        "finite_gradients": finite,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "trace_path": args.trace_path if args.profile else None,
        "status": "PASS",
    }


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def configure_memory_history(args):
    signature = inspect.signature(torch.cuda.memory._record_memory_history)
    required = {"enabled", "context", "stacks", "max_entries"}
    if not required <= set(signature.parameters):
        raise RuntimeError(f"Installed memory history API unsupported: {signature}")
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=args.memory_history_max_entries,
    )


def main() -> int:
    args = parser().parse_args()
    args.git_commit = git_commit()
    args.config_hash = hashlib.sha256(
        json.dumps(
            {
                key: getattr(args, key)
                for key in (
                    "implementation",
                    "seq_len",
                    "microbatch",
                    "accumulation_steps",
                    "layers",
                    "hidden",
                    "heads",
                    "ffn",
                    "vocab",
                    "seed",
                )
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if args.hidden % args.heads or args.layers < 1 or args.accumulation_steps < 1:
        raise ValueError("Hidden must divide heads; layers/accumulation must be positive")
    if args.steps < 1 or args.profile_steps < 1 or args.warmup < 0:
        raise ValueError("steps/profile_steps must be positive and warmup nonnegative")
    rank = int(os.environ.get("RANK", "0"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / f"environment.rank{rank}.json").exists():
        raise FileExistsError(f"Experiment output already exists for rank {rank}: {output_dir}")
    metadata = {
        "pid": os.getpid(),
        "rank": rank,
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda_build": torch.version.cuda,
        "git_commit": git_commit(),
        "argv": sys.argv,
        "config": vars(args),
        "comparison_scope": (
            "FSDP-only medium Transformer mechanism; not full Qwen TP/SP performance"
        ),
        "precision_policy": "All parameter/AG/reduce/compute/AdamW-state FP32; no AMP",
        "parameter_reshard_policy": (
            "Child blocks reshard after forward; root retains embedding/norm/head until backward"
        ),
        "input_policy": "GPU-resident fixed synthetic tokens/labels, rank-distinct DP replicas",
        "gradient_accumulation_policy": (
            "no_sync/set_requires_gradient_sync(False) before final microstep"
        ),
        "timing_policy": (
            "per-rank synchronize before/after optimizer step; max rank per step for aggregate"
        ),
        "nvtx_policy": (
            "CPU scopes label launches; layer backward hooks do not delimit GPU completion"
        ),
        "copy_markers": (
            "Only official PyTorch ranges/API copies; no fabricated copy-in/out markers"
        ),
        "official_scope_nvtx": bool(args.profile and args.emit_autograd_nvtx),
        "apis": {"FSDP1": source_record(FSDP), "FSDP2": source_record(fully_shard)},
    }
    if args.inspect_only:
        model = create_model(args, torch.device("cpu"))
        metadata["parameter_count"] = sum(p.numel() for p in model.parameters())
        metadata["status"] = "NOT_RUN"
        write_json(output_dir / f"environment.rank{rank}.json", metadata)
        return 0
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        metadata.update(status="BLOCKED", reason="CUDA/NCCL required")
        write_json(output_dir / f"environment.rank{rank}.json", metadata)
        return 2
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=args.timeout_seconds), device_id=device
    )
    metadata.update(
        rank=rank,
        world_size=dist.get_world_size(),
        nccl_version=torch.cuda.nccl.version(),
        gpu=torch.cuda.get_device_name(device),
    )
    model = create_model(args, device)
    metadata["parameter_count"] = sum(p.numel() for p in model.parameters())
    dense = copy.deepcopy(model) if args.correctness else None
    model = wrap_model(model, args, device)
    optimizer = optimizer_for(model)
    inputs = make_inputs(args, device)
    write_json(output_dir / f"environment.rank{rank}.json", metadata)
    rows = []
    profiling = False
    try:
        if args.correctness:
            evidence = correctness(model, dense, optimizer, args, inputs)
            write_json(output_dir / f"correctness.rank{rank}.json", evidence)
            if evidence["status"] != "PASS":
                raise AssertionError("Dense oracle mismatch; inspect correctness JSON")
            del dense
        for index in range(args.warmup):
            rows.append(train_step(model, optimizer, args, inputs, device, index, "warmup"))
        dist.barrier()
        if args.profile:
            configure_memory_history(args)
            torch.cuda.memory._dump_snapshot(str(output_dir / f"memory_start.rank{rank}.pickle"))
            torch.cuda.cudart().cudaProfilerStart()
            profiling = True
        measured_steps = args.profile_steps if args.profile else args.steps
        nvtx_context = (
            torch.autograd.profiler.emit_nvtx(record_shapes=False)
            if args.profile and args.emit_autograd_nvtx
            else nullcontext()
        )
        with nvtx_context:
            for index in range(measured_steps):
                rows.append(train_step(model, optimizer, args, inputs, device, index, "measured"))
                with (output_dir / f"steps.rank{rank}.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(rows[-1]) + "\n")
        if args.profile:
            dist.barrier()
            torch.cuda.cudart().cudaProfilerStop()
            profiling = False
            torch.cuda.memory._dump_snapshot(str(output_dir / f"memory_end.rank{rank}.pickle"))
            torch.cuda.memory._record_memory_history(enabled=None)
        write_json(output_dir / f"steps.rank{rank}.json", rows)
        with (output_dir / f"steps.rank{rank}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        measured = [row for row in rows if row["phase"] == "measured"]
        times = [row["step_time_ms"] for row in measured]
        summary = {
            "status": "PASS",
            "implementation": args.implementation,
            "rank": rank,
            "profile": args.profile,
            "mean_step_ms": statistics.mean(times),
            "p50_step_ms": statistics.median(times),
            "p95_step_ms": percentile(times, 95),
            "max_allocated_bytes": max(row["peak_allocated_bytes"] for row in measured),
            "max_reserved_bytes": max(row["peak_reserved_bytes"] for row in measured),
            "measured_steps": len(measured),
        }
        write_json(output_dir / f"summary.rank{rank}.json", summary)
        all_rows = [None] * dist.get_world_size()
        dist.all_gather_object(all_rows, measured)
        if rank == 0:
            maximums = [
                max(per_rank[index]["step_time_ms"] for per_rank in all_rows)
                for index in range(measured_steps)
            ]
            unique_tokens = measured[0]["unique_training_tokens_per_step"]
            write_json(
                output_dir / "summary.global.json",
                {
                    **summary,
                    "rank": "max_over_ranks_each_step",
                    "mean_step_ms": statistics.mean(maximums),
                    "p50_step_ms": statistics.median(maximums),
                    "p95_step_ms": percentile(maximums, 95),
                    "unique_training_tokens_per_step": unique_tokens,
                    "tokens_per_s": unique_tokens * 1000 / statistics.mean(maximums),
                    "max_allocated_bytes": max(
                        row["peak_allocated_bytes"] for group in all_rows for row in group
                    ),
                    "max_reserved_bytes": max(
                        row["peak_reserved_bytes"] for group in all_rows for row in group
                    ),
                    "max_rank_step_ms": maximums,
                    "performance_claim_allowed": not args.profile,
                },
            )
        return 0
    except Exception as exc:
        write_json(
            output_dir / f"failure.rank{rank}.json",
            {
                "status": "FAIL",
                "type": type(exc).__name__,
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if profiling:
            torch.cuda.cudart().cudaProfilerStop()
        if args.profile:
            torch.cuda.memory._record_memory_history(enabled=None)
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
