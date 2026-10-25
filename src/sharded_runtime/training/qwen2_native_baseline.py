"""Public PyTorch FSDP2/DTensor baseline, independent of the custom runtime.

The ordinary Qwen math modules are shared with the dense reference. Parameter
ownership, activation redistribution and gradient communication are exclusively
PyTorch fully_shard/DTensor operations; no custom runtime operator is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.nn import functional as F

from sharded_runtime.config import Config, ConfigError, load_config
from sharded_runtime.models.qwen2 import Qwen2RotaryEmbedding
from sharded_runtime.models.qwen2_factory import build_qwen_model
from sharded_runtime.observability.metadata import write_json
from sharded_runtime.observability.profiling import ProfileCapture
from sharded_runtime.training.performance import qwen_performance_summary
from sharded_runtime.training.qwen2_precision import resolve_qwen_compute_dtype


def local_tensor(value: Tensor) -> Tensor:
    """The optimizer/overflow checks inspect only this rank's native shards."""
    return value.to_local() if isinstance(value, DTensor) else value


def initialize_same_source(model: nn.Module, seed: int, checkpoint=None) -> None:
    """Materialize one CPU source tensor at a time, matching name-stable init.

    Startup constructs a dense CPU model per rank before native TP/FSDP. This
    deliberately simple startup is outside timing; unlike the custom runtime it
    does not directly initialize final 2D shards. No runtime code is called.
    """
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    if checkpoint is not None:
        shapes = {
            name: tuple(p.shape) for name, p in model.named_parameters(remove_duplicate=False)
        }
        checkpoint.validate(shapes, {"lm_head.weight": "model.embed_tokens.weight"} if tied else {})
    model.to_empty(device="cpu")
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.rotary_emb = Qwen2RotaryEmbedding(model.config, device="cpu")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if checkpoint is not None:
                value = checkpoint.get_tensor(name)
            elif name.endswith(".bias"):
                value = torch.zeros(parameter.shape, dtype=torch.float32)
            elif "layernorm.weight" in name or name == "model.norm.weight":
                value = torch.ones(parameter.shape, dtype=torch.float32)
            else:
                key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
                generator = torch.Generator().manual_seed((seed + key) % (2**63 - 1))
                value = torch.randn(parameter.shape, generator=generator)
                value.mul_(model.config.initializer_range)
                if name == "model.embed_tokens.weight":
                    padding = model.model.embed_tokens.padding_idx
                    if padding is not None:
                        value[padding].zero_()
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"Source shape differs for {name}")
            parameter.copy_(value)


class NativeQwenOutput(NamedTuple):
    # A dataclass is a pytree leaf in Torch2.9.1. NamedTuple allows native FSDP
    # to find logits and attach its root pre-backward hook.
    logits: Tensor
    last_hidden_state: Tensor


class NativeSequenceLinear(nn.Linear):
    """Preserve native SP layout through 3D linear's leading-axis flatten.

    Flattening [B,S,H] with Shard(S) cannot be represented by a single native
    DTensor Shard when B>1. Sequence-first [S,B,H] has a contiguous Shard(0),
    then the public linear operator and transpose restore [B,S,V]. No gather or
    custom autograd is required and the projection mathematics is unchanged.
    """

    def forward(self, inputs):
        if isinstance(inputs, DTensor) and inputs.placements == (Shard(1),):
            result = F.linear(inputs.transpose(0, 1).contiguous(), self.weight, self.bias)
            return result.transpose(0, 1).contiguous()
        return F.linear(inputs, self.weight, self.bias)


class NativeQwen2CausalLM(nn.Module):
    """Ordinary Qwen math with explicit full-context/local-SP input boundaries."""

    def __init__(self, dense_model: nn.Module) -> None:
        super().__init__()
        self.config = dense_model.config
        self.model = dense_model.model
        old_head = dense_model.lm_head
        self.lm_head = NativeSequenceLinear(
            old_head.in_features,
            old_head.out_features,
            bias=False,
            device="meta",
            dtype=old_head.weight.dtype,
        )
        self.lm_head.weight = old_head.weight
        self.tp_degree, self.tp_rank, self.sp_enabled = 1, 0, False

    def sequence_for_rank(self, value: Tensor) -> Tensor:
        if not self.sp_enabled:
            return value
        if value.shape[1] % self.tp_degree:
            raise ValueError("Global sequence must divide TP degree with SP")
        width = value.shape[1] // self.tp_degree
        return value[:, self.tp_rank * width : (self.tp_rank + 1) * width].contiguous()

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be the complete [batch, global sequence]")
        batch, seq = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        if position_ids.shape not in ((1, seq), (batch, seq)):
            raise ValueError("position_ids must describe the complete global sequence")
        hidden = self.model.embed_tokens(self.sequence_for_rank(input_ids))
        # Attention projections gather the sequence. RoPE and mask always describe
        # that complete sequence, rather than this rank's local residual shard.
        rotations = self.model.rotary_emb(hidden, position_ids.to(hidden.device))
        for index, layer in enumerate(self.model.layers):
            layer.self_attn.position_embeddings = rotations
            layer.self_attn.attention_mask = attention_mask
            with torch.profiler.record_function(f"pytorch_native/qwen/layer{index}"):
                hidden = layer(hidden)
        hidden = self.model.norm(hidden)
        return NativeQwenOutput(logits=self.lm_head(hidden), last_hidden_state=hidden)


class NativeSharedSequenceInput(ParallelStyle):
    """A native DTensor layout boundary with explicit collective dtypes.

    PrepareModuleInput does not expose redistribute's public forward_dtype and
    backward_dtype arguments. This style uses those arguments directly to gather
    norm activations in the autocast compute dtype and reduce their adjoints in
    FP32. The differentiable redistribution is entirely implemented by DTensor;
    this class contains no custom autograd Function or collective implementation.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        def prepare(_module, inputs):
            value = inputs[0]
            source = (
                value
                if isinstance(value, DTensor)
                else DTensor.from_local(value, device_mesh, [Shard(1)], run_check=False)
            )
            compute_dtype = (
                torch.get_autocast_dtype(value.device.type)
                if torch.is_autocast_enabled(value.device.type)
                else value.dtype
            )
            gathered = source.redistribute(
                placements=[Replicate()],
                forward_dtype=compute_dtype,
                backward_dtype=torch.float32,
            )
            return (gathered, *inputs[1:])

        module.register_forward_pre_hook(prepare)
        return module


def parallelize_native_qwen(
    model: NativeQwen2CausalLM, tp_mesh: DeviceMesh, *, sp_enabled: bool
) -> None:
    """TP GQA/SwiGLU and SP norms using public native layouts only."""
    degree = tp_mesh.size()
    heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    if heads % degree or kv_heads % degree or model.config.intermediate_size % degree:
        raise ValueError("Qwen attention heads, KV heads and MLP width must divide TP degree")
    model.tp_degree = degree
    model.tp_rank = tp_mesh.get_local_rank()
    model.sp_enabled = sp_enabled
    sequence_layout = Shard(1) if sp_enabled else Replicate()

    # Register the shared weight only once: applying two weight-sharding styles
    # separately to embedding and head would silently destroy their tie.
    embedding = model.model.embed_tokens
    head = model.lm_head
    tied = head.weight is embedding.weight
    shared = nn.Parameter(distribute_tensor(embedding.weight, tp_mesh, [Replicate()]))
    embedding.weight = shared
    head.weight = (
        shared if tied else nn.Parameter(distribute_tensor(head.weight, tp_mesh, [Replicate()]))
    )
    for module in (embedding, head):
        parallelize_module(
            module,
            tp_mesh,
            PrepareModuleInput(
                input_layouts=sequence_layout,
                desired_input_layouts=sequence_layout,
                use_local_output=False,
            ),
        )
        parallelize_module(
            module,
            tp_mesh,
            PrepareModuleOutput(
                output_layouts=sequence_layout,
                desired_output_layouts=sequence_layout,
                use_local_output=True,
            ),
        )
    for layer in model.model.layers:
        # Keep a single gathered DTensor shared by Q/K/V and gate/up. Returning
        # local input here could cause each projection's own backward reduction.
        plan = {
            "self_attn": NativeSharedSequenceInput()
            if sp_enabled
            else PrepareModuleInput(
                input_layouts=sequence_layout,
                desired_input_layouts=Replicate(),
                use_local_output=False,
            ),
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(output_layouts=sequence_layout),
            "mlp": NativeSharedSequenceInput()
            if sp_enabled
            else PrepareModuleInput(
                input_layouts=sequence_layout,
                desired_input_layouts=Replicate(),
                use_local_output=False,
            ),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(output_layouts=sequence_layout),
        }
        if sp_enabled:
            plan.update(
                {
                    "input_layernorm": SequenceParallel(use_local_output=True),
                    "post_attention_layernorm": SequenceParallel(use_local_output=True),
                }
            )
        parallelize_module(layer, tp_mesh, plan)
        layer.self_attn.num_heads //= degree
        layer.self_attn.num_key_value_heads //= degree
    if sp_enabled:
        parallelize_module(model.model.norm, tp_mesh, SequenceParallel(use_local_output=True))


def shard_native_qwen(
    model: NativeQwen2CausalLM,
    fsdp_mesh: DeviceMesh,
    *,
    reshard_after_forward: bool,
    forward_prefetch: bool,
    backward_prefetch: bool,
    compute_dtype: torch.dtype = torch.float32,
    precision_policy: str = "matched",
) -> dict:
    if precision_policy not in {"matched", "native_mixed"}:
        raise ValueError("precision_policy must be matched or native_mixed")
    parameter_dtype = None if precision_policy == "matched" else compute_dtype
    policy = MixedPrecisionPolicy(param_dtype=parameter_dtype, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(
            layer, mesh=fsdp_mesh, reshard_after_forward=reshard_after_forward, mp_policy=policy
        )
    # The custom final norm remains materialized after forward, so retain it here.
    fully_shard(model.model.norm, mesh=fsdp_mesh, reshard_after_forward=False, mp_policy=policy)
    # The embedding and head share a parameter owned only by this root group.
    # Both engines retain the tied embedding/head owner throughout forward and
    # backward. Native FSDP controls the exact unshard/reshard scheduling itself.
    fully_shard(model, mesh=fsdp_mesh, reshard_after_forward=False, mp_policy=policy)
    layers = list(model.model.layers)
    if forward_prefetch:
        model.set_modules_to_forward_prefetch([layers[0]])
        for current, following in zip(layers, layers[1:] + [model.model.norm], strict=True):
            current.set_modules_to_forward_prefetch([following])
    if backward_prefetch:
        model.model.norm.set_modules_to_backward_prefetch([layers[-1]])
        for index in range(1, len(layers)):
            layers[index].set_modules_to_backward_prefetch([layers[index - 1]])
    return {
        "precision_policy": precision_policy,
        "parameter_materialization_dtype": "torch.float32"
        if parameter_dtype is None
        else str(parameter_dtype),
        "gradient_reduce_dtype": "torch.float32",
        "persistent_parameter_dtype": "torch.float32",
        "optimizer_state_dtype": "torch.float32",
        "decoder_units": len(layers),
        "decoder_reshard_after_forward": reshard_after_forward,
        "final_norm_separate_unit": True,
        "final_norm_reshard_after_forward": False,
        "root_tied_weight_owner": True,
        "root_reshard_after_forward": False,
        "explicit_forward_prefetch": forward_prefetch,
        "explicit_backward_prefetch": backward_prefetch,
        "async_grad_rs": "native FSDP2 scheduling; no custom independent flag",
        "sp_forward_input_all_gather_dtype": str(compute_dtype),
        "sp_backward_input_gradient_reduce_scatter_dtype": "torch.float32",
        "shared_projection_gradient_accumulation_dtype": str(compute_dtype),
        "differences": [
            "Native FSDP shards parameter dimension 0; custom FSDP pads flattened "
            "TP-local tensors.",
            "Native FSDP2 controls physical communication streams and implicit backward prefetch.",
            "Native DTensor aggregates TP Partial replicated gradients before native "
            "FSDP reduction.",
            "Native shared QKV/gate-up input gradients accumulate in compute dtype "
            "before FP32 SP reduction; custom accumulates each branch's cast FP32 gradient.",
            "Native startup builds a dense CPU/GPU model before TP/FSDP, outside timing.",
        ],
    }


def build_native_qwen(config: Config, mesh: DeviceMesh, device: torch.device, *, policy="matched"):
    dense = build_qwen_model(config.model, config.training.seed)
    unique_numel = sum(parameter.numel() for parameter in dense.parameters())
    source = getattr(dense, "_qwen_weight_source", None)
    initialize_same_source(dense, config.training.seed, source)
    model = NativeQwen2CausalLM(dense).to(device)
    # Module.to may recreate aliases for some future/swap settings; enforce tie.
    if dense.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    parallelize_native_qwen(model, mesh["tp"], sp_enabled=config.parallel.sp_enabled)
    metadata = shard_native_qwen(
        model,
        mesh["fsdp"],
        reshard_after_forward=config.runtime.reshard_after_forward,
        forward_prefetch=config.runtime.forward_prefetch,
        backward_prefetch=config.runtime.backward_prefetch,
        compute_dtype=resolve_qwen_compute_dtype(config.training.dtype, device),
        precision_policy=policy,
    )
    return model, {**metadata, "global_unique_parameter_numel": unique_numel}


def native_loss_parts(model: NativeQwen2CausalLM, logits: Tensor, labels: Tensor):
    shifted = model.sequence_for_rank(F.pad(labels[:, 1:], (0, 1), value=-100))
    if logits.shape[:2] != shifted.shape:
        raise ValueError("Logits must have this rank's local sequence shape")
    numerator = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        shifted.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return numerator, shifted.ne(-100).sum()


@dataclass
class NativeStepResult:
    step: int
    loss: float
    valid_tokens: int
    overflow: bool
    loss_scale: float
    accumulation_steps: int


class NativeQwenTrainer:
    """Native DTensor optimizer leaves, token weighting and global AMP gate.

    FP16 scaling follows GradScaler defaults (128 initial scale as the custom
    trainer, factor2/0.5/2000). It unscales native local gradient shards explicitly
    since the generic foreach GradScaler op has no mixed Tensor/DTensor rule.
    This is trainer bookkeeping, not an autograd or collective implementation.
    """

    def __init__(self, model, mesh, config: Config):
        self.model, self.mesh = model, mesh
        self.device = next(model.parameters()).device
        self.compute_dtype = resolve_qwen_compute_dtype(config.training.dtype, self.device)
        self.dtype = config.training.dtype
        self.accumulation_steps = config.training.accumulation_steps
        self.step_id = 0
        parameters = list(model.parameters())
        if len(parameters) != len({id(parameter) for parameter in parameters}):
            raise RuntimeError("Native optimizer owns a shared parameter twice")
        if config.training.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(parameters, lr=config.training.lr, foreach=False)
        elif config.training.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(parameters, lr=config.training.lr)
        else:
            raise ValueError("Native Qwen optimizer must be sgd or adamw")
        self.loss_scale = 128.0 if self.dtype == "fp16" else 1.0
        self.growth_tracker = 0

    def sum_metric(self, value):
        result = value.detach().clone()
        if self.model.sp_enabled and self.mesh["tp"].size() > 1:
            dist.all_reduce(result, group=self.mesh["tp"].get_group())
        if self.mesh["fsdp"].size() > 1:
            dist.all_reduce(result, group=self.mesh["fsdp"].get_group())
        return result

    def backward_batch(self, input_ids, labels=None, *, attention_mask=None, position_ids=None):
        """Do not step the optimizer, exposing gradients for the independent oracle."""
        labels = input_ids if labels is None else labels
        batch = input_ids.shape[0]
        if labels.shape != input_ids.shape or batch % self.accumulation_steps:
            raise ValueError("Labels must match IDs and batch must divide accumulation_steps")
        micro = batch // self.accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)
        shifted = self.model.sequence_for_rank(F.pad(labels[:, 1:], (0, 1), value=-100))
        count = self.sum_metric(shifted.ne(-100).sum())
        numerator_total = torch.zeros((), device=self.device, dtype=torch.float32)
        for index in range(self.accumulation_steps):
            start, end = index * micro, (index + 1) * micro

            def slice_batch(value, start=start, end=end):
                return value[start:end] if value is not None and value.shape[0] == batch else value

            # Match the custom runtime: reduce each microbatch, then accumulate
            # FP32 sharded gradients, rather than using an optimized no_sync path.
            with torch.autocast(
                self.device.type,
                dtype=self.compute_dtype,
                enabled=self.dtype != "fp32",
                cache_enabled=False,
            ):
                output = self.model(
                    input_ids[start:end],
                    attention_mask=slice_batch(attention_mask),
                    position_ids=slice_batch(position_ids),
                )
                numerator, _ = native_loss_parts(self.model, output.logits, labels[start:end])
                loss = numerator * self.mesh["fsdp"].size() / count.clamp_min(1)
            numerator_total.add_(numerator.detach())
            (loss * self.loss_scale).backward()
            del output, numerator, loss
        if self.loss_scale != 1.0:
            with torch.no_grad():
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        local_tensor(parameter.grad).mul_(1.0 / self.loss_scale)
        reported = self.sum_metric(numerator_total) / count.clamp_min(1)
        checks = [
            torch.isfinite(local_tensor(p.grad)).all()
            for p in self.model.parameters()
            if p.grad is not None
        ]
        finite = (torch.stack(checks).all() & torch.isfinite(reported)).to(torch.int32)
        if dist.get_world_size() > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        return reported, count, bool(finite.item())

    def optimizer_step(self, finite: bool) -> None:
        if finite:
            self.optimizer.step()
            if self.dtype == "fp16":
                self.growth_tracker += 1
                if self.growth_tracker == 2000:
                    self.loss_scale *= 2.0
                    self.growth_tracker = 0
        elif self.dtype == "fp16":
            self.loss_scale *= 0.5
            self.growth_tracker = 0
        else:
            raise FloatingPointError("Nonfinite native Qwen loss/gradients")

    def train_step(self, *args, **kwargs) -> NativeStepResult:
        loss, count, finite = self.backward_batch(*args, **kwargs)
        self.optimizer_step(finite)
        result = NativeStepResult(
            self.step_id,
            float(loss.item()),
            int(count.item()),
            not finite,
            self.loss_scale,
            self.accumulation_steps,
        )
        self.step_id += 1
        return result


def _memory(device):
    return {
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "max_reserved": torch.cuda.max_memory_reserved(device),
        "allocated_end": torch.cuda.memory_allocated(device),
        "reserved_end": torch.cuda.memory_reserved(device),
    }


def _ownership(model, trainer):
    parameters = list(model.parameters())
    return {
        "parameter_bytes": sum(local_tensor(p).numel() * p.element_size() for p in parameters),
        "gradient_bytes": sum(
            local_tensor(p.grad).numel() * p.grad.element_size()
            for p in parameters
            if p.grad is not None
        ),
        "optimizer_tensor_bytes": sum(
            local_tensor(v).numel() * v.element_size()
            for state in trainer.optimizer.state.values()
            for v in state.values()
            if isinstance(v, Tensor)
        ),
        "persistent_shard_count": len(parameters),
        "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
        "gradient_dtypes": sorted({str(p.grad.dtype) for p in parameters if p.grad is not None}),
        "tied_identity_preserved": model.lm_head.weight is model.model.embed_tokens.weight,
        "dtensor_parameters": all(isinstance(p, DTensor) for p in parameters),
        "master_weight_strategy": "Native persistent FP32 DTensor shards are optimizer weights",
    }


def _topology(directory: Path):
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        output = result.stdout + result.stderr
        status = {"returncode": result.returncode, "command": "nvidia-smi topo -m"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        output, status = str(exc), {"error": repr(exc), "command": "nvidia-smi topo -m"}
    (directory / "gpu_topology.txt").write_text(output, encoding="utf-8")
    write_json(directory / "gpu_topology_status.json", status)


def run(
    config: Config,
    *,
    warmup=3,
    profile_mode="off",
    profile_detail="low_overhead",
    precision_policy="matched",
    peak_tflops_per_gpu=None,
):
    config.validate(int(os.environ.get("WORLD_SIZE", "1")), mode="runtime")
    if config.model.model_type != "qwen2" or config.parallel.device != "cuda":
        raise ConfigError("Native FSDP2 benchmark requires Qwen2 on CUDA")
    if warmup < 0 or profile_mode not in {"off", "torch", "nsight"}:
        raise ConfigError("Invalid warmup/profile mode")
    if profile_detail not in {"low_overhead", "full"}:
        raise ConfigError("profile_detail must be low_overhead or full")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = config.parallel.world_size
    directory = Path(config.debug.output_dir)
    if (directory / f"benchmark.rank{rank}.json").exists():
        raise ConfigError("Choose a fresh output directory to preserve native experiment records")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # Reject unsupported BF16 before constructing/materializing a large model.
    resolve_qwen_compute_dtype(config.training.dtype, device)
    created_pg = not dist.is_initialized()
    if created_pg:
        backend = config.parallel.backend
        kwargs = {}
        if world_size == 1 and "MASTER_ADDR" not in os.environ:
            kwargs = {"store": dist.HashStore(), "rank": 0, "world_size": 1}
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=config.debug.distributed_timeout_seconds),
            **kwargs,
        )
    directory.mkdir(parents=True, exist_ok=True)
    mesh = init_device_mesh(
        "cuda",
        (config.parallel.fsdp_degree, config.parallel.tp_degree),
        mesh_dim_names=("fsdp", "tp"),
    )
    coordinate = mesh.get_coordinate()
    rows, warmup_rows = [], []
    model = trainer = capture = None
    failure = None
    payload = {
        "engine": "pytorch_fsdp2_dtensor",
        "status": "initializing",
        "rank": rank,
        "world_size": world_size,
        "coordinate": {"fsdp": coordinate[0], "tp": coordinate[1]},
        "versions": {
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "nccl": torch.cuda.nccl.version() if dist.is_nccl_available() else None,
            "platform": platform.platform(),
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": torch.cuda.get_device_capability(device),
        "dtype": config.training.dtype,
        "optimizer": config.training.optimizer,
        "global_batch": config.training.local_batch_size
        * config.training.accumulation_steps
        * config.parallel.fsdp_degree,
        "micro_batch": config.training.local_batch_size,
        "sequence_length": config.model.seq_len,
        "accumulation_steps": config.training.accumulation_steps,
        "warmup": warmup,
        "profile_mode": profile_mode,
        "profile_detail": profile_detail,
        "runtime_switches_requested": asdict(config.runtime),
        "attention_backend": "PyTorch SDPA math; explicit grouped KV expansion",
        "precision_policy": {
            "compute": config.training.dtype,
            "linear_backward": config.training.dtype,
            "parameters": "fp32",
            "parameter_all_gather": "fp32"
            if precision_policy == "matched"
            else config.training.dtype,
            "gradient_reduce": "fp32",
            "optimizer_state": "fp32",
            "norm_accumulation": "fp32",
            "loss": "fp32",
            "sp_forward_input_all_gather": config.training.dtype,
            "sp_forward_output_reduce_scatter": config.training.dtype,
            "sp_backward_output_grad_all_gather": config.training.dtype,
            "sp_backward_input_grad_reduce_scatter": "fp32",
        },
        "loss_scaler": "FP16 dynamic scale128, growth2/2000, backoff0.5; "
        "native local-shard unscale",
        "oom": False,
        "nonfinite_loss": False,
        "nonfinite_detected": False,
        "errors": [],
    }
    try:
        torch.manual_seed(config.training.seed + rank)
        if rank == 0:
            _topology(directory)
        model, native = build_native_qwen(config, mesh, device, policy=precision_policy)
        payload["native_policy"] = native
        payload["model_source"] = {
            "initialization": "local_safetensors" if config.model.checkpoint_path else "random",
            "seed": config.training.seed,
            "config": model.config.to_dict(),
            "global_unique_parameter_numel": native["global_unique_parameter_numel"],
            "tied_aliases": {"lm_head.weight": "model.embed_tokens.weight"}
            if model.config.tie_word_embeddings
            else {},
            "memory_strategy": "dense CPU/GPU startup then native TP/FSDP; outside measured window",
        }
        trainer = NativeQwenTrainer(model, mesh, config)
        payload["optimizer_hyperparameters"] = trainer.optimizer.defaults
        generator = torch.Generator().manual_seed(config.training.seed + coordinate[0])
        ids = torch.randint(
            model.config.vocab_size,
            (
                config.training.local_batch_size * config.training.accumulation_steps,
                config.model.seq_len,
            ),
            generator=generator,
        )
        payload["input_batch_digest"] = hashlib.sha256(
            json.dumps(ids.tolist(), separators=(",", ":")).encode()
        ).hexdigest()
        ids = ids.to(device)
        mask = torch.ones_like(ids)
        positions = torch.arange(config.model.seq_len, device=device).unsqueeze(0)
        for _ in range(warmup):
            warmup_result = trainer.train_step(ids, attention_mask=mask, position_ids=positions)
            warmup_rows.append(asdict(warmup_result))
            if not math.isfinite(warmup_result.loss):
                payload["nonfinite_loss"] = True
                raise FloatingPointError("Nonfinite native warmup loss")
        payload["initialization_and_warmup_memory"] = _memory(device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        capture = ProfileCapture(directory, rank, device, mode=profile_mode, detail=profile_detail)
        capture.start()
        for step in range(config.training.steps):
            dist.barrier()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            start = time.perf_counter()
            scope = torch.profiler.record_function(f"pytorch_native/train_step{warmup + step}")
            with scope if profile_mode != "off" else nullcontext():
                result = trainer.train_step(ids, attention_mask=mask, position_ids=positions)
            end_event.record()
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            maximum = torch.tensor(seconds, device=device, dtype=torch.float64)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            rows.append(
                {
                    **asdict(result),
                    "measured_step": step,
                    "seconds": seconds,
                    "max_rank_seconds": float(maximum.item()),
                    "cuda_elapsed_ms": start_event.elapsed_time(end_event),
                    **_memory(device),
                }
            )
            capture.step(result.step)
            if not math.isfinite(result.loss):
                payload["nonfinite_loss"] = True
                raise FloatingPointError("Nonfinite native loss")
        payload["status"] = "completed"
    except BaseException as exc:
        failure = exc
        payload["status"] = "failed"
        payload["oom"] = (
            isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        )
        payload["errors"].append(repr(exc))
    finally:
        if capture is not None:
            try:
                capture.close()
            except Exception as exc:
                payload["errors"].append("profiling: " + repr(exc))
                if failure is None:
                    failure, payload["status"] = exc, "failed"
        payload.update(
            {
                "steps": rows,
                "warmup_steps": warmup_rows,
                "steady_steps_completed": len(rows),
                "optimizer_updates": sum(not row["overflow"] for row in rows),
                "overflow_steps": sum(row["overflow"] for row in rows),
                "warmup_overflow_steps": sum(row["overflow"] for row in warmup_rows),
                **_memory(device),
            }
        )
        payload["nonfinite_detected"] = any(row["overflow"] for row in warmup_rows + rows)
        if model is not None and trainer is not None:
            payload["ownership"] = _ownership(model, trainer)
            payload["performance"] = qwen_performance_summary(
                parameter_numel=payload["model_source"]["global_unique_parameter_numel"],
                layers=model.config.num_hidden_layers,
                hidden_size=model.config.hidden_size,
                sequence_length=config.model.seq_len,
                global_batch=payload["global_batch"],
                steps=rows,
                world_size=world_size,
                peak_tflops_per_gpu=peak_tflops_per_gpu,
                profile_mode=profile_mode,
            )
            payload["tokens_per_second"] = payload["performance"]["global_valid_tokens_per_second"]
        payload["timing_method"] = "Barrier before wall/CUDA timers; end CUDA synchronization "
        payload["timing_method"] += "inside timer; max-rank reduction, JSON output outside timer."
        payload["profiling_note"] = "Use off for performance. Instrumented trace and allocator "
        payload["profiling_note"] += (
            "snapshots are separate runs; snapshots exclude direct CUDA/NCCL memory."
        )
        write_json(directory / f"benchmark.rank{rank}.json", payload)
        write_json(
            directory / f"environment.rank{rank}.json",
            {
                "versions": payload["versions"],
                "config": config.to_dict(),
                "mesh": mesh.mesh.tolist(),
                "engine": payload["engine"],
            },
        )
        if created_pg:
            dist.destroy_process_group()
    if failure is not None:
        raise failure
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default="configs/qwen25_tiny.yaml")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-mode", choices=["off", "torch", "nsight"], default="off")
    parser.add_argument(
        "--profile-detail", choices=["low_overhead", "full"], default="low_overhead"
    )
    parser.add_argument(
        "--precision-policy", choices=["matched", "native_mixed"], default="matched"
    )
    parser.add_argument("--peak-tflops-per-gpu", type=float, default=None)
    args, overrides = parser.parse_known_args(argv)
    try:
        run(
            load_config(args.config, overrides),
            warmup=args.warmup,
            profile_mode=args.profile_mode,
            profile_detail=args.profile_detail,
            precision_policy=args.precision_policy,
            peak_tflops_per_gpu=args.peak_tflops_per_gpu,
        )
    except (ConfigError, ValueError) as exc:
        parser.error(str(exc))
    return 0
