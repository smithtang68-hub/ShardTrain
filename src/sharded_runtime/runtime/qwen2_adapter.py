"""Qwen registration using the existing UnitState, materializer, and communicator."""

import hashlib
import json
from types import MethodType

import torch
import torch.distributed as dist
from torch.nn import functional as F

from sharded_runtime.models.qwen2 import Qwen2RotaryEmbedding
from sharded_runtime.parallel.tp_linear import ParameterPartition
from sharded_runtime.runtime.bridge import PreBackward
from sharded_runtime.runtime.comm import RuntimeTPGroup
from sharded_runtime.runtime.engine import UnitState
from sharded_runtime.runtime.layout import ParameterSpec, ShardedParameter
from sharded_runtime.runtime.materializer import Materializer
from sharded_runtime.runtime.qwen2_operators import (
    ParameterSlot,
    RuntimeQwenEmbedding,
    RuntimeQwenLinear,
    RuntimeQwenNorm,
    RuntimeQwenProjectionGroup,
)


class TiedEmbeddingUnit(UnitState):
    """One shard/bridge owns both distant uses; sum their adjoints before one RS.

    The tied FullParam is deliberately retained across the forward and backward.
    Decoder units still reshard normally. No duplicated embedding optimizer leaf.
    """

    def embedding_complete(self, module, args, output):
        self.annotate("FWD")
        if self.runtime.forward_prefetch and len(self.runtime.units) > 1:
            self.runtime.prefetch(1, "FWD")
            self.annotate("FWD")
        return output

    def head_complete(self, module, args, output):
        if self.handles[0].forward_uses != 2:
            raise RuntimeError("Tied weight must have exactly embedding and head forward uses")
        for consumer in self.consumers:
            consumer.compute_weight = None
        self.annotate("FWD")
        self.state = "FWD_RETAINED"
        self.runtime.comm.event("tied_weight_retained", unit_id=self.unit_id)
        return PreBackward.apply(output, self)


def _source_tensor(model, name, parameter):
    source = getattr(model, "_qwen_weight_source", None)
    if source is not None:
        return source.get_tensor(name)
    if parameter.device.type != "meta":
        return parameter.detach().cpu()
    # Meta factory avoids a complete dense 1.5B model allocation on CPU/GPU.
    # Generate one CPU parameter at a time with a name-stable seed.
    seed = getattr(model, "_qwen_seed", 1234)
    key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    generator = torch.Generator().manual_seed((seed + key) % (2**63 - 1))
    if name.endswith(".bias"):
        return torch.zeros(parameter.shape)
    if "layernorm.weight" in name or name == "model.norm.weight":
        return torch.ones(parameter.shape)
    value = torch.randn(parameter.shape, generator=generator) * model.config.initializer_range
    if name == "model.embed_tokens.weight" and model.config.pad_token_id is not None:
        value[model.config.pad_token_id].zero_()
    return value


def _slot(runtime, model, unit, name, parameter, dim):
    partition = ParameterPartition(tuple(parameter.shape), dim, runtime.tp.rank, runtime.tp.degree)
    runtime.qwen_partitions[name] = partition
    value = partition.local_slice(_source_tensor(model, name, parameter)).contiguous()
    layout = {0: "column", 1: "row", None: "replicated"}[dim]
    shape_only = torch.empty(value.shape, device="meta", dtype=torch.float32)
    spec = ParameterSpec.create(
        name, unit.unit_id, shape_only, parameter.shape, layout, runtime.materializer.degree
    )
    shard = torch.zeros(spec.fsdp_shard_numel, dtype=torch.float32, device=runtime.device)
    valid = spec.valid_len(runtime.materializer.rank)
    if valid:
        shard[:valid].copy_(
            value.reshape(-1).narrow(0, runtime.materializer.rank * spec.fsdp_shard_numel, valid)
        )
    shard = torch.nn.Parameter(shard)
    # Initialization synchronization also uses the existing coordinator path.
    # FSDP ranks own DIFFERENT slices, so never broadcast their local shards.
    # Factory seed/checkpoint is identical on all ranks. TP replicas at the SAME
    # F coordinate do own the same slice, and may synchronize that slice.
    for role, group, degree in [
        ("tp", runtime.tp.process_group, runtime.tp.degree if dim is None else 1)
    ]:
        if degree > 1:
            runtime.comm.issue(
                role,
                f"qwen_init_{role}_shard",
                group,
                shard,
                shard,
                lambda async_, group=group: dist.broadcast(
                    shard, src=dist.get_global_rank(group, 0), group=group, async_op=async_
                ),
                channel="fsdp_ag",
                asynchronous=False,
            )
    handle = ShardedParameter(spec, shard, unit)

    def committed(leaf):
        if handle.grad_step == runtime.step_id:
            raise RuntimeError(f"Repeated Qwen shard commit: {name}")
        handle.grad_step = runtime.step_id
        if runtime.comm.streams:
            handle.grad_ready_event = torch.cuda.Event()
            handle.grad_ready_event.record(torch.cuda.current_stream(runtime.device))
        runtime.comm.event("leaf_grad_committed", param_id=name, unit_id=unit.unit_id)

    shard.register_post_accumulate_grad_hook(committed)
    slot = ParameterSlot(handle)
    unit.handles.append(handle)
    unit.consumers.append(slot)
    runtime.registry[name] = handle
    runtime.identities[name] = (id(shard), shard.data_ptr())
    return slot


def _hooks(runtime, unit, module):
    runtime.hooks.extend(
        [
            module.register_forward_pre_hook(unit.pre_forward),
            module.register_forward_hook(unit.post_forward),
        ]
    )


def _verify_initialization(runtime, model):
    """Reject inconsistent source weights before combining rank-local slices."""
    if runtime.tp.degree == runtime.materializer.degree == 1:
        return
    digest = hashlib.sha256()
    digest.update(json.dumps(model.config.to_dict(), sort_keys=True).encode())
    source = getattr(model, "_qwen_weight_source", None)
    if source is not None:
        # All ranks use the same shared checkpoint path, metadata and files.
        digest.update(str(source.path).encode())
        digest.update(json.dumps(source.shapes, sort_keys=True).encode())
    elif next(model.parameters()).device.type == "meta":
        digest.update(str(getattr(model, "_qwen_seed", 1234)).encode())
    else:
        for name, parameter in model.named_parameters():
            digest.update(name.encode())
            value = parameter.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode())
            # View the CPU bytes without materializing an additional state dict.
            digest.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
    token = torch.tensor(list(digest.digest()), dtype=torch.uint8, device=runtime.device)
    mismatched = False
    for role, group, degree, channel in [
        ("tp", runtime.tp.process_group, runtime.tp.degree, "sp_ag"),
        ("fsdp", runtime.materializer.group, runtime.materializer.degree, "fsdp_ag"),
    ]:
        if degree > 1:
            outputs = [torch.empty_like(token) for _ in range(degree)]
            runtime.comm.issue(
                role,
                "qwen_init_source_digest",
                group,
                token,
                outputs[0],
                lambda async_, group=group, outputs=outputs: dist.all_gather(
                    outputs, token, group=group, async_op=async_
                ),
                channel=channel,
                asynchronous=False,
                extra_outputs=outputs[1:],
            )
            mismatched |= any(not torch.equal(token, other) for other in outputs)
    mismatch = torch.tensor(int(mismatched), device=runtime.device, dtype=torch.int32)
    for role, group, degree, channel in [
        ("tp", runtime.tp.process_group, runtime.tp.degree, "sp_rs"),
        ("fsdp", runtime.materializer.group, runtime.materializer.degree, "fsdp_rs"),
    ]:
        if degree > 1:
            runtime.comm.issue(
                role,
                "qwen_init_source_mismatch",
                group,
                mismatch,
                mismatch,
                lambda async_, group=group: dist.all_reduce(
                    mismatch, op=dist.ReduceOp.MAX, group=group, async_op=async_
                ),
                channel=channel,
                asynchronous=False,
            )
    if mismatch.item():
        raise ValueError(
            "Qwen initialization source/config differs across ranks; "
            "use the same seed/checkpoint/weights"
        )


def _prepare_input(self, ids, embeds):
    if embeds is not None:
        raise ValueError(
            "ShardTrain Qwen training requires token IDs; inputs_embeds is reference-only"
        )
    if ids is None or ids.ndim != 2:
        raise ValueError("Qwen token IDs must have shape [batch, global sequence]")
    if self.sp and ids.shape[1] % self.tp.degree:
        raise ValueError("Qwen global sequence must be divisible by TP degree for SP")
    self.qwen_global_seq = ids.shape[1]
    return self.input_for_rank(ids), None


def _sequence_input(self, tensor):
    if not self.sp:
        return tensor
    if tensor.ndim < 2 or tensor.shape[1] % self.tp.degree:
        raise ValueError("Qwen sequence tensors must divide TP degree on dimension 1")
    width = tensor.shape[1] // self.tp.degree
    return tensor.narrow(1, self.tp.rank * width, width).clone()


def _sum_metric(self, value, kind):
    result = value.detach().clone()
    groups = []
    if self.sp:
        groups.append(("tp", self.tp.process_group, self.tp.degree, "sp_rs"))
    groups.append(("fsdp", self.materializer.group, self.materializer.degree, "fsdp_rs"))
    for role, group, degree, channel in groups:
        if degree > 1:
            self.comm.issue(
                role,
                kind + "_" + role,
                group,
                result,
                result,
                lambda async_, group=group: dist.all_reduce(result, group=group, async_op=async_),
                channel=channel,
                asynchronous=False,
            )
    return result


def _loss(self, logits, labels):
    if labels.ndim != 2 or labels.shape[1] != self.qwen_global_seq:
        raise ValueError("Qwen labels must describe the full global sequence")
    # Shift BEFORE SP slicing, including the next token on the shard boundary.
    shifted = F.pad(labels[:, 1:], (0, 1), value=-100)
    shifted = self.input_for_rank(shifted)
    if shifted.shape != logits.shape[:2]:
        raise ValueError("Qwen logits/label shard shape mismatch")
    count = shifted.ne(-100).sum()
    numerator = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        shifted.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    override = getattr(self, "qwen_denominator_override", None)
    self.qwen_denominator = (
        _sum_metric(self, count, "lm_valid_count") if override is None else override
    ).clamp_min(1)
    self.qwen_numerator = numerator.detach()
    # Materializer's unique FSDP average is compensated exactly once.
    return numerator * self.materializer.degree / self.qwen_denominator


def _reported_loss(self):
    return _sum_metric(self, self.qwen_numerator, "lm_loss_sum") / self.qwen_denominator


@torch.no_grad()
def register_qwen(runtime, model, mesh=None, *, sp=False):
    cfg = model.config
    runtime.sp = sp
    runtime.tp = RuntimeTPGroup(
        mesh.tp_group if mesh else None,
        runtime.comm,
        grad_group=mesh.tp_grad_group if mesh else None,
    )
    runtime.materializer = Materializer(mesh.fsdp_group if mesh else None, runtime.comm)
    runtime.fsdp_grad_group = mesh.fsdp_grad_group if mesh else None
    if sp and runtime.tp.degree == 1:
        raise ValueError("Qwen SP requires TP degree > 1")
    for name in ["num_attention_heads", "num_key_value_heads", "intermediate_size"]:
        if getattr(cfg, name) % runtime.tp.degree:
            raise ValueError(
                f"Qwen {name} must be divisible by TP degree (KV replication is unsupported)"
            )
    if runtime.units:
        raise RuntimeError("Runtime already registered")
    runtime.defer_sp_prefetch = runtime.communication_policy == "stream_ordered" and sp
    runtime.model, runtime.qwen_partitions = model, {}
    runtime.comm.phase = "INIT"
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    if tied != cfg.tie_word_embeddings:
        raise ValueError("Qwen weight tying config does not match actual Parameter aliasing")
    source = getattr(model, "_qwen_weight_source", None)
    if source is not None:
        source.validate(
            {n: tuple(p.shape) for n, p in model.named_parameters(remove_duplicate=False)},
            {"lm_head.weight": "model.embed_tokens.weight"} if tied else {},
        )
    _verify_initialization(runtime, model)
    unit = (TiedEmbeddingUnit if tied else UnitState)(runtime, 0, model.model.embed_tokens)
    runtime.units.append(unit)
    embedding_slot = _slot(
        runtime, model, unit, "model.embed_tokens.weight", model.model.embed_tokens.weight, None
    )
    model.model.embed_tokens = RuntimeQwenEmbedding(embedding_slot, cfg.pad_token_id)
    unit.block = model.model.embed_tokens
    unit.forward_prefetch_handle = embedding_slot.handle
    if tied:
        runtime.hooks.extend(
            [
                model.model.embed_tokens.register_forward_pre_hook(unit.pre_forward),
                model.model.embed_tokens.register_forward_hook(unit.embedding_complete),
            ]
        )
    else:
        _hooks(runtime, unit, model.model.embed_tokens)
    for layer_id, layer in enumerate(model.model.layers):
        unit = UnitState(runtime, len(runtime.units), layer)
        runtime.units.append(unit)
        prefix = f"model.layers.{layer_id}."
        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
            norm = getattr(layer, norm_name)
            slot = _slot(runtime, model, unit, prefix + norm_name + ".weight", norm.weight, None)
            setattr(layer, norm_name, RuntimeQwenNorm(slot, norm.eps))
        for parent_name, names in [
            ("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            ("mlp", ["gate_proj", "up_proj", "down_proj"]),
        ]:
            parent = getattr(layer, parent_name)
            for name in names:
                linear = getattr(parent, name)
                column = name not in {"o_proj", "down_proj"}
                slot = _slot(
                    runtime,
                    model,
                    unit,
                    prefix + parent_name + "." + name + ".weight",
                    linear.weight,
                    0 if column else 1,
                )
                bias_slot = (
                    _slot(
                        runtime,
                        model,
                        unit,
                        prefix + parent_name + "." + name + ".bias",
                        linear.bias,
                        0,
                    )
                    if linear.bias is not None
                    else None
                )
                setattr(
                    parent,
                    name,
                    RuntimeQwenLinear(slot, runtime.tp, bias_slot=bias_slot, column=column, sp=sp),
                )
        layer.self_attn.num_heads //= runtime.tp.degree
        layer.self_attn.num_key_value_heads //= runtime.tp.degree
        layer.self_attn.projection_group = RuntimeQwenProjectionGroup(
            (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj),
            runtime.tp,
            sp,
            "qkv",
        )
        layer.mlp.projection_group = RuntimeQwenProjectionGroup(
            (layer.mlp.gate_proj, layer.mlp.up_proj), runtime.tp, sp, "gate_up"
        )
        unit.forward_prefetch_handle = layer.self_attn.v_proj.weight_slot.handle
        unit.backward_prefetch_handle = layer.mlp.down_proj.weight_slot.handle
        _hooks(runtime, unit, layer)
    unit = UnitState(runtime, len(runtime.units), model.model.norm)
    runtime.units.append(unit)
    norm_slot = _slot(runtime, model, unit, "model.norm.weight", model.model.norm.weight, None)
    model.model.norm = RuntimeQwenNorm(norm_slot, cfg.rms_norm_eps)
    unit.block = model.model.norm
    unit.forward_prefetch_handle = unit.backward_prefetch_handle = norm_slot.handle
    _hooks(runtime, unit, model.model.norm)
    if tied:
        model.lm_head = RuntimeQwenLinear(embedding_slot, runtime.tp, replicated=True, tied=True)
        runtime.hooks.append(model.lm_head.register_forward_hook(runtime.units[0].head_complete))
    else:
        unit = UnitState(runtime, len(runtime.units), model.lm_head)
        runtime.units.append(unit)
        slot = _slot(runtime, model, unit, "lm_head.weight", model.lm_head.weight, None)
        model.lm_head = RuntimeQwenLinear(slot, runtime.tp, replicated=True)
        unit.block = model.lm_head
        _hooks(runtime, unit, model.lm_head)
    model.model.rotary_emb = Qwen2RotaryEmbedding(cfg, device=runtime.device)
    model._sharded_runtime = runtime
    runtime.prepare_qwen_input = MethodType(_prepare_input, runtime)
    runtime.input_for_rank = MethodType(_sequence_input, runtime)
    runtime.qwen_loss = MethodType(_loss, runtime)
    runtime.qwen_reported_loss = MethodType(_reported_loss, runtime)
    # Do not keep checkpoint readers or dense CPU parameter sources in the training graph.
    model.__dict__.pop("_qwen_weight_source", None)
    return runtime
