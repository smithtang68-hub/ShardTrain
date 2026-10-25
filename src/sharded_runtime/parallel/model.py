"""Constrained module conversion and parameter/layout ownership for synchronous M1."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from sharded_runtime.models.transformer_block import (
    CausalSelfAttention,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
)
from sharded_runtime.parallel.mappings import TPGroup
from sharded_runtime.parallel.shared_projection import SharedProjectionGroup
from sharded_runtime.parallel.tp_linear import (
    ColumnParallelLinear,
    ParameterPartition,
    RowParallelLinear,
)


@dataclass(frozen=True)
class TensorParallelPlan:
    model: nn.Module
    group: TPGroup
    sp: bool
    partitions: dict[str, ParameterPartition]

    def input_for_rank(self, full_batch: Tensor) -> Tensor:
        return self.group.sequence_shard(full_batch) if self.sp else full_batch

    def backward_loss(self, output: Tensor) -> Tensor:
        return output.square().mean() / (self.group.degree if self.sp else 1)

    def reported_loss(self, local_loss: Tensor) -> Tensor:
        return self.group.sum(local_loss.detach()) if self.sp else local_loss.detach()

    def finalize_gradients(self) -> None:
        # Called once per optimizer step, after ALL microbatch backwards, never per microbatch.
        if self.sp:
            for name, parameter in self.model.named_parameters():
                if self.partitions[name].dim is None:
                    if parameter.grad is None:
                        raise RuntimeError(f"Replicated parameter gradient is missing: {name}")
                    parameter.grad = self.group.sum(parameter.grad)

    def global_parameters(self, *, gradients: bool = False) -> dict[str, Tensor]:
        """Explicit correctness-only reconstruction. Never called by model forward/backward."""
        outputs = {}
        for name, parameter in self.model.named_parameters():
            tensor = parameter.grad if gradients else parameter
            if tensor is None:
                raise RuntimeError(f"No gradient available for {name}")
            tensor = tensor.detach()
            partition = self.partitions[name]
            outputs[name] = (
                tensor.clone()
                if partition.dim is None
                else self.group.gather(tensor, dim=partition.dim)
            )
        return outputs


def parallelize_tp_sp(
    model: nn.Module, tp_group: Any = None, *, sp: bool = False, broadcast_fn=None
) -> TensorParallelPlan:
    """Mutate a supported reference module BEFORE constructing its optimizer; return its plan."""
    if hasattr(model, "_tp_plan"):
        raise ValueError("Module is already TP-parallelized")
    if type(model) is TransformerBlock:
        blocks = [model]
    elif (
        type(model) is nn.Sequential
        and len(model)
        and all(type(block) is TransformerBlock for block in model)
    ):
        blocks = list(model)
    else:
        raise ValueError("M1 only supports TransformerBlock or Sequential of original Blocks")
    group = TPGroup.from_process_group(tp_group)
    if group.degree not in {1, 2, 4}:
        raise ValueError("TP supports degree 1, 2 or 4")
    if sp and group.degree == 1:
        raise ValueError("SP requires TP degree > 1")
    parameters = list(model.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in parameters}) != len(parameters):
        raise ValueError("Shared parameters are not supported by M1 conversion")
    if len(parameters) != 9 * len(blocks):
        raise ValueError("M1 requires exactly the original nine parameters per Block")
    if any(
        p.dtype != torch.float32 or not p.requires_grad or p.grad is not None for _, p in parameters
    ):
        raise ValueError("M1 conversion requires untrained, trainable FP32 parameters")
    for block in blocks:
        children = dict(block.named_children())
        expected_children = {
            "attention_norm": RMSNorm,
            "attention": CausalSelfAttention,
            "mlp_norm": RMSNorm,
            "mlp": SwiGLU,
        }
        if set(children) != set(expected_children) or any(
            type(children[name]) is not cls for name, cls in expected_children.items()
        ):
            raise ValueError("M1 requires original Block submodules without custom replacements")
        if block.attention.num_heads % group.degree:
            raise ValueError("Attention heads must be divisible by TP degree")
        expected_projections = (
            (block.attention, {"q_proj", "k_proj", "v_proj", "out_proj"}),
            (block.mlp, {"gate_proj", "up_proj", "down_proj"}),
        )
        for module, names in expected_projections:
            if set(dict(module.named_children())) != names:
                raise ValueError("M1 requires the original attention/MLP projection set")
            for name, linear in module.named_children():
                dim = 1 if name in {"out_proj", "down_proj"} else 0
                if type(linear) is not nn.Linear or linear.bias is not None:
                    raise ValueError("Only original bias-free Block projections are supported")
                if linear.weight.shape[dim] % group.degree:
                    raise ValueError("Projection partition dimension is not divisible by TP degree")
    with torch.no_grad():
        for _, parameter in parameters:
            if broadcast_fn is None:
                group.broadcast(parameter)
            else:
                broadcast_fn(parameter, group)
    for block in blocks:
        for module in (block.attention, block.mlp):
            for name, linear in list(module.named_children()):
                parallel_cls = (
                    RowParallelLinear if name in {"out_proj", "down_proj"} else ColumnParallelLinear
                )
                setattr(module, name, parallel_cls(linear, group, sp=sp))
        block.attention.num_heads //= group.degree
    partitions = {}
    for name, parameter in model.named_parameters():
        owner_name = name.rsplit(".", 1)[0] if "." in name else ""
        owner = model.get_submodule(owner_name)
        partitions[name] = (
            owner.partition
            if isinstance(owner, (ColumnParallelLinear, RowParallelLinear))
            else ParameterPartition(tuple(parameter.shape), None, group.rank, group.degree)
        )
    plan = TensorParallelPlan(model, group, sp, partitions)
    if sp:
        for block in blocks:
            block.attention.projection_group = SharedProjectionGroup(
                (block.attention.q_proj, block.attention.k_proj, block.attention.v_proj),
                group,
                "qkv",
            )
            block.mlp.projection_group = SharedProjectionGroup(
                (block.mlp.gate_proj, block.mlp.up_proj), group, "gate_up"
            )
    model._tp_plan = plan
    return plan
