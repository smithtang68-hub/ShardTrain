"""Per-parameter flat shards and rank-major packed materialization, without concatenation."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ParameterSpec:
    param_id: str
    unit_id: int
    global_shape: tuple[int, ...]
    tp_local_shape: tuple[int, ...]
    tp_layout: str
    tp_grad_reduce: str
    numel: int
    fsdp_shard_numel: int
    padding_numel: int
    dtype: torch.dtype

    @classmethod
    def create(cls, name, unit_id, weight, global_shape, layout, degree):
        n = weight.numel()
        q = (n + degree - 1) // degree
        return cls(
            name,
            unit_id,
            global_shape,
            tuple(weight.shape),
            layout,
            "sum" if layout == "replicated" else "none",
            n,
            q,
            degree * q - n,
            weight.dtype,
        )

    def valid_len(self, rank):
        return max(0, min(self.fsdp_shard_numel, self.numel - rank * self.fsdp_shard_numel))


@dataclass
class ShardedParameter:
    spec: ParameterSpec
    local_shard: nn.Parameter
    unit: object = None
    grad_ready_event: object = None
    grad_step: int = -1
    forward_uses: int = 0

    def weight(self):
        return self.unit.weight(self.spec.param_id)

    def mark_forward(self):
        if self.forward_uses or self.unit.state != "FWD_ACTIVE":
            raise RuntimeError(f"Repeated/early parameter consumer {self.spec.param_id}")
        self.forward_uses += 1


def initialize_shard(weight: Tensor, spec: ParameterSpec, rank: int) -> nn.Parameter:
    shard = torch.zeros(spec.fsdp_shard_numel, dtype=weight.dtype, device=weight.device)
    valid = spec.valid_len(rank)
    if valid:
        shard[:valid].copy_(weight.detach().reshape(-1)[rank * spec.fsdp_shard_numel :][:valid])
    return nn.Parameter(shard)


def offsets(specs):
    starts, length = [], 0
    for spec in specs:
        starts.append(length)
        length += spec.fsdp_shard_numel
    return starts, length


def copy_out(storage, specs, degree):
    starts, length = offsets(specs)
    assert storage.numel() == degree * length
    outputs = {}
    for spec, start in zip(specs, starts, strict=True):
        full = torch.empty(spec.numel, dtype=storage.dtype, device=storage.device)
        for rank in range(degree):
            valid = spec.valid_len(rank)
            if valid:
                full.narrow(0, rank * spec.fsdp_shard_numel, valid).copy_(
                    storage.narrow(0, rank * length + start, valid)
                )
        outputs[spec.param_id] = full.view(spec.tp_local_shape)
    return outputs


def pack_gradients(grads, specs, degree):
    starts, length = offsets(specs)
    packed = grads[0].new_zeros(degree * length)
    for grad, spec, start in zip(grads, specs, starts, strict=True):
        if tuple(grad.shape) != spec.tp_local_shape or grad.dtype != spec.dtype:
            raise RuntimeError(f"Invalid full gradient: {spec.param_id}")
        flat = grad.reshape(-1)
        for rank in range(degree):
            valid = spec.valid_len(rank)
            if valid:
                packed.narrow(0, rank * length + start, valid).copy_(
                    flat.narrow(0, rank * spec.fsdp_shard_numel, valid)
                )
    return packed
