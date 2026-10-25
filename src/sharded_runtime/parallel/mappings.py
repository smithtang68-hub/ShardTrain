"""Autograd maps for one logical replicated loss or disjoint SP token contributions."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class TPGroup:
    process_group: Any = None
    degree: int = 1
    rank: int = 0
    backend: str = "local"

    def __post_init__(self) -> None:
        if self.degree < 1 or not 0 <= self.rank < self.degree:
            raise ValueError("Invalid TP group degree/rank")
        if self.degree > 1 and self.process_group is None:
            raise ValueError("A non-singleton TP group requires an explicit ProcessGroup")

    @classmethod
    def from_process_group(cls, group: Any = None) -> "TPGroup":
        if not dist.is_initialized():
            if group is not None:
                raise ValueError("ProcessGroup supplied before distributed initialization")
            return cls()
        if group is None:
            raise ValueError("Pass the explicit TP ProcessGroup; WORLD is not an implicit TP group")
        return cls(
            group, dist.get_world_size(group), dist.get_rank(group), str(dist.get_backend(group))
        )

    def sum(self, x: Tensor) -> Tensor:
        if self.degree == 1:
            return x
        # Never mutate an upstream tensor or an expanded/shared backward gradient.
        output = x.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.process_group)
        return output

    def gather(self, x: Tensor, dim: int) -> Tensor:
        if self.degree == 1:
            return x
        source = x.contiguous()
        outputs = [torch.empty_like(source) for _ in range(self.degree)]
        dist.all_gather(outputs, source, group=self.process_group)
        return torch.cat(outputs, dim=dim)

    def sequence_shard(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[1] % self.degree:
            raise ValueError("SP requires [B,S,H] with nonempty sequence divisible by TP degree")
        size = x.shape[1] // self.degree
        return x.narrow(1, self.rank * size, size).contiguous()

    def reduce_scatter_sequence(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[1] % self.degree:
            raise ValueError("SP ReduceScatter requires a divisible full sequence in [B,S,H]")
        if self.degree == 1:
            return x
        if self.backend == "gloo":
            # Mathematical fallback only; this is not the NCCL ReduceScatter communication path.
            return self.sequence_shard(self.sum(x)).clone()
        if self.backend != "nccl":
            raise ValueError(f"Unsupported TP backend: {self.backend}")
        packed = x.movedim(1, 0).contiguous()
        output = torch.empty(
            (packed.shape[0] // self.degree, *packed.shape[1:]),
            dtype=x.dtype,
            device=x.device,
        )
        dist.reduce_scatter_tensor(output, packed, op=dist.ReduceOp.SUM, group=self.process_group)
        return output.movedim(0, 1).contiguous()

    def broadcast(self, x: Tensor) -> None:
        if self.degree > 1:
            dist.broadcast(
                x, src=dist.get_global_rank(self.process_group, 0), group=self.process_group
            )


class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return ctx.group.sum(gradient), None


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        return group.sum(x)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        # Same output adjoint is replicated on TP ranks, not independently counted p times.
        return gradient, None


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("SP AllGather expects nonempty sequence shards in [B,S_local,H]")
        ctx.group = group
        return group.gather(x, dim=1)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return ctx.group.reduce_scatter_sequence(gradient), None


class _ReduceScatterSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return group.reduce_scatter_sequence(x)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return ctx.group.gather(gradient, dim=1), None


def copy_to_tp(x: Tensor, group: TPGroup) -> Tensor:
    return _CopyToTP.apply(x, group)


def reduce_from_tp(x: Tensor, group: TPGroup) -> Tensor:
    return _ReduceFromTP.apply(x, group)


def gather_sequence(x: Tensor, group: TPGroup) -> Tensor:
    return _GatherSequence.apply(x, group)


def reduce_scatter_sequence(x: Tensor, group: TPGroup) -> Tensor:
    return _ReduceScatterSequence.apply(x, group)
