"""Bias-free Column/Row Linear with explicit layout transitions and stable local weights."""

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F

from sharded_runtime.parallel.mappings import (
    TPGroup,
    copy_to_tp,
    gather_sequence,
    reduce_from_tp,
    reduce_scatter_sequence,
)


@dataclass(frozen=True)
class ParameterPartition:
    global_shape: tuple[int, ...]
    dim: int | None
    rank: int
    degree: int

    def local_slice(self, tensor: Tensor) -> Tensor:
        if tuple(tensor.shape) != self.global_shape:
            raise ValueError(
                f"Expected logical shape {self.global_shape}, got {tuple(tensor.shape)}"
            )
        if self.dim is None:
            return tensor
        size = self.global_shape[self.dim] // self.degree
        return tensor.narrow(self.dim, self.rank * size, size)


def _validate_linear(linear: nn.Linear, degree: int, dim: int) -> None:
    if type(linear) is not nn.Linear or linear.bias is not None:
        raise ValueError("M1 TP Linear requires an ordinary bias-free nn.Linear")
    if linear.weight.shape[dim] % degree:
        raise ValueError("TP partition dimension must be divisible by TP degree")


class ColumnParallelLinear(nn.Module):
    def __init__(self, linear: nn.Linear, group: TPGroup, *, sp: bool = False) -> None:
        super().__init__()
        _validate_linear(linear, group.degree, dim=0)
        self.group = group
        self.sp = sp
        self.in_features = linear.in_features
        self.out_features = linear.out_features // group.degree
        self.partition = ParameterPartition(tuple(linear.weight.shape), 0, group.rank, group.degree)
        self.weight = nn.Parameter(
            self.partition.local_slice(linear.weight).detach().clone(),
            requires_grad=linear.weight.requires_grad,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Column Linear requires input feature size {self.in_features}")
        x = gather_sequence(x, self.group) if self.sp else copy_to_tp(x, self.group)
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    def __init__(self, linear: nn.Linear, group: TPGroup, *, sp: bool = False) -> None:
        super().__init__()
        _validate_linear(linear, group.degree, dim=1)
        self.group = group
        self.sp = sp
        self.in_features = linear.in_features // group.degree
        self.out_features = linear.out_features
        self.partition = ParameterPartition(tuple(linear.weight.shape), 1, group.rank, group.degree)
        self.weight = nn.Parameter(
            self.partition.local_slice(linear.weight).detach().clone(),
            requires_grad=linear.weight.requires_grad,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Row Linear requires a TP-local feature slice of size {self.in_features}"
            )
        partial = F.linear(x, self.weight)
        return (
            reduce_scatter_sequence(partial, self.group)
            if self.sp
            else reduce_from_tp(partial, self.group)
        )
