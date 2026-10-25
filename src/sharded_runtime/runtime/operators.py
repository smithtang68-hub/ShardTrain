"""Shard-aware parameter consumers. Save activations, never FullParam or full SP AG input."""

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from sharded_runtime.parallel.shared_projection import tagged


class ParameterLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, handle, group, column, sp):
        handle.unit.annotate("FWD")
        ctx.handle, ctx.group, ctx.column, ctx.sp = handle, group, column, sp
        # Clone only if needed to break a large view's ownership.
        local = x.contiguous()
        if local.untyped_storage().nbytes() != local.numel() * local.element_size():
            local = local.clone()
        ctx.save_for_backward(local)
        source = group.gather(local, 1) if column and sp else local
        handle.unit.prefetch_after_input(handle, "FWD")
        partial = F.linear(source, weight)
        if column and sp:
            group.retire_activation(source)
        if column:
            return partial
        with tagged(group, handle.spec.param_id.rsplit(".", 2)[-2] + "_output"):
            return group.reduce_scatter_sequence(partial) if sp else group.sum(partial)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        ctx.handle.unit.annotate("BWD")
        (local,) = ctx.saved_tensors
        group, sp, column = ctx.group, ctx.sp, ctx.column
        weight = ctx.handle.weight()  # pre_backward has provided a new ready generation.
        grad = gradient.contiguous()
        if grad.untyped_storage().nbytes() != grad.numel() * grad.element_size():
            grad = grad.clone()
        if not column and sp:
            with tagged(group, ctx.handle.spec.param_id.rsplit(".", 2)[-2] + "_row_dx"):
                grad = group.gather(grad, 1)
        ctx.handle.unit.prefetch_after_input(ctx.handle, "BWD")
        source = group.gather(local, 1) if column and sp else local
        dw = grad.reshape(-1, grad.shape[-1]).t() @ source.reshape(-1, source.shape[-1])
        dx = grad @ weight
        if column and sp:
            group.retire_activation(source)
        if not column and sp:
            group.retire_activation(grad)
        if column:
            dx = group.reduce_scatter_sequence(dx) if sp else group.sum(dx)
        return dx, dw, None, None, None, None


class ParameterRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, handle, eps):
        handle.unit.annotate("FWD")
        inverse = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        ctx.handle = handle
        ctx.save_for_backward(x, inverse)
        return x * inverse * gamma

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        ctx.handle.unit.annotate("BWD")
        x, inverse = ctx.saved_tensors
        gamma = ctx.handle.weight()
        scaled = gradient * gamma
        dx = inverse * (scaled - x * inverse.square() * (scaled * x).mean(-1, keepdim=True))
        dgamma = (gradient * x * inverse).sum(tuple(range(x.ndim - 1)))
        return dx, dgamma, None, None


class RuntimeLinear(nn.Module):
    def __init__(self, handle, group, column, sp):
        super().__init__()
        self.handle, self.group, self.column, self.sp = handle, group, column, sp
        self.compute_weight = None

    def forward(self, x):
        self.handle.mark_forward()
        if self.compute_weight is None:
            raise RuntimeError(f"Missing forward materialization {self.handle.spec.param_id}")
        return ParameterLinear.apply(
            x, self.compute_weight, self.handle, self.group, self.column, self.sp
        )


class RuntimeNorm(nn.Module):
    def __init__(self, handle, eps):
        super().__init__()
        self.handle, self.eps = handle, eps
        self.compute_weight = None

    def forward(self, x):
        self.handle.mark_forward()
        if self.compute_weight is None:
            raise RuntimeError(f"Missing norm materialization {self.handle.spec.param_id}")
        return ParameterRMSNorm.apply(x, self.compute_weight, self.handle, self.eps)
