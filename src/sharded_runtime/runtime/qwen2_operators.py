"""Qwen parameter consumers: compact saved activations, never saved full weights."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


def compact(x):
    value = x.contiguous()
    return (
        value.clone()
        if value.untyped_storage().nbytes() != value.numel() * value.element_size()
        else value
    )


def compute_input(x):
    if x.dtype in (torch.float16, torch.bfloat16, torch.float32) and torch.is_autocast_enabled(
        x.device.type
    ):
        return compact(x.to(torch.get_autocast_dtype(x.device.type)))
    return compact(x)


@dataclass
class ParameterSlot:
    handle: object
    compute_weight: object = None

    def forward_weight(self):
        if self.compute_weight is None or self.handle.unit.state != "FWD_ACTIVE":
            raise RuntimeError("Qwen consumer has no current forward materialization")
        return self.compute_weight


class QwenProjection(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, group, providers, sp, *weights):
        ctx.set_materialize_grads(False)
        ctx.group, ctx.providers, ctx.sp, ctx.input_dtype = group, providers, sp, x.dtype
        local = compute_input(x)
        ctx.compute_dtype = local.dtype
        providers[0].weight_slot.handle.unit.annotate("FWD")
        ctx.save_for_backward(local)
        ctx.local_lease = group.comm.buffer(local, "projection_local_saved")
        pending = group.issue_gather(local, 1) if sp else None
        providers[-1].weight_slot.handle.unit.prefetch_after_input(
            providers[-1].weight_slot.handle, "FWD"
        )
        source = group.finish_gather(pending) if sp else local
        outputs, offset = [], 0
        for provider in providers:
            weight = weights[offset]
            bias = weights[offset + 1] if provider.bias_slot else None
            outputs.append(
                F.linear(
                    source, weight.to(source.dtype), None if bias is None else bias.to(source.dtype)
                )
            )
            offset += 2 if provider.bias_slot else 1
        if sp:
            group.retire_activation(source)
        return tuple(outputs)

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *gradients):
        ctx.providers[0].weight_slot.handle.unit.annotate("BWD")
        (local,) = ctx.saved_tensors
        source = ctx.group.gather(local, 1) if ctx.sp else local
        dparameters, dx = [], None
        with torch.autocast(local.device.type, enabled=False):
            matrix = source.to(ctx.compute_dtype).reshape(-1, source.shape[-1])
            for gradient, provider in zip(gradients, ctx.providers, strict=True):
                if gradient is None:
                    raise RuntimeError("Unused Qwen projection branches are unsupported")
                weight = provider.weight_slot.handle.weight()
                grad = gradient.to(ctx.compute_dtype)
                dw = grad.reshape(-1, grad.shape[-1]).t() @ matrix
                # Use the forward's effective operands for the main GEMMs. Master
                # shard gradients remain FP32 without forcing the GEMMs to FP32.
                dparameters.append(dw.to(weight.dtype))
                contribution = (grad @ weight.to(ctx.compute_dtype)).to(ctx.input_dtype)
                dx = contribution if dx is None else dx + contribution
                if provider.bias_slot:
                    dparameters.append(
                        gradient.float()
                        .sum(tuple(range(gradient.ndim - 1)))
                        .to(ctx.compute_dtype)
                        .to(weight.dtype)
                    )
        if ctx.sp:
            ctx.group.retire_activation(source)
            dx = ctx.group.reduce_scatter_sequence(dx)
        elif ctx.group.degree > 1:
            dx = ctx.group.sum(dx)
        ctx.group.comm.retire(ctx.local_lease)
        return dx.to(ctx.input_dtype), None, None, None, *dparameters


class QwenRowLinear(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, weight, slot, group, sp, replicated):
        slot.handle.unit.annotate("FWD")
        ctx.slot, ctx.group, ctx.sp, ctx.replicated, ctx.input_dtype = (
            slot,
            group,
            sp,
            replicated,
            x.dtype,
        )
        local = compute_input(x)
        ctx.compute_dtype = local.dtype
        ctx.save_for_backward(local)
        slot.handle.unit.prefetch_after_input(slot.handle, "FWD")
        partial = F.linear(local, weight.to(local.dtype))
        if replicated:
            return partial
        return group.reduce_scatter_sequence(partial) if sp else group.sum(partial)

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gradient):
        ctx.slot.handle.unit.annotate("BWD")
        (local,) = ctx.saved_tensors
        grad = compact(gradient)
        if ctx.sp and not ctx.replicated:
            grad = ctx.group.gather(grad, 1)
        ctx.slot.handle.unit.prefetch_after_input(ctx.slot.handle, "BWD")
        weight = ctx.slot.handle.weight()
        with torch.autocast(local.device.type, enabled=False):
            matrix_grad = grad.to(ctx.compute_dtype)
            dw = matrix_grad.reshape(-1, grad.shape[-1]).t() @ local.to(ctx.compute_dtype).reshape(
                -1, local.shape[-1]
            )
            dx = matrix_grad @ weight.to(ctx.compute_dtype)
        if ctx.sp and not ctx.replicated:
            ctx.group.retire_activation(grad)
        return (
            dx.to(ctx.input_dtype),
            dw.to(weight.dtype),
            None,
            None,
            None,
            None,
        )


class QwenNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, slot, eps):
        slot.handle.unit.annotate("FWD")
        ctx.slot, ctx.input_dtype = slot, x.dtype
        value = compact(x.float())
        inverse = torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        normalized = (value * inverse).to(x.dtype)
        ctx.save_for_backward(value, inverse, normalized)
        result = normalized * weight
        slot.handle.unit.prefetch_after_input(slot.handle, "FWD")
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        ctx.slot.handle.unit.annotate("BWD")
        value, inverse, normalized = ctx.saved_tensors
        weight = ctx.slot.handle.weight()
        ctx.slot.handle.unit.prefetch_after_input(ctx.slot.handle, "BWD")
        scaled = gradient.float() * weight.float()
        dx = inverse * (scaled - value * inverse.square() * (scaled * value).mean(-1, keepdim=True))
        dw = (gradient.float() * normalized.float()).sum(tuple(range(gradient.ndim - 1)))
        return dx.to(ctx.input_dtype), dw.to(weight.dtype), None, None


class QwenEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, weight, slot, padding_idx):
        ctx.slot, ctx.padding_idx = slot, padding_idx
        ctx.save_for_backward(ids)
        result = F.embedding(ids, weight, padding_idx=padding_idx)
        slot.handle.unit.prefetch_after_input(slot.handle, "FWD")
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        ctx.slot.handle.unit.annotate("BWD")
        (ids,) = ctx.saved_tensors
        handle = ctx.slot.handle
        # Embedding adjoint uses shape metadata, not the forward FullParam.
        dw = torch.zeros(
            handle.spec.tp_local_shape, device=gradient.device, dtype=handle.spec.dtype
        )
        dw.index_add_(0, ids.reshape(-1), gradient.reshape(-1, gradient.shape[-1]).to(dw.dtype))
        if ctx.padding_idx is not None:
            dw[ctx.padding_idx].zero_()
        return None, dw, None, None


class RuntimeQwenLinear(nn.Module):
    def __init__(
        self,
        weight_slot,
        group,
        *,
        bias_slot=None,
        column=False,
        sp=False,
        replicated=False,
        tied=False,
    ):
        super().__init__()
        self.weight_slot, self.bias_slot, self.group = weight_slot, bias_slot, group
        self.column, self.sp, self.replicated, self.tied = column, sp, replicated, tied
        self.register_parameter("weight", weight_slot.handle.local_shard)
        self.register_parameter("bias", bias_slot.handle.local_shard if bias_slot else None)

    def mark(self):
        handle = self.weight_slot.handle
        if self.tied:
            if handle.forward_uses != 1 or handle.unit.state != "FWD_ACTIVE":
                raise RuntimeError("Tied LM head must consume the embedding owner exactly once")
            handle.forward_uses += 1
        else:
            handle.mark_forward()
        if self.bias_slot:
            self.bias_slot.handle.mark_forward()

    def forward(self, x):
        self.mark()
        if self.column:
            weights = [self.weight_slot.forward_weight()]
            if self.bias_slot:
                weights.append(self.bias_slot.forward_weight())
            return QwenProjection.apply(x, self.group, (self,), self.sp, *weights)[0]
        return QwenRowLinear.apply(
            x,
            self.weight_slot.forward_weight(),
            self.weight_slot,
            self.group,
            self.sp,
            self.replicated,
        )


class RuntimeQwenProjectionGroup:
    def __init__(self, projections, group, sp, role):
        self.projections, self.group, self.sp, self.role = tuple(projections), group, sp, role

    def __call__(self, x):
        weights = []
        for projection in self.projections:
            projection.mark()
            weights.append(projection.weight_slot.forward_weight())
            if projection.bias_slot:
                weights.append(projection.bias_slot.forward_weight())
        with self.group.tagged(self.role):
            return QwenProjection.apply(x, self.group, self.projections, self.sp, *weights)


class RuntimeQwenNorm(nn.Module):
    def __init__(self, slot, eps):
        super().__init__()
        self.slot, self.eps = slot, eps
        self.register_parameter("weight", slot.handle.local_shard)

    def forward(self, x):
        self.slot.handle.mark_forward()
        return QwenNorm.apply(x, self.slot.forward_weight(), self.slot, self.eps)


class RuntimeQwenEmbedding(nn.Module):
    def __init__(self, slot, padding_idx):
        super().__init__()
        self.slot, self.padding_idx = slot, padding_idx
        self.register_parameter("weight", slot.handle.local_shard)

    def forward(self, ids):
        self.slot.handle.mark_forward()
        return QwenEmbedding.apply(ids, self.slot.forward_weight(), self.slot, self.padding_idx)
