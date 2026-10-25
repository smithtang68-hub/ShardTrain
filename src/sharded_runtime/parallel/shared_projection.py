"""One activation gather and one summed adjoint for a fixed column-projection group."""

from contextlib import nullcontext

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


def tagged(group, role):
    return group.tagged(role) if hasattr(group, "tagged") else nullcontext()


def compact(tensor):
    result = tensor.contiguous()
    if result.untyped_storage().nbytes() != result.numel() * result.element_size():
        result = result.clone()
    return result


def retire(group, tensor):
    if hasattr(group, "retire_activation"):
        group.retire_activation(tensor)


class SharedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, providers, role, *weights):
        if not all(weight.requires_grad for weight in weights):
            raise RuntimeError("Shared projections require every weight to be trainable")
        ctx.set_materialize_grads(False)
        ctx.group, ctx.providers, ctx.role = group, providers, role
        ctx.backward_started = False
        local = compact(x)
        ctx.save_for_backward(local)
        # Providers are metadata handles
        runtime = hasattr(providers[0], "unit")
        if runtime:
            providers[0].unit.annotate("FWD")
            ctx.local_lease = group.comm.buffer(local, "projection_local_saved")
        with tagged(group, role):
            if runtime:
                pending = group.issue_gather(local, 1)
                providers[-1].unit.prefetch_after_input(providers[-1], "FWD")
                source = group.finish_gather(pending)
            else:
                source = group.gather(local, 1)
            if runtime:
                group.comm.event(
                    "projection_group_saved_local",
                    group_role=role,
                    shape=list(local.shape),
                    storage_bytes=local.untyped_storage().nbytes(),
                    storage_ptr=local.untyped_storage().data_ptr(),
                    full_storage_ptr=source.untyped_storage().data_ptr(),
                )
            outputs = tuple(F.linear(source, weight) for weight in weights)
            retire(group, source)
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        if ctx.backward_started:
            raise RuntimeError("Shared projections support a single backward only")
        ctx.backward_started = True
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("Unused shared-projection branches are unsupported")
        (local,) = ctx.saved_tensors
        runtime = hasattr(ctx.providers[0], "unit")
        if runtime:
            ctx.providers[0].unit.annotate("BWD")
        with tagged(ctx.group, ctx.role + "_wgrad"):
            source = ctx.group.gather(local, 1)
            matrix = source.reshape(-1, source.shape[-1])
            dweights, dx = [], None
            for gradient, provider in zip(gradients, ctx.providers, strict=True):
                grad = compact(gradient)
                # Reacquire the current backward generation
                weight = provider.weight() if runtime else provider.weight
                dweights.append(grad.reshape(-1, grad.shape[-1]).t() @ matrix)
                contribution = grad @ weight
                dx = contribution if dx is None else dx + contribution
            retire(ctx.group, source)
            if runtime:
                ctx.group.comm.retire(ctx.local_lease)
        with tagged(ctx.group, ctx.role + "_dx"):
            dx = ctx.group.reduce_scatter_sequence(dx)
        return (dx if ctx.needs_input_grad[0] else None, None, None, None, *dweights)


class SharedProjectionGroup:
    """Non-module adapter: original projection modules retain their parameter registration."""

    def __init__(self, projections, group, role):
        self.projections, self.group, self.role = tuple(projections), group, role

    def __call__(self, x):
        providers, weights = [], []
        for projection in self.projections:
            if hasattr(projection, "handle"):
                projection.handle.mark_forward()
                weight = projection.compute_weight
                if weight is None:
                    raise RuntimeError("Missing shared-projection forward materialization")
                providers.append(projection.handle)
            else:
                weight = projection.weight
                providers.append(projection)
            weights.append(weight)
        return SharedProjection.apply(x, self.group, tuple(providers), self.role, *weights)
