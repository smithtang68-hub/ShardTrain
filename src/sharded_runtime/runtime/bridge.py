"""Multi-output parameter materialization node and an early pre-backward boundary."""

import torch
from torch.autograd.function import once_differentiable


class UnitMaterialize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, unit, *shards):
        ctx.unit = unit
        ctx.set_materialize_grads(False)
        # Copy-out always uses Compute, even when this graph node is associated with
        # the RS stream so its backward can return safe leaf gradients independently.
        with unit.runtime.comm.stream_context("compute"):
            outputs = unit.runtime.materializer.finish(unit)
        if unit.runtime.comm.streams:
            unit.runtime.comm.wait_event(
                unit.lease.copy_out_ready_event,
                dependency_type="DATA_DEP",
                consumer="materialize_outputs",
            )
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grads):
        # Autograd schedules this one node only after all its output edges are ready.
        ctx.unit.annotate("BWD")
        if ctx.unit.runtime.comm.streams and ctx.unit.runtime.comm.diagnostics_enabled:
            ctx.unit.runtime.comm.event(
                "bridge_backward_stream",
                cuda_stream=torch.cuda.current_stream(ctx.unit.runtime.device).cuda_stream,
            )
        return (None, *ctx.unit.reduce(grads))


class PreBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, unit):
        ctx.unit = unit
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        ctx.unit.pre_backward()
        return gradient, None
