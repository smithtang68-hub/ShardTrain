"""Three-parameter P0 fixture uses the same production bridge and consumer Functions."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.runtime.comm import RuntimeTPGroup
from sharded_runtime.runtime.engine import UnitState
from sharded_runtime.runtime.layout import ParameterSpec, ShardedParameter, initialize_shard
from sharded_runtime.runtime.materializer import Materializer
from sharded_runtime.runtime.operators import RuntimeLinear, RuntimeNorm


class TinyUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Parameter(torch.ones(3))
        self.up = nn.Parameter(torch.randn(5, 3) * 0.1)
        self.down = nn.Parameter(torch.randn(3, 5) * 0.1)

    def forward(self, x):
        if hasattr(self, "runtime_norm"):
            return x + self.runtime_down(torch.tanh(self.runtime_up(self.runtime_norm(x))))
        z = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.norm
        return x + F.linear(torch.tanh(F.linear(z, self.up)), self.down)


def fixture_runtime(model, reshard):
    # Explicit P0 fixture adapter, not an advertised general Module wrapper.
    runtime = ShardedRuntime(reshard_after_forward=reshard)
    runtime.tp = RuntimeTPGroup(None, runtime.comm)
    runtime.materializer = Materializer(None, runtime.comm)
    for uid, block in enumerate(model):
        unit = UnitState(runtime, uid, block)
        runtime.units.append(unit)
        for attr, layout in (("norm", "replicated"), ("up", "column"), ("down", "row")):
            weight = getattr(block, attr)
            name = f"{uid}.{attr}"
            spec = ParameterSpec.create(name, uid, weight, tuple(weight.shape), layout, 1)
            handle = ShardedParameter(spec, initialize_shard(weight, spec, 0), unit)

            def committed(parameter, handle=handle):
                handle.grad_step = runtime.step_id

            handle.local_shard.register_post_accumulate_grad_hook(committed)
            consumer = (
                RuntimeNorm(handle, 1e-6)
                if attr == "norm"
                else RuntimeLinear(handle, runtime.tp, attr == "up", False)
            )
            delattr(block, attr)
            setattr(block, f"runtime_{attr}", consumer)
            unit.handles.append(handle)
            unit.consumers.append(consumer)
            runtime.registry[name] = handle
            runtime.identities[name] = (id(handle.local_shard), handle.local_shard.data_ptr())
        block.register_forward_pre_hook(unit.pre_forward)
        block.register_forward_hook(unit.post_forward)
    return runtime


@pytest.mark.parametrize("units", [1, 2])
@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize("reshard", [False, True])
def test_three_parameter_single_and_double_unit_three_sgd_steps(units, input_grad, reshard):
    dense = nn.Sequential(*(TinyUnit() for _ in range(units)))
    model = copy.deepcopy(dense)
    runtime = fixture_runtime(model, reshard)
    optimizer = runtime.bind_optimizer(torch.optim.SGD(runtime.sharded_parameters(), lr=0.01))
    dense_optimizer = torch.optim.SGD(dense.parameters(), lr=0.01)
    for step in range(3):
        runtime.begin_step(step)
        optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)
        x = torch.randn(2, 4, 3, requires_grad=input_grad)
        other = x.detach().clone().requires_grad_(input_grad)
        actual, expected = model(x), dense(other)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        expected.square().mean().backward()
        runtime.finalize_backward()
        if input_grad:
            torch.testing.assert_close(x.grad, other.grad)
        for name, grad in runtime.reconstruct_tp_local(gradients=True).items():
            torch.testing.assert_close(grad, dict(dense.named_parameters())[name].grad)
        optimizer.step()
        dense_optimizer.step()
        runtime.end_step()
        for name, value in runtime.reconstruct_tp_local().items():
            torch.testing.assert_close(value, dict(dense.named_parameters())[name])
