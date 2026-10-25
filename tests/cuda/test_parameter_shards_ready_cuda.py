"""Real CUDA writer readiness must survive begin_step on an independent stream.

F=1 intentionally excludes NCCL: delayed writes, materializer copy-in/copy-out and
the matrix consumer must be ordered by the published writer event itself.
"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sharded_runtime.config import ModelConfig
from sharded_runtime.models.reference import build_reference_model
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.runtime.engine import UnitState
from sharded_runtime.runtime.layout import ParameterSpec, ShardedParameter
from sharded_runtime.runtime.materializer import Materializer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
DELAY_CYCLES = 500_000_000


def writer_stream(runtime, role):
    return runtime.comm.streams["compute"] if role == "compute" else torch.cuda.Stream()


def finish_matrix(runtime, unit, param_id):
    unit.issue("FWD")
    weights = runtime.materializer.finish(unit)
    index = next(i for i, handle in enumerate(unit.handles) if handle.spec.param_id == param_id)
    return weights[index]


@pytest.mark.parametrize("writer_role", ["compute", "independent"])
def test_delayed_initialization_event_survives_begin_step_and_orders_matrix_copy(writer_role):
    runtime = ShardedRuntime(device="cuda", communication_policy="stream_ordered")
    runtime.materializer = Materializer(None, runtime.comm)
    assert runtime.materializer.degree == 1
    unit = UnitState(runtime, 0, nn.Identity())
    expected = torch.arange(64 * 128, dtype=torch.float32).remainder(37).reshape(64, 128) - 18
    inputs = torch.arange(3 * 128, dtype=torch.float32).remainder(11).reshape(3, 128) - 5
    shard = nn.Parameter(torch.full((expected.numel(),), -7.0, device="cuda"))
    value, activation = expected.cuda(), inputs.cuda()
    spec = ParameterSpec.create(
        "weight", 0, shard.view(expected.shape), expected.shape, "column", 1
    )
    handle = ShardedParameter(spec, shard, unit)
    unit.handles.append(handle)
    runtime.units.append(unit)
    runtime.registry["weight"] = handle
    runtime.identities["weight"] = (id(shard), shard.data_ptr())
    reader, writer = torch.cuda.Stream(), writer_stream(runtime, writer_role)
    try:
        # Warm every allocation/copy/GEMM/sleep kernel before imposing a delay;
        # CUDA lazy module loading may otherwise synchronize the whole device.
        runtime.publish_parameter_shards_ready("warmup_initialization")
        with torch.cuda.stream(reader):
            old_weight = finish_matrix(runtime, unit, "weight")
            F.linear(activation, old_weight)
            unit.release()
            unit.state = "SHARDED"
        torch.cuda._sleep(1)
        torch.cuda.synchronize()

        with torch.cuda.stream(writer), torch.no_grad():
            torch.cuda._sleep(DELAY_CYCLES)
            shard.copy_(value.flatten())
            runtime.publish_parameter_shards_ready("delayed_initialization")
            ready = runtime.parameter_shards_ready_event
        with torch.cuda.stream(reader):
            # No writer->reader wait here: the AG channel must consume readiness.
            runtime.begin_step(0)
            assert runtime.parameter_shards_ready_event is ready
            assert not ready.query(), "The delayed writer must still be pending"
            actual_weight = finish_matrix(runtime, unit, "weight")
            actual = F.linear(activation, actual_weight)
            unit.release()
            done = torch.cuda.Event()
            done.record(reader)
        # Synchronize only the final consumer AFTER all dependent work is queued.
        done.synchronize()
        torch.testing.assert_close(actual_weight.cpu(), expected, atol=0, rtol=0)
        torch.testing.assert_close(actual.cpu(), F.linear(inputs, expected), atol=0, rtol=0)
        assert runtime.parameter_shards_version == 0
        waits = [row for row in runtime.comm.events if row.get("consumer") == "parameter_shards"]
        assert waits and all(row["wait_stream"] == "fsdp_ag" for row in waits)
        assert not runtime.comm.live_buffers
    finally:
        # Cleanup also protects allocations if an identity/oracle assertion fails.
        torch.cuda.synchronize()


@pytest.mark.parametrize("writer_role", ["compute", "independent"])
def test_bound_optimizer_writer_event_and_version_survive_next_step(writer_role):
    torch.manual_seed(479)
    cfg = ModelConfig(layers=1, hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    model = build_reference_model(cfg).cuda()
    runtime = ShardedRuntime(
        device="cuda", async_grad_rs=True, communication_policy="stream_ordered"
    ).register(model)
    optimizer = runtime.bind_optimizer(
        torch.optim.SGD(runtime.sharded_parameters(), lr=0.125, foreach=False)
    )
    batch = torch.randn(2, 4, 8, device="cuda")
    with runtime.compute_context():
        runtime.begin_step(0)
        optimizer.zero_grad(set_to_none=True)
        model(batch).square().mean().backward()
        runtime.finalize_backward()
    unit = runtime.units[0]
    handle = next(handle for handle in unit.handles if handle.spec.tp_layout == "column")
    original = handle.local_shard.detach().cpu().view(handle.spec.tp_local_shape)
    gradient = handle.local_shard.grad.detach().cpu().view(handle.spec.tp_local_shape)
    expected = original - 0.125 * gradient
    assert not torch.equal(expected, original), "The optimizer must actually change this weight"
    expected_projection = F.linear(batch.cpu(), expected)
    reader, writer = torch.cuda.Stream(), writer_stream(runtime, writer_role)
    try:
        # Warm SGD kernels on separate persistent scratch parameters, without
        # incrementing this runtime's version or performing its actual update.
        scratch = [nn.Parameter(torch.zeros_like(p)) for p in runtime.sharded_parameters()]
        for parameter in scratch:
            parameter.grad = torch.ones_like(parameter)
        torch.optim.SGD(scratch, lr=0.125, foreach=False).step()
        F.linear(batch, original.cuda())
        torch.cuda._sleep(1)
        torch.cuda.synchronize()

        with torch.cuda.stream(writer):
            torch.cuda._sleep(DELAY_CYCLES)
            optimizer.step()
            ready = runtime.parameter_shards_ready_event
            assert runtime.parameter_shards_version == 1
            runtime.end_step()
        with torch.cuda.stream(reader):
            runtime.begin_step(1)
            assert runtime.parameter_shards_ready_event is ready
            assert not ready.query(), "The actual SGD update must still be pending"
            actual_weight = finish_matrix(runtime, unit, handle.spec.param_id)
            actual_projection = F.linear(batch, actual_weight)
            unit.release()
            done = torch.cuda.Event()
            done.record(reader)
        done.synchronize()
        torch.testing.assert_close(actual_weight.cpu(), expected, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(
            actual_projection.cpu(), expected_projection, atol=1e-6, rtol=1e-5
        )
        assert runtime.parameter_shards_version == 1
        updates = [
            row
            for row in runtime.comm.events
            if row["event_type"] == "parameter_shards_ready" and row["source"] == "optimizer_update"
        ]
        assert len(updates) == 1 and updates[0]["parameter_version"] == 1
        assert not any(
            row["event_type"] == "parameter_shards_ready" and row["step_id"] == 1
            for row in runtime.comm.events
        )
        assert not runtime.comm.live_buffers
    finally:
        torch.cuda.synchronize()
