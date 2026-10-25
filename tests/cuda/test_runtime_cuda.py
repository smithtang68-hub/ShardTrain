import copy
import itertools
import json
import weakref

import pytest
import torch

from sharded_runtime.config import ModelConfig
from sharded_runtime.models.reference import build_reference_model
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.runtime.comm import CommunicationCoordinator
from sharded_runtime.runtime.layout import ParameterSpec, copy_out
from sharded_runtime.runtime.materializer import MaterializationLease, Materializer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def test_conservative_keeps_pg_order_for_an_already_consumed_ticket(monkeypatch):
    comm = CommunicationCoordinator("cuda", communication_policy="conservative")
    source = torch.ones(4, device="cuda")
    output = torch.empty_like(source)
    waits = []
    original = torch.cuda.Stream.wait_event

    def observed(stream, event):
        waits.append((stream.cuda_stream, id(event)))
        return original(stream, event)

    monkeypatch.setattr(torch.cuda.Stream, "wait_event", observed)

    class Work:
        def wait(self):
            pass

    def operation(async_):
        output.copy_(source)
        return Work()

    first = comm.issue(
        "fsdp", "fsdp_ag", None, source, output, operation, channel="fsdp_ag", asynchronous=False
    )
    assert first.status == "CONSUMABLE"
    waits.clear()
    with comm.stream_context("sp_rs"):
        second = comm.issue(
            "tp", "sp_rs", None, source, output, operation, channel="sp_rs", asynchronous=False
        )
    assert (comm.streams["sp_rs"].cuda_stream, id(first.ready_event)) in waits
    edges = [
        row
        for row in comm.events
        if row["producer_ticket"] == first.op_id and row["consumer_ticket"] == second.op_id
    ]
    assert edges and all(row["dependency_type"] == "PG_ORDER" for row in edges)
    torch.cuda.synchronize()


def test_delayed_next_ag_does_not_block_current_sp_rs_or_its_compute_consumer(tmp_path):
    from types import SimpleNamespace

    comm = CommunicationCoordinator("cuda", communication_policy="stream_ordered")
    storage = torch.empty(6, device="cuda")
    source = torch.arange(6.0, device="cuda")
    # CUDA lazy module loading may synchronize the device on a kernel's first use.
    # Warm all test kernels before measuring independence from the delayed producer.
    source.square().add_(1)
    (source + 3).square().add_(1)
    storage.copy_(source)
    torch.cuda._sleep(1)
    torch.cuda.synchronize()
    lease = comm.buffer(storage, "fsdp_ag")
    spec = ParameterSpec.create("1.weight", 1, torch.ones(2, 3), (2, 3), "column", 1)
    comm.streams["compute"].wait_stream(torch.cuda.current_stream())

    class Work:
        def wait(self):
            pass

    def delayed_ag(async_):
        assert async_
        torch.cuda._sleep(500_000_000)  # Test-only bounded GPU delay, no NCCL claim.
        storage.copy_(source)
        return Work()

    with comm.stream_context("compute"):
        comm.unit_id = 1
        ticket = comm.issue(
            "fsdp",
            "fsdp_ag",
            None,
            source,
            storage,
            delayed_ag,
            channel="fsdp_ag",
            asynchronous=True,
        )
        unit = SimpleNamespace(
            unit_id=1,
            handles=[SimpleNamespace(spec=spec)],
            lease=MaterializationLease(1, 0, "FWD", 1, lease, ag_ticket=ticket),
        )
        comm.unit_id = 0
        current = source.square() + 1
        done = torch.cuda.Event()
        done.record(comm.streams["compute"])
        done.synchronize()
        assert not ticket.ready_event.query(), "Artificial AG must still be pending"
        assert not any(row["consumer"] == "copy_out/U1" for row in comm.events if "consumer" in row)
        with pytest.raises(RuntimeError, match="Premature"):
            comm.retire(lease)
        # Current attention output is independent of next unit's parameter storage.
        rs_output = torch.empty_like(source)

        def rs(async_):
            rs_output.copy_(source + 3)
            return Work()

        second = comm.issue(
            "tp", "sp_rs", None, source, rs_output, rs, channel="sp_rs", asynchronous=False
        )
        after_rs = rs_output.square().add_(1)
        rs_consumer_done = torch.cuda.Event()
        rs_consumer_done.record(comm.streams["compute"])
        rs_consumer_done.synchronize()
        assert not ticket.ready_event.query(), "SP RS and its consumer must not wait for next AG"
        assert second.status == "CONSUMABLE"
        comm.unit_id = 1
        weights = Materializer(None, comm).finish(unit)
        comm.finalize()
        for full in unit.lease.full_param_storages.values():
            comm.retire(full)
        comm.streams["compute"].synchronize()
        torch.testing.assert_close(weights[0], source.view(2, 3))
        torch.testing.assert_close(current, source.square() + 1)
        torch.testing.assert_close(rs_output, source + 3)
        torch.testing.assert_close(after_rs, (source + 3).square() + 1)
    edges = [
        row
        for row in comm.events
        if row["producer_ticket"] == ticket.op_id and row["consumer_ticket"] == second.op_id
    ]
    assert not edges
    copy_out_edges = [
        row
        for row in comm.events
        if row["event_type"] == "dependency_edge" and row.get("consumer") == "copy_out/U1"
    ]
    assert copy_out_edges and copy_out_edges[0]["producer_ticket"] == ticket.op_id
    assert copy_out_edges[0]["dependency_type"] == "DATA_DEP"
    timings = comm.gpu_wait_timings()
    assert not timings
    assert not comm.live_buffers and not comm.pending_last_reads
    (tmp_path / "delayed_ag_trace.json").write_text(
        json.dumps(
            dict(
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                actual_NCCL=False,
                current_compute_completed_with_ag_pending=True,
                current_sp_rs_and_consumer_completed_with_ag_pending=True,
                pg_order_gpu_waits=timings,
                events=comm.events,
            ),
            indent=2,
        )
    )


@pytest.mark.parametrize("consume_first", [False, True])
def test_host_launch_order_does_not_wait_for_unrelated_collective_completion(
    monkeypatch, consume_first
):
    # Real CUDA copies/streams/events, fake Work; NCCL semantics remain an M4 gate.
    comm = CommunicationCoordinator("cuda", communication_policy="stream_ordered")
    waits = []
    original_wait = torch.cuda.Stream.wait_event

    def observed_wait(stream, event):
        waits.append((stream.cuda_stream, id(event)))
        return original_wait(stream, event)

    monkeypatch.setattr(torch.cuda.Stream, "wait_event", observed_wait)

    class Work:
        def __init__(self):
            self.waits = 0

        def wait(self):
            self.waits += 1

    source = torch.arange(128, device="cuda", dtype=torch.float32)
    first_output, second_output = torch.empty_like(source), torch.empty_like(source)
    lease = comm.buffer(first_output, "test_ag")
    works = [Work(), Work()]

    def first_operation(async_):
        assert async_
        first_output.copy_(source * 2)
        return works[0]

    def second_operation(async_):
        assert async_
        second_output.copy_(source + 1)
        return works[1]

    comm.streams["compute"].wait_stream(torch.cuda.current_stream())
    with comm.stream_context("compute"):
        first = comm.issue(
            "fsdp",
            "fsdp_ag",
            None,
            source,
            first_output,
            first_operation,
            channel="fsdp_ag",
            asynchronous=True,
        )
        if consume_first:
            comm.consume(first)
        waits.clear()
        second = comm.issue(
            "tp",
            "sp_ag",
            None,
            source,
            second_output,
            second_operation,
            channel="sp_ag",
            asynchronous=True,
        )
        assert (comm.streams["sp_ag"].cuda_stream, id(first.ready_event)) not in waits
        assert (comm.streams["compute"].cuda_stream, id(first.ready_event)) not in waits
        assert first.status == ("CONSUMABLE" if consume_first else "ISSUED")
        assert works[0].waits == works[1].waits == 1
        if not consume_first:
            with pytest.raises(RuntimeError, match="Premature"):
                comm.retire(lease)
        comm.consume(second)
        assert (comm.streams["compute"].cuda_stream, id(second.ready_event)) in waits
        comm.finalize()
        comm.retire(lease)
        comm.streams["compute"].synchronize()
        torch.testing.assert_close(first_output, source * 2)
        torch.testing.assert_close(second_output, source + 1)
    dependencies = [
        row for row in comm.events if row["event_type"] == "collective_order_dependency"
    ]
    assert not dependencies
    launches = [row for row in comm.events if row["event_type"] == "collective_launch_order"]
    assert len(launches) == 1 and launches[0]["previous_op_id"] == first.op_id
    assert launches[0]["ordering_scope"] == "host_launch"
    assert not launches[0]["gpu_completion_dependency"]
    assert not comm.live_buffers


def test_five_stream_copy_out_last_read_retirement_and_independent_storage():
    comm = CommunicationCoordinator("cuda")
    assert len(comm.streams) == 5
    assert len({stream.cuda_stream for stream in comm.streams.values()}) == 5
    weight = torch.arange(1024 * 1024, dtype=torch.float32).reshape(1024, 1024)
    spec = ParameterSpec.create("weight", 0, weight, tuple(weight.shape), "column", 1)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    with comm.stream_context("fsdp_ag"):
        packed = weight.reshape(-1).cuda()
        ag = comm.buffer(packed, "fsdp_ag")
        producer = torch.cuda.Event()
        producer.record(comm.streams["fsdp_ag"])
    with comm.stream_context("compute"):
        comm.streams["compute"].wait_event(producer)
        full = copy_out(packed, [spec], 1)["weight"]
        assert full.data_ptr() != packed.data_ptr()
        peak_conversion = torch.cuda.memory_allocated() - before
        assert peak_conversion == 2 * weight.numel() * weight.element_size()
        pointer = packed.data_ptr()
        storage_ref = weakref.ref(packed.untyped_storage())
        comm.retire(ag)
        del packed
        assert storage_ref() is None
        result = full @ torch.ones(1024, device="cuda")
        ready = torch.cuda.Event()
        ready.record(comm.streams["compute"])
    # Remaining channels consume after the actual compute producer event.
    for channel in ("sp_ag", "fsdp_rs", "sp_rs"):
        with comm.stream_context(channel):
            comm.streams[channel].wait_event(ready)
            consumer = result.sum()
            result.record_stream(comm.streams[channel])
            done = torch.cuda.Event()
            done.record(comm.streams[channel])
        torch.cuda.current_stream().wait_event(done)
        assert torch.isfinite(consumer).item()
    torch.cuda.synchronize()
    torch.testing.assert_close(full.cpu(), weight)
    assert full.data_ptr() != pointer and not comm.live_buffers


@pytest.mark.parametrize("flags", list(itertools.product((False, True), repeat=3)))
@pytest.mark.parametrize("reshard", [False, True])
@pytest.mark.parametrize("policy", ["conservative", "stream_ordered"])
def test_single_gpu_four_block_autograd_events_and_three_updates(flags, reshard, policy):
    config = ModelConfig(layers=4, hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    reference = build_reference_model(config).cuda()
    model = copy.deepcopy(reference)
    runtime = ShardedRuntime(
        device="cuda",
        reshard_after_forward=reshard,
        forward_prefetch=flags[0],
        backward_prefetch=flags[1],
        async_grad_rs=flags[2],
        communication_policy=policy,
    ).register(model)
    optimizer = runtime.bind_optimizer(torch.optim.SGD(runtime.sharded_parameters(), lr=0.01))
    dense_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    for step in range(3):
        x = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
        dense_x = x.detach().clone().requires_grad_()
        target = torch.randn_like(x)
        with runtime.compute_context():
            runtime.begin_step(step)
            optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            actual, expected = model(x), reference(dense_x)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            runtime.local_loss(actual, target).backward()
            (expected - target).square().mean().backward()
            runtime.finalize_backward()
            torch.testing.assert_close(x.grad, dense_x.grad, atol=1e-5, rtol=1e-4)
            for name, param in runtime.registry.items():
                torch.testing.assert_close(
                    param.local_shard.grad.view(param.spec.tp_local_shape),
                    dict(reference.named_parameters())[name].grad,
                    atol=1e-5,
                    rtol=1e-4,
                )
            optimizer.step()
            dense_optimizer.step()
            runtime.end_step()
            for name, value in runtime.reconstruct_tp_local().items():
                torch.testing.assert_close(
                    value, dict(reference.named_parameters())[name], atol=1e-5, rtol=1e-4
                )
        torch.cuda.synchronize()
        assert not runtime.comm.live_buffers
        stream_events = [
            row for row in runtime.comm.events if row["event_type"] == "bridge_backward_stream"
        ]
        expected_stream = runtime.comm.streams["fsdp_rs" if flags[2] else "compute"].cuda_stream
        assert stream_events and all(row["cuda_stream"] == expected_stream for row in stream_events)
