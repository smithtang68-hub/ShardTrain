import weakref
from types import SimpleNamespace

import pytest
import torch

from sharded_runtime.parallel.shared_projection import SharedProjection, SharedProjectionGroup
from sharded_runtime.runtime.comm import CommunicationCoordinator, sequence_copy_out
from sharded_runtime.runtime.layout import ParameterSpec
from sharded_runtime.runtime.materializer import MaterializationLease, Materializer


class CountingGroup:
    def __init__(self):
        self.gathers, self.scatters, self.full_refs = 0, 0, []

    def gather(self, x, dim):
        self.gathers += 1
        full = torch.cat((x, x), dim=dim)
        self.full_refs.append(weakref.ref(full.untyped_storage()))
        return full

    def issue_gather(self, x, dim):
        return x, dim

    def finish_gather(self, pending):
        return self.gather(*pending)

    def reduce_scatter_sequence(self, x):
        self.scatters += 1
        a, b = x.chunk(2, dim=1)
        return a + b


def test_batch_first_position_coded_rank_major_copy_out():
    full = torch.arange(2 * 8 * 16).view(2, 8, 16)
    packed = torch.cat([full[:, :4].contiguous(), full[:, 4:].contiguous()], dim=0)
    actual = sequence_copy_out(packed, (2, 4, 16), 2)
    torch.testing.assert_close(actual, full, rtol=0, atol=0)
    assert actual.is_contiguous() and actual.untyped_storage().data_ptr() != packed.data_ptr()
    assert not torch.equal(packed.view(2, 8, 16), full)


@pytest.mark.parametrize("branches", [2, 3])
@pytest.mark.parametrize("input_grad", [False, True])
def test_shared_forward_backward_regather_sum_adjoint_and_saved_local(branches, input_grad):
    torch.manual_seed(37)
    # A small view of a large allocation must be made independently compact.
    parent = torch.randn(2, 8, 16, dtype=torch.float64)
    x = parent[:, :4].detach().requires_grad_(input_grad)
    reference_x = x.detach().clone().requires_grad_(input_grad)
    weights = [torch.randn(7, 16, dtype=torch.float64, requires_grad=True) for _ in range(branches)]
    reference_weights = [weight.detach().clone().requires_grad_() for weight in weights]
    projections = [SimpleNamespace(weight=weight) for weight in weights]
    group = CountingGroup()
    outputs = SharedProjectionGroup(projections, group, "qkv")(x)
    node = outputs[0].grad_fn
    assert len(node.saved_tensors) == 1
    saved = node.saved_tensors[0]
    assert saved.shape == x.shape and saved.untyped_storage().nbytes() == x.numel() * 8
    assert saved.untyped_storage().data_ptr() != parent.untyped_storage().data_ptr()
    assert all(ref() is None for ref in group.full_refs)
    assert group.gathers == 1 and group.scatters == 0
    full = torch.cat((reference_x, reference_x), dim=1)
    expected = [torch.nn.functional.linear(full, weight) for weight in reference_weights]
    for actual, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, reference)
    sum((i + 1) * value.square().sum() for i, value in enumerate(outputs)).backward()
    sum((i + 1) * value.square().sum() for i, value in enumerate(expected)).backward()
    for actual, reference in zip(weights, reference_weights, strict=True):
        torch.testing.assert_close(actual.grad, reference.grad)
    if input_grad:
        torch.testing.assert_close(x.grad, reference_x.grad)
    assert group.gathers == 2 and group.scatters == 1
    assert all(ref() is None for ref in group.full_refs)


def test_unused_frozen_and_repeated_backward_are_explicit_errors():
    x = torch.randn(2, 4, 3, requires_grad=True)
    weights = [torch.randn(2, 3, requires_grad=True) for _ in range(2)]
    providers = tuple(SimpleNamespace(weight=weight) for weight in weights)
    with pytest.raises(RuntimeError, match="trainable"):
        SharedProjection.apply(
            x, CountingGroup(), providers, "gate_up", weights[0], weights[1].detach()
        )
    outputs = SharedProjection.apply(x, CountingGroup(), providers, "gate_up", *weights)
    with pytest.raises(RuntimeError, match="Unused"):
        outputs[0].sum().backward()
    outputs = SharedProjection.apply(x, CountingGroup(), providers, "gate_up", *weights)
    loss = sum(value.sum() for value in outputs)
    loss.backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="single backward"):
        loss.backward()


def test_group_reacquires_backward_weights_and_does_not_hold_forward_storage():
    group = CountingGroup()
    group.comm = CommunicationCoordinator("cpu")
    unit = SimpleNamespace(
        annotate=lambda phase: None, prefetch_after_input=lambda handle, phase: None
    )

    class Materialize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, leaf):
            return leaf.clone()

        @staticmethod
        def backward(ctx, grad):
            return grad

    leaves = [torch.randn(2, 3, requires_grad=True) for _ in range(2)]
    old = [Materialize.apply(leaf) for leaf in leaves]
    new = [weight.detach().clone() for weight in old]
    providers = tuple(
        SimpleNamespace(unit=unit, weight=lambda weight=weight: weight) for weight in new
    )
    x = torch.randn(2, 4, 3, requires_grad=True)
    outputs = SharedProjection.apply(x, group, providers, "gate_up", *old)
    refs = [weakref.ref(weight.untyped_storage()) for weight in old]
    del old
    assert all(ref() is None for ref in refs)
    sum(value.sum() for value in outputs).backward()
    torch.testing.assert_close(x.grad, 2 * sum(weight.sum(0) for weight in new).expand_as(x))
    assert not group.comm.live_buffers


def test_delayed_next_ag_blocks_only_copy_out_and_protects_storage():
    comm = CommunicationCoordinator("cpu", communication_policy="stream_ordered")
    spec = ParameterSpec.create("1.weight", 1, torch.ones(2, 3), (2, 3), "column", 1)
    storage = torch.full((6,), -999.0)
    lease = comm.buffer(storage, "fsdp_ag")

    class Work:
        waited = False

        def wait(self):
            storage.copy_(torch.arange(6.0))
            self.waited = True

    work = Work()
    comm.unit_id, comm.generation = 1, 7
    ticket = comm.issue(
        "fsdp",
        "fsdp_ag",
        None,
        storage,
        storage,
        lambda _: work,
        channel="fsdp_ag",
        asynchronous=True,
    )
    unit = SimpleNamespace(
        unit_id=1,
        handles=[SimpleNamespace(spec=spec)],
        lease=MaterializationLease(1, 0, "FWD", 7, lease, ag_ticket=ticket),
    )
    # Unrelated current-unit simulation progresses without consuming the next ticket.
    independent = torch.ones(3).square() + 2
    assert torch.equal(independent, torch.full((3,), 3.0)) and not work.waited
    with pytest.raises(RuntimeError, match="Premature"):
        comm.retire(lease)
    weights = Materializer(None, comm).finish(unit)
    assert work.waited and ticket.status == "CONSUMABLE"
    torch.testing.assert_close(weights[0], torch.arange(6.0).view(2, 3))
    edges = [row for row in comm.events if row["event_type"] == "dependency_edge"]
    data = [row for row in edges if row["dependency_type"] == "DATA_DEP"]
    assert len(data) == 1 and data[0]["consumer"] == "copy_out/U1"
    assert data[0]["producer_ticket"] == ticket.op_id
    assert any(row["dependency_type"] == "BUFFER_LIFETIME" for row in edges)
    for full in unit.lease.full_param_storages.values():
        comm.retire(full)
    assert not comm.live_buffers
