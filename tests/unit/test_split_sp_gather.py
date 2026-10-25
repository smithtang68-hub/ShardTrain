"""Deferred SP consumption: independent parameter issue, batch order and lease ownership."""

from types import SimpleNamespace

import pytest
import torch

from sharded_runtime.parallel.shared_projection import SharedProjection
from sharded_runtime.runtime.comm import CommunicationCoordinator, RuntimeTPGroup
from sharded_runtime.runtime.qwen2_operators import QwenProjection


class DeferredWork:
    def __init__(self, complete):
        self.complete, self.waited = complete, False

    def wait(self):
        self.complete()
        self.waited = True


def delayed_group(monkeypatch, remote):
    comm = CommunicationCoordinator("cpu", communication_policy="stream_ordered")
    group = RuntimeTPGroup(None, comm)
    group.degree = 2
    works = []

    def all_gather(packed, local, *, group, async_op):
        assert async_op
        work = DeferredWork(lambda: packed.copy_(torch.cat((local, remote), dim=0)))
        works.append(work)
        return work

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", all_gather)
    return group, works


def test_split_gather_waits_only_its_output_and_retains_pending_buffers(monkeypatch):
    local = torch.arange(24.0).reshape(2, 3, 4)
    remote = local + 100
    group, works = delayed_group(monkeypatch, remote)
    pending = group.issue_gather(local, 1)
    sp_ticket = pending.ticket
    assert sp_ticket.status == "ISSUED" and not works[0].waited
    assert pending.input_lease.tensor is local
    assert {lease.owner for lease in group.comm.live_buffers.values()} == {
        "sp_ag_direct_input",
        "sp_ag_output",
    }
    with pytest.raises(RuntimeError, match="Premature"):
        group.comm.retire(pending.input_lease)

    # An independent prefetch can be submitted before SP's output is consumed.
    parameter = torch.empty(4)
    parameter_work = DeferredWork(lambda: parameter.fill_(11))
    parameter_ticket = group.comm.issue(
        "fsdp",
        "fsdp_ag",
        None,
        parameter,
        parameter,
        lambda _: parameter_work,
        channel="fsdp_ag",
        asynchronous=True,
    )
    full = group.finish_gather(pending)
    torch.testing.assert_close(full, torch.cat((local, remote), dim=1), rtol=0, atol=0)
    assert full.is_contiguous()
    assert works[0].waited and sp_ticket.status == "CONSUMABLE"
    assert not parameter_work.waited and parameter_ticket.status == "ISSUED"
    assert pending.input_lease is None and pending.packed_lease is None
    assert {lease.owner for lease in group.comm.live_buffers.values()} == {"sp_full_activation"}
    with pytest.raises(RuntimeError, match="already been finished"):
        group.finish_gather(pending)
    group.retire_activation(full)
    group.comm.consume(parameter_ticket)
    assert not group.comm.live_buffers


def test_split_gather_owner_guard_and_single_rank_wrapper():
    group = RuntimeTPGroup(None, CommunicationCoordinator("cpu"))
    other = RuntimeTPGroup(None, group.comm)
    local = torch.randn(2, 3, 4)
    pending = group.issue_gather(local, 1)
    with pytest.raises(RuntimeError, match="issuing group"):
        other.finish_gather(pending)
    assert group.finish_gather(pending) is local
    assert group.gather(local, 1) is local
    assert not group.comm.tickets and not group.comm.live_buffers


@pytest.mark.parametrize("operator", ["shared", "qwen"])
@pytest.mark.parametrize("branches", [2, 3])
def test_projection_group_prefetches_between_one_sp_issue_and_consumption(
    monkeypatch, operator, branches
):
    torch.manual_seed(109)
    local = torch.randn(2, 3, 4)
    remote = torch.randn_like(local)
    group, works = delayed_group(monkeypatch, remote)
    prefetch_work = DeferredWork(lambda: None)

    def prefetch(handle, phase):
        assert phase == "FWD" and len(works) == 1 and not works[0].waited
        assert group.comm.tickets[0].status == "ISSUED"
        shard = torch.ones(3)
        group.comm.issue(
            "fsdp",
            "fsdp_ag",
            None,
            shard,
            shard,
            lambda _: prefetch_work,
            channel="fsdp_ag",
            asynchronous=True,
        )

    unit = SimpleNamespace(annotate=lambda phase: None, prefetch_after_input=prefetch)
    weights = tuple(torch.randn(5 + i, 4, requires_grad=True) for i in range(branches))
    if operator == "shared":
        providers = tuple(SimpleNamespace(unit=unit) for _ in weights)
        outputs = SharedProjection.apply(local, group, providers, "qkv", *weights)
    else:
        providers = tuple(
            SimpleNamespace(
                weight_slot=SimpleNamespace(handle=SimpleNamespace(unit=unit)), bias_slot=None
            )
            for _ in weights
        )
        outputs = QwenProjection.apply(local, group, providers, True, *weights)
    full = torch.cat((local, remote), dim=1)
    for output, weight in zip(outputs, weights, strict=True):
        torch.testing.assert_close(output, torch.nn.functional.linear(full, weight))
    assert len(works) == 1 and works[0].waited and not prefetch_work.waited
    assert [ticket.kind for ticket in group.comm.tickets] == ["sp_ag", "fsdp_ag"]
    assert group.comm.tickets[1].status == "ISSUED"
    # Saved compact input outlives AG's direct-input lease until the autograd consumer.
    assert {lease.owner for lease in group.comm.live_buffers.values()} == {"projection_local_saved"}
    group.comm.retire(outputs[0].grad_fn.local_lease)
    group.comm.consume(group.comm.tickets[1])
    assert not group.comm.live_buffers
