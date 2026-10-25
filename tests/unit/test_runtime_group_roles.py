import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from sharded_runtime.runtime.comm import CommunicationCoordinator, RuntimeTPGroup

AUDIT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_qwen_gpu_acceptance.py"
audit_spec = importlib.util.spec_from_file_location("qwen_gpu_acceptance_group_roles", AUDIT_PATH)
audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit)


def configured_groups(monkeypatch, *, gradient_members=(0, 1)):
    activation, gradient = object(), object()
    members = {activation: [0, 1], gradient: list(gradient_members)}
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group: len(members[group]))
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: members[group])
    return activation, gradient


def test_replicated_gradients_use_distinct_group_and_can_run_on_compute(monkeypatch, tmp_path):
    activation, gradient = configured_groups(monkeypatch)
    called = []

    def all_reduce(tensor, *, group, async_op):
        called.append((group, async_op))
        tensor.mul_(2)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    comm = CommunicationCoordinator("cpu")
    tp = RuntimeTPGroup(activation, comm, grad_group=gradient)
    source = torch.arange(4.0)
    norm = tp.sum(
        source, kind="tp_norm_allreduce", group_role="norm_gradient_bucket", channel="compute"
    )
    activation_sum = tp.sum(source, kind="sp_rs_math_allreduce")
    second_norm = tp.sum(
        source, kind="replicated_embedding_sum", group_role="replicated_gradient"
    )
    for result in (norm, activation_sum, second_norm):
        torch.testing.assert_close(result, 2 * source)
        assert result.data_ptr() != source.data_ptr()
    torch.testing.assert_close(source, torch.arange(4.0))
    assert called == [(gradient, False), (activation, False), (gradient, False)]
    assert [(ticket.pg_id, ticket.pg_local_seq) for ticket in comm.tickets] == [
        ("tp_grad", 0), ("tp", 0), ("tp_grad", 1)
    ]
    assert [ticket.issue_stream for ticket in comm.tickets] == ["compute", "sp_rs", "sp_rs"]
    assert all(ticket.status == "CONSUMABLE" for ticket in comm.tickets)
    assert comm.tickets[0].group_role == "norm_gradient_bucket"
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in comm.events), encoding="utf-8")
    _, orders, summary = audit.audit_events(path, 0, 2, 2, "stream_ordered", False)
    assert len(orders["tp_grad"]) == 2 and len(orders["tp"]) == 1
    assert summary["collectives"] == 3


def test_gradient_group_membership_and_communicator_alias_are_rejected(monkeypatch):
    activation, gradient = configured_groups(monkeypatch, gradient_members=(1, 0))
    comm = CommunicationCoordinator("cpu")
    with pytest.raises(ValueError, match="identical rank order"):
        RuntimeTPGroup(activation, comm, grad_group=gradient)
    with pytest.raises(ValueError, match="distinct communicator"):
        RuntimeTPGroup(activation, comm, grad_group=activation)


@pytest.mark.parametrize("role,peer", [("tp_grad", 1), ("fsdp_grad", 2)])
def test_acceptance_audit_rejects_gradient_group_sequence_divergence(role, peer):
    row = (0, "gradient", 8, "torch.float32", 1, "BWD", 0, (8,), (4,))
    orders = [
        {name: [row] for name in ("world", "tp", "fsdp", "tp_grad", "fsdp_grad")}
        for _ in range(4)
    ]
    audit.audit_pg_sequences(orders, 2, 2)
    changed = deepcopy(orders)
    changed[peer][role] = []
    with pytest.raises(ValueError, match=f"divergence: {role}"):
        audit.audit_pg_sequences(changed, 2, 2)
