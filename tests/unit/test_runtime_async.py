import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sharded_runtime.config import ModelConfig
from sharded_runtime.models.reference import build_reference_model
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.runtime.comm import CommunicationCoordinator


@pytest.mark.parametrize("policy", ["conservative", "stream_ordered"])
def test_runner_smoke_and_runtime_configs_are_executable(tmp_path, policy):
    from sharded_runtime.config import load_config

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_matrix.py",
            "--backend",
            "cpu",
            "--max-ranks",
            "1",
            "--stages",
            "E0,E1",
            "--communication-policy",
            policy,
            "--dry-run",
            "--output",
            str(tmp_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plans = list(tmp_path.glob("run_*/plan.json"))
    assert len(plans) == 1
    plan = json.loads(plans[0].read_text())
    assert [item["stage"] for item in plan] == ["E0", "E1"]
    for item in plan:
        config = load_config(Path(item["directory"]) / "config.yaml")
        config.validate(1, mode="smoke" if item["stage"] == "E0" else "runtime")
        assert config.runtime.communication_policy == (
            "conservative" if item["stage"] == "E0" else policy
        )


class DelayedWork:
    def __init__(self, action, *, failure=False):
        self.action, self.failure, self.waited = action, failure, False

    def wait(self):
        if self.failure:
            raise RuntimeError("Injected delayed collective failure")
        if not self.waited:
            self.action()
        self.waited = True


def test_fake_delayed_ag_rs_no_premature_buffer_reuse_or_consumer():
    comm = CommunicationCoordinator("cpu")
    source = torch.arange(8.0)
    output = torch.full_like(source, -999)
    lease = comm.buffer(source, "rs_input")
    output_lease = comm.buffer(output, "rs_output")
    work = DelayedWork(lambda: output.copy_(source * 2))
    ticket = comm.issue(
        "fsdp",
        "fsdp_rs",
        None,
        source,
        output,
        lambda _: work,
        channel="fsdp_rs",
        asynchronous=True,
    )
    assert ticket.status == "ISSUED" and not work.waited
    assert torch.all(output == -999)
    with pytest.raises(RuntimeError, match="Premature"):
        comm.retire(lease)
    comm.consume(ticket)
    torch.testing.assert_close(output, source * 2)
    comm.retire(lease)
    comm.retire(output_lease)
    with pytest.raises(RuntimeError, match="Duplicate"):
        comm.retire(lease)
    assert not comm.live_buffers


def test_fake_out_of_order_failure_and_static_pg_divergence():
    comm = CommunicationCoordinator("cpu")
    x, a, b = torch.ones(2), torch.zeros(2), torch.zeros(2)
    works = [DelayedWork(lambda: a.fill_(1)), DelayedWork(lambda: b.fill_(2))]
    first = comm.issue(
        "fsdp", "fsdp_ag", None, x, a, lambda _: works[0], channel="fsdp_ag", asynchronous=True
    )
    second = comm.issue(
        "tp", "sp_ag", None, x, b, lambda _: works[1], channel="sp_ag", asynchronous=True
    )
    comm.consume(second)
    assert works[1].waited and not works[0].waited
    comm.finalize()
    assert first.status == second.status == "CONSUMABLE"
    broken = comm.issue(
        "fsdp",
        "fsdp_rs",
        None,
        x,
        b,
        lambda _: DelayedWork(lambda: None, failure=True),
        channel="fsdp_rs",
        asynchronous=True,
    )
    with pytest.raises(RuntimeError, match="Injected"):
        comm.consume(broken)
    assert broken.status == "FAILED"
    with pytest.raises(RuntimeError, match="Failed communication"):
        comm.consume(broken)
    comm.expected_sequence = [("fsdp", "fsdp_ag")] * 4
    with pytest.raises(RuntimeError, match="divergence"):
        comm.issue("tp", "sp_rs", None, x, b, lambda _: None, channel="sp_rs", asynchronous=False)


@pytest.mark.parametrize("flags", list(itertools.product((False, True), repeat=3)))
@pytest.mark.parametrize("policy", ["conservative", "stream_ordered"])
def test_local_prefetch_schedule_idempotence_u3_retention_and_sync_equivalence(flags, policy):
    config = ModelConfig(layers=4, hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    reference = build_reference_model(config)
    model = copy.deepcopy(reference)
    runtime = ShardedRuntime(
        forward_prefetch=flags[0],
        backward_prefetch=flags[1],
        async_grad_rs=flags[2],
        communication_policy=policy,
    ).register(model)
    runtime.begin_step(0)
    batch = torch.randn(2, 4, 8)
    actual, expected = model(batch), reference(batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    runtime.finalize_backward()
    for name, value in runtime.reconstruct_tp_local(gradients=True).items():
        torch.testing.assert_close(value, dict(reference.named_parameters())[name].grad)
    issues = [r for r in runtime.comm.events if r["event_type"] == "materialize_issue"]
    assert len(issues) == 7
    assert len({(r["unit_id"], r["phase"], r["materialization_generation"]) for r in issues}) == 7
    assert not any(r["unit_id"] == 3 and r["phase"] == "BWD" for r in issues)
    runtime.end_step()


def test_optimizer_guard_stale_generation_missing_grad_and_failed_step():
    config = ModelConfig(hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    model = build_reference_model(config)
    runtime = ShardedRuntime().register(model)
    optimizer = runtime.bind_optimizer(torch.optim.SGD(runtime.sharded_parameters(), lr=0.01))
    with pytest.raises(RuntimeError, match="Premature"):
        optimizer.step()
    runtime.begin_step(0)
    output = model(torch.randn(2, 4, 8))
    unit = runtime.units[0]
    unit.pre_backward()
    unit.lease.generation -= 1
    with pytest.raises(RuntimeError, match="Stale generation"):
        unit.handles[0].weight()
    unit.lease.generation += 1
    with pytest.raises(RuntimeError, match="Missing full parameter gradient"):
        unit.reduce([None] * 9)
    runtime.fail("injected missing grad")
    assert all(unit.state == "FAILED" for unit in runtime.units)
    with pytest.raises(RuntimeError, match="FAILED"):
        runtime.begin_step(1)
    del output


def test_v07_config_rejects_accumulation_and_unsupported_policies():
    from sharded_runtime.config import ConfigError, config_from_dict

    for data in (
        {"training": {"accumulation_steps": 2}},
        {"runtime": {"prefetch": "dynamic"}},
        {"runtime": {"materialization": "asynchronous"}},
        {"parallel": {"fsdp_degree": 3}},
        {"runtime": {"communication_policy": "unordered"}},
    ):
        config = config_from_dict(data)
        with pytest.raises(ConfigError):
            config.validate(config.parallel.world_size, mode="runtime")
    config = config_from_dict(
        {
            "parallel": {"fsdp_degree": 4, "tp_degree": 2, "sp_enabled": True},
            "runtime": {
                "param_grouping": "block",
                "reshard_after_forward": True,
                "forward_prefetch": True,
                "async_grad_rs": True,
                "communication_policy": "stream_ordered",
            },
        }
    )
    config.validate(8, mode="runtime")


def test_parameter_consumer_reuse_is_rejected():
    config = ModelConfig(hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    model = build_reference_model(config)
    runtime = ShardedRuntime().register(model)
    runtime.begin_step(0)
    unit = runtime.units[0]
    unit.pre_forward(unit.block, (torch.randn(2, 4, 8),))
    norm = unit.block.attention_norm
    norm(torch.randn(2, 4, 8))
    with pytest.raises(RuntimeError, match="Repeated/early parameter consumer"):
        norm(torch.randn(2, 4, 8))
