"""Fixed-source backward prefetch: real graph order and delayed collective consumers."""

import copy
from types import SimpleNamespace

import pytest
import torch

from sharded_runtime.config import ModelConfig
from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig
from sharded_runtime.models.reference import build_reference_model
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.runtime.engine import UnitState
from sharded_runtime.runtime.layout import ParameterSpec, initialize_shard
from sharded_runtime.runtime.materializer import MaterializationLease, Materializer


@pytest.mark.parametrize("kind", ["blocks", "qwen_tied", "qwen_untied"])
@pytest.mark.parametrize("reshard", [False, True])
def test_fixed_dispatch_real_autograd_sequence_and_dense_oracle(kind, reshard):
    torch.manual_seed(71)
    if kind == "blocks":
        model = build_reference_model(
            ModelConfig(layers=4, hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
        )
        batch = torch.randn(2, 4, 8)

        def loss(module):
            return module(batch).square().mean()
    else:
        model = Qwen2CausalLM(
            Qwen2ModelConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                tie_word_embeddings=kind == "qwen_tied",
            )
        )
        ids = torch.tensor([[1, 2, 3, 4]])

        def loss(module):
            return module(ids, labels=ids).loss
    reference = copy.deepcopy(model)
    runtime = ShardedRuntime(
        reshard_after_forward=reshard,
        forward_prefetch=True,
        backward_prefetch=True,
        async_grad_rs=True,
        communication_policy="stream_ordered",
    ).register(model)
    # Isolate scheduler/Autograd contracts at T=1; real SP mathematics is covered
    # by the independent multi-rank oracle, rather than fabricated peer activations.
    runtime.sp = True
    optimizer = runtime.bind_optimizer(torch.optim.SGD(runtime.sharded_parameters(), lr=0.01))
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        runtime.begin_step(step)
        loss(model).backward()
        loss(reference).backward()
        runtime.finalize_backward()
        for name, actual in runtime.reconstruct_tp_local(gradients=True).items():
            torch.testing.assert_close(actual, dict(reference.named_parameters())[name].grad)
        tokens = runtime.backward_prefetch_tokens
        assert set(tokens) == {(step, i) for i in range(1, len(runtime.units))}
        assert len({token["target_unit"] for token in tokens.values()}) == len(tokens)
        events = [e for e in runtime.comm.events if e["step_id"] == step]
        boundaries = [
            e for e in events if e["event_type"] == "prefetch_boundary" and e["phase"] == "BWD"
        ]
        assert [e["current_unit"] for e in boundaries] == list(
            range(len(runtime.units) - 1, 0, -1)
        )
        for boundary in boundaries[1:]:
            owner = boundary["gradient_owner"]
            assert boundary["scheduling_boundary"] == "previous_gradient_dispatch"
            assert owner == boundary["current_unit"] + 1 == boundary["target_unit"] + 2
            issued = [
                i for i, e in enumerate(events)
                if e["event_type"] == "gradient_bucket_issued" and e["unit_id"] == owner
            ]
            consumed = [
                i for i, e in enumerate(events)
                if e["event_type"] == "gradient_bucket_consumed" and e["unit_id"] == owner
            ]
            assert max(issued) < events.index(boundary) < min(consumed)
        if kind == "qwen_tied":
            assert tokens[(step, 1)]["status"] == "tied_retained"
            assert runtime.units[0].generation == step + 1
        optimizer.step()
        reference_optimizer.step()
        runtime.end_step()
        assert not runtime.comm.live_buffers


def _delayed_dispatch_runtime(monkeypatch):
    runtime = ShardedRuntime(
        backward_prefetch=True, async_grad_rs=True, communication_policy="stream_ordered"
    )
    runtime.sp = True
    runtime.materializer = Materializer(None, runtime.comm)
    runtime.materializer.degree = 2
    runtime.materializer.backend = "nccl"
    runtime.fsdp_grad_group = None
    actions, works = [], []

    class Work:
        def __init__(self, kind, action):
            self.kind, self.action, self.waited = kind, action, False
            works.append(self)

        def wait(self):
            if not self.waited:
                actions.append(("consume", self.kind))
                self.action()
                self.waited = True

    def gather(storage, slot, *, group, async_op):
        assert async_op
        target = runtime.comm.unit_id
        actions.append(("ag", target))
        return Work(("ag", target), lambda: storage.copy_(torch.cat((slot, slot))))

    def reduce_scatter(output, packed, *, group, async_op):
        assert async_op
        name = "norm" if output.numel() == 2 else "main"
        actions.append(("rs", name))
        # Independent F peer has the same gradient: sum=2*g and average=g.
        return Work(("rs", name), lambda: output.copy_(packed[: output.numel()] * 2))

    def tp_sum(gradient, **kwargs):
        assert kwargs["group_role"] == "norm_gradient_bucket"
        assert not next(work for work in works if work.kind == ("rs", "main")).waited
        assert actions.index(("ag", 1)) > actions.index(("rs", "main"))
        assert not next(work for work in works if work.kind == ("ag", 1)).waited
        actions.append(("tp_sum", "norm"))
        return gradient.clone() * 2

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", reduce_scatter)
    runtime.tp = SimpleNamespace(degree=2, sum=tp_sum)
    gradients = [torch.arange(1.0, 5.0), torch.arange(1.0, 17.0).reshape(4, 4)]
    for uid in range(4):
        unit = UnitState(runtime, uid, None)
        for name, gradient in zip(("norm", "matrix"), gradients, strict=True):
            spec = ParameterSpec.create(
                f"{uid}.{name}", uid, gradient, gradient.shape,
                "replicated" if name == "norm" else "column", 2,
            )
            unit.handles.append(SimpleNamespace(
                spec=spec, local_shard=initialize_shard(gradient, spec, 0), forward_uses=0,
            ))
        runtime.units.append(unit)
    runtime.begin_step(0)
    for unit in runtime.units:
        unit.forward_calls, unit.generation, unit.state = 1, 1, "RESHARDED"
        for handle in unit.handles:
            handle.forward_uses = 1
    owner = runtime.units[-1]
    owner.state = "FWD_RETAINED"
    owner.lease = MaterializationLease(owner.unit_id, 0, "FWD", 1)
    owner.lease.full_param_storages = {
        handle.spec.param_id: runtime.comm.buffer(gradient.clone(), "full_parameter")
        for handle, gradient in zip(owner.handles, gradients, strict=True)
    }
    owner.pre_backward()
    return runtime, gradients, actions, works


def test_actual_delayed_ag_is_issued_before_norm_sum_and_main_rs_consumption(monkeypatch):
    runtime, gradients, actions, works = _delayed_dispatch_runtime(monkeypatch)
    actual = runtime.units[3].reduce(gradients)
    assert actions.index(("rs", "norm")) < actions.index(("rs", "main"))
    assert actions.index(("rs", "main")) < actions.index(("ag", 1))
    assert actions.index(("ag", 1)) < actions.index(("tp_sum", "norm"))
    assert actions.index(("tp_sum", "norm")) < actions.index(("consume", ("rs", "main")))
    torch.testing.assert_close(actual[0], gradients[0][:2] * 2)
    torch.testing.assert_close(actual[1], gradients[1].flatten()[:8])
    assert not next(work for work in works if work.kind == ("ag", 1)).waited
    runtime.units[2].pre_backward()
    assert next(work for work in works if work.kind == ("ag", 2)).waited
    assert not next(work for work in works if work.kind == ("ag", 1)).waited


@pytest.mark.parametrize("invalid", ["completed", "entered", "stale_lease", "wrong_owner"])
def test_invalid_dispatch_fails_before_any_extra_ag_or_generation(monkeypatch, invalid):
    runtime, _, actions, _ = _delayed_dispatch_runtime(monkeypatch)
    owner, target = runtime.units[3], runtime.units[1]
    owner.state, owner.reductions = "GRAD_REDUCING", 1
    if invalid == "completed":
        target.state, target.backward_calls, target.reductions = "GRAD_READY", 1, 1
    elif invalid == "entered":
        target.state, target.backward_calls = "BWD_ACTIVE", 1
    elif invalid == "stale_lease":
        target.state = "FWD_RETAINED"
        target.lease = MaterializationLease(1, -1, "FWD", target.generation)
    else:
        runtime.backward_dispatch_expected = 2
    before = list(actions), target.generation
    with pytest.raises(RuntimeError):
        runtime.dispatch_backward_prefetch(owner)
    assert (actions, target.generation) == before
    assert (0, 2) not in runtime.backward_prefetch_tokens


@pytest.mark.parametrize("invalid", ["duplicate_token", "missing_token", "stale_generation"])
def test_entry_has_no_dynamic_fallback_and_token_cannot_be_reused(monkeypatch, invalid):
    runtime, _, actions, _ = _delayed_dispatch_runtime(monkeypatch)
    owner, current = runtime.units[3], runtime.units[2]
    owner.state, owner.reductions = "GRAD_REDUCING", 1
    runtime.dispatch_backward_prefetch(owner)
    before = list(actions)
    if invalid == "duplicate_token":

        def action():
            return runtime._queue_backward_prefetch(current, owner=owner)
    else:
        if invalid == "missing_token":
            del runtime.backward_prefetch_tokens[(0, 2)]
        else:
            runtime.units[1].generation += 1
        action = current.pre_backward
    with pytest.raises(RuntimeError, match="[Tt]oken"):
        action()
    assert actions == before
    assert current.backward_calls == 0


@pytest.mark.parametrize("policy,sp,async_rs,prefetch", [
    ("conservative", True, True, True),
    ("stream_ordered", False, True, True),
    ("stream_ordered", True, False, True),
    ("stream_ordered", True, True, False),
])
def test_other_modes_keep_existing_prefetch_source(policy, sp, async_rs, prefetch):
    runtime = ShardedRuntime(
        communication_policy=policy, async_grad_rs=async_rs, backward_prefetch=prefetch
    )
    runtime.sp = sp
    assert not runtime.gradient_dispatch_prefetch


def test_single_unit_dispatch_has_no_negative_target_and_closes_both_frontiers():
    model = build_reference_model(
        ModelConfig(layers=1, hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    )
    runtime = ShardedRuntime(
        backward_prefetch=True, async_grad_rs=True, communication_policy="stream_ordered"
    ).register(model)
    runtime.sp = True
    runtime.begin_step(0)
    model(torch.randn(2, 4, 8)).square().mean().backward()
    runtime.finalize_backward()
    assert not runtime.backward_prefetch_tokens
    assert runtime.backward_entry_expected == runtime.backward_dispatch_expected == -1
    assert not any(e["event_type"] == "prefetch_boundary" for e in runtime.comm.events)
    assert not runtime.comm.live_buffers
