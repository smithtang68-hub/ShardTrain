"""Real CUDA graph/leaf safety for fixed-source dispatch, independently of TP math."""

import copy

import pytest
import torch

from sharded_runtime.config import ModelConfig
from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig
from sharded_runtime.models.reference import build_reference_model
from sharded_runtime.runtime import ShardedRuntime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("tied", [False, True])
def test_cuda_dispatch_preserves_normal_autograd_leaf_grads_and_own_ag_consumers(tied):
    torch.manual_seed(72)
    if tied:
        model = Qwen2CausalLM(
            Qwen2ModelConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                tie_word_embeddings=True,
            )
        ).cuda()
        ids = torch.tensor([[1, 2, 3, 4]], device="cuda")

        def loss(module):
            return module(ids, labels=ids).loss

    else:
        model = build_reference_model(
            ModelConfig(layers=4, hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
        ).cuda()
        batch = torch.randn(2, 4, 8, device="cuda")

        def loss(module):
            return module(batch).square().mean()

    reference = copy.deepcopy(model)
    runtime = ShardedRuntime(
        device="cuda",
        forward_prefetch=True,
        backward_prefetch=True,
        async_grad_rs=True,
        communication_policy="stream_ordered",
    ).register(model)
    # Activate only the scheduler at T=1; the server oracle verifies real multi-rank SP.
    runtime.sp = True
    with runtime.compute_context():
        runtime.begin_step(0)
        loss(model).backward()
        runtime.finalize_backward()
        loss(reference).backward()
        for name, actual in runtime.reconstruct_tp_local(gradients=True).items():
            torch.testing.assert_close(
                actual, dict(reference.named_parameters())[name].grad, atol=2e-6, rtol=1e-4
            )
    torch.cuda.synchronize()
    events = runtime.comm.events
    boundaries = [
        e
        for e in events
        if e["event_type"] == "prefetch_boundary"
        and e.get("scheduling_boundary") == "previous_gradient_dispatch"
    ]
    assert len(boundaries) == len(runtime.units) - 2
    for boundary in boundaries:
        owner = boundary["gradient_owner"]
        consumed = next(
            e
            for e in events
            if e["event_type"] == "gradient_bucket_consumed" and e["unit_id"] == owner
        )
        assert events.index(boundary) < events.index(consumed)
    copy_out = [e for e in events if e["event_type"] == "copy_out_start"]
    for unit in runtime.units:
        assert unit.state == "GRAD_READY" and unit.lease is None
        for handle in unit.handles:
            assert handle.grad_step == 0 and handle.grad_ready_event is not None
    for consumer in copy_out:
        allocation = next(
            e
            for e in events
            if e["event_type"] == "buffer_allocate" and e["buffer_id"] == consumer["buffer_id"]
        )
        assert allocation["unit_id"] == consumer["unit_id"]
        assert allocation["materialization_generation"] == consumer["materialization_generation"]
        assert any(
            e["event_type"] == "dependency_edge"
            and e.get("consumer") == f"copy_out/U{consumer['unit_id']}"
            and e["buffer_id"] == consumer["buffer_id"]
            and e["dependency_type"] == "DATA_DEP"
            for e in events[: events.index(consumer)]
        )
    assert not runtime.comm.live_buffers
