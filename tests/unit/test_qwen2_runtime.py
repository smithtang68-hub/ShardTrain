"""Full Qwen gradients/updates through the shard runtime, including tied ownership."""

import copy

import pytest
import torch

from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig
from sharded_runtime.runtime import ShardedRuntime


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("reshard", [False, True])
@pytest.mark.parametrize("policy", ["conservative", "stream_ordered"])
def test_full_qwen_runtime_gradients_updates(tied, reshard, policy):
    torch.manual_seed(45)
    cfg = Qwen2ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=tied,
    )
    model = Qwen2CausalLM(cfg)
    reference = copy.deepcopy(model)
    runtime = ShardedRuntime(
        reshard_after_forward=reshard,
        communication_policy=policy,
        forward_prefetch=True,
        backward_prefetch=True,
        async_grad_rs=True,
    )
    runtime.register(model)
    optimizer = runtime.bind_optimizer(torch.optim.SGD(runtime.sharded_parameters(), lr=0.01))
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 0, 0, 0]])
    labels = ids.clone()
    labels[1, 3:] = -100
    mask = ids.ne(0)
    positions = torch.arange(17, 23).unsqueeze(0)
    identities = {name: id(handle.local_shard) for name, handle in runtime.registry.items()}
    assert (model.model.embed_tokens.weight is model.lm_head.weight) == tied
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        runtime.begin_step(step)
        actual = model(ids, labels=labels, attention_mask=mask, position_ids=positions)
        expected = reference(ids, labels=labels, attention_mask=mask, position_ids=positions)
        torch.testing.assert_close(actual.logits, expected.logits, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual.loss, expected.loss, atol=1e-6, rtol=1e-5)
        actual.loss.backward()
        expected.loss.backward()
        runtime.finalize_backward()
        for name, value in runtime.reconstruct_tp_local(gradients=True).items():
            torch.testing.assert_close(
                value, dict(reference.named_parameters())[name].grad, atol=1e-6, rtol=1e-5
            )
        optimizer.step()
        reference_optimizer.step()
        runtime.end_step()
        for name, value in runtime.reconstruct_tp_local().items():
            torch.testing.assert_close(
                value, dict(reference.named_parameters())[name], atol=1e-6, rtol=1e-5
            )
        assert not runtime.comm.live_buffers
        assert all(unit.lease is None for unit in runtime.units)
        assert identities == {
            name: id(handle.local_shard) for name, handle in runtime.registry.items()
        }


def test_runtime_rejects_local_sequence_and_config_alias_mismatch():
    model = Qwen2CausalLM(Qwen2ModelConfig())
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(ValueError, match="aliasing"):
        ShardedRuntime().register(model)


@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
@pytest.mark.parametrize("accumulation", [1, 2])
def test_accumulation_global_token_mean_and_optimizer(optimizer, accumulation):
    from sharded_runtime.training.qwen2_trainer import QwenTrainer

    torch.manual_seed(47)
    model = Qwen2CausalLM(
        Qwen2ModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    reference = copy.deepcopy(model)
    runtime = ShardedRuntime()
    runtime.register(model)
    trainer = QwenTrainer(
        model, runtime, lr=0.001, optimizer=optimizer, accumulation_steps=accumulation
    )
    opt_cls = torch.optim.SGD if optimizer == "sgd" else torch.optim.AdamW
    opt = opt_cls(reference.parameters(), lr=0.001)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 0, 0, 0]])
    labels = ids.clone()
    labels[1, 3:] = -100
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        expected = reference(ids, labels=labels, attention_mask=ids.ne(0))
        expected.loss.backward()
        opt.step()
        actual = trainer.train_step(ids, labels, attention_mask=ids.ne(0))
        assert actual.valid_tokens == 7
        assert not actual.overflow
        assert actual.loss == pytest.approx(expected.loss.item(), abs=1e-6)
        for name, value in runtime.reconstruct_tp_local().items():
            torch.testing.assert_close(
                value, dict(reference.named_parameters())[name], atol=1e-6, rtol=1e-5
            )
        assert not runtime.comm.live_buffers
