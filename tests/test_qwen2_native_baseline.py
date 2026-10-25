"""CPU math/boundary tests; these do not claim native multi-GPU acceptance."""

import hashlib

import pytest
import torch
from torch.nn import functional as F

from sharded_runtime.config import Config, ConfigError, ModelConfig
from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig, causal_lm_loss_parts
from sharded_runtime.models.qwen2_factory import build_qwen_model
from sharded_runtime.training.qwen2_native_baseline import (
    NativeQwen2CausalLM,
    initialize_same_source,
    native_loss_parts,
    run,
)


def tiny_model(*, tied=True):
    config = Qwen2ModelConfig(
        vocab_size=37,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=tied,
        pad_token_id=0,
    )
    dense = Qwen2CausalLM(config, device="meta", dtype=torch.float32)
    initialize_same_source(dense, 123)
    return dense


@pytest.mark.parametrize("tied", [True, False])
def test_name_stable_initialization_and_tied_owner(tied):
    dense = tiny_model(tied=tied)
    assert (dense.lm_head.weight is dense.model.embed_tokens.weight) == tied
    assert torch.count_nonzero(dense.model.embed_tokens.weight[0]) == 0
    for name, parameter in dense.named_parameters():
        if name.endswith(".bias"):
            expected = torch.zeros_like(parameter)
        elif "layernorm.weight" in name or name == "model.norm.weight":
            expected = torch.ones_like(parameter)
        else:
            seed = 123 + int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
            generator = torch.Generator().manual_seed(seed % (2**63 - 1))
            expected = torch.randn(parameter.shape, generator=generator) * 0.02
            if name == "model.embed_tokens.weight":
                expected[0].zero_()
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert torch.isfinite(dense.model.rotary_emb.inv_freq).all()
    assert dense.model.rotary_emb.inv_freq[0] == 1


@pytest.mark.parametrize("tied", [True, False])
def test_unparallelized_native_math_and_global_shift(tied):
    dense = tiny_model(tied=tied)
    native = NativeQwen2CausalLM(dense)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [3, 9, 6, 2, 1, 5, 7, 8]])
    labels = ids.clone()
    labels[0, 3] = -100
    labels[1, 6:] = -100
    mask = torch.ones_like(ids)
    positions = torch.arange(11, 19).unsqueeze(0)
    actual = native(ids, mask, positions)
    expected = dense(ids, mask, positions)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    full_num, full_count = causal_lm_loss_parts(expected.logits, labels)
    native.sp_enabled, native.tp_degree = True, 2
    parts = []
    for rank in range(2):
        native.tp_rank = rank
        parts.append(native_loss_parts(native, actual.logits[:, rank * 4 : (rank + 1) * 4], labels))
    torch.testing.assert_close(sum(value[0] for value in parts), full_num)
    assert sum(value[1] for value in parts) == full_count
    # The first token of SP rank1 supervises the final rank0 logit.
    shifted = F.pad(labels[:, 1:], (0, 1), value=-100)
    native.tp_rank = 0
    assert native.sequence_for_rank(shifted)[0, -1] == labels[0, 4]


def test_native_model_rejects_partial_position_context():
    native = NativeQwen2CausalLM(tiny_model())
    with pytest.raises(ValueError, match="complete global sequence"):
        native(torch.ones(2, 8, dtype=torch.long), position_ids=torch.arange(4).unsqueeze(0))
    native.sp_enabled, native.tp_degree = True, 2
    with pytest.raises(ValueError, match="divide TP"):
        native.sequence_for_rank(torch.zeros(2, 7))


def test_cpu_is_not_reported_as_native_fsdp2_acceptance():
    config = Config(model=ModelConfig(model_type="qwen2"))
    with pytest.raises(ConfigError, match="CUDA"):
        run(config)


def test_factory_initialization_matches_canonical_keys():
    config = ModelConfig(
        model_type="qwen2",
        vocab_size=37,
        hidden_size=16,
        num_heads=4,
        num_key_value_heads=2,
        mlp_hidden_size=32,
        layers=2,
    )
    dense = build_qwen_model(config, seed=313)
    initialize_same_source(dense, 313)
    assert dense.lm_head.weight is dense.model.embed_tokens.weight
    assert sum(p.numel() for p in dense.parameters()) > 0
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in dense.parameters())
