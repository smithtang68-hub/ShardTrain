import importlib.metadata
import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

save_file = pytest.importorskip("safetensors.torch").save_file

from sharded_runtime.models.qwen2_checkpoint import (  # noqa: E402
    CheckpointError,
    QwenCheckpoint,
    iter_checkpoint_tensors,
    load_qwen_checkpoint,
)


class TinyCanonicalModel(nn.Module):
    """Only checkpoint ownership is tested here; model math has separate HF tests."""

    def __init__(self, tied=True):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(7, 4)
        self.model.norm = nn.LayerNorm(4, bias=False)
        self.lm_head = nn.Linear(4, 7, bias=False)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight


def checkpoint_weights(model, *, omit_alias=True):
    weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    if omit_alias:
        weights.pop("lm_head.weight")
    return weights


def test_tied_single_file_load_keeps_actual_parameter_alias(tmp_path):
    torch.manual_seed(4)
    source = TinyCanonicalModel()
    file = tmp_path / "model.safetensors"
    weights = checkpoint_weights(source)
    save_file(weights, str(file))
    target = TinyCanonicalModel()
    parameter = target.model.embed_tokens.weight
    report = load_qwen_checkpoint(target, tmp_path)
    assert target.model.embed_tokens.weight is parameter
    assert target.lm_head.weight is parameter
    assert report.omitted_tied_aliases == ("lm_head.weight",)
    assert report.unique_tensor_count == 2
    assert report.checkpoint_tensor_count == 2
    assert report.copied_bytes == (7 * 4 + 4) * 4
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], tensor, rtol=0, atol=0)


def test_missing_lm_head_rejected_when_parameters_are_not_actually_tied(tmp_path):
    save_file(checkpoint_weights(TinyCanonicalModel()), str(tmp_path / "model.safetensors"))
    target = TinyCanonicalModel(tied=False)
    before = target.model.embed_tokens.weight.detach().clone()
    with pytest.raises(CheckpointError, match="Missing keys: lm_head.weight"):
        load_qwen_checkpoint(target, tmp_path)
    torch.testing.assert_close(target.model.embed_tokens.weight, before, rtol=0, atol=0)


def test_explicit_tied_keys_must_match_before_mutation(tmp_path):
    source = TinyCanonicalModel()
    weights = checkpoint_weights(source, omit_alias=False)
    weights["lm_head.weight"].add_(1)
    save_file(weights, str(tmp_path / "model.safetensors"))
    target = TinyCanonicalModel()
    before = target.model.embed_tokens.weight.detach().clone()
    with pytest.raises(CheckpointError, match="Conflicting checkpoint values for tied keys"):
        load_qwen_checkpoint(target, tmp_path)
    torch.testing.assert_close(target.model.embed_tokens.weight, before, rtol=0, atol=0)


def test_strict_diagnostics_collect_missing_unexpected_and_shape_keys(tmp_path):
    weights = checkpoint_weights(TinyCanonicalModel())
    weights.pop("model.norm.weight")
    weights["model.embed_tokens.weight"] = torch.zeros(8, 4)
    weights["typo.weight"] = torch.zeros(1)
    save_file(weights, str(tmp_path / "model.safetensors"))
    with pytest.raises(CheckpointError) as error:
        load_qwen_checkpoint(TinyCanonicalModel(), tmp_path)
    message = str(error.value)
    assert "Missing keys: model.norm.weight" in message
    assert "Unexpected keys: typo.weight" in message
    assert "model.embed_tokens.weight: checkpoint (8, 4), expected (7, 4)" in message


def test_index_shards_and_mapped_slice(tmp_path):
    weights = checkpoint_weights(TinyCanonicalModel())
    embed = "model.embed_tokens.weight"
    norm = "model.norm.weight"
    save_file({embed: weights[embed]}, str(tmp_path / "model-00001.safetensors"))
    save_file({norm: weights[norm]}, str(tmp_path / "model-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {embed: "model-00001.safetensors", norm: "model-00002.safetensors"}}
        ),
        encoding="utf-8",
    )
    reader = QwenCheckpoint(tmp_path)
    assert reader.shapes[embed] == (7, 4)
    assert reader.dtypes[embed] == "F32"
    torch.testing.assert_close(
        reader.get_tensor_slice(embed, (slice(2, 5), slice(None))),
        weights[embed][2:5],
        rtol=0,
        atol=0,
    )
    assert [name for name, _ in iter_checkpoint_tensors(tmp_path)] == sorted(weights)
    load_qwen_checkpoint(TinyCanonicalModel(), tmp_path)


def test_index_cannot_hide_unexpected_physical_keys(tmp_path):
    save_file({"a": torch.zeros(1), "hidden": torch.zeros(1)}, str(tmp_path / "part.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "part.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="Tensor hidden.*absent from the index"):
        QwenCheckpoint(tmp_path)


def test_index_missing_tensor_and_duplicate_keys_rejected(tmp_path):
    save_file({"a": torch.zeros(1)}, str(tmp_path / "part.safetensors"))
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps({"weight_map": {"a": "part.safetensors", "b": "part.safetensors"}}))
    with pytest.raises(CheckpointError, match="Index references absent tensors: b"):
        QwenCheckpoint(tmp_path)
    path.write_text('{"weight_map":{"a":"part.safetensors","a":"part.safetensors"}}')
    with pytest.raises(CheckpointError, match="Duplicate JSON key.*a"):
        QwenCheckpoint(tmp_path)


def test_index_rejects_directory_escape(tmp_path):
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}))
    with pytest.raises(CheckpointError, match="escapes index directory"):
        QwenCheckpoint(tmp_path)


def test_missing_file_or_pickle_file_has_clear_error(tmp_path):
    with pytest.raises(CheckpointError, match="does not exist"):
        QwenCheckpoint(tmp_path / "missing")
    path = tmp_path / "pytorch_model.bin"
    path.write_bytes(b"not a pickle")
    with pytest.raises(CheckpointError, match="Only local .safetensors"):
        QwenCheckpoint(path)


def test_cpu_dtype_conversion_is_reported(tmp_path):
    weights = {
        name: tensor.to(torch.bfloat16)
        for name, tensor in checkpoint_weights(TinyCanonicalModel()).items()
    }
    save_file(weights, str(tmp_path / "model.safetensors"))
    target = TinyCanonicalModel()
    report = load_qwen_checkpoint(target, tmp_path)
    assert set(report.converted_dtypes) == set(weights)
    for name, tensor in weights.items():
        torch.testing.assert_close(target.state_dict()[name], tensor.float(), rtol=0, atol=0)


def test_validated_missing_alias_accessor_maps_canonical_weight(tmp_path):
    weights = checkpoint_weights(TinyCanonicalModel())
    save_file(weights, str(tmp_path / "model.safetensors"))
    expected = {
        name: tuple(tensor.shape) for name, tensor in TinyCanonicalModel().state_dict().items()
    }
    reader = QwenCheckpoint(tmp_path, expected, {"lm_head.weight": "model.embed_tokens.weight"})
    torch.testing.assert_close(
        reader.get_tensor("lm_head.weight"), weights["model.embed_tokens.weight"]
    )
    with pytest.raises(CheckpointError, match="not found"):
        reader.get_tensor("missing")


def test_dense_loader_rejects_meta_model(tmp_path):
    target = TinyCanonicalModel().to("meta")
    with pytest.raises(CheckpointError, match="requires a CPU model"):
        load_qwen_checkpoint(target, tmp_path)


def test_metadata_validation_does_not_materialize_weights(tmp_path, monkeypatch):
    weights = checkpoint_weights(TinyCanonicalModel())
    save_file(weights, str(tmp_path / "model.safetensors"))
    reader = QwenCheckpoint(tmp_path)

    def unexpected_materialization(*args):
        raise AssertionError("Metadata validation materialized a weight")

    monkeypatch.setattr(reader, "get_tensor", unexpected_materialization)
    expected = {
        name: tuple(tensor.shape) for name, tensor in TinyCanonicalModel().state_dict().items()
    }
    assert reader.validate(expected, {"lm_head.weight": "model.embed_tokens.weight"}) == (
        "lm_head.weight",
    )


def test_full_tiny_qwen_checkpoint_roundtrip_logits_and_loss(tmp_path):
    from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig

    config = Qwen2ModelConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        use_cache=False,
    )
    torch.manual_seed(9)
    source = Qwen2CausalLM(config).eval()
    save_file(checkpoint_weights(source), str(tmp_path / "model.safetensors"))
    target = Qwen2CausalLM(config).eval()
    report = load_qwen_checkpoint(target, tmp_path)
    assert report.omitted_tied_aliases == ("lm_head.weight",)
    assert target.model.embed_tokens.weight is target.lm_head.weight
    tokens = torch.tensor([[1, 3, 2, 5]])
    actual = target(tokens, labels=tokens)
    expected = source(tokens, labels=tokens)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)


def test_hf_generated_sharded_checkpoint_load_and_logits(tmp_path):
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("HF checkpoint test requires pinned optional qwen dependencies")
    if version != "4.57.3":
        pytest.skip(f"HF checkpoint test pinned to 4.57.3, installed {version}")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig

    config = Qwen2ModelConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        use_cache=False,
    )
    hf_config = Qwen2Config(**config.to_dict())
    hf_config._attn_implementation = "sdpa"
    torch.manual_seed(12)
    reference = Qwen2ForCausalLM(hf_config).eval()
    reference.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="1KB")
    assert (tmp_path / "model.safetensors.index.json").is_file()
    target = Qwen2CausalLM(config).eval()
    report = load_qwen_checkpoint(target, tmp_path)
    assert report.omitted_tied_aliases == ("lm_head.weight",)
    assert target.model.embed_tokens.weight is target.lm_head.weight
    tokens = torch.tensor([[1, 3, 2, 5]])
    actual = target(tokens, labels=tokens)
    expected = reference(tokens, labels=tokens)
    torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.loss, expected.loss, atol=2e-6, rtol=2e-5)
    metrics_directory = os.environ.get("SHARDTRAIN_QWEN_METRICS_DIR")
    if metrics_directory:
        output = Path(metrics_directory)
        output.mkdir(parents=True, exist_ok=True)
        metrics = {
            "reference": "HF-generated tiny Qwen2 sharded safetensors, not official 1.5B weights",
            "transformers_version": version,
            "checkpoint_tensor_count": report.checkpoint_tensor_count,
            "unique_tensor_count": report.unique_tensor_count,
            "omitted_tied_aliases": report.omitted_tied_aliases,
            "logits_max_abs": float((actual.logits - expected.logits).detach().abs().max()),
            "loss_abs": float((actual.loss - expected.loss).detach().abs()),
            "atol": 2e-6,
            "rtol": 2e-5,
        }
        (output / "hf_checkpoint_cpu.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )


@pytest.mark.parametrize("tied", [False, True])
def test_checkpoint_meta_direct_runtime_shards_train_and_preserve_alias(tmp_path, tied):
    from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig
    from sharded_runtime.runtime import ShardedRuntime

    config = Qwen2ModelConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=tied,
        use_cache=False,
    )
    torch.manual_seed(37)
    reference = Qwen2CausalLM(config)
    save_file(checkpoint_weights(reference, omit_alias=tied), str(tmp_path / "model.safetensors"))
    model = Qwen2CausalLM(config, device="meta")
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    model._qwen_weight_source = QwenCheckpoint(tmp_path)
    runtime = ShardedRuntime(reshard_after_forward=True).register(model)
    assert not hasattr(model, "_qwen_weight_source")
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert (model.model.embed_tokens.weight is model.lm_head.weight) == tied
    canonical = dict(reference.named_parameters())
    assert set(runtime.registry) == set(canonical)
    identities = {
        name: (id(handle.local_shard), handle.local_shard.data_ptr())
        for name, handle in runtime.registry.items()
    }
    for name, tensor in runtime.reconstruct_tp_local().items():
        torch.testing.assert_close(tensor, canonical[name], atol=0, rtol=0)
    optimizer = runtime.bind_optimizer(torch.optim.SGD(runtime.sharded_parameters(), lr=0.03))
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    ids = torch.tensor([[1, 3, 4, 2], [5, 7, 6, 1]])
    labels = ids.clone()
    labels[0, 2] = -100

    def save_activation(tensor):
        pointer = tensor.untyped_storage().data_ptr()
        for unit in runtime.units:
            if unit.lease is not None:
                for lease in unit.lease.full_param_storages.values():
                    assert pointer != lease.tensor.untyped_storage().data_ptr(), (
                        "A runtime FullParam was retained in autograd saved tensors"
                    )
        return tensor

    for step in range(2):
        runtime.begin_step(step)
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        with torch.autograd.graph.saved_tensors_hooks(save_activation, lambda tensor: tensor):
            actual = model(ids, labels=labels)
            expected = reference(ids, labels=labels)
            torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(actual.loss, expected.loss, atol=2e-6, rtol=2e-5)
            actual.loss.backward()
            expected.loss.backward()
        runtime.finalize_backward()
        for name, handle in runtime.registry.items():
            torch.testing.assert_close(
                handle.local_shard.grad.reshape(handle.spec.tp_local_shape),
                canonical[name].grad,
                atol=3e-6,
                rtol=3e-5,
            )
        optimizer.step()
        reference_optimizer.step()
        runtime.end_step()
        assert not runtime.comm.live_buffers
        assert all(unit.lease is None for unit in runtime.units)
        for name, handle in runtime.registry.items():
            assert (id(handle.local_shard), handle.local_shard.data_ptr()) == identities[name]
        for name, tensor in runtime.reconstruct_tp_local().items():
            torch.testing.assert_close(tensor, canonical[name], atol=3e-6, rtol=3e-5)
