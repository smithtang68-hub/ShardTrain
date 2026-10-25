"""Tiny numerical checks against the pinned, independent HF Qwen2 reference."""

import importlib.metadata
import json
import os
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

from sharded_runtime.models.qwen2 import (
    Qwen2CausalLM,
    Qwen2ModelConfig,
    Qwen2RMSNorm,
    apply_rotary_pos_emb,
    causal_lm_loss_parts,
)

ATOL = 2e-6
RTOL = 2e-5


@pytest.fixture
def hf_classes():
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Install the qwen optional dependencies for HF numerical validation")
    if version != "4.57.3":
        pytest.skip(f"HF reference is pinned to transformers==4.57.3, installed {version}")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    return Qwen2Config, Qwen2ForCausalLM


def tiny_config(**overrides):
    values = dict(
        vocab_size=47,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        pad_token_id=0,
        use_cache=False,
    )
    return Qwen2ModelConfig(**(values | overrides))


def models(hf_classes, **overrides):
    config = tiny_config(**overrides)
    hf_config_class, hf_model_class = hf_classes
    hf_config = hf_config_class(**config.to_dict())
    hf_config._attn_implementation = "sdpa"
    torch.manual_seed(197)
    reference = hf_model_class(hf_config)
    actual = Qwen2CausalLM(config)
    actual.load_state_dict(reference.state_dict(), strict=True)
    return actual, reference


def max_error(actual, expected):
    return float((actual.detach().float() - expected.detach().float()).abs().max())


def assert_tensor(actual, expected):
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    return max_error(actual, expected)


def test_config_and_projection_shapes():
    config = tiny_config()
    model = Qwen2CausalLM(config)
    attn = model.model.layers[0].self_attn
    assert attn.q_proj.weight.shape == (32, 32)
    assert attn.k_proj.weight.shape == attn.v_proj.weight.shape == (16, 32)
    assert attn.q_proj.bias.shape == (32,)
    assert attn.k_proj.bias.shape == attn.v_proj.bias.shape == (16,)
    assert attn.o_proj.bias is None
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert Qwen2ModelConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="Sliding-window"):
        tiny_config(use_sliding_window=True)
    with pytest.raises(ValueError, match="divisible"):
        tiny_config(num_key_value_heads=3)
    with pytest.raises(ValueError, match="RoPE scaling"):
        tiny_config(rope_scaling={"rope_type": "yarn", "factor": 4.0})


def test_official_1p5b_structure_without_allocating_weights():
    path = Path(__file__).resolve().parents[2] / "configs" / "qwen25_1p5b_config.yaml"
    config = Qwen2ModelConfig.from_file(path)
    model = Qwen2CausalLM(config, device="meta")
    assert len(model.model.layers) == 28
    assert config.num_attention_heads == 12 and config.num_key_value_heads == 2
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_543_714_304
    assert len(list(model.parameters())) == 338
    assert len(model.state_dict()) == 339
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert all(parameter.device.type == "meta" for parameter in model.parameters())


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_config_file_formats(tmp_path, suffix):
    expected = tiny_config()
    path = tmp_path / ("config" + suffix)
    values = expected.to_dict()
    text = json.dumps(values) if suffix == ".json" else yaml.safe_dump(values)
    path.write_text(text, encoding="utf-8")
    assert Qwen2ModelConfig.from_file(path) == expected
    if suffix == ".json":
        assert Qwen2ModelConfig.from_json(path) == expected


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
@pytest.mark.parametrize("values", [None, [1, 2], "not a mapping"])
def test_config_file_rejects_non_mapping(tmp_path, suffix, values):
    path = tmp_path / ("config" + suffix)
    text = json.dumps(values) if suffix == ".json" else yaml.safe_dump(values)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        Qwen2ModelConfig.from_file(path)


def test_config_file_rejects_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="must use"):
        Qwen2ModelConfig.from_file(tmp_path / "config.txt")


def test_rmsnorm_fp32_accumulation(hf_classes):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as HFNorm

    torch.manual_seed(11)
    norm, reference = Qwen2RMSNorm(32).half(), HFNorm(32).half()
    weights = torch.randn(32).half()
    norm.weight.data.copy_(weights)
    reference.weight.data.copy_(weights)
    x = (torch.randn(2, 5, 32) * 250).half().requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    output, ref_output = norm(x), reference(ref_x)
    torch.testing.assert_close(output, ref_output, atol=0, rtol=0)
    output.float().square().sum().backward()
    ref_output.float().square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, atol=0, rtol=0)
    assert torch.isfinite(output).all()  # FP16 square accumulation would overflow.


def test_rope_global_positions_and_swiglu(hf_classes):
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb as hf_apply_rope,
    )

    actual, reference = models(hf_classes)
    x = torch.randn(2, 7, 32)
    positions = torch.stack((torch.arange(7) + 101, torch.arange(7) + 211))
    cos, sin = actual.model.rotary_emb(x, positions)
    ref_cos, ref_sin = reference.model.rotary_emb(x, positions)
    assert_tensor(cos, ref_cos)
    assert_tensor(sin, ref_sin)
    q, k = torch.randn(2, 4, 7, 8), torch.randn(2, 2, 7, 8)
    q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)
    ref_q, ref_k = hf_apply_rope(q, k, ref_cos, ref_sin)
    assert_tensor(q_rotated, ref_q)
    assert_tensor(k_rotated, ref_k)
    assert_tensor(actual.model.layers[0].mlp(x), reference.model.layers[0].mlp(x))


def test_next_token_loss_hand_calculation_and_empty_valid_tokens():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]], requires_grad=True)
    labels = torch.tensor([[0, 1, -100]])
    numerator, count = causal_lm_loss_parts(logits, labels)
    assert count.item() == 1
    assert_tensor(numerator, F.cross_entropy(logits[:, :1].reshape(1, 2), torch.tensor([1])))
    numerator.backward()
    assert torch.count_nonzero(logits.grad[:, 1:]) == 0
    empty_logits = torch.randn(2, 3, 11, requires_grad=True)
    numerator, count = causal_lm_loss_parts(empty_logits, torch.full((2, 3), -100))
    assert count == 0 and numerator == 0
    (numerator / count.clamp_min(1)).backward()
    assert torch.count_nonzero(empty_logits.grad) == 0
    single_logits = torch.randn(1, 1, 11, requires_grad=True)
    numerator, count = causal_lm_loss_parts(single_logits, torch.zeros(1, 1, dtype=torch.long))
    numerator.backward()
    assert numerator == 0 and count == 0 and torch.count_nonzero(single_logits.grad) == 0


@pytest.mark.parametrize("tied", [True, False])
def test_full_model_hf_logits_loss_gradients_and_sgd_step(hf_classes, tied):
    actual, reference = models(hf_classes, tie_word_embeddings=tied)
    input_ids = torch.tensor([[3, 5, 7, 9, 0, 0, 0, 0], [0, 0, 11, 13, 17, 19, 23, 0]])
    mask = input_ids.ne(0).long()
    positions = torch.stack((torch.arange(8) + 7, torch.arange(8) + 19))
    labels = input_ids.masked_fill(mask.eq(0), -100)
    labels[1, 4] = -100
    actual_modules = dict(actual.named_modules())
    reference_modules = dict(reference.named_modules())
    observed, ref_observed = {}, {}
    handles = []
    for name in actual_modules:
        if name.endswith(("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")):

            def capture(storage, key):
                def hook(_module, _args, result):
                    storage[key] = result[0] if isinstance(result, tuple) else result

                return hook

            handles.append(actual_modules[name].register_forward_hook(capture(observed, name)))
            handles.append(
                reference_modules[name].register_forward_hook(capture(ref_observed, name))
            )
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        output = actual(input_ids, attention_mask=mask, position_ids=positions, labels=labels)
        expected = reference(
            input_ids, attention_mask=mask, position_ids=positions, labels=labels, use_cache=False
        )
    metrics = {
        "transformers": "4.57.3",
        "dtype": "float32",
        "tied": tied,
        "atol": ATOL,
        "rtol": RTOL,
        "logits_max_abs": assert_tensor(output.logits, expected.logits),
        "loss_abs": assert_tensor(output.loss, expected.loss),
    }
    metrics["modules_max_abs"] = max(
        assert_tensor(observed[name], ref_observed[name]) for name in observed
    )
    for handle in handles:
        handle.remove()
    output.loss.backward()
    expected.loss.backward()
    ref_parameters = dict(reference.named_parameters())
    gradient_errors = []
    for name, parameter in actual.named_parameters():
        assert parameter.grad is not None, name
        gradient_errors.append(assert_tensor(parameter.grad, ref_parameters[name].grad))
    metrics["gradient_max_abs"] = max(gradient_errors)
    if tied:
        assert actual.lm_head.weight is actual.model.embed_tokens.weight
        assert reference.lm_head.weight is reference.model.embed_tokens.weight
    optimizer = torch.optim.SGD(actual.parameters(), lr=0.03)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    optimizer.step()
    reference_optimizer.step()
    metrics["updated_parameter_max_abs"] = max(
        assert_tensor(parameter, ref_parameters[name])
        for name, parameter in actual.named_parameters()
    )
    print("QWEN_REFERENCE_METRICS=" + json.dumps(metrics, sort_keys=True))
    metrics_directory = os.environ.get("SHARDTRAIN_QWEN_METRICS_DIR")
    if metrics_directory:
        target = Path(metrics_directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"hf_cpu_tied{int(tied)}.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )


def test_inputs_embeds_input_gradient_reference(hf_classes):
    actual, reference = models(hf_classes)
    x = torch.randn(2, 6, 32, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    labels = torch.arange(12).reshape(2, 6) + 1
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        output = actual(inputs_embeds=x, labels=labels)
        expected = reference(inputs_embeds=ref_x, labels=labels, use_cache=False)
    assert_tensor(output.logits, expected.logits)
    output.loss.backward()
    expected.loss.backward()
    assert_tensor(x.grad, ref_x.grad)


@pytest.mark.parametrize("tied", [True, False])
def test_all_padding_sample_hf_sdpa_logits_and_gradients(hf_classes, tied):
    """Keep padded logits in the oracle and verify ignored targets give zero gradient."""
    actual, reference = models(
        hf_classes,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        max_position_embeddings=128,
        pad_token_id=None,
        tie_word_embeddings=tied,
    )
    # Same replica-3 token generation/mask as the F4 distributed fixture. Keep
    # nonzero token IDs in the fully padded row so attention semantics are visible.
    generator = torch.Generator().manual_seed(2718 + 3 * 100)
    tokens = torch.randint(1, 64, (2, 8), generator=generator)
    mask = torch.ones_like(tokens)
    mask[1] = 0
    labels = tokens.masked_fill(mask.eq(0), -100)
    labels[0, 7] = -100
    positions = torch.arange(17, 25).expand(2, -1)
    x = actual.model.embed_tokens(tokens).detach().requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    observed, ref_observed = [], []

    def capture_attention(storage):
        def hook(_module, _args, output):
            storage.append(output[0] if isinstance(output, tuple) else output)

        return hook

    handles = []
    for layer, ref_layer in zip(actual.model.layers, reference.model.layers, strict=True):
        handles.append(layer.self_attn.register_forward_hook(capture_attention(observed)))
        handles.append(ref_layer.self_attn.register_forward_hook(capture_attention(ref_observed)))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        output = actual(inputs_embeds=x, attention_mask=mask, position_ids=positions, labels=labels)
        expected = reference(
            inputs_embeds=ref_x,
            attention_mask=mask,
            position_ids=positions,
            labels=labels,
            use_cache=False,
        )
    metrics = {
        "transformers": "4.57.3",
        "attention_backend": "SDPA math",
        "dtype": "float32",
        "hardware": "CPU",
        "tied": tied,
        "atol": ATOL,
        "rtol": RTOL,
        "logits_max_abs": assert_tensor(output.logits, expected.logits),
        "all_padding_logits_max_abs": assert_tensor(output.logits[1], expected.logits[1]),
        "loss_abs": assert_tensor(output.loss, expected.loss),
    }
    assert torch.isfinite(output.loss)
    for observed_attention, ref_attention in zip(observed, ref_observed, strict=True):
        assert_tensor(observed_attention, ref_attention)
        assert torch.count_nonzero(observed_attention[1]) == 0
        assert torch.count_nonzero(ref_attention[1]) == 0
    for handle in handles:
        handle.remove()
    output.loss.backward()
    expected.loss.backward()
    metrics["input_loss_gradient_max_abs"] = assert_tensor(x.grad, ref_x.grad)
    assert torch.count_nonzero(x.grad[1]) == torch.count_nonzero(ref_x.grad[1]) == 0
    ref_parameters = dict(reference.named_parameters())

    def parameter_gradient_error():
        errors = []
        for name, parameter in actual.named_parameters():
            if parameter.grad is None:
                # inputs_embeds bypasses an untied input embedding leaf.
                assert not tied and name == "model.embed_tokens.weight"
                assert ref_parameters[name].grad is None
                continue
            assert ref_parameters[name].grad is not None, name
            errors.append(assert_tensor(parameter.grad, ref_parameters[name].grad))
        return max(errors)

    metrics["parameter_loss_gradient_max_abs"] = parameter_gradient_error()

    # An objective on padding logits also checks their input/parameter derivatives,
    # which the ignored-label LM loss alone cannot exercise.
    actual.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    pad_x = x.detach()[1:].clone().requires_grad_()
    ref_pad_x = pad_x.detach().clone().requires_grad_()
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        padded = actual(inputs_embeds=pad_x, attention_mask=mask[1:], position_ids=positions[1:])
        ref_padded = reference(
            inputs_embeds=ref_pad_x,
            attention_mask=mask[1:],
            position_ids=positions[1:],
            use_cache=False,
        )
    padded.logits.square().mean().backward()
    ref_padded.logits.square().mean().backward()
    assert torch.count_nonzero(pad_x.grad) > 0
    metrics["padding_logits_input_gradient_max_abs"] = assert_tensor(pad_x.grad, ref_pad_x.grad)
    metrics["padding_logits_parameter_gradient_max_abs"] = parameter_gradient_error()
    print("QWEN_ALL_PADDING_REFERENCE_METRICS=" + json.dumps(metrics, sort_keys=True))
    metrics_directory = os.environ.get("SHARDTRAIN_QWEN_METRICS_DIR")
    if metrics_directory:
        target = Path(metrics_directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"hf_cpu_all_padding_tied{int(tied)}.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )


def test_fp16_math_sdpa_cpu_reference(hf_classes):
    """CPU half-precision numerical smoke; this does not validate V100 AMP/NCCL."""
    actual, reference = models(hf_classes)
    actual.half()
    reference.half()
    tokens = torch.tensor([[1, 3, 5, 7, 11, 13]])
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        output = actual(tokens, labels=tokens)
        expected = reference(tokens, labels=tokens, use_cache=False)
    torch.testing.assert_close(output.logits, expected.logits, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(output.loss, expected.loss, atol=3e-3, rtol=3e-3)
    output.loss.backward()
    expected.loss.backward()
    ref_parameters = dict(reference.named_parameters())
    differences = []
    for name, parameter in actual.named_parameters():
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, ref_parameters[name].grad, atol=3e-3, rtol=3e-3)
        differences.append(max_error(parameter.grad, ref_parameters[name].grad))
    metrics = {
        "transformers": "4.57.3",
        "dtype": "float16",
        "hardware": "CPU",
        "attention_backend": "SDPA math, explicit KV repeat",
        "atol": 3e-3,
        "rtol": 3e-3,
        "logits_max_abs": max_error(output.logits, expected.logits),
        "loss_abs": max_error(output.loss, expected.loss),
        "gradient_max_abs": max(differences),
        "V100_AMP_validated": False,
    }
    print("QWEN_FP16_REFERENCE_METRICS=" + json.dumps(metrics, sort_keys=True))
    metrics_directory = os.environ.get("SHARDTRAIN_QWEN_METRICS_DIR")
    if metrics_directory:
        target = Path(metrics_directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "hf_cpu_fp16.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )


def test_causal_padding_mask_and_dropout_semantics():
    torch.manual_seed(17)
    model = Qwen2CausalLM(tiny_config(attention_dropout=0.2))
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    model.eval()
    first = model(tokens).logits
    changed = tokens.clone()
    changed[:, 4:] = 9
    assert_tensor(first[:, :4], model(changed).logits[:, :4])
    mask = torch.tensor([[1, 1, 0, 1, 1, 1]])
    padded = model(tokens, attention_mask=mask).logits
    changed = tokens.clone()
    changed[:, 2] = 11
    assert_tensor(padded[:, 3:], model(changed, attention_mask=mask).logits[:, 3:])
    torch.manual_seed(31)
    assert_tensor(first, model(tokens).logits)
    model.train()
    torch.manual_seed(31)
    dropped = model(tokens).logits
    torch.manual_seed(32)
    assert not torch.equal(dropped, model(tokens).logits)
    with pytest.raises(ValueError, match="use_cache=False"):
        model(tokens, use_cache=True)


@pytest.mark.parametrize("layers", [4, 12, 28])
def test_depth_training_closed_loop(layers):
    model = Qwen2CausalLM(
        tiny_config(num_hidden_layers=layers, hidden_size=16, intermediate_size=32, vocab_size=23)
    )
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(tokens, labels=tokens)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()
