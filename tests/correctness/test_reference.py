import copy
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

from sharded_runtime.config import ModelConfig
from sharded_runtime.models.reference import build_reference_model, synthetic_batch
from sharded_runtime.training.trainer import ReferenceTrainer


def functional_oracle(x, parameters, config):
    """Independent attention path (SDPA math) and functional parameters, no model forward."""
    for layer in range(config.layers):
        p = {
            name.removeprefix(f"{layer}."): value
            for name, value in parameters.items()
            if name.startswith(f"{layer}.")
        }

        def norm(value, weight):
            return (
                value
                / torch.sqrt(value.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)
                * weight
            )

        z = norm(x, p["attention_norm.weight"])
        batch, seq, hidden = z.shape
        q, k, v = [
            F.linear(z, p[f"attention.{name}_proj.weight"])
            .reshape(batch, seq, config.num_heads, hidden // config.num_heads)
            .transpose(1, 2)
            for name in ("q", "k", "v")
        ]
        # This oracle runs on CPU; the implementation uses explicit masked softmax.
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        x = x + F.linear(
            attention.transpose(1, 2).reshape(batch, seq, hidden), p["attention.out_proj.weight"]
        )
        z = norm(x, p["mlp_norm.weight"])
        gate = F.linear(z, p["mlp.gate_proj.weight"])
        activated = gate * torch.sigmoid(gate)
        x = x + F.linear(
            activated * F.linear(z, p["mlp.up_proj.weight"]), p["mlp.down_proj.weight"]
        )
    return x


@pytest.mark.parametrize("layers", [1, 2])
def test_outputs_all_gradients_and_three_sgd_updates_against_oracle(layers):
    config = ModelConfig(layers=layers, hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    model = build_reference_model(config, 1234)
    oracle_params = {
        name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()
    }
    trainer = ReferenceTrainer(model, lr=0.01)
    for step in range(3):
        model.zero_grad(set_to_none=True)
        batch = synthetic_batch(config, 2, 4321 + step)
        x = batch.clone().requires_grad_()
        oracle_x = batch.clone().requires_grad_()
        actual = model(x)
        expected = functional_oracle(oracle_x, oracle_params, config)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, oracle_x.grad, rtol=3e-5, atol=3e-6)
        for name, value in model.named_parameters():
            torch.testing.assert_close(value.grad, oracle_params[name].grad, rtol=3e-5, atol=3e-6)
        # The trainer clears model gradients and performs its own full train step.
        result = trainer.train_step(batch)
        assert result.step == step
        assert result.loss == pytest.approx(expected.square().mean().item(), abs=2e-6)
        with torch.no_grad():
            for name, value in model.named_parameters():
                oracle_params[name].sub_(0.01 * oracle_params[name].grad)
                oracle_params[name].grad = None
                torch.testing.assert_close(value, oracle_params[name], rtol=3e-5, atol=3e-6)


def test_causal_prefix_has_no_future_dependency():
    config = ModelConfig(hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    model = build_reference_model(config)
    x = synthetic_batch(config, 1, 4).requires_grad_()
    original = model(x)
    changed = x.detach().clone()
    changed[:, 2:] += 10
    torch.testing.assert_close(original[:, :2], model(changed)[:, :2], rtol=0, atol=0)
    original[:, :2].sum().backward()
    assert torch.count_nonzero(x.grad[:, 2:]) == 0


@pytest.mark.parametrize("accumulation", [1, 2])
def test_accumulation_matches_full_batch_over_three_steps(accumulation):
    config = ModelConfig(hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    full = build_reference_model(config)
    accumulated = copy.deepcopy(full)
    full_trainer = ReferenceTrainer(full, lr=0.01)
    accumulated_trainer = ReferenceTrainer(accumulated, lr=0.01, accumulation_steps=accumulation)
    for step in range(3):
        batch = synthetic_batch(config, 4, 40 + step)
        a, b = full_trainer.train_step(batch), accumulated_trainer.train_step(batch)
        assert a.loss == pytest.approx(b.loss, abs=2e-6)
        for p, q in zip(full.parameters(), accumulated.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(p, q, rtol=2e-5, atol=2e-6)


def test_fixed_seed_and_rng_isolation():
    config = ModelConfig(hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    torch.manual_seed(999)
    before = torch.random.get_rng_state().clone()
    first = build_reference_model(config, 12)
    second = build_reference_model(config, 12)
    synthetic_batch(config, 1, 99)
    assert torch.equal(torch.random.get_rng_state(), before)
    for p, q in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(p, q)


def test_invalid_batch_shape_and_nonfinite_input():
    config = ModelConfig(hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    model = build_reference_model(config)
    trainer = ReferenceTrainer(model, lr=0.01, accumulation_steps=2)
    with pytest.raises(ValueError, match="divisible"):
        trainer.train_step(torch.zeros(3, 4, 8))
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(2, 4, 7))
    with pytest.raises(FloatingPointError, match="Non-finite"):
        trainer.train_step(torch.full((2, 4, 8), float("nan")))


def test_cpu_golden_reference():
    fixture = yaml.safe_load(
        (Path(__file__).parent / "golden_reference.yaml").read_text(encoding="utf-8")
    )
    config = ModelConfig(**fixture["model"])
    model = build_reference_model(config, fixture["seed"])
    x = synthetic_batch(config, 1, fixture["input_seed"]).requires_grad_()
    y = model(x)
    y.square().mean().backward()
    torch.testing.assert_close(y, torch.tensor(fixture["output"]), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        x.grad, torch.tensor(fixture["input_gradient"]), rtol=3e-5, atol=3e-6
    )
    for name, param in model.named_parameters():
        torch.testing.assert_close(
            param.grad, torch.tensor(fixture["gradients"][name]), rtol=3e-5, atol=3e-6
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=fixture["lr"])
    optimizer.step()
    for name, param in model.named_parameters():
        torch.testing.assert_close(
            param, torch.tensor(fixture["updated_parameters"][name]), rtol=3e-5, atol=3e-6
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_outputs_input_gradients_parameter_gradients_and_three_updates():
    config = ModelConfig(hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    cpu = build_reference_model(config, 1234)
    cuda = copy.deepcopy(cpu).to("cuda")
    cpu_trainer = ReferenceTrainer(cpu, lr=0.01)
    cuda_trainer = ReferenceTrainer(cuda, lr=0.01)
    for step in range(3):
        cpu.zero_grad(set_to_none=True)
        cuda.zero_grad(set_to_none=True)
        batch = synthetic_batch(config, 2, 4321 + step)
        x = batch.clone().requires_grad_()
        cuda_x = batch.to("cuda").requires_grad_()
        cpu_y, cuda_y = cpu(x), cuda(cuda_x)
        torch.testing.assert_close(cpu_y, cuda_y.cpu(), rtol=3e-5, atol=3e-6)
        cpu_y.square().mean().backward()
        cuda_y.square().mean().backward()
        torch.testing.assert_close(x.grad, cuda_x.grad.cpu(), rtol=3e-5, atol=3e-6)
        for p, q in zip(cpu.parameters(), cuda.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad.cpu(), rtol=3e-5, atol=3e-6)
        a = cpu_trainer.train_step(batch)
        b = cuda_trainer.train_step(batch.to("cuda"))
        assert a.loss == pytest.approx(b.loss, abs=3e-6)
        for p, q in zip(cpu.parameters(), cuda.parameters(), strict=True):
            torch.testing.assert_close(p, q.cpu(), rtol=3e-5, atol=3e-6)
