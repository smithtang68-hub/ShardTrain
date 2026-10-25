"""Regenerate the tiny CPU fixture using the independent functional test oracle.

Review changes to this fixture; do not regenerate it merely to make a failing test pass.
Run from the repository root with the development dependencies installed.
"""

import runpy
from dataclasses import asdict
from pathlib import Path

import yaml

from sharded_runtime.config import ModelConfig
from sharded_runtime.models.reference import build_reference_model, synthetic_batch

ROOT = Path(__file__).resolve().parents[1]


def main():
    oracle = runpy.run_path(str(ROOT / "tests/correctness/test_reference.py"))["functional_oracle"]
    config = ModelConfig(hidden_size=4, num_heads=2, mlp_hidden_size=6, seq_len=3)
    seed, input_seed, lr = 1234, 4321, 0.01
    model = build_reference_model(config, seed)
    parameters = {name: p.detach().clone().requires_grad_() for name, p in model.named_parameters()}
    x = synthetic_batch(config, 1, input_seed).requires_grad_()
    output = oracle(x, parameters, config)
    output.square().mean().backward()
    fixture = {
        "provenance": "CPU functional SDPA oracle; all parameter gradients and one SGD update",
        "model": asdict(config),
        "seed": seed,
        "input_seed": input_seed,
        "lr": lr,
        "output": output.detach().tolist(),
        "input_gradient": x.grad.tolist(),
        "gradients": {name: p.grad.tolist() for name, p in parameters.items()},
        "updated_parameters": {
            name: (p - lr * p.grad).detach().tolist() for name, p in parameters.items()
        },
    }
    target = ROOT / "tests/correctness/golden_reference.yaml"
    target.write_text(yaml.safe_dump(fixture, sort_keys=False), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
