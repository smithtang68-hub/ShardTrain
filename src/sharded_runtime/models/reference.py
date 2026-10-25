"""Seeded reference construction and deterministic synthetic activation batches."""

import torch
from torch import Tensor, nn

from sharded_runtime.config import ModelConfig
from sharded_runtime.models.transformer_block import TransformerBlock


def build_reference_model(config: ModelConfig, seed: int = 1234) -> nn.Module:
    # Logical weights are initialized on CPU once, then transferred to the target device.
    # Preserve the caller's RNG; data generation uses its own generator too.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return nn.Sequential(*(TransformerBlock(config) for _ in range(config.layers)))


def synthetic_batch(
    config: ModelConfig, batch_size: int, seed: int, *, device: torch.device | str = "cpu"
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        batch_size, config.seq_len, config.hidden_size, generator=generator, dtype=torch.float32
    ).to(device)
