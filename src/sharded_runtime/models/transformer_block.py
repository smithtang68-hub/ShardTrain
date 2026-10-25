"""FP32 pre-norm causal decoder block with separate Q/K/V and SwiGLU projections."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sharded_runtime.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        group = getattr(self, "projection_group", None)
        projected = (
            group(x) if group else [proj(x) for proj in (self.q_proj, self.k_proj, self.v_proj)]
        )
        batch, seq, local_hidden = projected[0].shape
        q, k, v = (
            tensor.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
            for tensor in projected
        )
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones((seq, seq), dtype=torch.bool, device=x.device).triu(diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attended = scores.softmax(dim=-1) @ v
        return self.out_proj(
            attended.transpose(1, 2).contiguous().reshape(batch, seq, local_hidden)
        )


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.mlp_hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.mlp_hidden_size, bias=False)
        self.down_proj = nn.Linear(config.mlp_hidden_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        group = getattr(self, "projection_group", None)
        gate, up = group(x) if group else (self.gate_proj(x), self.up_proj(x))
        return self.down_proj(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected [batch, sequence, {self.hidden_size}], got {tuple(x.shape)}"
            )
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))
