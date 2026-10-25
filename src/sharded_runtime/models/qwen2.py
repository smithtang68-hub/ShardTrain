"""Training-only Qwen2.5 adapter with Hugging Face checkpoint parameter names.

The model equations follow the documented Qwen2 architecture; numerical reference
tests use transformers 4.57.3. No HF/FSDP/Megatron implementation is embedded here.
Attention deliberately uses PyTorch SDPA math plus explicit KV head expansion, so
the training path does not require FlashAttention or native BF16 support.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Qwen2ModelConfig:
    """Validated subset of the HF Qwen2Config for full-sequence training."""

    vocab_size: int = 128
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    initializer_range: float = 0.02
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    use_cache: bool = False
    use_sliding_window: bool = False
    model_type: str = "qwen2"
    rope_scaling: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        sizes = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
        )
        if any(size <= 0 for size in sizes):
            raise ValueError("Qwen2 dimensions must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("Qwen2 RoPE requires an even head dimension")
        if self.model_type != "qwen2" or self.hidden_act != "silu":
            raise ValueError("Only the Qwen2 decoder with SiLU SwiGLU is supported")
        if self.use_sliding_window:
            raise ValueError("Sliding-window attention is outside this training adapter's scope")
        if self.rope_scaling is not None:
            raise ValueError("RoPE scaling/YaRN is outside this training adapter's scope")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("rope_theta and rms_norm_eps must be positive")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.pad_token_id is not None and not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id is outside the vocabulary")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Qwen2ModelConfig:
        if not isinstance(values, dict):
            raise ValueError("Qwen2 configuration must be a mapping")
        # HF config files also contain serialization/inference metadata, not model equations.
        fields = cls.__dataclass_fields__
        accepted = {key: value for key, value in values.items() if key in fields}
        if accepted.get("num_key_value_heads") is None:
            accepted["num_key_value_heads"] = accepted.get("num_attention_heads", 4)
        rope_parameters = values.get("rope_parameters")
        if rope_parameters:
            if rope_parameters.get("rope_type", "default") != "default":
                raise ValueError("Only default Qwen2 RoPE is supported")
            accepted["rope_theta"] = rope_parameters.get("rope_theta", 1000000.0)
        if values.get(
            "head_dim", accepted.get("hidden_size", 32) // accepted.get("num_attention_heads", 4)
        ) != (accepted.get("hidden_size", 32) // accepted.get("num_attention_heads", 4)):
            raise ValueError("A custom head_dim is unsupported for Qwen2.5")
        if "layer_types" in values and values["layer_types"] is not None:
            if any(kind != "full_attention" for kind in values["layer_types"]):
                raise ValueError("Only full causal attention layers are supported")
        return cls(**accepted)

    @classmethod
    def from_json(cls, path: str | Path) -> Qwen2ModelConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_file(cls, path: str | Path) -> Qwen2ModelConfig:
        """Read a project YAML configuration or an external HF JSON configuration."""
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls.from_json(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
        raise ValueError("Qwen2 configuration must use .yaml, .yml or .json")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **factory_kwargs: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.variance_epsilon = eps
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        accumulated = x.float()
        scale = torch.rsqrt(accumulated.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (accumulated * scale).to(x.dtype)
        return self.weight * normalized


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2ModelConfig, device: Any = None) -> None:
        super().__init__()
        exponents = torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=device)
        frequencies = config.rope_theta ** (-exponents / config.head_dim)
        self.register_buffer("inv_freq", frequencies, persistent=False)

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        # Always global positions supplied by the outer model, including when x is an SP shard.
        with torch.autocast(device_type=x.device.type, enabled=False):
            angles = position_ids.float().unsqueeze(-1) * self.inv_freq.float()
            angles = torch.cat((angles, angles), dim=-1)
            cos, sin = angles.cos(), angles.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    def rotate(x: Tensor) -> Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


def causal_attention_mask(x: Tensor, attention_mask: Tensor | None) -> Tensor:
    """Global SDPA causal/padding mask for [B, H, S, D] queries.

    Boolean masks give zero attention for queries with no allowed keys, matching
    transformers 4.57.3's SDPA backend on Torch >=2.5. HF's eager backend uses a
    finite-minimum additive mask and gives a uniform average in these rows;
    fully padded logits therefore require the SDPA reference for comparison.
    """
    batch, _, seq, _ = x.shape
    allowed = torch.ones((seq, seq), dtype=torch.bool, device=x.device).tril()
    allowed = allowed.unsqueeze(0).unsqueeze(0).expand(batch, 1, seq, seq)
    minimum = torch.finfo(x.dtype).min
    if attention_mask is None:
        return allowed
    attention_mask = attention_mask.to(x.device)
    if attention_mask.ndim == 2:
        if attention_mask.shape != (batch, seq):
            raise ValueError("attention_mask must describe the full global [batch, sequence]")
        allowed = allowed & attention_mask[:, None, None, :].bool()
        return allowed
    if attention_mask.ndim != 4 or attention_mask.shape[-2:] != (seq, seq):
        raise ValueError("attention_mask must have shape [B,S] or [B,1,S,S]")
    if attention_mask.shape[0] not in (1, batch):
        raise ValueError("attention_mask batch size is incompatible")
    if attention_mask.dtype == torch.bool:
        allowed = allowed & attention_mask
        return allowed
    if not attention_mask.is_floating_point():
        raise ValueError("A 4-D attention mask must be boolean or additive floating point")
    return attention_mask.to(x.dtype).masked_fill(~allowed, minimum)


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2ModelConfig, layer_idx: int, **factory_kwargs: Any) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=True, **factory_kwargs
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
            **factory_kwargs,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
            **factory_kwargs,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False, **factory_kwargs
        )
        self.position_embeddings: tuple[Tensor, Tensor] | None = None
        self.attention_mask: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        group = getattr(self, "projection_group", None)
        projected = (
            group(x) if group is not None else [self.q_proj(x), self.k_proj(x), self.v_proj(x)]
        )
        batch, seq, _ = projected[0].shape
        q = projected[0].reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            projected[1]
            .reshape(batch, seq, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            projected[2]
            .reshape(batch, seq, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        if self.position_embeddings is None:
            raise RuntimeError("Qwen2Attention requires the full-sequence position context")
        cos, sin = self.position_embeddings
        if cos.shape[-2] != seq:
            raise ValueError("QKV sequence length differs from global RoPE positions")
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        groups = self.num_heads // self.num_key_value_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        mask = causal_attention_mask(q, self.attention_mask)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
        flat = attended.transpose(1, 2).contiguous().reshape(batch, seq, -1)
        return self.o_proj(flat)


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2ModelConfig, **factory_kwargs: Any) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False, **factory_kwargs
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False, **factory_kwargs
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False, **factory_kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        group = getattr(self, "projection_group", None)
        gate, up = group(x) if group is not None else (self.gate_proj(x), self.up_proj(x))
        return self.down_proj(F.silu(gate) * up)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2ModelConfig, layer_idx: int, **factory_kwargs: Any) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config, layer_idx, **factory_kwargs)
        self.mlp = Qwen2MLP(config, **factory_kwargs)
        self.input_layernorm = Qwen2RMSNorm(
            config.hidden_size, config.rms_norm_eps, **factory_kwargs
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, config.rms_norm_eps, **factory_kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2ModelConfig, **factory_kwargs: Any) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id, **factory_kwargs
        )
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, i, **factory_kwargs)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps, **factory_kwargs)
        self.rotary_emb = Qwen2RotaryEmbedding(config, device=factory_kwargs.get("device"))


@dataclass
class Qwen2CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None
    last_hidden_state: Tensor | None = None


def causal_lm_loss_parts(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Return FP32 next-token loss sum and valid count, avoiding a mean of local means."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("labels must match the complete logits [batch, sequence]")
    shifted_labels = labels[:, 1:].contiguous()
    count = shifted_labels.ne(-100).sum()
    if shifted_labels.numel() == 0:
        return logits.float().sum() * 0.0, count
    numerator = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return numerator, count


class Qwen2CausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen2ModelConfig | Any,
        *,
        device: Any = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, Qwen2ModelConfig):
            config = Qwen2ModelConfig.from_dict(config.to_dict())
        self.config = config
        self.model = Qwen2Model(config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype
        )
        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        use_cache: bool = False,
    ) -> Qwen2CausalLMOutput:
        if use_cache:
            raise ValueError("This adapter supports training with use_cache=False only")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        source = input_ids if input_ids is not None else inputs_embeds
        assert source is not None
        batch, seq = source.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(seq, device=source.device).unsqueeze(0)
        if position_ids.shape not in ((1, seq), (batch, seq)):
            raise ValueError("position_ids must describe the complete global sequence")
        if inputs_embeds is not None and inputs_embeds.shape[-1] != self.config.hidden_size:
            raise ValueError("inputs_embeds hidden size differs from config.hidden_size")
        runtime = getattr(self, "_sharded_runtime", None)
        if runtime is not None:
            input_ids, inputs_embeds = runtime.prepare_qwen_input(input_ids, inputs_embeds)
        hidden = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden, position_ids.to(hidden.device))
        for layer_index, layer in enumerate(self.model.layers):
            layer.self_attn.position_embeddings = position_embeddings
            layer.self_attn.attention_mask = attention_mask
            range_name = f"shardtrain/qwen/layer{layer_index}"
            with torch.profiler.record_function(range_name):
                cuda_range = hidden.is_cuda
                if cuda_range:
                    torch.cuda.nvtx.range_push(range_name)
                try:
                    hidden = layer(hidden)
                finally:
                    if cuda_range:
                        torch.cuda.nvtx.range_pop()
        hidden = self.model.norm(hidden)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            if runtime is not None:
                loss = runtime.qwen_loss(logits, labels)
            else:
                numerator, count = causal_lm_loss_parts(logits, labels)
                loss = numerator / count.clamp_min(1)
        return Qwen2CausalLMOutput(logits=logits, loss=loss, last_hidden_state=hidden)


Qwen2Config = Qwen2ModelConfig
Qwen2ForCausalLM = Qwen2CausalLM
