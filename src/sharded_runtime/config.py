"""Typed config and pure validation; importing this module does not initialize torch."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints

import yaml


class ConfigError(ValueError):
    """Invalid or unsupported configuration."""


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 1
    hidden_size: int = 128
    num_heads: int = 4
    mlp_hidden_size: int = 256
    seq_len: int = 16
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    model_type: str = "toy"
    hf_config_path: str = ""
    checkpoint_path: str = ""
    vocab_size: int = 128
    num_key_value_heads: int = 2
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True


@dataclass(frozen=True)
class ParallelConfig:
    fsdp_degree: int = 1
    tp_degree: int = 1
    sp_enabled: bool = False
    device: str = "cpu"
    backend: str = "gloo"

    @property
    def world_size(self) -> int:
        return self.fsdp_degree * self.tp_degree


@dataclass(frozen=True)
class RuntimeConfig:
    reshard_after_forward: bool = False
    materialization: str = "synchronous"
    prefetch: str = "none"
    param_grouping: str = "submodule"
    forward_prefetch: bool = False
    backward_prefetch: bool = False
    async_grad_rs: bool = False
    communication_policy: str = "conservative"


@dataclass(frozen=True)
class TrainingConfig:
    dtype: str = "fp32"
    optimizer: str = "sgd"
    lr: float = 0.001
    steps: int = 3
    local_batch_size: int = 2
    accumulation_steps: int = 1
    seed: int = 1234


@dataclass(frozen=True)
class DebugConfig:
    log_level: str = "INFO"
    distributed_timeout_seconds: int = 30
    output_dir: str = "runs/reference"


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self, world_size: int | None = None, *, mode: str = "reference") -> None:
        if mode not in {"reference", "train", "smoke", "runtime"}:
            raise ConfigError(f"Unknown execution mode: {mode}")
        positive = {
            "model.layers": self.model.layers,
            "model.hidden_size": self.model.hidden_size,
            "model.num_heads": self.model.num_heads,
            "model.mlp_hidden_size": self.model.mlp_hidden_size,
            "model.seq_len": self.model.seq_len,
            "parallel.fsdp_degree": self.parallel.fsdp_degree,
            "parallel.tp_degree": self.parallel.tp_degree,
            "training.steps": self.training.steps,
            "training.local_batch_size": self.training.local_batch_size,
            "training.accumulation_steps": self.training.accumulation_steps,
            "debug.distributed_timeout_seconds": self.debug.distributed_timeout_seconds,
        }
        for name, value in positive.items():
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{name} must be a positive integer; got {value!r}")
        p, m = self.parallel, self.model
        qwen = m.model_type == "qwen2"
        if m.model_type not in {"toy", "qwen2"}:
            raise ConfigError("model.model_type must be toy or qwen2")
        if qwen and mode not in {"runtime", "smoke"}:
            raise ConfigError("Qwen2 training requires the ShardTrain runtime mode")
        if world_size is not None and world_size != p.world_size:
            raise ConfigError(
                f"WORLD_SIZE={world_size} does not match fsdp_degree * tp_degree={p.world_size}"
            )
        if m.hidden_size % m.num_heads:
            raise ConfigError("model.hidden_size must be divisible by model.num_heads")
        for name, value in {
            "model.hidden_size": m.hidden_size,
            "model.num_heads": m.num_heads,
            "model.mlp_hidden_size": m.mlp_hidden_size,
        }.items():
            if value % p.tp_degree:
                raise ConfigError(f"{name} must be divisible by parallel.tp_degree")
        if p.sp_enabled and p.tp_degree == 1:
            raise ConfigError("SP requires parallel.tp_degree > 1; SP is not a mesh dimension")
        if p.sp_enabled and m.seq_len % p.tp_degree:
            raise ConfigError("model.seq_len must be divisible by TP degree when SP is enabled")
        if p.device not in {"cpu", "cuda"}:
            raise ConfigError("parallel.device must be cpu or cuda (index comes from LOCAL_RANK)")
        if p.backend not in {"gloo", "nccl"}:
            raise ConfigError("parallel.backend must be gloo or nccl")
        if p.device == "cpu" and p.backend != "gloo":
            raise ConfigError("CPU requires the Gloo backend")
        if p.device == "cuda" and p.world_size > 1 and p.backend != "nccl":
            raise ConfigError("Multi-process CUDA requires NCCL in the supported configuration")
        if m.dropout != 0.0:
            raise ConfigError("Only dropout=0 is supported by the deterministic reference")
        for name, value in {
            "model.rms_norm_eps": m.rms_norm_eps,
            "training.lr": self.training.lr,
        }.items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ConfigError(f"{name} must be finite and positive")
        if type(self.training.seed) is not int or not 0 <= self.training.seed < 2**32:
            raise ConfigError("training.seed must be an integer in [0, 2**32)")
        if qwen:
            if self.training.dtype not in {
                "fp32",
                "fp16",
                "bf16",
            } or self.training.optimizer not in {
                "sgd",
                "adamw",
            }:
                raise ConfigError("Qwen2 supports fp32/fp16/bf16 AMP with sgd/adamw")
            if self.training.dtype == "fp16" and p.device != "cuda":
                raise ConfigError("Qwen2 fp16 AMP requires CUDA")
        elif self.training.dtype != "fp32" or self.training.optimizer != "sgd":
            raise ConfigError("M0/M1 support only FP32 with SGD")
        if self.runtime.reshard_after_forward and mode != "runtime":
            raise ConfigError(
                "reshard_after_forward is implemented in M3; set it to false for M0/M1"
            )
        if mode != "runtime" and (
            self.runtime.materialization != "synchronous"
            or self.runtime.prefetch != "none"
            or self.runtime.forward_prefetch
            or self.runtime.backward_prefetch
            or self.runtime.async_grad_rs
            or self.runtime.communication_policy != "conservative"
        ):
            raise ConfigError("Async materialization/prefetch is not implemented in M0/M1")
        if self.runtime.param_grouping != "submodule" and not (
            mode == "runtime" and self.runtime.param_grouping == "block"
        ):
            raise ConfigError("Use param_grouping=block for runtime, submodule for the legacy path")
        if self.debug.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ConfigError("debug.log_level must be DEBUG, INFO, WARNING or ERROR")
        if not self.debug.output_dir.strip():
            raise ConfigError("debug.output_dir must not be empty")
        if mode == "reference" and p.world_size != 1:
            raise ConfigError(
                "M0 reference training requires fsdp_degree=tp_degree=1; "
                "use --smoke-groups for topology/collective checks only"
            )
        if mode == "train":
            if p.fsdp_degree != 1:
                raise ConfigError("FSDP training is implemented in M2; M1 requires fsdp_degree=1")
            if p.tp_degree not in {1, 2}:
                raise ConfigError("M1 training supports only tp_degree=1 or 2")
        if mode == "runtime":
            if self.runtime.communication_policy not in {"conservative", "stream_ordered"}:
                raise ConfigError("communication_policy must be conservative or stream_ordered")
            if p.fsdp_degree not in {1, 2, 4, 8} or p.tp_degree not in {1, 2, 4}:
                raise ConfigError("v0.7 runtime supports F=1/2/4/8 and T=1/2/4")
            if self.training.accumulation_steps != 1 and not qwen:
                raise ConfigError("v0.7 runtime does not support microbatch gradient accumulation")
            if self.runtime.materialization != "synchronous":
                raise ConfigError(
                    "Use independent async switches; keep materialization=synchronous"
                )
            if self.runtime.prefetch != "none":
                raise ConfigError("Use independent forward_prefetch/backward_prefetch flags")


_SECTIONS = {
    "model": ModelConfig,
    "parallel": ParallelConfig,
    "runtime": RuntimeConfig,
    "training": TrainingConfig,
    "debug": DebugConfig,
}


def config_from_dict(data: dict[str, Any]) -> Config:
    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping")
    unknown = set(data) - set(_SECTIONS)
    if unknown:
        raise ConfigError(f"Unknown config sections: {sorted(unknown)}")
    resolved = {}
    for name, cls in _SECTIONS.items():
        section = data.get(name, {})
        if not isinstance(section, dict):
            raise ConfigError(f"{name} must be a mapping")
        types = get_type_hints(cls)
        unknown = set(section) - {f.name for f in fields(cls)}
        if unknown:
            raise ConfigError(f"Unknown fields in {name}: {sorted(unknown)}")
        values = {}
        for key, value in section.items():
            expected = types[key]
            if expected is float and type(value) in {int, float}:
                value = float(value)
            elif type(value) is not expected:
                raise ConfigError(f"{name}.{key} must be {expected.__name__}; got {value!r}")
            values[key] = value
        resolved[name] = cls(**values)
    return Config(**resolved)


def load_config(filename: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    try:
        data = yaml.safe_load(Path(filename).read_text(encoding="utf-8")) if filename else {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read configuration {filename}: {exc}") from exc
    if data is None:
        data = {}
    # Validate keys/types before applying overrides so typos are never hidden.
    data = config_from_dict(data).to_dict()
    overrides = overrides or []
    if len(overrides) % 2:
        raise ConfigError("Overrides require pairs: --section.field value")
    for flag, raw in zip(overrides[::2], overrides[1::2], strict=True):
        if not flag.startswith("--") or flag.count(".") != 1:
            raise ConfigError(f"Invalid override {flag!r}; expected --section.field value")
        section, key = flag[2:].split(".")
        if section not in data or key not in data[section]:
            raise ConfigError(f"Unknown config field: {flag}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid value for {flag}: {raw!r}") from exc
        data[section][key] = value
    return config_from_dict(data)
