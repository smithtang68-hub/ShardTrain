"""Strict, CPU-only safetensors loading without constructing a full state dict.

The reader inspects every key and shape before training starts, then maps only the
requested tensor (or slice). Runtime registration can copy directly into a local
shard; the dense convenience loader is deliberately restricted to CPU modules.
No pickle-based weight file or implicit network download is accepted here.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn


class CheckpointError(ValueError):
    """A checkpoint cannot be loaded without missing or ambiguous weights."""


@dataclass(frozen=True)
class CheckpointLoadReport:
    checkpoint_path: str
    checkpoint_tensor_count: int
    unique_tensor_count: int
    copied_bytes: int
    omitted_tied_aliases: tuple[str, ...]
    converted_dtypes: tuple[str, ...]


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"Duplicate JSON key in checkpoint index: {key}")
        result[key] = value
    return result


class QwenCheckpoint:
    """Inspect a local HF safetensors checkpoint and map weights on demand.

    ``expected_shapes`` should describe the complete *canonical HF* model before
    TP/FSDP partitioning. ``tied_aliases`` maps an omitted alias to its physical
    checkpoint key and must come from verified parameter ownership, not merely
    ``config.tie_word_embeddings``. Shapes and all physical shard-file keys are
    checked; an index cannot hide additional tensors in its files.
    """

    def __init__(
        self,
        path: str | Path,
        expected_shapes: Mapping[str, tuple[int, ...]] | None = None,
        tied_aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self._files: dict[str, Path] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._dtypes: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._inspect()
        if expected_shapes is not None:
            self.validate(expected_shapes, tied_aliases)
        elif tied_aliases:
            raise CheckpointError("tied_aliases require expected_shapes for validation")

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._files))

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self._shapes)

    @property
    def dtypes(self) -> dict[str, str]:
        return dict(self._dtypes)

    def _inspect(self) -> None:
        if self.path.is_dir():
            index = self.path / "model.safetensors.index.json"
            single = self.path / "model.safetensors"
            if index.is_file():
                source = index
            elif single.is_file():
                source = single
            else:
                candidates = sorted(self.path.glob("*.safetensors"))
                if len(candidates) != 1:
                    raise CheckpointError(
                        f"Expected model.safetensors or its index in {self.path}; "
                        f"found {len(candidates)} standalone safetensors files"
                    )
                source = candidates[0]
        elif self.path.is_file():
            source = self.path
        else:
            raise CheckpointError(f"Checkpoint path does not exist: {self.path}")

        weight_map: dict[str, Path] | None = None
        if source.name.endswith(".index.json"):
            try:
                data = json.loads(
                    source.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise CheckpointError(f"Cannot read checkpoint index {source}: {exc}") from exc
            if not isinstance(data, dict) or not isinstance(data.get("weight_map"), dict):
                raise CheckpointError("Checkpoint index must contain a weight_map object")
            raw_map = data["weight_map"]
            if not raw_map:
                raise CheckpointError("Checkpoint weight_map must not be empty")
            root = source.parent.resolve()
            weight_map = {}
            for name, filename in raw_map.items():
                if not isinstance(name, str) or not isinstance(filename, str):
                    raise CheckpointError("Checkpoint weight_map keys and values must be strings")
                file_path = (root / filename).resolve()
                if not file_path.is_relative_to(root):
                    raise CheckpointError(f"Checkpoint shard escapes index directory: {filename}")
                if file_path.suffix != ".safetensors" or not file_path.is_file():
                    raise CheckpointError(f"Missing or unsupported checkpoint shard: {filename}")
                weight_map[name] = file_path
            files = sorted(set(weight_map.values()))
        elif source.suffix == ".safetensors":
            files = [source]
        else:
            raise CheckpointError("Only local .safetensors files or .index.json are supported")

        for file_path in files:
            try:
                with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        if name in self._files:
                            raise CheckpointError(
                                f"Duplicate tensor across checkpoint shards: {name}"
                            )
                        if weight_map is not None and weight_map.get(name) != file_path:
                            raise CheckpointError(
                                f"Tensor {name} in {file_path.name} is absent from the index "
                                "or mapped to a different shard"
                            )
                        tensor_slice = handle.get_slice(name)
                        self._files[name] = file_path
                        self._shapes[name] = tuple(tensor_slice.get_shape())
                        self._dtypes[name] = tensor_slice.get_dtype()
            except CheckpointError:
                raise
            except Exception as exc:
                raise CheckpointError(
                    f"Cannot inspect safetensors file {file_path}: {exc}"
                ) from exc
        if not self._files:
            raise CheckpointError("Checkpoint contains no tensors")
        if weight_map is not None:
            missing = sorted(set(weight_map) - self._files.keys())
            if missing:
                raise CheckpointError(f"Index references absent tensors: {', '.join(missing)}")

    def validate(
        self,
        expected_shapes: Mapping[str, tuple[int, ...]],
        tied_aliases: Mapping[str, str] | None = None,
    ) -> tuple[str, ...]:
        """Check all names/shapes atomically before any parameter is modified."""
        expected = {name: tuple(shape) for name, shape in expected_shapes.items()}
        aliases = dict(tied_aliases or {})
        for alias, canonical in aliases.items():
            if alias == canonical or alias not in expected or canonical not in expected:
                raise CheckpointError(f"Invalid tied alias: {alias} -> {canonical}")
            if expected[alias] != expected[canonical] or canonical in aliases:
                raise CheckpointError(f"Inconsistent or chained tied alias: {alias} -> {canonical}")
        missing = sorted(
            name
            for name in expected
            if name not in self._files and not (name in aliases and aliases[name] in self._files)
        )
        unexpected = sorted(self._files.keys() - expected.keys())
        mismatched = [
            f"{name}: checkpoint {self._shapes[name]}, expected {shape}"
            for name, shape in expected.items()
            if name in self._shapes and self._shapes[name] != shape
        ]
        errors = []
        if missing:
            errors.append("Missing keys: " + ", ".join(missing))
        if unexpected:
            errors.append("Unexpected keys: " + ", ".join(unexpected))
        if mismatched:
            errors.append("Shape mismatches: " + "; ".join(mismatched))
        if errors:
            raise CheckpointError("Checkpoint validation failed. " + " | ".join(errors))
        for alias, canonical in aliases.items():
            if alias in self._files and canonical in self._files:
                if not torch.equal(self.get_tensor(alias), self.get_tensor(canonical)):
                    raise CheckpointError(
                        f"Conflicting checkpoint values for tied keys: {canonical}, {alias}"
                    )
        self._aliases = aliases
        return tuple(sorted(name for name in aliases if name not in self._files))

    def _physical_name(self, name: str) -> str:
        if name in self._files:
            return name
        if name in self._aliases and self._aliases[name] in self._files:
            return self._aliases[name]
        raise CheckpointError(f"Checkpoint tensor not found: {name}")

    def get_tensor(self, name: str) -> torch.Tensor:
        """Return a mapped CPU tensor; callers should treat it as read-only."""
        physical = self._physical_name(name)
        with safe_open(str(self._files[physical]), framework="pt", device="cpu") as handle:
            return handle.get_tensor(physical)

    def get_tensor_slice(self, name: str, index: Any) -> torch.Tensor:
        """Map only an indexed region, useful for vocabulary/TP local shards."""
        physical = self._physical_name(name)
        with safe_open(str(self._files[physical]), framework="pt", device="cpu") as handle:
            return handle.get_slice(physical)[index]

    def iter_tensors(
        self, names: Iterator[str] | None = None
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield one tensor at a time; do not retain yielded tensors as a state dict."""
        for name in self.keys if names is None else names:
            yield name, self.get_tensor(name)


def iter_checkpoint_tensors(path: str | Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Iterate local checkpoint weights on CPU without eagerly loading a state dict."""
    yield from QwenCheckpoint(path).iter_tensors()


def load_qwen_checkpoint(model: nn.Module, path: str | Path) -> CheckpointLoadReport:
    """Strictly copy checkpoint weights into an existing CPU model.

    Alias permission is derived from actual tensor identity. If both tied keys are
    serialized, their values must match. Metadata and tied values are validated
    before copying, so those errors cannot partially mutate the target model.
    ``QwenCheckpoint`` is the preferred API for direct runtime shard registration.
    """
    targets = model.state_dict(keep_vars=True)
    if any(tensor.device.type != "cpu" for tensor in targets.values()):
        raise CheckpointError(
            "Dense checkpoint loading requires a CPU model; use QwenCheckpoint "
            "for memory-controlled direct shard registration"
        )
    aliases: dict[str, str] = {}
    identity: dict[int, str] = {}
    for name, tensor in targets.items():
        tensor_id = id(tensor)
        if tensor_id in identity:
            aliases[name] = identity[tensor_id]
        else:
            identity[tensor_id] = name
    reader = QwenCheckpoint(path)
    omitted = reader.validate(
        {name: tuple(tensor.shape) for name, tensor in targets.items()}, aliases
    )
    copied_bytes = 0
    converted = []
    with torch.no_grad():
        for name, target in targets.items():
            if name in aliases:
                continue
            tensor = reader.get_tensor(name)
            if tensor.dtype != target.dtype:
                converted.append(name)
            target.copy_(tensor)
            copied_bytes += target.numel() * target.element_size()
    return CheckpointLoadReport(
        checkpoint_path=str(reader.path),
        checkpoint_tensor_count=len(reader.keys),
        unique_tensor_count=len(identity),
        copied_bytes=copied_bytes,
        omitted_tied_aliases=omitted,
        converted_dtypes=tuple(converted),
    )
