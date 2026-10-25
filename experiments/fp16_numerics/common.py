"""Frozen metrics, immutable outputs, and bounded tensor snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

BIAS = "model.layers.3.self_attn.k_proj.bias"
ATOL = RTOL = 0.003
OPTIMIZER = dict(
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    foreach=False,
    fused=None,
    capturable=False,
    amsgrad=False,
    maximize=False,
    differentiable=False,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")


def cpu(value):
    return value.detach().cpu().clone()


def tensor_hash(value):
    value = cpu(value).contiguous()
    digest = hashlib.sha256(f"{value.dtype}/{tuple(value.shape)}".encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def named_hash(parameters):
    digest = hashlib.sha256()
    for name, value in sorted(parameters.items()):
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def distribution(actual, expected, *, atol=ATOL, rtol=RTOL, topk=3):
    """Original allclose plus exact and magnitude metrics, without raising."""
    a, b = cpu(actual).float().reshape(-1), cpu(expected).float().reshape(-1)
    if a.shape != b.shape or not a.numel():
        raise ValueError("Comparison requires equal nonempty tensor shapes")
    finite = torch.isfinite(a) & torch.isfinite(b)
    error = (a - b).abs()
    clean_error = error[finite].double()
    clean_a, clean_b = a[finite].double(), b[finite].double()
    anorm, bnorm = float(clean_a.norm()), float(clean_b.norm())
    p = (
        torch.quantile(clean_error, torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64)).tolist()
        if clean_error.numel()
        else [None] * 3
    )
    values, indices = torch.topk(torch.nan_to_num(error, nan=float("inf")), min(topk, len(error)))
    return {
        "numel": len(a),
        "shape": list(actual.shape),
        "bitwise_equal": torch.equal(a, b),
        "passed": bool(torch.allclose(a, b, atol=atol, rtol=rtol)) and bool(finite.all()),
        "atol": atol,
        "rtol": rtol,
        "actual_nonfinite": int((~torch.isfinite(a)).sum()),
        "reference_nonfinite": int((~torch.isfinite(b)).sum()),
        "tolerance_failed_elements": int((~torch.isclose(a, b, atol=atol, rtol=rtol)).sum()),
        "max_abs": float(clean_error.max()) if clean_error.numel() else None,
        "p50_abs": p[0],
        "p95_abs": p[1],
        "p99_abs": p[2],
        "actual_l2": anorm,
        "reference_l2": bnorm,
        "relative_l2": float((clean_a - clean_b).norm()) / bnorm if bnorm else None,
        "cosine": float(torch.dot(clean_a, clean_b)) / (anorm * bnorm) if anorm and bnorm else None,
        "sign_disagreement_nonzero": int(((clean_a * clean_b) < 0).sum()),
        "topk": [
            {
                "flat_index": int(i),
                "actual": float(a[i]) if torch.isfinite(a[i]) else None,
                "reference": float(b[i]) if torch.isfinite(b[i]) else None,
                "abs_error": float(v) if torch.isfinite(v) else None,
            }
            for i, v in zip(indices, values, strict=True)
        ],
    }


def optimizer_options(optimizer):
    group = optimizer.param_groups[0]
    options = {key: group.get(key) for key in OPTIMIZER}
    if options != OPTIMIZER:
        raise AssertionError(("Unexpected optimizer configuration", options, OPTIMIZER))
    return options


def pack(parameter, optimizer, *, gradient=None):
    """Real pre/post tensor state; empty initial state is explicit, not missing."""
    grad = parameter.grad if gradient is None else gradient
    return {
        "parameter": cpu(parameter),
        "gradient": None if grad is None else cpu(grad),
        "state_initialized": bool(optimizer.state.get(parameter)),
        "state": {
            key: cpu(value) if torch.is_tensor(value) else value
            for key, value in optimizer.state.get(parameter, {}).items()
        },
        "options": optimizer_options(optimizer),
        "parameter_dtype": str(parameter.dtype),
        "gradient_dtype": None if grad is None else str(grad.dtype),
    }


def global_index(local_index, spec, partition, fsdp_rank):
    """Independent row-major inverse for published TP-axis/FSDP-flat ownership."""
    index = fsdp_rank * spec.fsdp_shard_numel + local_index
    coordinates = []
    for width in reversed(spec.tp_local_shape):
        coordinates.append(index % width)
        index //= width
    coordinates.reverse()
    if partition.dim is not None:
        coordinates[partition.dim] += partition.rank * spec.tp_local_shape[partition.dim]
    result = 0
    for coordinate, width in zip(coordinates, spec.global_shape, strict=True):
        result = result * width + coordinate
    return result
