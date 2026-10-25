"""Q9: actual FSDP1/FSDP2 representation and small-model correctness evidence.

No ShardTrain runtime operator is used. Run with torchrun on CUDA (FP32):
  torchrun --standalone --nproc-per-node=4 .../q09_param_representation.py \
    --implementation all --cases R1 R2 R3 --output-dir results/q09

The FSDP1 FlatParameter metadata read below is version-checked, read-only
instrumentation. Gradient gathering is ONLY a correctness observer after backward;
it is not part of the measured training algorithm or a performance benchmark.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecisionPolicy, ShardingStrategy, fully_shard
from torch.distributed.tensor import DTensor

IMPLEMENTATIONS = ("fsdp1_flat", "fsdp1_orig", "fsdp2")


class TinyNet(nn.Module):
    """Deliberately small FP32 model; Q9 is NOT the full-Qwen performance test."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(31, 16)
        self.up = nn.Linear(16, 24)
        self.down = nn.Linear(24, 8)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.down(torch.tanh(self.up(self.embedding(token_ids))))


def canonical(name: str) -> str:
    return ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_record(value: object) -> dict:
    try:
        path = Path(inspect.getfile(inspect.unwrap(value)))
        return {
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "signature": str(inspect.signature(value)),
        }
    except (OSError, TypeError, ValueError) as exc:
        return {"unavailable": str(exc)}


def environment(args: argparse.Namespace) -> dict:
    nccl_version = None
    if torch.cuda.is_available() and dist.is_nccl_available():
        nccl_version = torch.cuda.nccl.version()
    return {
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "nccl_version": nccl_version,
        "nccl_available": dist.is_nccl_available(),
        "python": sys.version,
        "git_commit": git_commit(),
        "argv": sys.argv,
        "config": vars(args),
        "sources": {
            "FSDP1": source_record(FSDP),
            "FSDP2": source_record(fully_shard),
            "get_model_state_dict": source_record(get_model_state_dict),
            "set_model_state_dict": source_record(set_model_state_dict),
            "state_dict_options": source_record(StateDictOptions),
        },
        "primary_source_links": {
            "FSDP1": "https://github.com/pytorch/pytorch/blob/v2.9.1/torch/distributed/fsdp/fully_sharded_data_parallel.py",
            "FSDP1_flat_metadata": "https://github.com/pytorch/pytorch/blob/v2.9.1/torch/distributed/fsdp/_flat_param.py",
            "FSDP2": "https://github.com/pytorch/pytorch/blob/v2.9.1/torch/distributed/fsdp/_fully_shard/_fully_shard.py",
            "state_dict": "https://github.com/pytorch/pytorch/blob/v2.9.1/torch/distributed/checkpoint/state_dict.py",
        },
    }


def check_api() -> None:
    """Fail explicitly if the installed API is not the validated PyTorch 2.9 API."""
    required = {
        FSDP: {"use_orig_params", "sharding_strategy", "device_id"},
        fully_shard: {"mesh", "reshard_after_forward", "mp_policy"},
        StateDictOptions: {"full_state_dict", "cpu_offload", "broadcast_from_rank0"},
    }
    for function, names in required.items():
        missing = names - set(inspect.signature(function).parameters)
        if missing:
            raise RuntimeError(f"Installed API {function}: missing {sorted(missing)}")


def make_model(seed: int, device: torch.device, frozen: bool = False) -> TinyNet:
    torch.manual_seed(seed)
    model = TinyNet().to(device=device, dtype=torch.float32)
    model.embedding.weight.requires_grad_(not frozen)
    return model


def data_for_rank(seed: int, rank: int, device: torch.device) -> tuple:
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000 + rank)
    ids = torch.randint(0, 31, (4, 8), generator=generator)
    target = torch.randn(4, 8, 8, generator=generator)
    return ids.to(device), target.to(device)


def parameter_group(name: str) -> str:
    """Embedding/up and down have different decay; members of each unit agree."""
    return "down" if canonical(name).startswith("down.") else "embedding_up"


def sharded_model(model: TinyNet, implementation: str, case: str, device: torch.device):
    if implementation == "fsdp2":
        mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("fsdp",))
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32),
        )
        return model
    use_orig = implementation == "fsdp1_orig"
    options = {
        "use_orig_params": use_orig,
        "device_id": device,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "limit_all_gathers": True,
    }
    if implementation == "fsdp1_flat" and case in {"R3", "R4"}:
        # A single root flat leaf cannot span optimizer groups with different decay.
        # Make one FSDP unit per homogeneous group/module, then wrap the root.
        model.embedding = FSDP(model.embedding, **options)
        model.up = FSDP(model.up, **options)
        model.down = FSDP(model.down, **options)
    return FSDP(model, **options)


def parameter_rows(model: nn.Module, before: dict, phase: str) -> list[dict]:
    rows = []
    for name, param in model.named_parameters():
        name = canonical(name)
        is_dtensor = isinstance(param, DTensor)
        local = param.to_local() if is_dtensor else param
        row = {
            "phase": phase,
            "name": name,
            "python_type": f"{type(param).__module__}.{type(param).__qualname__}",
            "is_flat_parameter": bool(getattr(param, "_is_flat_param", False)),
            "is_dtensor": is_dtensor,
            "python_id": id(param),
            "original_parameter_id": before.get(name, {}).get("id"),
            "same_identity_as_original": id(param) == before.get(name, {}).get("id"),
            "shape": list(param.shape),
            "local_shape": list(local.shape),
            "local_numel": local.numel(),
            "local_bytes": local.numel() * local.element_size(),
            "dtype": str(param.dtype),
            "requires_grad": param.requires_grad,
            "placements": [str(p) for p in param.placements] if is_dtensor else None,
            "mesh": param.device_mesh.mesh.tolist() if is_dtensor else None,
            "original_shape": before.get(name, {}).get("shape"),
        }
        if row["is_flat_parameter"]:
            fields = ("_fqns", "_shapes", "_numels", "_numels_with_padding", "_is_padding_mask")
            missing = [field for field in fields if not hasattr(param, field)]
            if missing:
                raise RuntimeError(f"Installed FlatParameter metadata changed: {missing}")
            row["flat_original_names"] = list(param._fqns)
            row["flat_original_shapes"] = [list(shape) for shape in param._shapes]
            row["flat_original_numels"] = list(param._numels)
            row["flat_numels_with_padding"] = list(param._numels_with_padding)
            row["flat_is_padding_mask"] = list(param._is_padding_mask)
        rows.append(row)
    return rows


def internal_flat_rows(model: nn.Module) -> list[dict]:
    """Expose FSDP1's remaining flat handles even when orig params are public."""
    rows = []
    for prefix, module in model.named_modules():
        if not isinstance(module, FSDP):
            continue
        handle = getattr(module, "_handle", None)
        if handle is None:
            continue
        flat = handle.flat_param
        rows.append(
            {
                "module": canonical(prefix),
                "flat_python_id": id(flat),
                "local_flat_shape": list(flat.shape),
                "original_names": list(flat._fqns),
                "original_shapes": [list(shape) for shape in flat._shapes],
                "numels_with_padding": list(flat._numels_with_padding),
                "is_padding_mask": list(flat._is_padding_mask),
                "exposed_original_parameter_ids": (
                    [id(param) for param in flat._params] if flat._params is not None else None
                ),
            }
        )
    return rows


def sharded_state_rows(model: nn.Module) -> list[dict]:
    state = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=False, cpu_offload=False)
    )
    rows = []
    for name, value in state.items():
        row = {
            "name": canonical(name),
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
        if isinstance(value, DTensor):
            row.update(
                global_shape=list(value.shape),
                local_shape=list(value.to_local().shape),
                placements=[str(p) for p in value.placements],
                mesh=value.device_mesh.mesh.tolist(),
            )
        elif hasattr(value, "local_shards"):
            row.update(
                global_shape=list(value.size()),
                local_shards=[
                    {
                        "shape": list(shard.tensor.shape),
                        "offsets": list(shard.metadata.shard_offsets),
                        "sizes": list(shard.metadata.shard_sizes),
                        "placement": str(shard.metadata.placement),
                    }
                    for shard in value.local_shards()
                ],
            )
        else:
            row["shape"] = list(value.shape)
        rows.append(row)
    return rows


def optimizer_for(model: nn.Module, multi_group: bool) -> torch.optim.Optimizer:
    if not multi_group:
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.003,
            weight_decay=0.01,
            foreach=False,
        )
    groups = {"embedding_up": [], "down": []}
    for name, param in model.named_parameters():
        if param.requires_grad:
            groups[parameter_group(name)].append(param)
    return torch.optim.AdamW(
        [
            {"params": groups["embedding_up"], "weight_decay": 0.0, "label": "embedding_up"},
            {"params": groups["down"], "weight_decay": 0.2, "label": "down"},
        ],
        lr=0.003,
        foreach=False,
    )


def optimizer_rows(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[dict]:
    by_id = {id(param): canonical(name) for name, param in model.named_parameters()}
    return [
        {
            "group": index,
            "label": group.get("label", "all"),
            "weight_decay": group["weight_decay"],
            "learning_rate": group["lr"],
            "parameters": [by_id[id(param)] for param in group["params"]],
            "parameter_ids": [id(param) for param in group["params"]],
        }
        for index, group in enumerate(optimizer.param_groups)
    ]


def full_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    state = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=False)
    )
    return {canonical(name): tensor.detach().clone() for name, tensor in state.items()}


def full_gradients(model: nn.Module, implementation: str) -> dict[str, torch.Tensor | None]:
    if implementation == "fsdp2":
        return {
            canonical(name): None
            if param.grad is None
            else param.grad.full_tensor().detach().clone()
            for name, param in model.named_parameters()
        }
    if implementation == "fsdp1_orig":
        with FSDP.summon_full_params(model, writeback=False, with_grads=True):
            return {
                canonical(name): None if param.grad is None else param.grad.detach().clone()
                for name, param in model.named_parameters()
            }
    result = {}
    for prefix, module in model.named_modules():
        if not isinstance(module, FSDP):
            continue
        handle = getattr(module, "_handle", None)
        if handle is None:
            continue
        flat = handle.flat_param
        if flat.grad is None:
            for name in flat._fqns:
                result[canonical(".".join(part for part in (prefix, name) if part))] = None
            continue
        # use_orig_params=False does not support summon_full_params(with_grads=True).
        # Read the already-reduced equal-sized local flat gradient, gather ONLY
        # for this oracle, strip final shard padding, and decode verified metadata.
        full = torch.empty(
            flat.grad.numel() * dist.get_world_size(), device=flat.device, dtype=flat.dtype
        )
        dist.all_gather_into_tensor(full, flat.grad.detach().reshape(-1).contiguous())
        if any(flat._is_padding_mask):
            raise RuntimeError("Unexpected intra-flat alignment padding in use_orig_params=False")
        offset = 0
        for name, shape, numel in zip(flat._fqns, flat._shapes, flat._numels, strict=True):
            qualified = canonical(".".join(part for part in (prefix, name) if part))
            result[qualified] = full[offset : offset + numel].reshape(shape).clone()
            offset += numel
    return result


def error_row(name: str, observed, expected, atol: float, rtol: float) -> dict:
    if observed is None or expected is None:
        return {"name": name, "pass": observed is None and expected is None, "max_abs": None}
    same_shape = tuple(observed.shape) == tuple(expected.shape)
    if not same_shape:
        return {
            "name": name,
            "pass": False,
            "shape": list(observed.shape),
            "expected_shape": list(expected.shape),
            "max_abs": None,
        }
    return {
        "name": name,
        "pass": bool(torch.allclose(observed, expected, atol=atol, rtol=rtol)),
        "max_abs": float((observed - expected).abs().max().item()),
        "max_expected_abs": float(expected.abs().max().item()),
    }


def frozen_gradient_row(name: str, observed, expected_shape: tuple) -> dict:
    """Accept a strict zero view exposed ONLY by full-gradient observation.

    FSDP1 _use_unsharded_grad_views assigns all flat gradient slices, including
    frozen parameters, inside summon_full_params(with_grads=True). After leaving
    this context _use_sharded_grad_views restores frozen .grad=None. Therefore a
    full observer's zero Tensor and the dense model's None express the same lack
    of trainable gradient. Nonzero/NaN/wrong-shape frozen views still fail.
    """
    if observed is None:
        return {"name": name, "pass": True, "observed_representation": "None"}
    nonzero = int(torch.count_nonzero(observed).item())
    finite = bool(torch.isfinite(observed).all().item())
    return {
        "name": name,
        "pass": tuple(observed.shape) == expected_shape and nonzero == 0 and finite,
        "observed_representation": "full_observer_zero_tensor_view",
        "shape": list(observed.shape),
        "expected_shape": list(expected_shape),
        "nonzero_elements": nonzero,
        "finite": finite,
        "max_abs": float(observed.abs().max().item()),
        "scope": "Full-gradient observer; optimizer boundary checked independently",
    }


def dense_step(
    dense: nn.Module, ids: torch.Tensor, targets: torch.Tensor, optimizer: torch.optim.Optimizer
) -> tuple:
    output = dense(ids)
    loss = F.mse_loss(output, targets)
    loss.backward()
    for param in dense.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad)
            param.grad.div_(dist.get_world_size())
    gradients = {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in dense.named_parameters()
    }
    optimizer.step()
    return output.detach(), loss.detach(), gradients


def run_case(
    args: argparse.Namespace, implementation: str, case: str, device: torch.device
) -> dict:
    rank = dist.get_rank()
    frozen = case == "R2"
    base = make_model(args.seed, device, frozen=frozen)
    dense = copy.deepcopy(base)
    before = {name: {"id": id(p), "shape": list(p.shape)} for name, p in base.named_parameters()}
    result = {
        "run_id": args.run_id,
        "experiment_id": f"Q9_{case}_{implementation}",
        "implementation": implementation,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "fsdp_group_size": dist.get_world_size(),
        "tp_group_size": 1,
        "sp_group_size": 1,
        "model_id": "Q9_TinyNet_embedding31x16_up16x24_down24x8",
        "precision": "fp32",
        "optimizer": "AdamW(foreach=False)",
        "seed": args.seed,
        "local_microbatch": 4,
        "seq_len": 8,
        "accumulation_steps": 1,
        "unique_training_tokens_per_step": 4 * 8 * dist.get_world_size(),
        "input_semantics": "rank-dependent DP inputs; mean gradients over equal-sized replicas",
        "git_commit": git_commit(),
        "config_hash": hashlib.sha256(
            json.dumps(
                {"case": case, "implementation": implementation, "seed": args.seed}, sort_keys=True
            ).encode()
        ).hexdigest(),
        "parameter_metadata": [],
        "original_parameter_metadata": parameter_rows(base, before, "before_wrap"),
    }
    if implementation == "fsdp1_flat" and frozen:
        try:
            sharded_model(base, implementation, case, device)
        except ValueError as exc:
            expected = "uniform `requires_grad`" in str(exc)
            result.update(
                status="PASS" if expected else "FAIL",
                outcome="EXPECTED_LIMITATION" if expected else "UNEXPECTED_ERROR",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                correctness=(
                    "Not trained: expected pre-collective mixed-frozen root-flatten rejection"
                ),
            )
            return result
        result.update(status="FAIL", outcome="EXPECTED_ERROR_NOT_RAISED")
        return result
    if implementation == "fsdp1_flat" and case == "R3":
        root_flat_metadata = []
        probe = make_model(args.seed, device)
        probe_before = {
            name: {"id": id(p), "shape": list(p.shape)} for name, p in probe.named_parameters()
        }
        probe = sharded_model(probe, implementation, "R1", device)
        root_flat_metadata = parameter_rows(probe, probe_before, "single_root_flat_group_probe")
        if (
            len(root_flat_metadata) != 1
            or len({parameter_group(name) for name in root_flat_metadata[0]["flat_original_names"]})
            != 2
        ):
            raise RuntimeError("Single-root flat optimizer group probe did not span both groups")
        result["single_root_flat_optimizer_group_limitation"] = {
            "status": "UNREPRESENTABLE_AT_THIS_WRAPPING_GRANULARITY",
            "reason": "One optimizer leaf spans original down and embedding/up parameter groups",
            "metadata": root_flat_metadata,
            "workaround_used": (
                "Separate homogeneous FSDP units for embedding/up/down before root wrapping"
            ),
        }
        del probe
    model = sharded_model(base, implementation, case, device)
    result["parameter_metadata"] += parameter_rows(model, before, "after_wrap")
    if implementation.startswith("fsdp1"):
        result["internal_flat_handles_after_wrap"] = internal_flat_rows(model)
        with FSDP.summon_full_params(model, writeback=False):
            result["parameter_metadata"] += parameter_rows(model, before, "summon_full_params")
    multi_group = case in {"R3", "R4"}
    optimizer = optimizer_for(model, multi_group)
    dense_optimizer = optimizer_for(dense, multi_group)
    result["optimizer_groups"] = optimizer_rows(model, optimizer)
    result["dense_optimizer_groups"] = optimizer_rows(dense, dense_optimizer)
    ids, targets = data_for_rank(args.seed, rank, device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    output = model(ids)
    loss = F.mse_loss(output, targets)
    loss.backward()
    result["parameter_metadata"] += parameter_rows(model, before, "after_backward")
    observed_gradients = full_gradients(model, implementation)
    expected_output, expected_loss, expected_gradients = dense_step(
        dense, ids, targets, dense_optimizer
    )
    checks = [error_row("output", output.detach(), expected_output, args.atol, args.rtol)]
    checks.append(error_row("loss", loss.detach(), expected_loss, args.atol, args.rtol))
    dense_parameters = dict(dense.named_parameters())
    for name in sorted(expected_gradients):
        if not dense_parameters[name].requires_grad and expected_gradients[name] is None:
            checks.append(
                frozen_gradient_row(
                    f"gradient/{name}",
                    observed_gradients.get(name),
                    tuple(dense_parameters[name].shape),
                )
            )
            continue
        checks.append(
            error_row(
                f"gradient/{name}",
                observed_gradients.get(name),
                expected_gradients[name],
                args.atol,
                args.rtol,
            )
        )
    if set(observed_gradients) != set(expected_gradients):
        checks.append(
            {
                "name": "gradient_keys",
                "pass": False,
                "observed": sorted(observed_gradients),
                "expected": sorted(expected_gradients),
            }
        )
    optimizer_parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            checks.append(
                {
                    "name": f"frozen_optimizer_boundary/{canonical(name)}",
                    "pass": parameter.grad is None and id(parameter) not in optimizer_parameter_ids,
                    "gradient_is_none": parameter.grad is None,
                    "excluded_from_optimizer": id(parameter) not in optimizer_parameter_ids,
                }
            )
    optimizer.step()
    torch.cuda.synchronize(device)
    result["step_and_oracle_wall_ms"] = (time.perf_counter() - start) * 1000
    observed_parameters = full_parameters(model)
    result["sharded_state_dict_representation_after_update"] = sharded_state_rows(model)
    for name, param in dense.named_parameters():
        checks.append(
            error_row(
                f"updated_parameter/{name}",
                observed_parameters[name],
                param.detach(),
                args.atol,
                args.rtol,
            )
        )
    result["parameter_metadata"] += parameter_rows(model, before, "after_update")
    if frozen:
        checks.append(
            {
                "name": "frozen_embedding_unchanged",
                "pass": bool(
                    torch.equal(
                        observed_parameters["embedding.weight"],
                        make_model(args.seed, device).embedding.weight,
                    )
                ),
            }
        )
    result.update(
        status="PASS" if all(check["pass"] for check in checks) else "FAIL",
        outcome="TRAINED_AND_VERIFIED",
        loss=float(loss.detach().item()),
        dense_loss=float(expected_loss.item()),
        finite_gradients=all(
            g is None or bool(torch.isfinite(g).all().item()) for g in observed_gradients.values()
        ),
        checks=checks,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        timing_scope=(
            "Includes dense oracle and observer collectives; not a performance measurement"
        ),
    )
    del optimizer, model, dense_optimizer, dense
    return result


def clone_cpu_model_state(state: dict) -> dict[str, torch.Tensor]:
    """Independent CPU value reference; the state-dict setter may mutate its input."""
    if not isinstance(state, dict) or any(not isinstance(name, str) for name in state):
        raise TypeError("R4 checkpoint model state must map parameter names to tensors")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise TypeError("R4 model-only checkpoint contains a non-Tensor value")
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def model_state_devices(state: dict) -> dict:
    return {
        name: {
            "device": str(tensor.device),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "tensor_id": id(tensor),
        }
        for name, tensor in state.items()
    }


def roundtrip(args: argparse.Namespace, implementation: str, device: torch.device) -> dict:
    """Optional full MODEL state portability only; optimizer resume is not claimed."""
    start = time.perf_counter()
    rank = dist.get_rank()
    path = Path(args.checkpoint_dir) / f"{implementation}.pt"
    if args.roundtrip_action == "save" and path.exists():
        raise FileExistsError(f"Refusing to replace checkpoint {path}")
    model = sharded_model(make_model(args.seed, device), implementation, "R4", device)
    options = StateDictOptions(full_state_dict=True, cpu_offload=False)
    timings = {}
    input_state_observation = None
    if args.roundtrip_action == "save":
        ids, targets = data_for_rank(args.seed, rank, device)
        optimizer = optimizer_for(model, True)
        F.mse_loss(model(ids), targets).backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        export_start = time.perf_counter()
        full_state = get_model_state_dict(model, options=options)
        torch.cuda.synchronize(device)
        timings["state_export_wall_ms"] = (time.perf_counter() - export_start) * 1000
        if rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            io_start = time.perf_counter()
            torch.save(
                {
                    "model": {name: tensor.detach().cpu() for name, tensor in full_state.items()},
                    "world_size": dist.get_world_size(),
                    "seed": args.seed,
                    "optimizer_state_saved": False,
                },
                path,
            )
            timings["checkpoint_io_wall_ms"] = (time.perf_counter() - io_start) * 1000
        dist.barrier()
        status = "PASS"
        checks = []
        origin_world_size = dist.get_world_size()
    else:
        io_start = time.perf_counter()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        timings["checkpoint_io_wall_ms"] = (time.perf_counter() - io_start) * 1000
        # Public set_model_state_dict may replace CPU mapping values with CUDA
        # tensors. Preserve the original checkpoint values in a separate clone
        # that the API never receives, then give the API its own independent copy.
        reference_state = clone_cpu_model_state(checkpoint["model"])
        api_input_state = clone_cpu_model_state(reference_state)
        api_input_before = model_state_devices(api_input_state)
        torch.cuda.synchronize(device)
        import_start = time.perf_counter()
        incompat = set_model_state_dict(model, model_state_dict=api_input_state, options=options)
        torch.cuda.synchronize(device)
        timings["state_import_wall_ms"] = (time.perf_counter() - import_start) * 1000
        api_input_after = model_state_devices(api_input_state)
        input_state_observation = {
            "before": api_input_before,
            "after": api_input_after,
            "mapping_keys_unchanged": api_input_before.keys() == api_input_after.keys(),
            "changed_device_names": [
                name
                for name in api_input_before
                if name in api_input_after
                and api_input_before[name]["device"] != api_input_after[name]["device"]
            ],
            "replaced_tensor_names": [
                name
                for name in api_input_before
                if name in api_input_after
                and api_input_before[name]["tensor_id"] != api_input_after[name]["tensor_id"]
            ],
            "reference_device": "cpu",
            "reference_is_separate_from_api_input": True,
        }
        origin_world_size = checkpoint["world_size"]
        full_state = get_model_state_dict(model, options=options)
        checks = [
            error_row(
                f"restored_parameter/{name}",
                tensor.detach().cpu(),
                reference_state[name],
                0.0,
                0.0,
            )
            for name, tensor in full_state.items()
        ]
        checks.append(
            {
                "name": "state_dict_keys",
                "pass": not incompat.missing_keys
                and not incompat.unexpected_keys
                and full_state.keys() == reference_state.keys(),
            }
        )
        dense = make_model(args.seed, device)
        dense.load_state_dict(reference_state)
        ids, _ = data_for_rank(args.seed, rank, device)
        with torch.no_grad():
            checks.append(
                error_row("restored_forward", model(ids), dense(ids), args.atol, args.rtol)
            )
        status = "PASS" if all(check["pass"] for check in checks) else "FAIL"
    torch.cuda.synchronize(device)
    return {
        "run_id": args.run_id,
        "experiment_id": f"Q9_R4_{implementation}_{args.roundtrip_action}",
        "implementation": implementation,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "origin_world_size": origin_world_size,
        "action": args.roundtrip_action,
        "status": status,
        "checkpoint_path": str(path),
        "checkpoint_bytes": path.stat().st_size,
        "state_format": "public FULL model state_dict; portable CPU tensors",
        "scope": "model only; no sharded optimizer-state resume or distributed checkpoint claim",
        "entire_action_wall_ms": (time.perf_counter() - start) * 1000,
        "state_api_timings": timings,
        "input_state_api_observation": input_state_observation,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("all", *IMPLEMENTATIONS), default="all")
    parser.add_argument(
        "--cases", nargs="+", choices=("R1", "R2", "R3"), default=["R1", "R2", "R3"]
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="q09_param_representation")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--atol", type=float, default=3e-6)
    parser.add_argument("--rtol", type=float, default=3e-5)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="API/CPU model audit; no GPU experiment is marked PASS",
    )
    parser.add_argument("--roundtrip-action", choices=("save", "load"))
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args()
    if args.roundtrip_action and not args.checkpoint_dir:
        parser.error("--roundtrip-action requires --checkpoint-dir")
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rank = int(os.environ.get("RANK", "0"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / f"environment.rank{rank}.json").exists():
        raise FileExistsError(f"Experiment output already exists for rank {rank}: {output_dir}")
    check_api()
    write_json(output_dir / f"environment.rank{rank}.json", environment(args))
    if args.inspect_only:
        model = make_model(args.seed, torch.device("cpu"))
        write_json(
            output_dir / f"results.rank{rank}.json",
            {
                "status": "NOT_RUN",
                "reason": "inspect-only: no distributed GPU experiment executed",
                "cpu_parameter_shapes": {
                    name: list(p.shape) for name, p in model.named_parameters()
                },
            },
        )
        return 0
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        write_json(
            output_dir / f"results.rank{rank}.json",
            {"status": "BLOCKED", "reason": "CUDA GPU and NCCL-enabled PyTorch required"},
        )
        return 2
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=args.timeout_seconds), device_id=device
    )
    results = []
    implementations = IMPLEMENTATIONS if args.implementation == "all" else (args.implementation,)
    try:
        for implementation in implementations:
            if args.roundtrip_action:
                results.append(roundtrip(args, implementation, device))
                write_json(output_dir / f"results.rank{rank}.json", results)
                dist.barrier()
                continue
            for case in args.cases:
                result = run_case(args, implementation, case, device)
                results.append(result)
                write_json(output_dir / f"results.rank{rank}.json", results)
                torch.cuda.synchronize(device)
                dist.barrier()
        rows = [
            {
                key: result.get(key)
                for key in (
                    "run_id",
                    "experiment_id",
                    "implementation",
                    "rank",
                    "world_size",
                    "status",
                    "outcome",
                    "loss",
                    "finite_gradients",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                )
            }
            for result in results
        ]
        with (output_dir / f"summary.rank{rank}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        metadata = [
            {"experiment_id": result["experiment_id"], "rank": rank, **row}
            for result in results
            for row in result.get("parameter_metadata", [])
        ]
        write_json(output_dir / f"parameters.rank{rank}.json", metadata)
        if metadata:
            fieldnames = sorted({key for row in metadata for key in row})
            with (output_dir / f"parameters.rank{rank}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    {
                        key: json.dumps(value) if isinstance(value, (list, dict)) else value
                        for key, value in row.items()
                    }
                    for row in metadata
                )
        return 0 if all(result["status"] == "PASS" for result in results) else 1
    except Exception as exc:
        write_json(
            output_dir / f"failure.rank{rank}.json",
            {
                "status": "FAIL",
                "type": type(exc).__name__,
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
