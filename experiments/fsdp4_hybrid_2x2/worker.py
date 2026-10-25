"""Experiment-only audits around unchanged trainers and official fully_shard.

All source and full-parameter reconstruction is outside the training timer.
Native TP1 skips only the TP plan; FSDP remains the official implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(PROJECT),
    str(PROJECT / "src"),
    str(PROJECT / "experiments/fsdp_interview/scripts"),
]
from experiments.fsdp4_hybrid_2x2.contracts import CASES, groups, save, sha  # noqa:E402


def tensor_record(value):
    import torch

    value = value.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=h.hexdigest())


def native_implementation_proof(model):
    """Observe real official wrapper types and imported definition bytes."""
    import inspect

    from torch.distributed.fsdp import FSDPModule, fully_shard
    from torch.distributed.tensor import DTensor

    unwrapped = inspect.unwrap(fully_shard)
    definition = inspect.getsourcefile(unwrapped)
    wrappers = [name for name, module in model.named_modules() if isinstance(module, FSDPModule)]
    proof = dict(
        root_is_official_fsdp_module=isinstance(model, FSDPModule),
        model_type_module=type(model).__module__,
        model_type_qualname=type(model).__qualname__,
        model_mro=[f"{cls.__module__}.{cls.__qualname__}" for cls in type(model).__mro__],
        fsdp_wrapped_module_count=len(wrappers),
        fsdp_wrapped_module_names=wrappers,
        all_parameters_official_dtensor=all(isinstance(p, DTensor) for p in model.parameters()),
        fully_shard_module=fully_shard.__module__,
        fully_shard_qualname=fully_shard.__qualname__,
        fully_shard_definition_source=definition,
        fully_shard_definition_sha256=sha(definition) if definition else None,
        scope="Actual returned model wrappers and loaded source bytes; "
        "external official Git match separate",
    )
    if not proof["root_is_official_fsdp_module"] or not proof["all_parameters_official_dtensor"]:
        raise AssertionError("Native did not return actual official FSDPModule/DTensor parameters")
    return proof


@contextlib.contextmanager
def native_tp1_scope():
    """Never route Native through a custom registry/materializer/collective."""
    import sharded_runtime.training.qwen2_native_baseline as native

    original = native.parallelize_native_qwen

    def plan(model, tp_mesh, *, sp_enabled):
        if tp_mesh.size() == 1:
            if sp_enabled:
                raise ValueError("TP1 must disable SP")
            model.tp_degree, model.tp_rank, model.sp_enabled = 1, 0, False
            model._p1_tp_plan_disabled = True
            return
        return original(model, tp_mesh, sp_enabled=sp_enabled)

    native.parallelize_native_qwen = plan
    try:
        yield
    finally:
        native.parallelize_native_qwen = original


def placement_audit(case, role_groups, *, device):
    import torch
    import torch.distributed as dist

    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    manifest_path = os.environ["SHARDTRAIN_PLACEMENT_MANIFEST"]
    manifest = json.loads(Path(manifest_path).read_text())
    m0 = manifest["placements"]["M0"]
    uuids = m0["ordered_gpu_uuids"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(uuids) or local != rank:
        raise AssertionError("Four-rank M0 UUID visibility/LOCAL_RANK mismatch")
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    if uuid != uuids[rank]:
        raise AssertionError("Actual CUDA UUID differs from M0")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    by_uuid = dict(tuple(p.strip() for p in line.split(",")) for line in query.stdout.splitlines())
    pci = by_uuid[uuid]
    devices = {d["uuid"]: d for d in m0["ranks"]}
    if pci.lower() != devices[uuid]["pci_bus_id"].lower():
        raise AssertionError("Actual PCI differs from physical UUID record")
    expected, observed = groups(case), {}
    for role, pg in role_groups.items():
        axis = role.removesuffix("_grad")
        members = next(g for g in expected[axis] if rank in g)
        actual = dist.get_process_group_ranks(pg)
        if actual != members or dist.get_rank(pg) != members.index(rank):
            raise AssertionError(f"PG ordered identity mismatch {role}: {actual}/{members}")
        observed[role] = dict(
            members=actual,
            group_rank=dist.get_rank(pg),
            process_group_name=pg.group_name,
            physical_uuids=[uuids[r] for r in actual],
        )
    return dict(
        status="PASS",
        case=case,
        mapping_id="M0",
        rank=rank,
        local_rank=local,
        actual_uuid=uuid,
        actual_pci_bus_id=pci,
        actual_pg_members=observed,
        placement_manifest_sha256=sha(manifest_path),
        source_manifest_sha256=os.environ["SHARDTRAIN_SOURCE_MANIFEST_SHA256"],
    )


class InitialStateAudit:
    """Hash actual source and actual reconstructed globals, not seed strings."""

    def __init__(self):
        self.source = {}

    def capture(self, name, value):
        row = tensor_record(value)
        if name in self.source and self.source[name] != row:
            raise AssertionError("Canonical source changed within initialization")
        self.source[name] = row

    def self_reconstructed(self, runtime):
        import torch
        import torch.distributed as dist

        reconstructed = {}
        with torch.no_grad():
            for name, handle in sorted(runtime.registry.items()):
                spec = handle.spec
                shards = [
                    torch.empty_like(handle.local_shard) for _ in range(runtime.materializer.degree)
                ]
                dist.all_gather(shards, handle.local_shard, group=runtime.materializer.group)
                local = torch.cat([v[: spec.valid_len(r)] for r, v in enumerate(shards)]).view(
                    spec.tp_local_shape
                )
                part = runtime.qwen_partitions[name]
                if part.dim is not None and part.degree > 1:
                    pieces = [torch.empty_like(local) for _ in range(part.degree)]
                    dist.all_gather(pieces, local, group=runtime.tp.process_group)
                    local = torch.cat(pieces, dim=part.dim)
                reconstructed[name] = tensor_record(local)
                del local, shards
        return self.finish(reconstructed)

    def native_reconstructed(self, model):
        from torch.distributed.tensor import DTensor

        reconstructed = {}
        for name, parameter in sorted(model.named_parameters()):
            if not isinstance(parameter, DTensor):
                raise AssertionError("Native parameter is not an official FSDP DTensor")
            reconstructed[name] = tensor_record(parameter.full_tensor())
        return self.finish(reconstructed)

    def finish(self, reconstructed):
        if reconstructed != self.source:
            bad = [
                n
                for n in self.source.keys() | reconstructed.keys()
                if self.source.get(n) != reconstructed.get(n)
            ]
            raise AssertionError(f"Actual global parameter differs from canonical source: {bad}")
        return dict(
            status="PASS",
            parameters=self.source,
            parameter_count=len(self.source),
            global_state_sha256=hashlib.sha256(
                json.dumps(self.source, sort_keys=True).encode()
            ).hexdigest(),
            scope="Every actual global parameter reconstructed; "
            "initialization only, not gradient equivalence",
        )


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--communication-audit", action="store_true")
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    import train_workload as workload

    import sharded_runtime.runtime.qwen2_adapter as adapter
    import sharded_runtime.training.qwen2_native_baseline as native

    case = CASES[args.case]
    parsed = workload.parse()
    from sharded_runtime.config import load_config

    config = load_config(parsed.config)
    if (
        config.parallel.fsdp_degree,
        config.parallel.tp_degree,
        config.parallel.sp_enabled,
        config.training.local_batch_size,
    ) != (case["F"], case["T"], case["SP"], case["microbatch"]):
        raise ValueError("Case/config identity mismatch")
    state, audit = InitialStateAudit(), {}
    original_source, original_init = adapter._source_tensor, native.initialize_same_source
    original_register, original_native = (
        workload.ShardedRuntime.register,
        workload.build_native_qwen,
    )
    trainers, old_initializers = [], []
    for cls in [workload.QwenTrainer, workload.NativeQwenTrainer]:
        original = cls.__init__
        old_initializers.append((cls, original))

        def initialize_trainer(trainer, *v, _old=original, **kw):
            _old(trainer, *v, **kw)
            trainers.append(trainer)

        cls.__init__ = initialize_trainer

    def source(model, name, parameter):
        value = original_source(model, name, parameter)
        state.capture(name, value)
        return value

    def initialize(model, seed, checkpoint=None):
        original_init(model, seed, checkpoint)
        for name, parameter in model.named_parameters():
            state.capture(name, parameter)

    def register(runtime, model, context, **kwargs):
        result = original_register(runtime, model, context, **kwargs)
        audit.update(
            placement_audit(
                args.case,
                {
                    "fsdp": context.fsdp_group,
                    "tp": context.tp_group,
                    "fsdp_grad": context.fsdp_grad_group,
                    "tp_grad": context.tp_grad_group,
                },
                device=runtime.device,
            )
        )
        audit["initial_global_state"] = state.self_reconstructed(runtime)
        audit["initial_local_state"] = {
            n: tensor_record(h.local_shard) for n, h in runtime.registry.items()
        }
        return result

    def build(cfg, mesh, device, **kwargs):
        model, meta = original_native(cfg, mesh, device, **kwargs)
        audit["native_implementation"] = native_implementation_proof(model)
        audit.update(
            placement_audit(
                args.case,
                {"fsdp": mesh.get_group("fsdp"), "tp": mesh.get_group("tp")},
                device=device,
            )
        )
        audit["initial_global_state"] = state.native_reconstructed(model)
        audit["initial_local_state"] = {
            n: tensor_record(v.to_local()) for n, v in model.named_parameters()
        }
        meta.update(
            actual_fsdp_api="torch.distributed.fsdp.fully_shard",
            tp_plan_disabled=case["T"] == 1,
            tp_sp_enabled=case["SP"],
        )
        return model, meta

    destination = Path(parsed.output_dir) / f"placement.rank{os.environ['RANK']}.json"
    if destination.exists():
        raise FileExistsError(destination)
    adapter._source_tensor, native.initialize_same_source = source, initialize
    workload.ShardedRuntime.register, workload.build_native_qwen = register, build
    probe = None
    old_steps = []
    if args.communication_audit:
        from .collective_probe import CollectiveProbe

        probe = CollectiveProbe()
        for cls in [workload.QwenTrainer, workload.NativeQwenTrainer]:
            old = cls.train_step
            old_steps.append((cls, old))

            def train_step(trainer, *v, _old=old, **kw):
                probe.training_step = trainer.step_id
                try:
                    return _old(trainer, *v, **kw)
                finally:
                    probe.training_step = None

            cls.train_step = train_step
    try:
        with native_tp1_scope(), probe if probe is not None else contextlib.nullcontext():
            result = workload.run(parsed)
        if len(trainers) != 1:
            raise AssertionError("Expected exactly one real training optimizer")
        audit["optimizer_end"] = optimizer_end_audit(
            trainers[0].optimizer, parsed.warmup + parsed.steps
        )
        audit["status"] = result["status"]
    except BaseException as error:
        audit.update(status="FAIL", error=repr(error))
        raise
    finally:
        adapter._source_tensor, native.initialize_same_source = original_source, original_init
        workload.ShardedRuntime.register, workload.build_native_qwen = (
            original_register,
            original_native,
        )
        for cls, old in old_steps:
            cls.train_step = old
        for cls, old in old_initializers:
            cls.__init__ = old
        save(destination, audit)
        if probe is not None:
            save(
                Path(parsed.output_dir) / f"collective_probe.rank{os.environ['RANK']}.json",
                dict(
                    status="CAPTURED",
                    rows=probe.rows,
                    case=args.case,
                    coverage="PARTIAL_DISPATCHER_INVOCATIONS;"
                    "autograd-thread/async coverage requires actual crosscheck",
                    actual_process_groups=audit.get("actual_pg_members"),
                    scope="Capacity-only actual dispatcher calls; no formal throughput claim; "
                    "calls outside trainer excluded",
                ),
            )


def optimizer_end_audit(optimizer, expected):
    """Read every actual Adam leaf counter and local bytes once, after all timers."""
    from collections import Counter

    import torch

    steps, state_bytes, cpu_step_bytes = [], 0, 0
    param_bytes, grad_bytes, leaves = 0, 0, 0

    def local(value):
        return value.to_local() if hasattr(value, "to_local") else value

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            leaves += 1
            value = local(parameter)
            param_bytes += value.numel() * value.element_size()
            if parameter.grad is not None:
                grad = local(parameter.grad)
                grad_bytes += grad.numel() * grad.element_size()
            state = optimizer.state[parameter]
            step = state.get("step")
            if step is None:
                raise AssertionError("Adam leaf has no actual step counter")
            steps.append(int(local(step).item()) if isinstance(step, torch.Tensor) else int(step))
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    value = local(value)
                    size = value.numel() * value.element_size()
                    if key == "step" and value.device.type == "cpu":
                        cpu_step_bytes += size
                    else:
                        state_bytes += size
    if len(steps) != leaves or set(steps) != {expected}:
        raise AssertionError(
            f"Actual Adam leaf counters differ from expected{expected}: {set(steps)}"
        )
    return dict(
        status="PASS",
        leaf_count=leaves,
        expected_updates=expected,
        step_min=min(steps),
        step_max=max(steps),
        step_histogram=dict(Counter(steps)),
        parameter_bytes=param_bytes,
        gradient_bytes=grad_bytes,
        optimizer_gpu_state_bytes_excluding_cpu_steps=state_bytes,
        optimizer_cpu_step_scalar_bytes=cpu_step_bytes,
        scope="Actual post-timing optimizer states; not loop-index inferred updates",
    )


if __name__ == "__main__":
    main()
