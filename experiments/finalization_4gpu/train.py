"""Reuse the existing workload; add placement audit outside the measured step.

No production math, mesh, trainer, or Runtime implementation is replaced here.
The profile-only collector intentionally disables ATen tracing and alloc history.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "experiments/fsdp_interview/scripts"))

from experiments.finalization_4gpu.placement import (  # noqa: E402
    load,
    placement_for,
    worker_expectation,
)


def digest_parameters(named_parameters):
    """Hash real initial local shards in bounded CPU chunks, before timing starts."""
    import torch

    result = hashlib.sha256()
    inventory = []
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        tensor = parameter.detach()
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        metadata = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        result.update(json.dumps(metadata, sort_keys=True).encode())
        flat = tensor.contiguous().reshape(-1)
        elements = max(1, 16 * 2**20 // tensor.element_size())
        for start in range(0, flat.numel(), elements):
            raw = flat[start : start + elements].view(torch.uint8).cpu().numpy().tobytes()
            result.update(raw)
        inventory.append(metadata)
    return {
        "sha256": result.hexdigest(),
        "inventory": inventory,
        "scope": "initial local shards, not global dense state",
    }


class LeanProfileCapture:
    """States-only snapshots: no allocator history, traceback, or all-ATen NVTX."""

    def __init__(self, output, rank, device, **_kwargs):
        self.output, self.rank, self.device = Path(output), rank, device
        self.closed = False

    def _snapshot(self, phase):
        import torch

        path = self.output / f"memory.{phase}.rank{self.rank}.pickle"
        with path.open("xb") as stream:
            pickle.dump(torch.cuda.memory._snapshot(), stream, protocol=pickle.HIGHEST_PROTOCOL)

    def start(self):
        self._snapshot("start")

    def step(self, _step):
        pass

    def close(self):
        if not self.closed:
            self._snapshot("end")
            self.closed = True


def main():
    import torch
    import torch.distributed as dist
    import train_workload as workload

    manifest = load(os.environ["SHARDTRAIN_PLACEMENT_MANIFEST"])
    mapping_id = os.environ["SHARDTRAIN_MAPPING_ID"]
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    expected = worker_expectation(manifest, mapping_id, rank)
    if local != expected["local_rank"] or int(os.environ["WORLD_SIZE"]) != 4:
        raise RuntimeError("This placement requires exactly four single-node logical ranks")
    torch.cuda.set_device(local)
    properties = torch.cuda.get_device_properties(local)
    actual_uuid = str(getattr(properties, "uuid", "UNKNOWN"))
    if actual_uuid != "UNKNOWN" and not actual_uuid.startswith("GPU-"):
        actual_uuid = "GPU-" + actual_uuid
    if actual_uuid != expected["uuid"]:
        raise RuntimeError(f"GPU UUID mismatch rank{rank}: {actual_uuid} != {expected['uuid']}")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    actual_devices = dict(
        tuple(value.strip() for value in row.split(","))
        for row in query.stdout.splitlines()
        if row.strip()
    )
    actual_pci = actual_devices.get(actual_uuid)
    if not actual_pci or (
        expected.get("pci_bus_id") and actual_pci.lower() != expected["pci_bus_id"].lower()
    ):
        raise RuntimeError("Actual UUID/PCI identity differs from the physical placement manifest")
    args = workload.parse()
    audit = {
        "status": "INITIALIZING",
        "mapping_id": mapping_id,
        "expected": expected,
        "actual_uuid": actual_uuid,
        "actual_pci_bus_id": actual_pci,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_mesh": [[0, 1], [2, 3]],
        "ordered_groups": placement_for(manifest, mapping_id)["groups"],
        "source_manifest_sha256": os.environ["SHARDTRAIN_SOURCE_MANIFEST_SHA256"],
        "placement_manifest_sha256": hashlib.sha256(
            Path(os.environ["SHARDTRAIN_PLACEMENT_MANIFEST"]).read_bytes()
        ).hexdigest(),
        "initial_local_state": None,
        "actual_pg_members": {},
        "profile_mode": "cuda+essential_nvtx; allocator history off; states-only snapshots"
        if args.profile
        else "off",
    }

    def inspect_groups(groups):
        for role, group in groups.items():
            observed = dist.get_process_group_ranks(group)
            axis = role.removesuffix("_grad")
            expected_members = next(
                pair for pair in manifest["logical_groups"][axis] if rank in pair
            )
            if observed != expected_members or dist.get_rank(group) != expected_members.index(rank):
                raise RuntimeError(f"Ordered logical PG mismatch: {role}/{observed}")
            audit["actual_pg_members"][role] = {
                "members": observed,
                "group_rank": dist.get_rank(group),
            }

    original_register = workload.ShardedRuntime.register

    def audited_register(runtime, model, context, **kwargs):
        result = original_register(runtime, model, context, **kwargs)
        inspect_groups(
            {
                "tp": context.tp_group,
                "fsdp": context.fsdp_group,
                "tp_grad": context.tp_grad_group,
                "fsdp_grad": context.fsdp_grad_group,
            }
        )
        audit["initial_local_state"] = digest_parameters(
            (handle.spec.param_id, handle.local_shard) for handle in runtime.registry.values()
        )
        return result

    original_native = workload.build_native_qwen

    def audited_native(config, mesh, device, **kwargs):
        model, native = original_native(config, mesh, device, **kwargs)
        inspect_groups({"tp": mesh.get_group("tp"), "fsdp": mesh.get_group("fsdp")})
        audit["initial_local_state"] = digest_parameters(model.named_parameters())
        return model, native

    workload.ShardedRuntime.register = audited_register
    workload.build_native_qwen = audited_native
    if args.profile:
        workload.ProfileCapture = LeanProfileCapture
        args.profile_detail = "low_overhead"  # Select our states-only collector branch.
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"placement.rank{rank}.json"
    if destination.exists():
        raise FileExistsError(destination)
    try:
        payload = workload.run(args)
        audit["status"] = payload["status"]
    except BaseException as error:
        audit.update(status="FAIL", error=repr(error))
        raise
    finally:
        workload.ShardedRuntime.register = original_register
        workload.build_native_qwen = original_native
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(audit, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
