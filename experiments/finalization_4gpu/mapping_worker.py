"""Placement gates and reversible observers around the unchanged HF FP32 oracle.

CUDA UUIDs are observed after LOCAL_RANK binding; logical seeds never use UUIDs.
This is correctness instrumentation, not a performance workload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from placement import load, validate_manifest, visible_order, worker_expectation  # noqa: E402


def import_fixture(relative, name):
    path = PROJECT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{value.dtype}/{tuple(value.shape)}".encode())
    digest.update(value.view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def input_hashes(factory, steps, f):
    return [
        {
            "step": step,
            "fsdp_coordinate": replica,
            "tensors": [tensor_hash(value) for value in factory(replica, step)],
        }
        for step in range(steps)
        for replica in range(f)
    ]


def normalize_uuid(value):
    value = str(value)
    return value if value.startswith("GPU-") else "GPU-" + value


def verify_source_metadata(payload):
    path = os.environ.get("SHARDTRAIN_SOURCE_MANIFEST")
    expected = os.environ.get("SHARDTRAIN_SOURCE_MANIFEST_SHA256")
    if not path or not expected or not Path(path).is_file():
        raise RuntimeError("BLOCKED_ENVIRONMENT: saved candidate source manifest is required")
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise AssertionError("Candidate source manifest content differs from the launcher hash")
    payload["source_manifest_sha256"] = actual
    payload["source_manifest_path"] = path


def verify_context(
    context, manifest, mapping_id, group_roles=("tp", "fsdp", "tp_grad", "fsdp_grad")
):
    """No guessed UUID/PCI identity and no physical-index-dependent random seed."""
    import torch
    import torch.distributed as dist

    validate_manifest(manifest)
    worker = context.worker
    expected = worker_expectation(manifest, mapping_id, worker.rank)
    if worker.world_size != 4 or worker.local_rank != worker.rank:
        raise AssertionError("Four single-node torchrun ranks with LOCAL_RANK=RANK required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != visible_order(manifest, mapping_id):
        raise AssertionError("CUDA_VISIBLE_DEVICES differs from the ordered manifest UUIDs")
    if context.device.type != "cuda" or torch.cuda.current_device() != expected["cuda_ordinal"]:
        raise AssertionError("LOCAL_RANK binding differs from manifest CUDA ordinal")
    props = torch.cuda.get_device_properties(context.device)
    uuid = getattr(props, "uuid", None)
    if uuid is None:
        raise RuntimeError("BLOCKED_ENVIRONMENT: installed Torch does not expose actual GPU UUID")
    uuid = normalize_uuid(uuid)
    if uuid != expected["uuid"]:
        raise AssertionError(f"Actual UUID {uuid} differs from {expected['uuid']}")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    devices = dict(
        tuple(piece.strip() for piece in line.split(",")) for line in query.stdout.splitlines()
    )
    pci = devices.get(uuid)
    if not pci or (expected.get("pci_bus_id") and pci.lower() != expected["pci_bus_id"].lower()):
        raise AssertionError("Actual UUID-to-PCI mapping differs from manifest")
    if (context.coordinate.fsdp, context.coordinate.tp) != (expected["f"], expected["t"]):
        raise AssertionError("Logical coordinate differs from rank=f*T+t")
    groups = []
    for role in group_roles:
        base = role.removesuffix("_grad")
        wanted = next(pair for pair in manifest["logical_groups"][base] if worker.rank in pair)
        group = getattr(context, role + "_group")
        members = dist.get_process_group_ranks(group)
        if members != wanted or dist.get_rank(group) != wanted.index(worker.rank):
            raise AssertionError(f"Wrong ordered PG members/local rank for {role}")
        groups.append(
            {
                "pg_role": base,
                "pg_instance_label": role,
                "ordered_logical_members": members,
                "group_local_rank": dist.get_rank(group),
                "ordered_physical_uuids": [
                    worker_expectation(manifest, mapping_id, r)["uuid"] for r in members
                ],
            }
        )
    row = {
        "logical_rank": worker.rank,
        "local_rank": worker.local_rank,
        "visible_device": torch.cuda.current_device(),
        "actual_uuid": uuid,
        "pci_bus_id": pci,
        "f": context.coordinate.fsdp,
        "t": context.coordinate.tp,
        "mapping_id": mapping_id,
        "groups": groups,
    }
    all_rows = [None] * 4
    dist.all_gather_object(all_rows, row)
    if {item["actual_uuid"] for item in all_rows} != set(manifest["allowed_gpu_uuids"]):
        raise AssertionError("Workers do not cover the four allocated UUIDs exactly once")
    return {
        "status": "PASS",
        "local": row,
        "all_ranks": all_rows,
        "diagnostic_collective": "WORLD object allgather outside model/training inventory",
    }


class RuntimeObserver:
    """Read-only postconditions; wraps only acceptance hooks and restores them."""

    def __init__(self, payload):
        self.payload = payload
        self.payload.setdefault("lifecycle", [])
        self.runtime = None
        self.identities = None
        self.generations = None

    def registered(self, runtime):
        self.runtime = runtime
        self.identities = dict(runtime.identities)
        self.generations = [unit.generation for unit in runtime.units]
        self.payload["initial_shard_fingerprints"] = {
            name: {
                "sha256": tensor_hash(handle.local_shard),
                "shape": list(handle.local_shard.shape),
                "dtype": str(handle.local_shard.dtype),
                "valid_numel": handle.spec.valid_len(runtime.materializer.rank),
                "tp_local_shape": list(handle.spec.tp_local_shape),
                "padding_numel": handle.spec.padding_numel,
            }
            for name, handle in runtime.registry.items()
        }
        if runtime.model.lm_head.weight is not runtime.model.model.embed_tokens.weight:
            raise AssertionError("Tied head is not the actual embedding owner")
        if (
            sum(name == "model.embed_tokens.weight" for name in runtime.registry) != 1
            or "lm_head.weight" in runtime.registry
        ):
            raise AssertionError("Tied parameter has duplicated optimizer ownership")

    def finalized(self, runtime):
        if not runtime.optimizer_ready or runtime.comm.live_buffers:
            raise AssertionError(
                "finalize_backward did not publish an empty-buffer optimizer boundary"
            )
        if any(unit.state != "GRAD_READY" or unit.reductions != 1 for unit in runtime.units):
            raise AssertionError("Missing unit gradient readiness or duplicate reduction")
        if any(ticket.status != "CONSUMABLE" for ticket in runtime.comm.tickets):
            raise AssertionError("Outstanding collective ticket after finalize_backward")
        self.payload["lifecycle"].append(
            {
                "boundary": "finalize_backward",
                "microstep": runtime.step_id,
                "optimizer_ready": True,
                "live_buffers": 0,
                "ticket_count": len(runtime.comm.tickets),
                "generations": [unit.generation for unit in runtime.units],
                "gradient_commits": {
                    name: handle.grad_step for name, handle in runtime.registry.items()
                },
            }
        )

    def ended(self, runtime):
        if (
            runtime.active
            or runtime.comm.live_buffers
            or any(unit.lease is not None or unit.state != "SHARDED" for unit in runtime.units)
        ):
            raise AssertionError("Live buffer/FullParam lease or active state survived end_step")
        if self.identities != runtime.identities or self.identities != {
            name: (id(h.local_shard), h.local_shard.data_ptr())
            for name, h in runtime.registry.items()
        }:
            raise AssertionError("Stable optimizer leaf identity/storage changed")
        generations = [unit.generation for unit in runtime.units]
        if any(now <= before for now, before in zip(generations, self.generations, strict=True)):
            raise AssertionError("Materialization generation failed to advance")
        self.generations = generations
        issues = [
            row
            for row in runtime.comm.events
            if row["event_type"] == "collective_issue"
            and row["step_id"] == runtime.step_id
            and row["phase"] in {"FWD", "BWD"}
        ]
        per_group = {}
        contract = []
        first_consumers = {}
        for row in runtime.comm.events:
            if (
                row["event_type"] == "dependency_edge"
                and row.get("dependency_type") == "DATA_DEP"
                and row.get("producer_ticket") is not None
                and row.get("consumer")
            ):
                first_consumers.setdefault(row["producer_ticket"], row)
        for issue in issues:
            first_consumer = first_consumers.get(issue["op_id"])
            ordinal = per_group.get(issue["pg_id"], 0)
            per_group[issue["pg_id"]] = ordinal + 1
            contract.append(
                {
                    **{
                        key: issue.get(key)
                        for key in (
                            "pg_id",
                            "kind",
                            "unit_id",
                            "phase",
                            "group_members",
                            "group_role",
                            "numel",
                            "dtype",
                            "input_shape",
                            "output_shape",
                            "issue_stream",
                        )
                    },
                    "relative_pg_ordinal": ordinal,
                    "first_consumer": None
                    if first_consumer is None
                    else {key: first_consumer.get(key) for key in ("consumer", "unit_id", "phase")},
                }
            )
        if not contract or any(row["first_consumer"] is None for row in contract):
            raise AssertionError("Missing model collective or first-consumer evidence")
        self.payload.setdefault("training_collective_contract", []).append(contract)
        self.payload["lifecycle"].append(
            {
                "boundary": "end_step",
                "microstep": runtime.step_id,
                "stable_leaf_identity": True,
                "live_buffers": 0,
                "live_fullparam_leases": 0,
                "generations": generations,
            }
        )


@contextmanager
def observed_fixture(module, args, payload):
    manifest = load(args.placement_manifest)
    observer = RuntimeObserver(payload)
    original_session, original_runtime = module.WorkerSession, module.ShardedRuntime

    class Session(original_session):
        def __enter__(self):
            result = super().__enter__()
            try:
                payload["placement"] = verify_context(self.context, manifest, args.mapping_id)
            except BaseException:
                super().__exit__(*sys.exc_info())
                raise
            return result

    class Runtime(original_runtime):
        def register(self, *values, **kwargs):
            result = super().register(*values, **kwargs)
            observer.registered(self)
            return result

        def finalize_backward(self, **kwargs):
            result = super().finalize_backward(**kwargs)
            observer.finalized(self)
            return result

        def end_step(self):
            result = super().end_step()
            observer.ended(self)
            return result

    module.WorkerSession, module.ShardedRuntime = Session, Runtime
    try:
        yield observer
    finally:
        module.WorkerSession, module.ShardedRuntime = original_session, original_runtime


def run(args):
    rank = int(os.environ.get("RANK", "0"))
    path = Path(args.output) / f"mapping_{args.mode}.rank{rank}.json"
    if path.exists():
        raise FileExistsError(path)
    payload = {
        "status": "FAIL",
        "mapping_id": args.mapping_id,
        "rank": rank,
        "mode": args.mode,
        "errors": [],
        "profile_mode": "off",
        "performance_claim_allowed": False,
        "placement_manifest_sha256": hashlib.sha256(
            Path(args.placement_manifest).read_bytes()
        ).hexdigest(),
        "source_manifest_sha256": os.environ.get("SHARDTRAIN_SOURCE_MANIFEST_SHA256"),
    }
    try:
        verify_source_metadata(payload)
        if args.mode == "startup":
            from sharded_runtime.config import Config, DebugConfig, ParallelConfig, RuntimeConfig
            from sharded_runtime.launch.bootstrap import WorkerSession

            cfg = Config(
                parallel=ParallelConfig(
                    fsdp_degree=2, tp_degree=2, sp_enabled=True, backend="nccl", device="cuda"
                ),
                runtime=RuntimeConfig(communication_policy="stream_ordered"),
                debug=DebugConfig(output_dir=args.output, distributed_timeout_seconds=180),
            )
            with WorkerSession(cfg, "runtime") as session:
                payload["placement"] = verify_context(
                    session.context, load(args.placement_manifest), args.mapping_id
                )
        elif args.mode.startswith("native_"):
            from types import SimpleNamespace

            from sharded_runtime.distributed.context import WorkerEnvironment

            module = import_fixture(
                "tests/distributed/qwen2_native_worker.py", "_finalization_native_fixture"
            )
            payload["input_fingerprints"] = [
                {
                    "fsdp_coordinate": f,
                    "accumulation_steps": 1,
                    "tensors": [tensor_hash(value) for value in module.token_batch(f, 1)],
                }
                for f in range(2)
            ]
            payload["oracle_source"] = "tests/distributed/qwen2_native_worker.py"
            payload["oracle_sha256"] = hashlib.sha256(
                (PROJECT / payload["oracle_source"]).read_bytes()
            ).hexdigest()
            original_mesh, original_build = module.init_device_mesh, module.build_native_qwen

            def checked_mesh(*values, **kwargs):
                mesh = original_mesh(*values, **kwargs)
                f, t = mesh.get_coordinate()
                context = SimpleNamespace(
                    worker=WorkerEnvironment.read(),
                    device=module.torch.device("cuda", int(os.environ["LOCAL_RANK"])),
                    coordinate=SimpleNamespace(fsdp=f, tp=t),
                    tp_group=mesh["tp"].get_group(),
                    fsdp_group=mesh["fsdp"].get_group(),
                )
                payload["placement"] = verify_context(
                    context, load(args.placement_manifest), args.mapping_id, ("tp", "fsdp")
                )
                return mesh

            def checked_build(*values, **kwargs):
                model, policy = original_build(*values, **kwargs)
                payload["initial_shard_fingerprints"] = {
                    name: {
                        "sha256": tensor_hash(value.to_local()),
                        "shape": list(value.to_local().shape),
                        "dtype": str(value.dtype),
                    }
                    for name, value in model.named_parameters()
                }
                return model, policy

            module.init_device_mesh, module.build_native_qwen = checked_mesh, checked_build
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(PROJECT / payload["oracle_source"]),
                    "--f",
                    "2",
                    "--t",
                    "2",
                    "--sp",
                    "--dtype",
                    args.mode.removeprefix("native_"),
                    "--optimizer",
                    "adamw",
                    "--accumulation",
                    "1",
                    "--output-dir",
                    args.output,
                ]
                module.main()
            finally:
                module.init_device_mesh, module.build_native_qwen = original_mesh, original_build
                sys.argv = old_argv
            payload["scope"] = (
                "Unchanged 2-layer native FSDP2/TP/SP HF global-gradient/update oracle; "
                "FP16 includes overflow skip/recovery; not ten-update independent rollout"
            )
        else:
            module = import_fixture(
                "tests/distributed/qwen2_worker.py", "_finalization_fp32_fixture"
            )
            payload["input_fingerprints"] = input_hashes(module.token_batch, args.steps, 2)
            payload["oracle_source"] = "tests/distributed/qwen2_worker.py"
            payload["oracle_sha256"] = hashlib.sha256(
                (PROJECT / payload["oracle_source"]).read_bytes()
            ).hexdigest()
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(PROJECT / payload["oracle_source"]),
                    "--f",
                    "2",
                    "--t",
                    "2",
                    "--sp",
                    "--layers",
                    "4",
                    "--steps",
                    str(args.steps),
                    "--backend",
                    "nccl",
                    "--device",
                    "cuda",
                    "--policy",
                    "stream_ordered",
                    "--flags",
                    "111",
                    "--profile-mode",
                    "off",
                    "--debug.output_dir",
                    args.output,
                ]
                with observed_fixture(module, args, payload):
                    module.main()
            finally:
                sys.argv = old_argv
        payload["status"] = "PASS"
    except BaseException as exc:
        payload["errors"].append(repr(exc))
        raise
    finally:
        write_json(path, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--placement-manifest", required=True)
    parser.add_argument("--mapping-id", choices=["M0", "M1"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", choices=["startup", "fp32", "native_fp32", "native_fp16"], required=True
    )
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    run(args)


if __name__ == "__main__":
    main()
