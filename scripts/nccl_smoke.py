"""E0: actual NCCL in-place packed AG alias and out-of-place RS before any model run."""

import argparse
import os
import platform
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

from sharded_runtime.config import ParallelConfig
from sharded_runtime.distributed.context import DistributedContext, WorkerEnvironment
from sharded_runtime.observability.logging import make_rank_logger
from sharded_runtime.observability.metadata import write_json
from sharded_runtime.runtime.layout import ParameterSpec, copy_out, offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    worker = WorkerEnvironment.read()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    logger = make_rank_logger(directory, worker.rank, "INFO")
    parallel = ParallelConfig(args.fsdp, args.tp, args.tp > 1, "cuda", "nccl")
    with DistributedContext(parallel, worker, 60, logger) as mesh:
        rank = dist.get_rank(mesh.fsdp_group)
        shapes = [(1,), (3, 5), (2, 7)]
        values = [
            torch.arange(
                torch.tensor(shape).prod().item(), device=mesh.device, dtype=torch.float32
            ).reshape(shape)
            + j * 100
            for j, shape in enumerate(shapes)
        ]
        specs = [
            ParameterSpec.create(str(j), 0, value, tuple(value.shape), "column", args.fsdp)
            for j, value in enumerate(values)
        ]
        starts, length = offsets(specs)
        packed = torch.full((args.fsdp * length,), -777.0, device=mesh.device)
        slot = packed.narrow(0, rank * length, length)
        for value, spec, start in zip(values, specs, starts, strict=True):
            valid = spec.valid_len(rank)
            slot[start:][:valid].copy_(value.reshape(-1)[rank * spec.fsdp_shard_numel :][:valid])
            slot[start + valid :][: spec.fsdp_shard_numel - valid].zero_()
        assert slot.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
        assert slot.storage_offset() == rank * length
        work = dist.all_gather_into_tensor(packed, slot, group=mesh.fsdp_group, async_op=True)
        work.wait()
        full = copy_out(packed, specs, args.fsdp)
        for spec, value in zip(specs, values, strict=True):
            torch.testing.assert_close(full[spec.param_id], value, atol=0, rtol=0)
            assert full[spec.param_id].data_ptr() != packed.data_ptr()
        source = torch.arange(args.fsdp * length, dtype=torch.float32, device=mesh.device)
        source = source + rank
        output = torch.empty(length, device=mesh.device)
        assert source.untyped_storage().data_ptr() != output.untyped_storage().data_ptr()
        dist.reduce_scatter_tensor(output, source, group=mesh.fsdp_group)
        expected = (
            torch.arange(args.fsdp * length, dtype=torch.float32, device=mesh.device) * args.fsdp
            + args.fsdp * (args.fsdp - 1) / 2
        )[rank * length :][:length]
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        torch.cuda.synchronize(mesh.device)
        environment = dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            nccl=torch.cuda.nccl.version(),
            python=platform.python_version(),
            rank=worker.rank,
            local_rank=worker.local_rank,
            fsdp_local_rank=rank,
            GPU=torch.cuda.get_device_name(),
            visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        if worker.rank == 0:
            environment["topology"] = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            ).stdout
        write_json(
            directory / f"smoke.rank{worker.rank}.json",
            dict(
                status="passed",
                environment=environment,
                inplace_AG_alias_verified=True,
                out_of_place_RS_verified=True,
            ),
        )


if __name__ == "__main__":
    main()
