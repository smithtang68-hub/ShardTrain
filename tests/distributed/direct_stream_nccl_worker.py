"""Real four-rank NCCL completion and allocator-lifetime checks for the coordinator.

Run with torchrun --standalone --nproc-per-node=4 and --output-dir NEW_DIRECTORY.
Nsight Systems may wrap torchrun. This worker checks arithmetic, producer readiness,
two independent consumer streams and early buffer release under allocation pressure.
It never substitutes a fake Work or backend and does not measure model throughput.
"""

import argparse
import hashlib
import inspect
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from sharded_runtime.distributed.nccl_ordering import prepare_nccl_launch_order
from sharded_runtime.observability.profiling import ProfileCapture
from sharded_runtime.runtime.comm import CommunicationCoordinator

GROUPS = (("fsdp0", (0, 2)), ("fsdp1", (1, 3)), ("tp0", (0, 1)), ("tp1", (2, 3)))
SEQUENCE = (
    ("fsdp", "ag", "fsdp_ag"),
    ("tp", "list_ag", "sp_ag"),
    ("fsdp", "rs", "fsdp_rs"),
    ("tp", "ar", "sp_rs"),
    ("world", "broadcast", "compute"),
    ("fsdp", "ag", "sp_rs"),
    ("tp", "rs", "sp_ag"),
)


def stream_info(stream):
    return dict(cuda_stream=int(stream.cuda_stream), torch_stream_id=int(stream.stream_id))


def payload(kind, rank, members, numel, device, offset):
    length = numel * len(members) if kind == "rs" else numel
    source = torch.arange(length, dtype=torch.float32, device=device).add_(rank * 1000 + offset)
    cpu_base = torch.arange(numel, dtype=torch.float32)
    if kind in {"ag", "list_ag"}:
        expected = torch.cat([cpu_base + member * 1000 + offset for member in members])
    elif kind == "rs":
        start = members.index(rank) * numel
        expected = (cpu_base + start + offset) * len(members) + 1000 * sum(members)
    elif kind == "ar":
        expected = (cpu_base + offset) * len(members) + 1000 * sum(members)
    else:
        expected = cpu_base + members[0] * 1000 + offset
    if kind == "list_ag":
        outputs = [torch.empty(numel, dtype=torch.float32, device=device) for _ in members]
    elif kind == "ag":
        outputs = [torch.empty(numel * len(members), dtype=torch.float32, device=device)]
    elif kind == "rs":
        outputs = [torch.empty(numel, dtype=torch.float32, device=device)]
    else:
        outputs = [source]
    return source, outputs, expected


def collective(kind, source, outputs, group, members, async_op):
    if kind == "ag":
        return dist.all_gather_into_tensor(outputs[0], source, group=group, async_op=async_op)
    if kind == "list_ag":
        return dist.all_gather(outputs, source, group=group, async_op=async_op)
    if kind == "rs":
        return dist.reduce_scatter_tensor(outputs[0], source, group=group, async_op=async_op)
    if kind == "ar":
        return dist.all_reduce(outputs[0], group=group, async_op=async_op)
    return dist.broadcast(outputs[0], src=members[0], group=group, async_op=async_op)


def submit(comm, producer, consumer, group, members, spec, call_id, numel, rank):
    role, kind, channel = spec
    label = f"direct_stream_contract/{kind}/{channel}/rank{rank}/call{call_id}"
    torch.cuda.nvtx.range_push(label)
    try:
        with torch.cuda.stream(producer):
            source, outputs, expected = payload(
                kind, rank, members, numel, comm.device, offset=17 * call_id
            )
            # Lease every distinct storage, including all list-AG outputs. The
            # coordinator must protect each tensor on the actual execution stream.
            tensors = {tensor.data_ptr(): tensor for tensor in (source, *outputs)}
            leases = [comm.buffer(tensor, "direct_stream_test") for tensor in tensors.values()]
            comm.unit_id, comm.generation = call_id, call_id
            begin = time.monotonic_ns()
            ticket = comm.issue(
                role,
                kind,
                group,
                source,
                outputs[0],
                lambda async_: collective(kind, source, outputs, group, members, async_),
                channel=channel,
                asynchronous=True,
                group_role=role,
                extra_outputs=tuple(outputs[1:]),
            )
            cpu_ns = time.monotonic_ns() - begin
        if (
            ticket.execution_mode != "nccl_current_stream_v291"
            or ticket.api_async_op is not False
            or ticket.work_returned
            or ticket.ready_event is None
            or ticket.collective_cuda_stream != comm.streams[channel].cuda_stream
        ):
            raise AssertionError("Coordinator did not use the qualified current-stream contract")
        row = dict(
            call_id=call_id,
            op_id=ticket.op_id,
            group=role,
            members=list(members),
            kind=kind,
            channel=channel,
            nvtx=label,
            producer=stream_info(producer),
            collective_stream=stream_info(comm.streams[channel]),
            consumer=stream_info(consumer),
            cpu_issue_ns=cpu_ns,
            execution_mode=ticket.execution_mode,
            api_async_op=ticket.api_async_op,
            work_returned=ticket.work_returned,
            readiness_query_after_issue=ticket.ready_event.query(),
            output_count=len(outputs),
        )
        with torch.cuda.stream(consumer):
            comm.consume(ticket, consumer=f"independent_consumer/{call_id % 2}")
            # Keep only this copied result for the final CPU oracle. Retire and
            # drop the collective buffers while both writes and reads may remain queued.
            observed = torch.cat(outputs) if kind == "list_ag" else outputs[0].clone()
            for lease in leases:
                comm.retire(lease)
        return row, observed, expected
    finally:
        torch.cuda.nvtx.range_pop()


def allocation_pressure(producer, device, numel, rank):
    with torch.cuda.stream(producer):
        for length in (numel, 2 * numel, numel, numel, 2 * numel, numel):
            scratch = torch.empty(length, dtype=torch.float32, device=device)
            scratch.fill_(-rank - 77)
            del scratch


def run(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or rank != local_rank:
        raise ValueError("This worker requires four ranks on one node, LOCAL_RANK=RANK")
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("Real CUDA/NCCL is required; no backend fallback")
    if str(torch.__version__).split("+", 1)[0] != "2.9.1":
        raise RuntimeError("This current-stream contract is qualified only for PyTorch 2.9.1")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"direct_stream_contract.rank{rank}.json"
    if path.exists():
        raise ValueError("Choose a fresh output directory to preserve previous artifacts")
    contract = prepare_nccl_launch_order()
    props = torch.cuda.get_device_properties(device)
    result = dict(
        status="running",
        rank=rank,
        pid=os.getpid(),
        torch_version=str(torch.__version__),
        torch_git_version=torch.version.git_version,
        cuda_build=torch.version.cuda,
        launch_order_contract=contract,
        gpu=dict(
            name=props.name,
            capability=[props.major, props.minor],
            total_memory=props.total_memory,
            multiprocessor_count=props.multi_processor_count,
        ),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        coordinator_sha256=hashlib.sha256(
            Path(inspect.getfile(CommunicationCoordinator)).read_bytes()
        ).hexdigest(),
        rounds=args.rounds,
        payload_numel=args.numel,
        no_launch_device_synchronization=True,
        early_collective_storage_release=True,
        checks=[],
        errors=[],
    )
    capture = ProfileCapture(directory, rank, device, mode="nsight")
    owned = []
    try:
        capture.start()
        dist.init_process_group("nccl", timeout=timedelta(seconds=90), device_id=device)
        groups = {"world": (dist.group.WORLD, tuple(range(4)))}
        for name, members in GROUPS:
            group = dist.new_group(list(members), backend="nccl", timeout=timedelta(seconds=90))
            if rank in members:
                groups["fsdp" if name.startswith("fsdp") else "tp"] = (group, members)
                owned.append(group)
        # Warm each actual communicator and basic allocation kernels once. This
        # synchronization is outside the tested launch/consumer/retirement window.
        for role in ("fsdp", "tp", "world"):
            source = torch.ones(8, device=device)
            dist.all_reduce(source, group=groups[role][0], async_op=False)
        torch.cuda.synchronize(device)
        del source
        comm = CommunicationCoordinator(
            device, rank, directory, communication_policy="stream_ordered"
        )
        producer = torch.cuda.Stream(device=device)
        consumers = [torch.cuda.Stream(device=device) for _ in range(2)]
        result["stream_registry"] = {
            name: stream_info(stream) for name, stream in comm.streams.items()
        }
        result["producer"] = stream_info(producer)
        result["consumers"] = [stream_info(stream) for stream in consumers]
        pending = []
        for step in range(args.rounds):
            comm.step_id = step
            for spec in SEQUENCE:
                call_id = len(pending)
                group, members = groups[spec[0]]
                row, observed, expected = submit(
                    comm, producer, consumers[call_id % 2], group, members,
                    spec, call_id, args.numel, rank,
                )
                pending.append((row, observed, expected))
                allocation_pressure(producer, device, args.numel, rank)
        # This is the only device sync in the checked window. CPU result transfers
        # and assertions happen after all collective submissions and queued reads.
        torch.cuda.synchronize(device)
        for row, observed, expected in pending:
            row["correct"] = torch.equal(observed.cpu(), expected)
            result["checks"].append(row)
            if not row["correct"]:
                raise AssertionError(f"Arithmetic/lifetime failure in {row['nvtx']}")
        if comm.live_buffers or any(ticket.status != "CONSUMABLE" for ticket in comm.tickets):
            raise AssertionError("Unretired buffers or unconsumed collective tickets")
        if any(
            row["dependency_type"] == "PG_ORDER"
            for row in comm.events
            if row["event_type"] == "dependency_edge"
        ):
            raise AssertionError("stream_ordered mode inserted a GPU PG-order wait")
        comm.gpu_wait_timings()
        result.update(
            status="passed", live_buffers=len(comm.live_buffers),
            outstanding_last_reads=len(comm.pending_last_reads),
            checked_calls=len(pending), pg_order_completion_waits=0,
            max_memory_allocated=torch.cuda.max_memory_allocated(device),
            max_memory_reserved=torch.cuda.max_memory_reserved(device),
        )
        capture.step(args.rounds)
    except BaseException as error:
        result["status"] = "failed"
        result["errors"].append(repr(error))
        raise
    finally:
        try:
            capture.close()
        finally:
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            for group in reversed(owned):
                dist.destroy_process_group(group)
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--numel", type=int, default=32768)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 32 or not 8 <= args.numel <= 1_048_576:
        parser.error("rounds must be 1..32 and numel 8..1048576")
    run(args)


if __name__ == "__main__":
    main()
