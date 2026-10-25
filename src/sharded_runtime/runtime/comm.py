"""Single-path collective submission, stream dependencies and explicit buffer retirement."""

import json
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import torch
import torch.distributed as dist


def serialized(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        with self.submission_lock:
            return method(self, *args, **kwargs)

    return invoke


@dataclass
class BufferLease:
    buffer_id: int
    tensor: object
    owner: str
    ready_event: object = None
    status: str = "ACTIVE"
    generation: int = 0

    def retire(self, stream=None):
        if self.status != "ACTIVE":
            raise RuntimeError(f"Duplicate buffer retirement {self.buffer_id}")
        if self.tensor is not None and self.tensor.is_cuda and stream is not None:
            self.tensor.record_stream(stream)
        self.tensor = None
        self.status = "RETIRED"


@dataclass
class CommunicationTicket:
    op_id: int
    pg_id: str
    pg_local_seq: int
    unit_id: int | None
    phase: str
    kind: str
    work: object = None
    ready_event: object = None
    status: str = "ISSUED"
    input_leases: tuple = ()
    output_leases: tuple = ()
    group_members: tuple = ()
    issue_stream: str = "cpu"
    generation: int = 0
    group_role: str | None = None
    execution_mode: str = "cpu"
    logical_cuda_stream: int | None = None
    collective_cuda_stream: int | None = None
    api_async_op: bool | None = None
    work_returned: bool = False


class CommunicationCoordinator:
    channels = ("compute", "fsdp_ag", "sp_ag", "fsdp_rs", "sp_rs")

    def __init__(
        self,
        device,
        rank=0,
        directory=None,
        *,
        communication_policy="conservative",
        diagnostics_enabled=True,
    ):
        if communication_policy not in {"conservative", "stream_ordered"}:
            raise ValueError("communication_policy must be conservative or stream_ordered")
        self.communication_policy = communication_policy
        self.diagnostics_enabled = diagnostics_enabled
        self.device = torch.device(device)
        self.rank = rank
        self.streams = {}
        if self.device.type == "cuda":
            self.streams = {name: torch.cuda.Stream(device=self.device) for name in self.channels}
        self._stream_names = {stream.cuda_stream: name for name, stream in self.streams.items()}
        self.tickets = []
        # History is for diagnostics; ownership checks only inspect current operations.
        self._issued_tickets = {}
        self._buffer_users = {}
        self._storage_leases = {}
        self._lease_storage_ptrs = {}
        self._group_metadata = {}
        self.next_op_id = 0
        self.events = []
        self._diagnostics_file = None
        self._diagnostics_flushed_events = 0
        self.sequence = {}
        self.expected_sequence = None
        self.step_id = -1
        self.next_buffer = 0
        self.live_buffers = {}
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.unit_id = None
        self.phase = "FWD"
        self.generation = 0
        self.group_role = None
        self.submission_lock = threading.RLock()
        self.gpu_waits = []
        self.pending_last_reads = []
        self.nccl_launch_order = None
        # This is a qualified backend behavior, not a promise for arbitrary newer builds.
        # In PyTorch 2.9.1, ordinary NCCL collectives with async_op=False enqueue on
        # the current CUDA stream and return None. Their GPU execution is still async.
        self.torch_build_version = str(torch.__version__)
        self.current_stream_version_qualified = self.torch_build_version.split("+", 1)[0] == "2.9.1"
        if self.streams and self.diagnostics_enabled:
            self.event(
                "cuda_stream_registry",
                torch_build_version=self.torch_build_version,
                current_stream_version_qualified=self.current_stream_version_qualified,
                streams={
                    name: {
                        "cuda_stream": stream.cuda_stream,
                        "torch_stream_id": stream.stream_id,
                    }
                    for name, stream in self.streams.items()
                },
            )

    def stream_name(self, stream=None):
        if not self.streams:
            return "cpu"
        stream = stream or torch.cuda.current_stream(self.device)
        handle = stream.cuda_stream
        name = self._stream_names.get(handle)
        registered = self.streams.get(name)
        if registered is not None and registered.cuda_stream == handle:
            return name
        # A caller can replace a logical stream. Cache only the current registry,
        # never every externally supplied CUDA stream observed by diagnostics.
        for name, registered in self.streams.items():
            if registered.cuda_stream == handle:
                self._stream_names = {value.cuda_stream: key for key, value in self.streams.items()}
                return name
        self._stream_names.pop(handle, None)
        return "caller"

    def dependency(self, dependency_type, *, producer_ticket=None, consumer_ticket=None, **fields):
        if dependency_type not in {"DATA_DEP", "PG_ORDER", "BUFFER_LIFETIME"}:
            raise ValueError("Unknown dependency type")
        if not self.diagnostics_enabled:
            return
        self.event(
            "dependency_edge",
            dependency_type=dependency_type,
            producer_ticket=producer_ticket,
            consumer_ticket=consumer_ticket,
            **fields,
        )

    def wait_event(self, event, *, dependency_type, stream=None, **fields):
        stream = stream or torch.cuda.current_stream(self.device)
        if dependency_type not in {"DATA_DEP", "PG_ORDER", "BUFFER_LIFETIME"}:
            raise ValueError("Unknown dependency type")
        if self.diagnostics_enabled:
            wait_stream = self.stream_name(stream)
            self.dependency(dependency_type, wait_stream=wait_stream, **fields)
        if dependency_type == "PG_ORDER" and self.diagnostics_enabled:
            # Deferred GPU timing: no host synchronization in the submission path.
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            stream.wait_event(event)
            end.record(stream)
            self.gpu_waits.append(
                (
                    start,
                    end,
                    dict(
                        fields,
                        wait_stream=wait_stream,
                        step_id=self.step_id,
                        rank=self.rank,
                        unit_id=self.unit_id,
                        phase=self.phase,
                    ),
                )
            )
        else:
            stream.wait_event(event)

    def gpu_wait_timings(self):
        """Call after an external synchronization, e.g. correctness-worker step boundary."""
        rows = []
        for start, end, fields in self.gpu_waits:
            if end.query():
                rows.append(
                    dict(fields, dependency_type="PG_ORDER", gpu_wait_ms=start.elapsed_time(end))
                )
        pending = []
        for event, buffer_id, generation in self.pending_last_reads:
            if event.query():
                self.event(
                    "buffer_last_read_complete",
                    buffer_id=buffer_id,
                    generation=generation,
                    dependency_type="BUFFER_LIFETIME",
                    last_read_complete=True,
                )
            else:
                pending.append((event, buffer_id, generation))
        self.pending_last_reads = pending
        return rows

    def stream_context(self, channel):
        return torch.cuda.stream(self.streams[channel]) if self.streams else nullcontext()

    @serialized
    def event(self, event_type, **fields):
        if not self.diagnostics_enabled:
            return
        row = dict(
            step_id=self.step_id,
            rank=self.rank,
            unit_id=self.unit_id,
            phase=self.phase,
            materialization_generation=self.generation,
            CPU_timestamp=time.monotonic(),
            event_type=event_type,
            op_id=None,
            pg_id=None,
            pg_seq=None,
            buffer_id=None,
            generation=self.generation,
            group_role=self.group_role,
            pg_local_seq=None,
            group_members=None,
            op_kind=None,
            producer_ticket=None,
            consumer_ticket=None,
            dependency_type=None,
            issue_stream=None,
            wait_stream=None,
            work_issued=False,
            work_wait_called=False,
            consumer_ready=False,
            last_read_complete=False,
        )
        row.update(fields)
        ticket_id = row["op_id"]
        if ticket_id is None:
            ticket_id = row["consumer_ticket"]
        if ticket_id is None:
            ticket_id = row["producer_ticket"]
        if ticket_id is not None and 0 <= ticket_id < len(self.tickets):
            ticket = self.tickets[ticket_id]
            for key, value in dict(
                op_id=ticket.op_id,
                op_kind=ticket.kind,
                pg_id=ticket.pg_id,
                pg_local_seq=ticket.pg_local_seq,
                group_members=list(ticket.group_members),
                issue_stream=ticket.issue_stream,
                group_role=ticket.group_role,
                execution_mode=ticket.execution_mode,
                logical_cuda_stream=ticket.logical_cuda_stream,
                collective_cuda_stream=ticket.collective_cuda_stream,
                api_async_op=ticket.api_async_op,
                work_returned=ticket.work_returned,
            ).items():
                if row.get(key) is None:
                    row[key] = value
        self.events.append(row)
        if self.streams and os.environ.get("SHARDTRAIN_NVTX") == "1":
            torch.cuda.nvtx.mark(
                f"shardtrain/{event_type}/U{row['unit_id']}/{row['phase']}"
                f"/step{row['step_id']}/op{row['op_id']}/pg{row['pg_id']}"
                f"/{row['dependency_type']}"
            )

    @serialized
    def flush_diagnostics(self, *, release_events=False):
        """Persist queued rows in order at an existing, untimed experiment boundary.

        Submission only builds the in-memory event record and emits its NVTX mark.
        JSON serialization and file I/O happen here, without a GPU synchronization.
        By default history remains available to existing diagnostic consumers.
        A benchmark may release rows only after their sole on-disk copy has been
        fully written and flushed, at an untimed boundary. Without a directory,
        the in-memory history is the only copy and is always retained.
        """
        if not self.diagnostics_enabled or self.directory is None:
            return
        if not self.events:
            return
        if self._diagnostics_file is None:
            self._diagnostics_file = (
                self.directory / f"runtime_events.rank{self.rank}.jsonl"
            ).open("a", encoding="utf-8", buffering=1024 * 1024)
        while self._diagnostics_flushed_events < len(self.events):
            row = self.events[self._diagnostics_flushed_events]
            self._diagnostics_file.write(json.dumps(row) + "\n")
            self._diagnostics_flushed_events += 1
        self._diagnostics_file.flush()
        if release_events:
            self.events.clear()
            self._diagnostics_flushed_events = 0

    @serialized
    def close_diagnostics(self):
        """Flush all recorded rows and close the log; later recording may reopen it."""
        self.flush_diagnostics()
        if self._diagnostics_file is not None:
            self._diagnostics_file.close()
            self._diagnostics_file = None

    def _register_issued(self, ticket):
        self._issued_tickets[ticket.op_id] = ticket
        for lease in (*ticket.input_leases, *ticket.output_leases):
            self._buffer_users.setdefault(lease.buffer_id, set()).add(ticket.op_id)

    def _release_issued(self, ticket):
        if self._issued_tickets.pop(ticket.op_id, None) is None:
            return
        for lease in (*ticket.input_leases, *ticket.output_leases):
            users = self._buffer_users.get(lease.buffer_id)
            if users is not None:
                users.discard(ticket.op_id)
                if not users:
                    del self._buffer_users[lease.buffer_id]

    def _process_group_metadata(self, group):
        if not dist.is_initialized():
            return "local", (self.rank,), 1
        # None means the current WORLD; its identity can change after reinitialization.
        group = dist.group.WORLD if group is None else group
        key = id(group)
        cached = self._group_metadata.get(key)
        if cached is None or cached[0] is not group:
            backend = dist.get_backend(group)
            members = tuple(dist.get_process_group_ranks(group))
            cached = group, backend, members, len(members)
            self._group_metadata[key] = cached
        return cached[1:]

    @serialized
    def buffer(self, tensor, owner):
        self.next_buffer += 1
        lease = BufferLease(self.next_buffer, tensor, owner, generation=self.generation)
        self.live_buffers[lease.buffer_id] = lease
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        self._lease_storage_ptrs[lease.buffer_id] = pointer
        self._storage_leases.setdefault(pointer, {})[lease.buffer_id] = lease
        if self.diagnostics_enabled:
            self.event(
                "buffer_allocate",
                buffer_id=lease.buffer_id,
                owner=owner,
                bytes=storage.nbytes(),
                shape=list(tensor.shape),
            )
        return lease

    @serialized
    def retire(self, lease):
        if lease.status != "ACTIVE":
            raise RuntimeError(f"Duplicate buffer retirement {lease.buffer_id}")
        if self.live_buffers.get(lease.buffer_id) is not lease:
            raise RuntimeError(f"Unknown buffer lease {lease.buffer_id}")
        if self._buffer_users.get(lease.buffer_id):
            raise RuntimeError(f"Premature buffer retirement {lease.buffer_id}")
        stream = torch.cuda.current_stream(self.device) if self.streams else None
        if stream is not None and self.diagnostics_enabled:
            lease.ready_event = torch.cuda.Event()
            lease.ready_event.record(stream)
            self.pending_last_reads.append((lease.ready_event, lease.buffer_id, lease.generation))
        if self.diagnostics_enabled:
            self.dependency(
                "BUFFER_LIFETIME",
                buffer_id=lease.buffer_id,
                owner=lease.owner,
                wait_stream=self.stream_name(stream),
                last_read_complete=stream is None,
                retirement="record_stream" if stream else "last_read_complete",
                generation=lease.generation,
            )
        lease.retire(stream)
        self.live_buffers.pop(lease.buffer_id)
        pointer = self._lease_storage_ptrs.pop(lease.buffer_id)
        leases = self._storage_leases[pointer]
        del leases[lease.buffer_id]
        if not leases:
            del self._storage_leases[pointer]
        if self.diagnostics_enabled:
            self.event("buffer_retire", buffer_id=lease.buffer_id, owner=lease.owner)

    @serialized
    def issue(
        self,
        pg_id,
        kind,
        group,
        inputs,
        output,
        operation,
        *,
        channel,
        asynchronous,
        group_role=None,
        extra_outputs=(),
    ):
        # All tensors touched by the collective need lifetime protection on its
        # execution stream. List all_gather callers provide the remaining outputs.
        outputs = (output, *extra_outputs)
        backend, members, group_size = self._process_group_metadata(group)
        current_stream_mode = bool(
            self.streams and backend == "nccl" and self.current_stream_version_qualified
        )
        seq = self.sequence.get(pg_id, 0)
        if self.expected_sequence is not None:
            index = self.next_op_id
            if index >= len(self.expected_sequence) or self.expected_sequence[index] != (
                pg_id,
                kind,
            ):
                raise RuntimeError(f"ProcessGroup order divergence at {index}: {pg_id}/{kind}")
        self.sequence[pg_id] = seq + 1
        ticket = CommunicationTicket(self.next_op_id, pg_id, seq, self.unit_id, self.phase, kind)
        ticket.group_members = members
        ticket.issue_stream, ticket.generation = channel, self.generation
        ticket.group_role = group_role or self.group_role
        if self.streams:
            ticket.logical_cuda_stream = self.streams[channel].cuda_stream
            ticket.execution_mode = (
                "nccl_current_stream_v291" if current_stream_mode else "backend_work"
            )
            ticket.api_async_op = not current_stream_mode
            if current_stream_mode:
                ticket.collective_cuda_stream = ticket.logical_cuda_stream
        else:
            ticket.api_async_op = asynchronous
        self.next_op_id += 1
        ticket.input_leases = tuple(
            self._storage_leases.get(inputs.untyped_storage().data_ptr(), {}).values()
        )
        output_leases = {}
        for tensor in outputs:
            output_leases.update(self._storage_leases.get(tensor.untyped_storage().data_ptr(), {}))
        # Match the original allocation order even when outputs share storage.
        ticket.output_leases = tuple(output_leases[key] for key in sorted(output_leases))
        self.tickets.append(ticket)
        self._register_issued(ticket)
        try:
            if self.diagnostics_enabled:
                self.event(
                    "collective_issue",
                    op_id=ticket.op_id,
                    pg_id=pg_id,
                    pg_seq=seq,
                    kind=kind,
                    numel=inputs.numel(),
                    dtype=str(inputs.dtype),
                    op_kind=kind,
                    pg_local_seq=seq,
                    group_members=list(ticket.group_members),
                    group_role=ticket.group_role,
                    issue_stream=channel,
                    backend=backend,
                    input_shape=list(inputs.shape),
                    output_shape=list(output.shape),
                    input_buffer_ids=[lease.buffer_id for lease in ticket.input_leases],
                    output_buffer_ids=[lease.buffer_id for lease in ticket.output_leases],
                    torch_build_version=self.torch_build_version,
                    current_stream_version_qualified=self.current_stream_version_qualified,
                    extra_output_count=len(extra_outputs),
                )
            if self.streams:
                if (
                    self.communication_policy == "stream_ordered"
                    and dist.is_initialized()
                    and backend == "nccl"
                    and group_size > 1
                    and self.nccl_launch_order is None
                ):
                    from sharded_runtime.distributed.nccl_ordering import require_nccl_launch_order

                    self.nccl_launch_order = require_nccl_launch_order()
                    if self.diagnostics_enabled:
                        self.event("nccl_launch_order_contract", **self.nccl_launch_order)
                # The conservative baseline explicitly serializes GPU completion.
                # stream_ordered instead uses NCCL >= 2.26 implicit HOST launch ordering
                # (CUDA runtime/driver >= 12.3), prepared before communicator creation.
                # The submission lock and per-PG sequence still serialize host calls.
                if self.communication_policy == "conservative" and len(self.tickets) > 1:
                    # CONSUMABLE means readiness was inherited by a particular consumer
                    # stream, not by every later caller. Keep the transitive total order
                    # even when the previous ticket was already consumed elsewhere.
                    self.consume(
                        self.tickets[-2],
                        dependency_type="PG_ORDER",
                        consumer_ticket=ticket.op_id,
                        consumer=kind,
                    )
                producer = torch.cuda.Event()
                producer.record(torch.cuda.current_stream(self.device))
                with self.stream_context(channel):
                    if self.diagnostics_enabled:
                        self.wait_event(
                            producer,
                            dependency_type="DATA_DEP",
                            consumer_ticket=ticket.op_id,
                            consumer="collective_input",
                            buffer_ids=[lease.buffer_id for lease in ticket.input_leases],
                        )
                    else:
                        torch.cuda.current_stream(self.device).wait_event(producer)
                    if self.communication_policy == "stream_ordered" and len(self.tickets) > 1:
                        previous = self.tickets[-2]
                        if previous.status == "FAILED":
                            raise RuntimeError(f"Failed communication {previous.op_id}")
                        # Host order is not a data-ready event. Waiting on previous.ready_event
                        # here would make current SP RS inherit next-unit prefetch completion,
                        # then indirectly stall current compute through SP RS consumption.
                        # Preserve each operation's own input/Work/output dependencies below.
                        if self.diagnostics_enabled:
                            self.event(
                                "collective_launch_order",
                                op_id=ticket.op_id,
                                previous_op_id=previous.op_id,
                                pg_id=ticket.pg_id,
                                pg_local_seq=ticket.pg_local_seq,
                                issue_stream=channel,
                                ordering_scope="host_launch",
                                gpu_completion_dependency=False,
                            )
                    inputs.record_stream(torch.cuda.current_stream(self.device))
                    for tensor in outputs:
                        tensor.record_stream(torch.cuda.current_stream(self.device))
                    ticket.work = operation(ticket.api_async_op)
                    ticket.work_returned = ticket.work is not None
                    if self.diagnostics_enabled:
                        self.event(
                            "work_issued",
                            op_id=ticket.op_id,
                            work_issued=ticket.work_returned,
                            issue_stream=channel,
                        )
                    if current_stream_mode:
                        if ticket.work is not None:
                            raise RuntimeError(
                                "Qualified PyTorch 2.9.1 current-stream NCCL collective "
                                "unexpectedly returned Work; this backend must be requalified"
                            )
                        # The ordinary NCCL kernel is queued on this stream. Recording
                        # its completion event does not wait for the GPU on the CPU.
                    else:
                        # Work.wait joins the backend's internal stream to THIS stream;
                        # it is not interpreted as all GPU consumers having completed.
                        ticket.work.wait()
                    if self.diagnostics_enabled:
                        self.dependency(
                            "DATA_DEP",
                            producer_ticket=ticket.op_id,
                            consumer="collective_ready_event",
                            work_wait_called=not current_stream_mode,
                            completion_source=(
                                "current_stream_event"
                                if current_stream_mode
                                else "work_stream_join"
                            ),
                            wait_stream=channel,
                            buffer_ids=[lease.buffer_id for lease in ticket.output_leases],
                        )
                    ticket.ready_event = torch.cuda.Event()
                    ticket.ready_event.record(torch.cuda.current_stream(self.device))
            else:
                ticket.work = operation(asynchronous)
                ticket.work_returned = ticket.work is not None
                if self.diagnostics_enabled:
                    self.event(
                        "work_issued",
                        op_id=ticket.op_id,
                        work_issued=ticket.work_returned,
                        issue_stream="cpu",
                    )
            if not asynchronous:
                self.consume(ticket)
            return ticket
        except BaseException:
            ticket.status = "FAILED"
            self._release_issued(ticket)
            if self.diagnostics_enabled:
                self.event("collective_failed", op_id=ticket.op_id)
            raise

    @serialized
    def consume(
        self,
        ticket,
        *,
        dependency_type="DATA_DEP",
        consumer_ticket=None,
        consumer="collective_output",
    ):
        if ticket.status == "FAILED":
            raise RuntimeError(f"Failed communication {ticket.op_id}")
        try:
            if ticket.ready_event is not None:
                if self.diagnostics_enabled:
                    self.wait_event(
                        ticket.ready_event,
                        dependency_type=dependency_type,
                        producer_ticket=ticket.op_id,
                        consumer_ticket=consumer_ticket,
                        consumer=consumer,
                        producer_kind=ticket.kind,
                        producer_unit=ticket.unit_id,
                        consumer_unit=self.unit_id,
                        pg_id=ticket.pg_id,
                        pg_local_seq=ticket.pg_local_seq,
                        group_members=list(ticket.group_members),
                        issue_stream=ticket.issue_stream,
                        group_role=ticket.group_role,
                        buffer_ids=[lease.buffer_id for lease in ticket.output_leases],
                    )
                else:
                    self.wait_event(ticket.ready_event, dependency_type=dependency_type)
            elif ticket.work is not None:
                if self.diagnostics_enabled:
                    self.dependency(
                        dependency_type,
                        producer_ticket=ticket.op_id,
                        consumer_ticket=consumer_ticket,
                        consumer=consumer,
                        wait_stream="cpu",
                        work_wait_called=True,
                        pg_id=ticket.pg_id,
                        pg_local_seq=ticket.pg_local_seq,
                        group_members=list(ticket.group_members),
                        issue_stream=ticket.issue_stream,
                        group_role=ticket.group_role,
                        buffer_ids=[lease.buffer_id for lease in ticket.output_leases],
                    )
                elif dependency_type not in {"DATA_DEP", "PG_ORDER", "BUFFER_LIFETIME"}:
                    raise ValueError("Unknown dependency type")
                ticket.work.wait()
        except BaseException:
            ticket.status = "FAILED"
            self._release_issued(ticket)
            if self.diagnostics_enabled:
                self.event("collective_failed", op_id=ticket.op_id)
            raise
        ticket.status = "CONSUMABLE"
        self._release_issued(ticket)
        # A completed Work may itself own output tensors. Keep readiness metadata, not
        # that strong reference: buffers retire independently after their last consumer.
        ticket.work = None
        if self.diagnostics_enabled:
            self.event(
                "collective_consumer_ready",
                op_id=ticket.op_id,
                pg_id=ticket.pg_id,
                pg_seq=ticket.pg_local_seq,
                pg_local_seq=ticket.pg_local_seq,
                group_members=list(ticket.group_members),
                consumer_ready=True,
                wait_stream=self.stream_name(),
                dependency_type=dependency_type,
                producer_ticket=ticket.op_id,
                consumer_ticket=consumer_ticket,
            )

    @serialized
    def finalize(self):
        # Consume in original host issue order, without scanning historical tickets.
        while self._issued_tickets:
            ticket = next(iter(self._issued_tickets.values()))
            if ticket.status == "ISSUED":
                self.consume(ticket)
            else:
                self._release_issued(ticket)

    @serialized
    def compact_completed_history(self):
        """Release old container references at an existing end-step boundary.

        Full diagnostic history is retained whenever diagnostics are enabled. With
        diagnostics off, op IDs remain monotonic but list positions are no longer
        op IDs after compaction. External ticket references retain every field.
        CONSUMABLE only describes an inherited stream dependency, not global GPU
        completion: no events or leases are cleared, queried or synchronized.
        Keep the latest ticket for conservative cross-step PG ordering, all failed
        or issued tickets, and every ticket whose leases have not retired.
        """
        if self.diagnostics_enabled or len(self.tickets) < 2:
            return 0
        latest = self.tickets[-1]
        retained = [
            ticket
            for ticket in self.tickets
            if ticket is latest
            or ticket.status != "CONSUMABLE"
            or any(
                lease.status != "RETIRED" for lease in (*ticket.input_leases, *ticket.output_leases)
            )
        ]
        removed = len(self.tickets) - len(retained)
        self.tickets[:] = retained
        return removed


@dataclass
class SequenceGather:
    """Pending SP gather, retaining its input and packed output until copy-out."""

    group: object
    local_shape: tuple
    dim: int
    local: object = None
    input_lease: object = None
    packed_lease: object = None
    ticket: object = None
    finished: bool = False


class RuntimeTPGroup:
    """Out-of-place SP collectives; AG input is the existing compact activation, no copy-in."""

    def __init__(self, group, coordinator, *, grad_group=None):
        from sharded_runtime.parallel.mappings import TPGroup

        self.base = TPGroup.from_process_group(group)
        self.degree, self.rank = self.base.degree, self.base.rank
        self.process_group, self.backend = group, self.base.backend
        self.grad_process_group = group if grad_group is None else grad_group
        if dist.is_initialized() and grad_group is not None:
            if dist.get_process_group_ranks(group) != dist.get_process_group_ranks(grad_group):
                raise ValueError("TP activation and gradient groups must have identical rank order")
            if self.degree > 1 and grad_group is group:
                raise ValueError("TP gradient group must be a distinct communicator")
        self.comm = coordinator

    @contextmanager
    def tagged(self, role):
        previous = self.comm.group_role
        self.comm.group_role = role
        try:
            yield
        finally:
            self.comm.group_role = previous

    def sequence_shard(self, x):
        return self.base.sequence_shard(x).clone()

    def sum(self, x, *, kind="tp_allreduce", group_role=None, channel="sp_rs"):
        if self.degree == 1:
            return x.clone()
        replicated_gradient = kind == "tp_norm_allreduce" or group_role == "replicated_gradient"
        group = self.grad_process_group if replicated_gradient else self.process_group
        pg_id = "tp_grad" if replicated_gradient else "tp"
        output = x.contiguous().clone()
        ticket = self.comm.issue(
            pg_id,
            kind,
            group,
            output,
            output,
            lambda async_: dist.all_reduce(output, group=group, async_op=async_),
            channel=channel,
            asynchronous=False,
            group_role=group_role or ("tp_replicated_gradient" if replicated_gradient else None),
        )
        assert ticket.status == "CONSUMABLE"
        return output

    def issue_gather(self, x, dim):
        if self.degree == 1:
            return SequenceGather(self, tuple(x.shape), dim, local=x)
        # x must already be compact/contiguous. No hidden copy-in in this path.
        if not x.is_contiguous():
            raise RuntimeError("SP AG requires an existing contiguous LocalActivation")
        if x.untyped_storage().nbytes() != x.numel() * x.element_size():
            raise RuntimeError("SP AG input aliases a larger activation storage")
        input_lease = self.comm.buffer(x, "sp_ag_direct_input")
        if self.comm.diagnostics_enabled:
            self.comm.event(
                "sp_ag_direct_input",
                shape=list(x.shape),
                storage_bytes=x.untyped_storage().nbytes(),
                copy_in=False,
                buffer_id=input_lease.buffer_id,
            )
        packed = torch.empty(
            (self.degree * x.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device
        )
        lease = self.comm.buffer(packed, "sp_ag_output")
        ticket = self.comm.issue(
            "tp",
            "sp_ag",
            self.process_group,
            x,
            packed,
            lambda async_: dist.all_gather_into_tensor(
                packed, x, group=self.process_group, async_op=async_
            ),
            channel="sp_ag",
            asynchronous=True,
        )
        return SequenceGather(
            self, tuple(x.shape), dim, input_lease=input_lease, packed_lease=lease, ticket=ticket
        )

    def finish_gather(self, pending):
        if pending.group is not self:
            raise RuntimeError("SP gather must be finished by its issuing group")
        if pending.finished:
            raise RuntimeError("SP gather has already been finished")
        if self.degree == 1:
            output, pending.local = pending.local, None
            pending.finished = True
            return output
        self.comm.consume(pending.ticket, consumer="sp_gather_copy_out")
        # B>1 rank-major to [B,S,H]; explicit compact destination, not a rank-major view.
        output = sequence_copy_out(
            pending.packed_lease.tensor, pending.local_shape, self.degree, pending.dim
        )
        output._runtime_activation_lease = self.comm.buffer(output, "sp_full_activation")
        self.comm.retire(pending.input_lease)
        self.comm.retire(pending.packed_lease)
        pending.input_lease, pending.packed_lease, pending.ticket = None, None, None
        pending.finished = True
        return output

    def gather(self, x, dim):
        return self.finish_gather(self.issue_gather(x, dim))

    def retire_activation(self, tensor):
        lease = getattr(tensor, "_runtime_activation_lease", None)
        if lease is not None:
            self.comm.retire(lease)

    def reduce_scatter_sequence(self, x):
        if self.degree == 1:
            return x.clone()
        if self.backend == "gloo":
            return self.sequence_shard(self.sum(x, kind="sp_rs_math_allreduce"))
        # Sequence-major packing is required by NCCL's dimension-zero RS convention.
        packed = x.movedim(1, 0).contiguous()
        output = torch.empty(
            (packed.shape[0] // self.degree, *packed.shape[1:]), dtype=x.dtype, device=x.device
        )
        lease = self.comm.buffer(packed, "sp_rs_input")
        self.comm.issue(
            "tp",
            "sp_rs",
            self.process_group,
            packed,
            output,
            lambda async_: dist.reduce_scatter_tensor(
                output, packed, group=self.process_group, async_op=async_
            ),
            channel="sp_rs",
            asynchronous=False,
        )
        result = output.movedim(0, 1).contiguous()
        self.comm.retire(lease)
        return result


def sequence_copy_out(packed, local_shape, degree, dim=1):
    """Restore batch-first sequences from process-group-local rank-major AG output."""
    output = packed.new_empty(
        (*local_shape[:dim], local_shape[dim] * degree, *local_shape[dim + 1 :])
    )
    for rank in range(degree):
        output.narrow(dim, rank * local_shape[dim], local_shape[dim]).copy_(
            packed.narrow(0, rank * local_shape[0], local_shape[0])
        )
    return output
