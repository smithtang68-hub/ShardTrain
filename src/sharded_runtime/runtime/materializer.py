"""Packed FSDP AG and independent copy-out storages, with destination-major gradient RS."""

from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from sharded_runtime.runtime.layout import copy_out, offsets, pack_gradients


@dataclass
class MaterializationLease:
    unit_id: int
    step_id: int
    phase: str
    generation: int
    ag_storage: object = None
    full_param_storages: dict = field(default_factory=dict)
    ag_ticket: object = None
    copy_out_ready_event: object = None
    fullparam_last_consumer_event: object = None


class Materializer:
    def __init__(self, group, coordinator):
        self.group, self.comm = group, coordinator
        self.degree = dist.get_world_size(group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
        self.backend = str(dist.get_backend(group)) if dist.is_initialized() else "local"

    @torch.no_grad()
    def issue(self, unit, phase):
        specs = [h.spec for h in unit.handles]
        starts, length = offsets(specs)
        first = unit.handles[0].local_shard
        with self.comm.stream_context("fsdp_ag"):
            # Persistent shards depend on initialization/optimizer writes, not on the
            # activation currently queued on compute. Allocate on this stream too, so
            # allocator reuse follows its own stream's prior reads before copy-in.
            if self.comm.streams:
                unit.runtime.wait_parameter_shards_ready()
            storage = first.new_empty(self.degree * length)
            lease = MaterializationLease(
                unit.unit_id,
                unit.runtime.step_id,
                phase,
                unit.generation,
                self.comm.buffer(storage, "fsdp_ag"),
            )
            slot = storage.narrow(0, self.rank * length, length)
            for handle, start in zip(unit.handles, starts, strict=True):
                slot.narrow(0, start, handle.spec.fsdp_shard_numel).copy_(handle.local_shard)
            if self.degree > 1:
                if self.backend == "nccl":
                    # Alias contract intentionally isolated for M4 in-place smoke.
                    def operation(async_):
                        return dist.all_gather_into_tensor(
                            storage, slot, group=self.group, async_op=async_
                        )
                else:
                    # Gloo mathematical path: direct output views, no FullParam-sized cat.
                    chunks = [storage.narrow(0, r * length, length) for r in range(self.degree)]

                    def operation(async_):
                        return dist.all_gather(chunks, slot, group=self.group, async_op=async_)

                lease.ag_ticket = self.comm.issue(
                    "fsdp",
                    "fsdp_ag",
                    self.group,
                    slot,
                    storage,
                    operation,
                    channel="fsdp_ag",
                    asynchronous=unit.runtime.asynchronous,
                    group_role="parameter_materialization",
                )
            elif self.comm.streams:
                lease.copy_out_ready_event = torch.cuda.Event()
                lease.copy_out_ready_event.record(torch.cuda.current_stream(first.device))
        return lease

    def finish(self, unit):
        lease = unit.lease
        if lease.full_param_storages:
            raise RuntimeError(f"Duplicate copy-out U{unit.unit_id}")
        if lease.ag_ticket:
            self.comm.consume(lease.ag_ticket, consumer=f"copy_out/U{unit.unit_id}")
        elif lease.copy_out_ready_event is not None:
            self.comm.wait_event(
                lease.copy_out_ready_event,
                dependency_type="DATA_DEP",
                consumer=f"copy_out/U{unit.unit_id}",
                buffer_id=lease.ag_storage.buffer_id,
            )
        if self.comm.diagnostics_enabled:
            self.comm.event(
                "copy_out_start",
                unit_id=unit.unit_id,
                producer_ticket=lease.ag_ticket.op_id if lease.ag_ticket else None,
                buffer_id=lease.ag_storage.buffer_id,
                issue_stream=self.comm.stream_name(),
            )
        weights = copy_out(lease.ag_storage.tensor, [h.spec for h in unit.handles], self.degree)
        for name, weight in weights.items():
            lease.full_param_storages[name] = self.comm.buffer(weight, "full_parameter")
        if self.comm.streams:
            lease.copy_out_ready_event = torch.cuda.Event()
            lease.copy_out_ready_event.record(torch.cuda.current_stream(self.comm.device))
        self.comm.event("copy_out_ready", unit_id=unit.unit_id)
        # No strong AG tensor reference survives copy-out; allocator is protected by record_stream.
        self.comm.retire(lease.ag_storage)
        lease.ag_storage = None
        return tuple(weights[h.spec.param_id] for h in unit.handles)

    def _issue_gradient_bucket(self, unit, grads, specs, bucket):
        """Queue RS without joining its output to the graph's gradient-consumer stream."""
        group = getattr(unit.runtime, "fsdp_grad_group", self.group)
        packed = pack_gradients(grads, specs, self.degree)
        source_lease = self.comm.buffer(packed, f"fsdp_{bucket}_rs_input")
        starts, length = offsets(specs)
        output = packed.new_empty(length)
        ticket = None
        issued_ready_event = None
        if self.degree == 1:
            output.copy_(packed)
            if self.comm.streams:
                producer = torch.cuda.current_stream(self.comm.device)
                packed.record_stream(producer)
                output.record_stream(producer)
                issued_ready_event = torch.cuda.Event()
                issued_ready_event.record(producer)
        elif self.backend == "gloo":
            ticket = self.comm.issue(
                "fsdp_grad",
                "fsdp_rs_math_allreduce",
                group,
                packed,
                packed,
                lambda async_: dist.all_reduce(packed, group=group, async_op=async_),
                channel="fsdp_rs",
                asynchronous=unit.runtime.async_grad_rs,
                group_role=f"{bucket}_gradient_bucket",
            )
        else:
            ticket = self.comm.issue(
                "fsdp_grad",
                "fsdp_rs",
                group,
                packed,
                output,
                lambda async_: dist.reduce_scatter_tensor(
                    output, packed, group=group, async_op=async_
                ),
                channel="fsdp_rs",
                asynchronous=unit.runtime.async_grad_rs,
                group_role=f"{bucket}_gradient_bucket",
            )
        if self.comm.diagnostics_enabled:
            self.comm.event(
                "gradient_bucket_issued",
                bucket=bucket,
                parameters=[spec.param_id for spec in specs],
                producer_ticket=ticket.op_id if ticket else None,
                source_buffer_id=source_lease.buffer_id,
                output_numel=length,
            )
        return output, packed, source_lease, starts, ticket, issued_ready_event

    def _consume_gradient_bucket(self, data, bucket):
        output, packed, source_lease, starts, ticket, issued_ready_event = data
        if ticket is not None:
            self.comm.consume(ticket, consumer=f"{bucket}_gradient_bucket")
        elif issued_ready_event is not None:
            self.comm.wait_event(
                issued_ready_event,
                dependency_type="DATA_DEP",
                consumer=f"{bucket}_gradient_bucket",
                buffer_id=source_lease.buffer_id,
            )
        if self.degree > 1 and self.backend == "gloo":
            output.copy_(packed.narrow(0, self.rank * output.numel(), output.numel()))
        output.div_(self.degree)  # One FSDP average per parameter, across disjoint buckets.
        if self.comm.streams:
            output.record_stream(torch.cuda.current_stream(self.comm.device))
        self.comm.retire(source_lease)
        self.comm.event(
            "gradient_bucket_consumed",
            bucket=bucket,
            producer_ticket=ticket.op_id if ticket else None,
        )
        return output, starts

    def reduce(self, unit, grads):
        specs = [h.spec for h in unit.handles]
        if len(grads) != len(specs) or any(g is None for g in grads):
            raise RuntimeError(f"Missing full parameter gradient U{unit.unit_id}")
        split_norm = (
            unit.runtime.communication_policy == "stream_ordered" and unit.runtime.async_grad_rs
        )
        norm_indices = [
            i
            for i, spec in enumerate(specs)
            if split_norm and spec.tp_grad_reduce == "sum" and len(spec.tp_local_shape) == 1
        ]
        main_indices = [i for i in range(len(specs)) if i not in norm_indices]
        buckets = {}
        # Small replicated vectors precede the large matrix/embedding bucket on one RS PG.
        # A separate TP-gradient PG can start their SUM while the main RS is still running.
        for name, indices in (("norm", norm_indices), ("main", main_indices)):
            if indices:
                buckets[name] = self._issue_gradient_bucket(
                    unit, [grads[i] for i in indices], [specs[i] for i in indices], name
                )
        if getattr(unit.runtime, "gradient_dispatch_prefetch", False):
            # Fixed previous-unit ownership; submit parameter AG before Norm/main waits.
            unit.runtime.dispatch_backward_prefetch(unit)
        result = [None] * len(specs)
        norm_ready = None
        if norm_indices:
            with self.comm.stream_context("compute"):
                output, starts = self._consume_gradient_bucket(buckets["norm"], "norm")
                for i, start in zip(norm_indices, starts, strict=True):
                    spec = specs[i]
                    gradient = output.narrow(0, start, spec.fsdp_shard_numel)
                    if not unit.runtime.sp:
                        gradient = gradient / unit.runtime.tp.degree
                    result[i] = unit.runtime.tp.sum(
                        gradient,
                        kind="tp_norm_allreduce",
                        group_role="norm_gradient_bucket",
                        channel="compute",
                    )
                if self.comm.streams:
                    norm_ready = torch.cuda.Event()
                    norm_ready.record(torch.cuda.current_stream(self.comm.device))
                if self.comm.diagnostics_enabled:
                    self.comm.event(
                        "norm_gradient_branch_ready",
                        unit_id=unit.unit_id,
                        parameters=[specs[i].param_id for i in norm_indices],
                        main_rs_ticket=(
                            buckets["main"][4].op_id
                            if "main" in buckets and buckets["main"][4] is not None
                            else None
                        ),
                    )
        if main_indices:
            output, starts = self._consume_gradient_bucket(buckets["main"], "main")
            for i, start in zip(main_indices, starts, strict=True):
                spec = specs[i]
                gradient = output.narrow(0, start, spec.fsdp_shard_numel)
                if spec.tp_grad_reduce == "sum":
                    # Large replicated embedding gradients keep the post-RS shard SUM.
                    if not unit.runtime.sp:
                        gradient = gradient / unit.runtime.tp.degree
                    gradient = unit.runtime.tp.sum(gradient, kind="tp_norm_allreduce")
                result[i] = gradient
        if norm_ready is not None:
            self.comm.wait_event(
                norm_ready,
                dependency_type="DATA_DEP",
                consumer=f"autograd_norm_shard/U{unit.unit_id}",
                producer_unit=unit.unit_id,
            )
        if self.comm.streams:
            # Norm results were allocated on Compute, but AccumulateGrad reads on this
            # Function's original RS stream. Preserve both readiness and allocator lifetime.
            for gradient in result:
                gradient.record_stream(torch.cuda.current_stream(self.comm.device))
        return tuple(result)
