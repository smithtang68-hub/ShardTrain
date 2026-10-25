"""Reversible, version-qualified communication scheduling and event collection.

The policies add CUDA event edges immediately before the real NCCL call on the
runtime's actual execution stream. They never replace a collective, change its
math, synchronize the CPU, or retain input/output tensors. C1/C2/C3 require a C0
cross-rank host-order contract. This module does not modify the main runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

POLICIES = {
    "C0": "existing flags111 scheduling, no extra dependencies",
    "C1": "all FWD/BWD communication serial; compute remains independent",
    "C2": "FSDP AG/RS may overlap; SP/norm separated from all FSDP communication",
    "C3": "communication serial; FSDP also gated against runtime compute stream",
}


def family(kind):
    if kind == "fsdp_ag":
        return "fsdp_parameter_ag"
    if kind in {"fsdp_rs", "fsdp_rs_math_allreduce"}:
        return "fsdp_gradient_rs"
    if kind == "sp_ag":
        return "sp_activation_ag"
    if kind in {"sp_rs", "sp_rs_math_allreduce"}:
        return "sp_activation_rs"
    if kind == "tp_norm_allreduce":
        return "tp_norm_allreduce"
    return "other"


def signature(row):
    """Same role/shape on every rank in the regular F2/T2 experiment mesh."""
    return {
        key: row.get(key)
        for key in (
            "process_group_label",
            "collective_type",
            "unit_id",
            "phase",
            "channel",
            "input_shape",
            "output_shapes",
            "dtype",
            "group_size",
            "group_role",
        )
    }


def dependency_predecessors(policy, current_family, tails):
    """Pure planner: only point from already submitted operations to this op."""
    if policy not in POLICIES:
        raise ValueError(f"Unsupported policy {policy}")
    if policy == "C0" or not tails:
        return []
    if policy in {"C1", "C3"}:
        return [max(tails.values(), key=lambda item: item["ordinal"])]
    is_fsdp = current_family.startswith("fsdp_")
    candidates = [
        value for value in tails.values() if value["family"].startswith("fsdp_") != is_fsdp
    ]
    # Separate stream tails are necessary: AG and RS need not complete in order.
    return sorted(candidates, key=lambda item: item["ordinal"])


def read_events(paths):
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def validate_schedule(paths):
    """Qualify a C0 template before using additional cross-communicator edges.

    Every participating rank must have an identical FWD/BWD host ordinal template
    for each microstep. Actual PG members/sequence/payload must also match. A
    common global ordinal makes every added edge monotonic on the *global*
    collective DAG, rather than merely on a rank-local chain.
    """
    rows = [row for row in read_events(paths) if row.get("phase") in {"FWD", "BWD"}]
    if not rows:
        raise ValueError("No FWD/BWD communication events to qualify")
    ranks = sorted({row["rank"] for row in rows})
    expected_ranks = set(range(rows[0].get("world_size", len(ranks))))
    if set(ranks) != expected_ranks:
        raise ValueError(f"Missing rank evidence: found {ranks}, expected {expected_ranks}")
    per_step = defaultdict(list)
    per_collective = defaultdict(list)
    for row in rows:
        per_step[(row["rank"], row["runtime_microstep"])].append(row)
        pg_key = (
            row["process_group_label"],
            tuple(row["group_members"]),
            row["collective_sequence"],
        )
        per_collective[pg_key].append(row)
    templates = []
    for key, values in sorted(per_step.items()):
        values.sort(key=lambda item: item["host_ordinal"])
        templates.append((key, [signature(value) for value in values]))
    template = templates[0][1]
    for key, candidate in templates:
        if candidate != template:
            raise ValueError(f"Global host-order template differs at rank/microstep {key}")
    for key, values in per_collective.items():
        members = set(key[1])
        if {row["rank"] for row in values} != members or len(values) != len(members):
            raise ValueError(f"Incomplete/duplicate collective evidence for {key}")
        if len({json.dumps(signature(row), sort_keys=True) for row in values}) != 1:
            raise ValueError(f"Collective shape/type divergence for {key}")
    return {
        "status": "QUALIFIED",
        "num_ranks": len(ranks),
        "validated_microsteps": len(per_step),
        "template": template,
        "proof": "all ranks have a common host ordinal; added event edges only point forward",
        "scope": "regular equal-shape mesh and exact recorded architecture/configuration",
    }


def runtime_layout(runtime):
    """Capture actual parameter padding/buckets; no dense materialization."""
    units = []
    degree = runtime.materializer.degree
    for unit in runtime.units:
        parameters = []
        for handle in unit.handles:
            spec = handle.spec
            parameters.append(
                {
                    "name": spec.param_id,
                    "global_shape": list(spec.global_shape),
                    "tp_local_shape": list(spec.tp_local_shape),
                    "tp_local_numel": spec.numel,
                    "shard_numel": spec.fsdp_shard_numel,
                    "padding_numel": spec.padding_numel,
                    "dtype": str(spec.dtype),
                    "element_size": handle.local_shard.element_size(),
                    "valid_shard_numel": spec.valid_len(runtime.materializer.rank),
                    "tp_grad_reduce": spec.tp_grad_reduce,
                }
            )
        tied = type(unit).__name__ == "TiedEmbeddingUnit"
        retains_forward = tied or not runtime.reshard_after_forward or unit is runtime.units[-1]
        units.append(
            {
                "unit_id": unit.unit_id,
                "unit_class": type(unit).__name__,
                "parameters": parameters,
                "retains_forward_parameters": retains_forward,
                "expected_parameter_ag_per_microstep": 1 + int(not retains_forward)
                if degree > 1
                else 0,
                "expected_parameter_ag_fwd": int(degree > 1),
                "expected_parameter_ag_bwd": int(degree > 1 and not retains_forward),
                "expected_gradient_rs_buckets": (
                    int(
                        any(
                            p["tp_grad_reduce"] == "sum" and len(p["tp_local_shape"]) == 1
                            for p in parameters
                        )
                    )
                    + int(
                        any(
                            not (p["tp_grad_reduce"] == "sum" and len(p["tp_local_shape"]) == 1)
                            for p in parameters
                        )
                    )
                    if runtime.communication_policy == "stream_ordered" and runtime.async_grad_rs
                    else int(bool(parameters))
                )
                if degree > 1
                else 0,
                "padded_full_parameter_bytes": sum(
                    p["shard_numel"] * degree * p["element_size"] for p in parameters
                ),
                "unpadded_tp_local_parameter_bytes": sum(
                    p["tp_local_numel"] * p["element_size"] for p in parameters
                ),
            }
        )
    return {
        "fsdp_degree": degree,
        "tp_degree": runtime.tp.degree,
        "sp_enabled": runtime.sp,
        "reshard_after_forward": runtime.reshard_after_forward,
        "async_grad_rs": runtime.async_grad_rs,
        "units": units,
        "accumulation_semantics": "runtime performs AG/RS each microstep; no no_sync suppression",
    }


class Collector:
    def __init__(
        self,
        *,
        policy="C0",
        output_dir=None,
        run_id="unassigned",
        experiment_id="unassigned",
        warmup_steps=0,
        contract=None,
        gpu_timing=False,
        injection=None,
        patch_trainer=True,
        record_events=True,
        guarded_baseline=False,
    ):
        if policy not in POLICIES:
            raise ValueError(f"Unsupported policy {policy}")
        if guarded_baseline and policy != "C0":
            raise ValueError("guarded_baseline applies only to the C0 control")
        if (policy != "C0" or guarded_baseline) and (
            not contract or contract.get("status") != "QUALIFIED"
        ):
            raise RuntimeError("BLOCKED: controlled scheduling requires a cross-rank C0 contract")
        self.policy, self.output_dir = policy, Path(output_dir) if output_dir else None
        self.run_id, self.experiment_id = run_id, experiment_id
        self.warmup_steps, self.contract = warmup_steps, contract
        self.gpu_timing, self.injection, self.patch_trainer = gpu_timing, injection, patch_trainer
        self.record_events = record_events
        self.guarded_baseline = guarded_baseline
        self.rows, self.layouts, self.stream_registries = [], {}, {}
        self.event_count = 0
        self._unit_layers = {}
        self._tails, self._timings = defaultdict(dict), []
        self._guard_positions = {}
        self._trainer_steps, self._step_context = {}, {}
        self._rank = None
        self._installed = False

    def begin_step(self, step, measured=None, runtime=None):
        """Untimed runner boundary; step is optimizer step, runtime counts microsteps.

        measured=True derives measured index from warmup_steps; an integer can
        provide the exact measured index. measured=False/None denotes warmup.
        """
        if runtime is None:
            raise ValueError("begin_step requires the actual runtime")
        comm = runtime.comm
        self._rank = comm.rank
        index = step - self.warmup_steps if measured is True else measured
        if measured is False or measured is None:
            index = None
        self._step_context[id(comm)] = {
            "optimizer_step": step,
            "measured_step": index,
            "initial_microstep": runtime.step_id,
        }
        self._capture_layout(runtime)

    def _capture_layout(self, runtime):
        key = id(runtime.comm)
        if key in self.layouts:
            return
        self.layouts[key] = runtime_layout(runtime)
        mapping = {}
        for unit in runtime.units:
            names = [handle.spec.param_id for handle in unit.handles]
            matches = [re.search(r"(?:^|\.)layers\.(\d+)\.", name) for name in names]
            layers = {int(match.group(1)) for match in matches if match}
            mapping[unit.unit_id] = next(iter(layers)) if len(layers) == 1 else None
        self._unit_layers[key] = mapping
        if self.injection is not None:
            self.injection.prepare(runtime)

    def _check_contract(self, comm, row):
        if (self.policy == "C0" and not self.guarded_baseline) or row["phase"] not in {
            "FWD",
            "BWD",
        }:
            return
        key = id(comm)
        step, position = self._guard_positions.get(key, (row["runtime_microstep"], 0))
        template = self.contract["template"]
        if step != row["runtime_microstep"]:
            if position != len(template):
                raise RuntimeError(
                    f"Incomplete microstep scheduling contract {step}: {position}/{len(template)}"
                )
            step, position = row["runtime_microstep"], 0
        if position >= len(template) or signature(row) != template[position]:
            raise RuntimeError(
                f"Collective scheduling contract diverges at microstep {step}, ordinal {position}"
            )
        self._guard_positions[key] = step, position + 1

    def _issue(
        self,
        original,
        comm,
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
        import torch
        import torch.distributed as dist

        # Reuse the runtime RLock: guard/metadata/GPU edges and original issue form
        # one host submission operation even when autograd calls from another thread.
        with comm.submission_lock:
            if (
                not self.record_events
                and self.policy == "C0"
                and self.injection is None
                and not self.guarded_baseline
            ):
                return original(
                    comm,
                    pg_id,
                    kind,
                    group,
                    inputs,
                    output,
                    operation,
                    channel=channel,
                    asynchronous=asynchronous,
                    group_role=group_role,
                    extra_outputs=extra_outputs,
                )
            backend, members, degree = comm._process_group_metadata(group)
            training = comm.phase in {"FWD", "BWD"}
            if training and (self.policy != "C0" or self.guarded_baseline):
                if not (
                    comm.streams
                    and backend == "nccl"
                    and comm.current_stream_version_qualified
                    and comm.communication_policy == "stream_ordered"
                ):
                    raise RuntimeError(
                        "BLOCKED: policy requires qualified Torch 2.9.1 direct NCCL streams"
                    )
            rank = comm.rank
            self._rank = rank
            context = self._step_context.get(id(comm), {})
            outputs = (output, *extra_outputs)
            row = {
                "run_id": self.run_id,
                "experiment_id": self.experiment_id,
                "implementation": "shardtrain",
                "policy": self.policy,
                "rank": rank,
                "world_size": dist.get_world_size() if dist.is_initialized() else 1,
                "host_ordinal": self.event_count,
                "runtime_op_id": comm.next_op_id,
                "process_group_label": pg_id,
                "process_group_id": id(group),
                "communicator_id_or_label": f"{pg_id}/ranks={','.join(map(str, members))}",
                "group_members": list(members),
                "group_size": degree,
                "collective_sequence": comm.sequence.get(pg_id, 0),
                "runtime_microstep": comm.step_id,
                "step": context.get("optimizer_step"),
                "optimizer_step": context.get("optimizer_step"),
                "measured_step": context.get("measured_step"),
                "microstep_in_optimizer_step": (
                    comm.step_id - context["initial_microstep"] - 1
                    if training and context
                    else None
                ),
                "unit_id": comm.unit_id,
                "layer": self._unit_layers.get(id(comm), {}).get(comm.unit_id),
                "phase": comm.phase,
                "group_role": group_role or comm.group_role,
                "collective_type": kind,
                "family": family(kind),
                "backend": backend,
                "channel": channel,
                "dtype": str(inputs.dtype),
                "input_shape": list(inputs.shape),
                "output_shapes": [list(value.shape) for value in outputs],
                "input_bytes": inputs.numel() * inputs.element_size(),
                "output_bytes": sum(value.numel() * value.element_size() for value in outputs),
                "materialization_generation": comm.generation,
                "dependencies": [],
                "cpu_clock": "time.monotonic_ns (single Linux host CLOCK_MONOTONIC)",
                "cpu_wrapper_entry_ns": time.monotonic_ns(),
                "cpu_collective_submit_ns": None,
                "cpu_collective_return_ns": None,
                "gpu_start_timestamp": None,
                "gpu_end_timestamp": None,
                "gpu_timestamp_source": (
                    "unmatched until NSYS correlation; CUDA Events give relative duration only"
                ),
                "status": "ISSUED",
            }
            self._check_contract(comm, row)
            if id(comm) not in self.stream_registries:
                self.stream_registries[id(comm)] = {
                    name: {"cuda_stream_ptr": value.cuda_stream, "torch_stream_id": value.stream_id}
                    for name, value in comm.streams.items()
                }
            predecessors = (
                dependency_predecessors(self.policy, row["family"], self._tails[id(comm)])
                if training
                else []
            )
            compute_tail = None
            if training and self.policy == "C3" and row["family"].startswith("fsdp_"):
                compute_tail = torch.cuda.Event()
                compute_tail.record(comm.streams["compute"])
            start = end = None

            def instrumented_operation(async_op):
                nonlocal start, end
                current = torch.cuda.current_stream(comm.device) if comm.streams else None
                row["actual_submission_stream_ptr"] = current.cuda_stream if current else None
                row["logical_stream_ptr"] = (
                    comm.streams[channel].cuda_stream if comm.streams else None
                )
                row["execution_mode"] = (
                    "nccl_current_stream_v291"
                    if (
                        comm.streams and backend == "nccl" and comm.current_stream_version_qualified
                    )
                    else "backend_internal_stream_unmatched"
                )
                if (
                    training
                    and self.policy != "C0"
                    and current.cuda_stream != comm.streams[channel].cuda_stream
                ):
                    raise RuntimeError(
                        "BLOCKED: actual submission stream differs from qualified channel"
                    )
                for predecessor in predecessors:
                    current.wait_event(predecessor["event"])
                    row["dependencies"].append(
                        {
                            "type": "EXPERIMENT_COMM_ORDER",
                            "producer_host_ordinal": predecessor["ordinal"],
                            "producer_family": predecessor["family"],
                            "consumer_stream": channel,
                            "cpu_synchronize": False,
                        }
                    )
                if compute_tail is not None:
                    current.wait_event(compute_tail)
                    row["dependencies"].append(
                        {"type": "EXPERIMENT_COMPUTE_BEFORE_FSDP", "cpu_synchronize": False}
                    )
                if self.injection is not None:
                    self.injection.before_collective(row, comm)
                if (
                    self.gpu_timing
                    and current is not None
                    and row["execution_mode"] == "nccl_current_stream_v291"
                ):
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record(current)
                label = (
                    f"interview/collective/{self.experiment_id}/rank{rank}/op{row['runtime_op_id']}"
                    f"/pg{pg_id}/seq{row['collective_sequence']}/{kind}/U{comm.unit_id}/{comm.phase}"
                )
                row["nvtx_label"] = label
                if current is not None and self.record_events:
                    torch.cuda.nvtx.range_push(label)
                try:
                    row["cpu_collective_submit_ns"] = time.monotonic_ns()
                    work = operation(async_op)
                    row["cpu_collective_return_ns"] = time.monotonic_ns()
                finally:
                    if current is not None and self.record_events:
                        torch.cuda.nvtx.range_pop()
                if end is not None:
                    end.record(current)
                return work

            try:
                ticket = original(
                    comm,
                    pg_id,
                    kind,
                    group,
                    inputs,
                    output,
                    instrumented_operation,
                    channel=channel,
                    asynchronous=asynchronous,
                    group_role=group_role,
                    extra_outputs=extra_outputs,
                )
                if training and ticket.ready_event is not None:
                    self._tails[id(comm)][channel] = {
                        "ordinal": row["host_ordinal"],
                        "event": ticket.ready_event,
                        "family": row["family"],
                    }
                if compute_tail is not None:
                    comm.streams["compute"].wait_event(ticket.ready_event)
                    row["dependencies"].append(
                        {"type": "EXPERIMENT_FSDP_BEFORE_COMPUTE", "cpu_synchronize": False}
                    )
                row["status"] = "SUBMITTED"
                if start is not None:
                    self._timings.append((row, start, end))
                return ticket
            except BaseException as exc:
                row["status"], row["failure_reason"] = "FAILED", repr(exc)
                raise
            finally:
                self.event_count += 1
                if self.record_events:
                    self.rows.append(row)

    def _train(self, original, trainer, *args, **kwargs):
        comm = trainer.runtime.comm
        key = id(trainer)
        step = self._trainer_steps.get(key, 0)
        self._trainer_steps[key] = step + 1
        self._step_context[id(comm)] = {
            "optimizer_step": step,
            "measured_step": step - self.warmup_steps if step >= self.warmup_steps else None,
            "initial_microstep": trainer.runtime.step_id,
        }
        self._capture_layout(trainer.runtime)
        return original(trainer, *args, **kwargs)

    def __enter__(self):
        from sharded_runtime.runtime.comm import CommunicationCoordinator
        from sharded_runtime.training.qwen2_trainer import QwenTrainer

        if self._installed or getattr(
            CommunicationCoordinator.issue, "_interview_instrumentation", False
        ):
            raise RuntimeError("Nested communication instrumentation is unsupported")
        self._original_issue = CommunicationCoordinator.issue
        self._original_train = QwenTrainer.train_step
        collector = self

        def issue(comm, *args, **kwargs):
            return collector._issue(collector._original_issue, comm, *args, **kwargs)

        issue._interview_instrumentation = True
        CommunicationCoordinator.issue = issue
        if self.patch_trainer:

            def train(trainer, *args, **kwargs):
                return collector._train(collector._original_train, trainer, *args, **kwargs)

            QwenTrainer.train_step = train
        self._installed = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        from sharded_runtime.runtime.comm import CommunicationCoordinator
        from sharded_runtime.training.qwen2_trainer import QwenTrainer

        CommunicationCoordinator.issue = self._original_issue
        QwenTrainer.train_step = self._original_train
        self._installed = False
        self._exit_exception = repr(exc) if exc else None
        validation_error = None
        if exc is None:
            try:
                if self.injection is not None:
                    self.injection.validate_hits()
                if self.contract is not None:
                    expected = len(self.contract["template"])
                    for microstep, position in self._guard_positions.values():
                        if position != expected:
                            raise RuntimeError(
                                f"Incomplete final scheduling microstep {microstep}: "
                                f"{position}/{expected}"
                            )
            except BaseException as error:
                validation_error = error
                self._exit_exception = repr(error)
        self.flush()
        self._timings.clear()
        self._tails.clear()
        if validation_error is not None:
            raise validation_error
        return False

    def report(self):
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "rank": self._rank,
            "policy": self.policy,
            "policy_description": POLICIES[self.policy],
            "event_count": self.event_count,
            "recorded_event_count": len(self.rows),
            "record_events": self.record_events,
            "guarded_baseline": self.guarded_baseline,
            "gpu_timing": self.gpu_timing,
            "status": "FAILED" if getattr(self, "_exit_exception", None) else "RECORDED",
            "exception": getattr(self, "_exit_exception", None),
            "runtime_layouts": list(self.layouts.values()),
            "stream_registries": list(self.stream_registries.values()),
            "injection": self.injection.summary() if self.injection else None,
            "limits": [
                "CUDA stream pointers are not NSYS integer stream IDs; match CUDA API correlation",
                "C3 gates runtime compute stream; ancillary packing/copy kernels "
                "may run on communication streams",
                "event durations include NCCL residency, not measured physical wire traffic",
            ],
        }

    def flush(self, *, output_dir=None, release_events=False):
        """Existing synchronized, untimed boundary only; does not synchronize GPU.

        Writes append-only event rows and a replaceable manifest. Keep raw run
        directories unique. release_events also releases completed timing pairs.
        """
        for row, start, end in self._timings:
            if end.query():
                row["gpu_collective_event_elapsed_ms"] = start.elapsed_time(end)
        directory = Path(output_dir) if output_dir else self.output_dir
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            suffix = f"rank{self._rank if self._rank is not None else 'unknown'}"
            flushed = getattr(self, "_flushed_rows", 0)
            with (directory / f"communication_events.{suffix}.jsonl").open(
                "a", encoding="utf-8"
            ) as target:
                for row in self.rows[flushed:]:
                    target.write(json.dumps(row, sort_keys=True) + "\n")
            self._flushed_rows = len(self.rows)
            with (directory / f"communication_manifest.{suffix}.json").open(
                "w", encoding="utf-8"
            ) as target:
                json.dump(self.report(), target, indent=2, sort_keys=True)
            if release_events:
                self.rows.clear()
                self._flushed_rows = 0
                self._timings.clear()


def instrumentation(**kwargs):
    return Collector(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualify", nargs="+", required=True, help="C0 per-rank communication_events JSONL"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate_schedule(args.qualify)
    with Path(args.output).open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
