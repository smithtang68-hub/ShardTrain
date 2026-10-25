"""Static supported-Block runtime; stable shard leaves, genuine reshard and new BWD leases."""

from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn

from sharded_runtime.models.transformer_block import RMSNorm
from sharded_runtime.parallel.model import parallelize_tp_sp
from sharded_runtime.parallel.shared_projection import SharedProjectionGroup
from sharded_runtime.parallel.tp_linear import ColumnParallelLinear, RowParallelLinear
from sharded_runtime.runtime.bridge import PreBackward, UnitMaterialize
from sharded_runtime.runtime.comm import CommunicationCoordinator, RuntimeTPGroup
from sharded_runtime.runtime.layout import ParameterSpec, ShardedParameter, initialize_shard
from sharded_runtime.runtime.materializer import Materializer
from sharded_runtime.runtime.operators import RuntimeLinear, RuntimeNorm


class UnitState:
    def __init__(self, runtime, unit_id, block):
        self.runtime, self.unit_id, self.block = runtime, unit_id, block
        self.handles, self.consumers = [], []
        self.generation = 0
        self.state = "SHARDED"
        self.lease = None
        self.forward_calls = self.backward_calls = self.reductions = 0
        self.forward_prefetch_handle = self.backward_prefetch_handle = None

    def prefetch_after_input(self, handle, phase):
        if not self.runtime.defer_sp_prefetch:
            return
        if phase == "BWD" and self.runtime.async_grad_rs:
            # Async backward has one fixed entry/gradient-dispatch scheduling source.
            return
        boundary = self.forward_prefetch_handle if phase == "FWD" else self.backward_prefetch_handle
        if handle is not boundary:
            return
        enabled = (
            self.runtime.forward_prefetch if phase == "FWD" else self.runtime.backward_prefetch
        )
        target = self.unit_id + (1 if phase == "FWD" else -1)
        if enabled and 0 <= target < len(self.runtime.units):
            self.annotate(phase)
            self.runtime.comm.event(
                "prefetch_boundary", param_id=handle.spec.param_id, target_unit=target
            )
            self.runtime.prefetch(target, phase)
            self.annotate(phase)

    def annotate(self, phase):
        comm = self.runtime.comm
        comm.unit_id, comm.phase, comm.generation = self.unit_id, phase, self.generation

    def issue(self, phase):
        if self.lease is not None:
            raise RuntimeError(f"Duplicate materialization U{self.unit_id} {phase}")
        self.generation += 1
        self.annotate(phase)
        self.state = f"{phase}_MATERIALIZING"
        self.lease = self.runtime.materializer.issue(self, phase)
        self.runtime.comm.event("materialize_issue")

    def release(self):
        if self.lease is None:
            raise RuntimeError(f"Missing lease U{self.unit_id}")
        if self.runtime.comm.streams:
            self.lease.fullparam_last_consumer_event = torch.cuda.Event()
            self.lease.fullparam_last_consumer_event.record(
                torch.cuda.current_stream(self.runtime.device)
            )
        for lease in self.lease.full_param_storages.values():
            self.runtime.comm.retire(lease)
        self.lease.full_param_storages.clear()
        self.lease = None
        self.runtime.comm.event("fullparams_retired", unit_id=self.unit_id)

    def weight(self, name):
        if self.state != "BWD_ACTIVE" or self.lease is None:
            raise RuntimeError(f"Stale/early weight consumer {name} U{self.unit_id} {self.state}")
        if self.lease.step_id != self.runtime.step_id or self.lease.generation != self.generation:
            raise RuntimeError(f"Stale generation {name}")
        return self.lease.full_param_storages[name].tensor

    def pre_forward(self, module, args):
        if not self.runtime.active or self.forward_calls:
            raise RuntimeError(f"Forward outside begin_step or repeated Unit U{self.unit_id}")
        self.forward_calls += 1
        if self.lease is None:
            self.issue("FWD")
        self.annotate("FWD")
        channel = "fsdp_rs" if self.runtime.async_grad_rs else "compute"
        with self.runtime.comm.stream_context(channel):
            weights = UnitMaterialize.apply(self, *(h.local_shard for h in self.handles))
        if len(self.handles) == 1:
            weights = (weights,) if isinstance(weights, torch.Tensor) else weights
        for consumer, weight in zip(self.consumers, weights, strict=True):
            consumer.compute_weight = weight
        self.state = "FWD_ACTIVE"
        self.runtime.comm.event("unit_forward")
        if (
            not self.runtime.defer_sp_prefetch
            and self.runtime.forward_prefetch
            and self.unit_id + 1 < len(self.runtime.units)
        ):
            self.runtime.prefetch(self.unit_id + 1, "FWD")

    def post_forward(self, module, args, output):
        for consumer in self.consumers:
            consumer.compute_weight = None
        self.annotate("FWD")
        if self.runtime.reshard_after_forward and self.unit_id != len(self.runtime.units) - 1:
            self.release()
            self.state = "RESHARDED"
        else:
            self.state = "FWD_RETAINED"
        self.runtime.comm.event(self.state)
        return PreBackward.apply(output, self)

    def pre_backward(self):
        if not self.runtime.active or self.backward_calls:
            raise RuntimeError(
                f"Repeated/invalid backward U{self.unit_id}; retain_graph unsupported"
            )
        dispatch_prefetch = self.runtime.gradient_dispatch_prefetch
        if dispatch_prefetch:
            self.runtime.validate_backward_prefetch_entry(self)
        self.backward_calls += 1
        if self.lease is None:
            self.issue("BWD")
        self.annotate("BWD")
        if not self.lease.full_param_storages:
            self.runtime.materializer.finish(self)
        self.state = "BWD_ACTIVE"
        self.runtime.comm.event("unit_pre_backward")
        if dispatch_prefetch:
            self.runtime.backward_prefetch_entered(self)
        elif (
            self.runtime.backward_prefetch
            and self.unit_id
            and (not self.runtime.defer_sp_prefetch or self.runtime.async_grad_rs)
        ):
            self.runtime.comm.event(
                "prefetch_boundary",
                target_unit=self.unit_id - 1,
                scheduling_boundary="backward_unit_entry",
            )
            self.runtime.prefetch(self.unit_id - 1, "BWD")
            self.annotate("BWD")

    def reduce(self, grads):
        if self.reductions or self.state != "BWD_ACTIVE":
            raise RuntimeError(f"Duplicate/early gradient RS U{self.unit_id} {self.state}")
        self.reductions += 1
        self.annotate("BWD")
        self.state = "GRAD_REDUCING"
        self.runtime.comm.event(
            "unit_grads_ready", parameters=[h.spec.param_id for h in self.handles]
        )
        result = self.runtime.materializer.reduce(self, grads)
        self.release()
        self.state = "GRAD_READY"
        return result


class ShardedRuntime:
    def __init__(
        self,
        *,
        device="cpu",
        reshard_after_forward=True,
        forward_prefetch=False,
        backward_prefetch=False,
        async_grad_rs=False,
        communication_policy="conservative",
        directory=None,
        diagnostics_enabled=True,
    ):
        self.device = torch.device(device)
        self.reshard_after_forward = reshard_after_forward
        self.forward_prefetch, self.backward_prefetch = forward_prefetch, backward_prefetch
        self.async_grad_rs = async_grad_rs
        self.asynchronous = forward_prefetch or backward_prefetch or async_grad_rs
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.communication_policy = communication_policy
        self.comm = CommunicationCoordinator(
            self.device,
            rank,
            directory,
            communication_policy=communication_policy,
            diagnostics_enabled=diagnostics_enabled,
        )
        self.units, self.registry, self.hooks = [], {}, []
        self.step_id = -1
        self.active = self.optimizer_ready = self.failed = False
        self.identities = {}
        self.sp = False
        self.defer_sp_prefetch = False
        self.parameter_shards_ready_event = None
        self.parameter_shards_version = 0
        self.backward_prefetch_tokens = {}
        self.backward_entry_expected = self.backward_dispatch_expected = None

    @property
    def gradient_dispatch_prefetch(self):
        return (
            self.communication_policy == "stream_ordered"
            and self.sp
            and self.async_grad_rs
            and self.backward_prefetch
        )

    @staticmethod
    def _tied_backward_unit(unit):
        # The supported tied embedding has an early head entry and one late reduction.
        return unit.unit_id == 0 and callable(getattr(unit, "head_complete", None))

    def _validate_dispatch_lease(self, unit):
        lease = unit.lease
        if lease is None or lease.step_id != self.step_id or lease.generation != unit.generation:
            raise RuntimeError(f"Missing/stale gradient-dispatch lease U{unit.unit_id}")
        return lease

    def _validate_prefetch_token(self, current):
        token = self.backward_prefetch_tokens.get((self.step_id, current.unit_id))
        if token is None:
            raise RuntimeError(f"Missing gradient-dispatch token U{current.unit_id}")
        target = self.units[current.unit_id - 1]
        if (
            token["current_generation"] != current.generation
            or token["target_unit"] != target.unit_id
            or token["target_generation"] != target.generation
            or token["parameter_version"] != self.parameter_shards_version
        ):
            raise RuntimeError(f"Stale gradient-dispatch token U{current.unit_id}")
        self._validate_dispatch_lease(target)
        return token

    def _queue_backward_prefetch(self, current, *, owner=None):
        key = (self.step_id, current.unit_id)
        if key in self.backward_prefetch_tokens:
            raise RuntimeError(f"Duplicate gradient-dispatch token U{current.unit_id}")
        self._validate_dispatch_lease(current)
        target = self.units[current.unit_id - 1]
        tied = self._tied_backward_unit(target)
        if target.forward_calls != 1 or target.reductions:
            raise RuntimeError(f"Completed/unvisited gradient-dispatch target U{target.unit_id}")
        if tied:
            if (
                target.backward_calls != 1
                or target.state != "BWD_ACTIVE"
                or len(target.handles) != 1
                or target.handles[0].forward_uses != 2
            ):
                raise RuntimeError("Invalid retained tied backward target")
            self._validate_dispatch_lease(target)
        elif target.backward_calls or target.state not in {
            "RESHARDED",
            "FWD_RETAINED",
            "FWD_MATERIALIZING",
            "BWD_MATERIALIZING",
        }:
            raise RuntimeError(f"Entered/completed gradient-dispatch target U{target.unit_id}")
        elif target.lease is None:
            if target.state != "RESHARDED":
                raise RuntimeError(f"Missing retained gradient-dispatch target U{target.unit_id}")
        else:
            self._validate_dispatch_lease(target)
        source = (
            "previous_gradient_dispatch" if owner is not None else "backward_gradient_dispatch_seed"
        )
        annotated = owner if owner is not None else current
        annotated.annotate("BWD")
        self.comm.event(
            "prefetch_boundary",
            scheduling_boundary=source,
            gradient_owner=owner.unit_id if owner is not None else None,
            current_unit=current.unit_id,
            target_unit=target.unit_id,
            token_step=self.step_id,
        )
        retained = target.lease is not None
        self.prefetch(target.unit_id, "BWD")
        self.backward_prefetch_tokens[key] = {
            "current_generation": current.generation,
            "target_unit": target.unit_id,
            "target_generation": target.generation,
            "parameter_version": self.parameter_shards_version,
            "gradient_owner": owner.unit_id if owner is not None else None,
            "scheduling_boundary": source,
            "status": "tied_retained" if tied else "retained" if retained else "issued",
        }
        annotated.annotate("BWD")
        self.comm.event(
            "backward_prefetch_token",
            current_unit=current.unit_id,
            **self.backward_prefetch_tokens[key],
        )

    def validate_backward_prefetch_entry(self, unit):
        if self._tied_backward_unit(unit):
            if unit.state != "FWD_RETAINED":
                raise RuntimeError("Tied backward entry must retain its forward lease")
            self._validate_dispatch_lease(unit)
            return
        if unit.unit_id != self.backward_entry_expected:
            raise RuntimeError(
                f"Gradient-dispatch entry order: expected U{self.backward_entry_expected}, "
                f"got U{unit.unit_id}"
            )
        self._validate_dispatch_lease(unit)
        last = len(self.units) - 1
        if unit.unit_id != last:
            # Incoming AG and outgoing lookahead were both assigned by fixed sources.
            self._validate_prefetch_token(self.units[unit.unit_id + 1])
            if unit.unit_id:
                self._validate_prefetch_token(unit)

    def backward_prefetch_entered(self, unit):
        if self._tied_backward_unit(unit):
            self.comm.event("backward_prefetch_tied_entry", scheduling_excluded=True)
            return
        self.backward_entry_expected -= 1
        if unit.unit_id == len(self.units) - 1 and unit.unit_id:
            self._queue_backward_prefetch(unit)

    def dispatch_backward_prefetch(self, owner):
        """One deterministic source, between RS submission and any gradient consumption."""
        if owner.unit_id != self.backward_dispatch_expected:
            raise RuntimeError(
                f"Gradient-dispatch reduction order: expected U{self.backward_dispatch_expected}, "
                f"got U{owner.unit_id}"
            )
        if owner.state != "GRAD_REDUCING" or owner.backward_calls != 1 or owner.reductions != 1:
            raise RuntimeError(f"Invalid gradient-dispatch owner U{owner.unit_id}")
        if owner.unit_id >= 2:
            current = self.units[owner.unit_id - 1]
            if current.backward_calls or current.reductions:
                raise RuntimeError(
                    f"Gradient dispatch arrived after current entry U{current.unit_id}"
                )
            if self.backward_entry_expected != current.unit_id:
                raise RuntimeError(f"Invalid gradient-dispatch frontier U{current.unit_id}")
            self._validate_prefetch_token(owner)
            self._queue_backward_prefetch(current, owner=owner)
        elif owner.unit_id == 0 and self._tied_backward_unit(owner):
            if self.backward_entry_expected != 0:
                raise RuntimeError("Tied reduction preceded decoder backward completion")
            self.backward_entry_expected = -1
        self.backward_dispatch_expected -= 1

    def register(self, model, mesh=None, *, sp=False):
        if self.units:
            raise RuntimeError("Runtime already registered")
        add_close_callback = getattr(mesh, "add_close_callback", None)
        if add_close_callback is not None:
            add_close_callback(self.comm.close_diagnostics)
        from sharded_runtime.models.qwen2 import Qwen2CausalLM

        if isinstance(model, Qwen2CausalLM):
            from sharded_runtime.runtime.qwen2_adapter import register_qwen

            return register_qwen(self, model, mesh, sp=sp)
        if getattr(model, "_tp_plan", None) is not None:
            raise ValueError(
                "Register the original dense model, before TP or optimizer construction"
            )
        self.sp = sp
        tp_group = mesh.tp_group if mesh else None
        fsdp_group = mesh.fsdp_group if mesh else None
        # One canonical dense source supplies all F replicas before TP slicing.
        model.to(self.device)
        self.comm.phase = "INIT"
        if dist.is_initialized():
            with torch.no_grad():
                for parameter in model.parameters():
                    self.comm.issue(
                        "world",
                        "init_broadcast",
                        dist.group.WORLD,
                        parameter,
                        parameter,
                        lambda async_, parameter=parameter: dist.broadcast(
                            parameter, src=0, async_op=async_
                        ),
                        channel="fsdp_ag",
                        asynchronous=False,
                    )

        def broadcast_tp(parameter, group):
            if group.degree > 1:
                self.comm.issue(
                    "tp",
                    "init_tp_broadcast",
                    group.process_group,
                    parameter,
                    parameter,
                    lambda async_: dist.broadcast(
                        parameter,
                        src=dist.get_global_rank(group.process_group, 0),
                        group=group.process_group,
                        async_op=async_,
                    ),
                    channel="fsdp_ag",
                    asynchronous=False,
                )

        plan = parallelize_tp_sp(model, tp_group, sp=sp, broadcast_fn=broadcast_tp)
        self.tp = RuntimeTPGroup(
            tp_group, self.comm, grad_group=mesh.tp_grad_group if mesh else None
        )
        self.defer_sp_prefetch = (
            self.communication_policy == "stream_ordered" and sp and self.tp.degree > 1
        )
        self.materializer = Materializer(fsdp_group, self.comm)
        self.fsdp_grad_group = mesh.fsdp_grad_group if mesh else fsdp_group
        self.model = model
        blocks = list(model) if type(model) is nn.Sequential else [model]
        for unit_id, block in enumerate(blocks):
            unit = UnitState(self, unit_id, block)
            self.units.append(unit)
            prefix = f"{unit_id}." if type(model) is nn.Sequential else ""
            for local_name, parameter in list(block.named_parameters()):
                name = prefix + local_name
                owner_path = local_name.rsplit(".", 1)[0]
                owner = block.get_submodule(owner_path)
                partition = plan.partitions[name]
                layout = {0: "column", 1: "row", None: "replicated"}[partition.dim]
                spec = ParameterSpec.create(
                    name,
                    unit_id,
                    parameter,
                    partition.global_shape,
                    layout,
                    self.materializer.degree,
                )
                handle = ShardedParameter(
                    spec, initialize_shard(parameter, spec, self.materializer.rank), unit
                )

                def committed(parameter, handle=handle):
                    if handle.grad_step == self.step_id:
                        raise RuntimeError(f"Repeated gradient commit {handle.spec.param_id}")
                    handle.grad_step = self.step_id
                    if self.comm.streams:
                        handle.grad_ready_event = torch.cuda.Event()
                        handle.grad_ready_event.record(torch.cuda.current_stream(self.device))
                    self.comm.event(
                        "leaf_grad_committed",
                        param_id=handle.spec.param_id,
                        unit_id=handle.spec.unit_id,
                    )

                handle.local_shard.register_post_accumulate_grad_hook(committed)
                if type(owner) is RMSNorm:
                    consumer = RuntimeNorm(handle, owner.eps)
                elif type(owner) in {ColumnParallelLinear, RowParallelLinear}:
                    consumer = RuntimeLinear(handle, self.tp, layout == "column", sp)
                else:
                    raise ValueError(f"Unknown trainable consumer {name}: {type(owner)}")
                parent_path, child_name = (
                    owner_path.rsplit(".", 1) if "." in owner_path else ("", owner_path)
                )
                setattr(block.get_submodule(parent_path), child_name, consumer)
                unit.handles.append(handle)
                unit.consumers.append(consumer)
                self.registry[name] = handle
                self.identities[name] = (id(handle.local_shard), handle.local_shard.data_ptr())
            if sp:
                block.attention.projection_group = SharedProjectionGroup(
                    (block.attention.q_proj, block.attention.k_proj, block.attention.v_proj),
                    self.tp,
                    "qkv",
                )
                block.mlp.projection_group = SharedProjectionGroup(
                    (block.mlp.gate_proj, block.mlp.up_proj), self.tp, "gate_up"
                )
            if self.defer_sp_prefetch:
                # Explicit supported-Block boundaries, independent of parameter ordering.
                unit.forward_prefetch_handle = block.attention.v_proj.handle
                unit.backward_prefetch_handle = block.mlp.down_proj.handle
            self.hooks.extend(
                [
                    block.register_forward_pre_hook(unit.pre_forward),
                    block.register_forward_hook(unit.post_forward),
                ]
            )
        # TP plan owns no parameter tensors, but its old trainer API must not be used.
        model._sharded_runtime = self
        return self

    def sharded_parameters(self):
        return [handle.local_shard for handle in self.registry.values()]

    def bind_optimizer(self, optimizer):
        expected = {id(p) for p in self.sharded_parameters()}
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("Optimizer must hold exactly the persistent local shards")

        def guard(opt, args, kwargs):
            if not self.active or not self.optimizer_ready or self.failed:
                raise RuntimeError("Premature optimizer: call finalize_backward first")

        optimizer.register_step_pre_hook(guard)

        def updated(opt, args, kwargs):
            self.parameter_shards_version += 1
            self.publish_parameter_shards_ready("optimizer_update")

        optimizer.register_step_post_hook(updated)
        return optimizer

    def publish_parameter_shards_ready(self, source):
        """Publish the actual parameter writer, independently of activation work."""
        if self.comm.streams:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self.parameter_shards_ready_event = event
        self.comm.event(
            "parameter_shards_ready",
            source=source,
            parameter_version=self.parameter_shards_version,
        )

    def wait_parameter_shards_ready(self):
        if self.comm.streams:
            if self.parameter_shards_ready_event is None:
                raise RuntimeError("Parameter AG has no published shard readiness")
            self.comm.wait_event(
                self.parameter_shards_ready_event,
                dependency_type="DATA_DEP",
                consumer="parameter_shards",
                parameter_version=self.parameter_shards_version,
            )

    def begin_step(self, step_id):
        if self.active or self.failed or step_id != self.step_id + 1:
            raise RuntimeError("Invalid step boundary or FAILED runtime")
        self.step_id, self.active, self.optimizer_ready = step_id, True, False
        self.comm.step_id = step_id
        # Keep the actual optimizer writer across steps, even when the next model
        # entry uses another stream. Only initialization needs an entry publication.
        if not self.comm.streams or self.parameter_shards_ready_event is None:
            self.publish_parameter_shards_ready("step_entry")
        self.backward_prefetch_tokens.clear()
        self.backward_entry_expected = self.backward_dispatch_expected = len(self.units) - 1
        for unit in self.units:
            if unit.lease is not None or unit.state != "SHARDED":
                raise RuntimeError(f"Live lease/state across steps U{unit.unit_id}")
            unit.forward_calls = unit.backward_calls = unit.reductions = 0
            for handle in unit.handles:
                handle.forward_uses = 0

    def prefetch(self, unit_id, phase):
        unit = self.units[unit_id]
        if unit.lease is None:
            unit.issue(phase)
        else:
            self.comm.event("prefetch_skipped_retained", unit_id=unit_id, phase=phase)

    def finalize_backward(self, *, allow_nonfinite=False):
        if not self.active or self.optimizer_ready:
            raise RuntimeError("Invalid finalize_backward boundary")
        if self.gradient_dispatch_prefetch and (
            self.backward_entry_expected != -1 or self.backward_dispatch_expected != -1
        ):
            raise RuntimeError("Incomplete gradient-dispatch schedule")
        self.comm.finalize()
        for unit in self.units:
            if unit.state != "GRAD_READY" or unit.reductions != 1:
                raise RuntimeError(f"Missing gradient readiness U{unit.unit_id}")
        checked_names, invalid_gradients = [], []
        for name, handle in self.registry.items():
            if handle.grad_step != self.step_id:
                raise RuntimeError(f"Missing shard Autograd commit {name}")
            if handle.grad_ready_event is not None:
                self.comm.wait_event(
                    handle.grad_ready_event,
                    dependency_type="DATA_DEP",
                    consumer="optimizer",
                    param_id=name,
                )
            gradient = handle.local_shard.grad
            if gradient is None or gradient.shape != handle.local_shard.shape:
                raise RuntimeError(f"Missing/invalid shard .grad {name}")
            valid = handle.spec.valid_len(self.materializer.rank)
            invalid = None if allow_nonfinite else ~torch.isfinite(gradient).all()
            if valid < gradient.numel():
                # NaN/Inf in padding also count as nonzero, including during AMP
                # overflow handling. An empty tail is valid without a CUDA read.
                nonzero_padding = torch.count_nonzero(gradient[valid:]).ne(0)
                invalid = nonzero_padding if invalid is None else invalid | nonzero_padding
            if invalid is not None:
                checked_names.append(name)
                invalid_gradients.append(invalid)
        if invalid_gradients:
            # Keep the checks and their parameter identities on-device until one
            # batched host read. Avoid one implicit scalar synchronization per leaf.
            invalid_values = torch.stack(invalid_gradients).to(device="cpu").tolist()
            for name, invalid in zip(checked_names, invalid_values, strict=True):
                if invalid:
                    raise RuntimeError(f"Nonfinite gradient or nonzero padding {name}")
        if self.comm.live_buffers:
            raise RuntimeError("Unretired buffer at optimizer boundary")
        self.optimizer_ready = True
        self.comm.event("optimizer_ready")

    def end_step(self):
        if not self.optimizer_ready:
            raise RuntimeError("Premature optimizer/end_step")
        for name, handle in self.registry.items():
            if (id(handle.local_shard), handle.local_shard.data_ptr()) != self.identities[name]:
                raise RuntimeError(f"Shard identity/storage replaced {name}")
        for unit in self.units:
            unit.state = "SHARDED"
        self.active = False
        self.comm.event("step_end")
        self.comm.compact_completed_history()

    @contextmanager
    def compute_context(self):
        if not self.comm.streams:
            yield
            return
        caller = torch.cuda.current_stream(self.device)
        compute = self.comm.streams["compute"]
        entry = torch.cuda.Event()
        entry.record(caller)
        self.comm.wait_event(
            entry, dependency_type="DATA_DEP", stream=compute, consumer="compute_context_inputs"
        )
        with torch.cuda.stream(compute):
            yield
        exit_event = torch.cuda.Event()
        exit_event.record(compute)
        self.comm.wait_event(
            exit_event,
            dependency_type="DATA_DEP",
            stream=caller,
            consumer="compute_context_outputs",
        )

    def fail(self, reason):
        self.failed = True
        self.comm.event("runtime_failed", reason=str(reason))
        for unit in self.units:
            unit.state = "FAILED"

    def input_for_rank(self, batch):
        return self.tp.sequence_shard(batch) if self.sp else batch

    def local_loss(self, output, target):
        return (output - target).square().mean() / (self.tp.degree if self.sp else 1)

    def reported_loss(self, loss):
        self.comm.unit_id, self.comm.phase = None, "METRIC"
        value = self.tp.sum(loss.detach(), kind="loss_tp_sum") if self.sp else loss.detach().clone()
        if self.materializer.degree > 1:
            self.comm.issue(
                "fsdp",
                "loss_fsdp_sum",
                self.materializer.group,
                value,
                value,
                lambda async_: dist.all_reduce(
                    value, group=self.materializer.group, async_op=async_
                ),
                channel="fsdp_rs",
                asynchronous=False,
            )
            value.div_(self.materializer.degree)
        return value

    def verification_gather(self, tensor, group, axis):
        """Reference-only communication also enters the same PG sequence diagnostics."""
        self.comm.unit_id, self.comm.phase = None, "VERIFY"
        degree = dist.get_world_size(group) if dist.is_initialized() else 1
        outputs = [torch.empty_like(tensor) for _ in range(degree)]
        if degree == 1:
            outputs[0].copy_(tensor)
        else:
            self.comm.issue(
                axis,
                "verification_ag",
                group,
                tensor,
                outputs[0],
                lambda async_: dist.all_gather(outputs, tensor, group=group, async_op=async_),
                channel="fsdp_ag" if axis == "fsdp" else "sp_ag",
                asynchronous=False,
                extra_outputs=outputs[1:],
            )
        return outputs

    def reconstruct_tp_local(self, *, gradients=False):
        """Correctness-only reconstruction, not model compute."""
        results = {}
        for name, handle in self.registry.items():
            shard = handle.local_shard.grad if gradients else handle.local_shard
            if shard is None:
                raise RuntimeError(f"Missing gradient {name}")
            gathered = self.verification_gather(shard, self.materializer.group, "fsdp")
            full = torch.empty(handle.spec.numel, device=shard.device, dtype=shard.dtype)
            for rank, part in enumerate(gathered):
                valid = handle.spec.valid_len(rank)
                full[rank * handle.spec.fsdp_shard_numel :][:valid].copy_(part[:valid])
            results[name] = full.view(handle.spec.tp_local_shape)
        return results
