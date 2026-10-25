"""Causal-LM optimizer owner with FP32 shards, AMP, and global token weighting."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn import functional as F

from sharded_runtime.runtime.qwen2_adapter import _sum_metric
from sharded_runtime.training.qwen2_precision import resolve_qwen_compute_dtype


@contextmanager
def training_range(name, device):
    with torch.profiler.record_function("shardtrain/qwen/" + name):
        if device.type == "cuda":
            torch.cuda.nvtx.range_push("shardtrain/qwen/" + name)
        try:
            yield
        finally:
            if device.type == "cuda":
                torch.cuda.nvtx.range_pop()


@dataclass
class QwenStepResult:
    step: int
    loss: float
    valid_tokens: int
    overflow: bool
    loss_scale: float
    accumulation_steps: int


class QwenTrainer:
    def __init__(
        self, model, runtime, lr=0.001, *, dtype="fp32", optimizer="sgd", accumulation_steps=1
    ):
        self.compute_dtype = resolve_qwen_compute_dtype(dtype, runtime.device)
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be positive")
        self.model, self.runtime, self.dtype = model, runtime, dtype
        self.accumulation_steps, self.step_id = accumulation_steps, 0
        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(runtime.sharded_parameters(), lr=lr)
        elif optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(runtime.sharded_parameters(), lr=lr, foreach=False)
        else:
            raise ValueError("Qwen optimizer must be sgd or adamw")
        runtime.bind_optimizer(self.optimizer)
        self.scaler = torch.amp.GradScaler("cuda", enabled=dtype == "fp16", init_scale=128.0)

    def _globally_finite(self, reported=None, *, return_tensor=False):
        runtime = self.runtime
        finite = (
            torch.stack([torch.isfinite(p.grad).all() for p in runtime.sharded_parameters()])
            .all()
            .to(torch.int32)
        )
        if reported is not None:
            # Loss validity participates in the same rank-wide decision as the
            # gradients. Keep the result on-device until the metric read below.
            finite.mul_(torch.isfinite(reported).to(torch.int32))
        runtime.comm.unit_id, runtime.comm.phase = None, "AMP"
        for role, group, degree, channel in [
            ("tp", runtime.tp.process_group, runtime.tp.degree, "sp_rs"),
            ("fsdp", runtime.materializer.group, runtime.materializer.degree, "fsdp_rs"),
        ]:
            if degree > 1:
                runtime.comm.issue(
                    role,
                    "amp_finite_" + role,
                    group,
                    finite,
                    finite,
                    lambda async_, group=group: dist.all_reduce(
                        finite, op=dist.ReduceOp.MIN, group=group, async_op=async_
                    ),
                    channel=channel,
                    asynchronous=False,
                )
        return finite if return_tensor else bool(finite.item())

    def _read_step_metrics(self, reported, global_count, count_upper_bound):
        finite = self._globally_finite(reported, return_tensor=True)
        values = [finite.to(torch.float64), reported.detach().to(torch.float64)]
        exact_count = count_upper_bound <= 2**53
        if exact_count:
            # Every integer up to this shape-derived bound is exactly representable
            # in FP64. Larger counts retain their int64 host-read path below.
            values.append(global_count.to(torch.float64))
        host_values = torch.stack(values).to(device="cpu").tolist()
        valid_tokens = (
            int(host_values[2]) if exact_count else int(global_count.to(device="cpu").item())
        )
        return bool(host_values[0]), float(host_values[1]), valid_tokens

    def train_step(self, input_ids, labels=None, *, attention_mask=None, position_ids=None):
        runtime = self.runtime
        labels = input_ids if labels is None else labels
        batch = input_ids.shape[0]
        if labels.shape != input_ids.shape or batch % self.accumulation_steps:
            raise ValueError("Labels must match input IDs and batch must divide accumulation_steps")
        micro = batch // self.accumulation_steps
        numerator_total = torch.zeros((), device=runtime.device)
        with runtime.compute_context():
            try:
                self.optimizer.zero_grad(set_to_none=True)
                runtime.comm.unit_id, runtime.comm.phase = None, "METRIC"
                shifted = runtime.input_for_rank(F.pad(labels[:, 1:], (0, 1), value=-100))
                global_count = _sum_metric(runtime, shifted.ne(-100).sum(), "lm_accum_valid_count")
                runtime.qwen_denominator_override = global_count.clamp_min(1)
                for index in range(self.accumulation_steps):
                    start, end = index * micro, (index + 1) * micro

                    def context_slice(value, start=start, end=end):
                        return (
                            value[start:end]
                            if value is not None and value.shape[0] == batch
                            else value
                        )

                    runtime.begin_step(runtime.step_id + 1)
                    with torch.autocast(
                        runtime.device.type,
                        dtype=self.compute_dtype,
                        enabled=self.dtype != "fp32",
                        cache_enabled=False,
                    ):
                        with training_range("model_forward", runtime.device):
                            output = self.model(
                                input_ids[start:end],
                                attention_mask=context_slice(attention_mask),
                                position_ids=context_slice(position_ids),
                            )
                        with training_range("loss", runtime.device):
                            loss = runtime.qwen_loss(output.logits, labels[start:end])
                    numerator_total.add_(runtime.qwen_numerator)
                    with training_range("backward", runtime.device):
                        self.scaler.scale(loss).backward()
                        # Wait for all leaf writes before unscaling. FP16 overflow is
                        # decided globally below; padding and ownership stay checked.
                        runtime.finalize_backward(allow_nonfinite=self.scaler.is_enabled())
                    if index + 1 < self.accumulation_steps:
                        runtime.end_step()
                    del output, loss
                runtime.qwen_numerator = numerator_total
                runtime.comm.unit_id, runtime.comm.phase = None, "METRIC"
                reported = runtime.qwen_reported_loss()
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                finite, reported_loss, valid_tokens = self._read_step_metrics(
                    reported, global_count, labels.numel() * runtime.materializer.degree
                )
                loss_scale = None
                with training_range("optimizer", runtime.device):
                    if finite:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    elif self.scaler.is_enabled():
                        # Every rank skips the same update and adopts the same scale,
                        # even when only a TP/FSDP subset observed local overflow.
                        loss_scale = self.scaler.get_scale() * self.scaler.get_backoff_factor()
                        self.scaler.update(new_scale=loss_scale)
                        # Explicit new_scale does not reset GradScaler's growth
                        # counter. Apply the standard overflow reset on every rank.
                        scaler_state = self.scaler.state_dict()
                        # Reuse this public snapshot's stored FP32 scale, including
                        # any rounding from a user-selected backoff factor.
                        loss_scale = scaler_state["scale"]
                        scaler_state["_growth_tracker"] = 0
                        self.scaler.load_state_dict(scaler_state)
                    else:
                        raise RuntimeError("Nonfinite Qwen loss/gradients")
                runtime.end_step()
                if loss_scale is None:
                    loss_scale = self.scaler.get_scale()
                result = QwenStepResult(
                    self.step_id,
                    reported_loss,
                    valid_tokens,
                    not finite,
                    loss_scale,
                    self.accumulation_steps,
                )
                self.step_id += 1
                return result
            except BaseException as exc:
                runtime.fail(exc)
                raise
            finally:
                runtime.__dict__.pop("qwen_denominator_override", None)
