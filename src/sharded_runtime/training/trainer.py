"""Reference or TP/SP trainer with explicit, equal-size microbatch normalization."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sharded_runtime.parallel.model import TensorParallelPlan


@dataclass(frozen=True)
class StepResult:
    step: int
    loss: float
    gradient_l2: float
    parameter_l2: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        accumulation_steps: int = 1,
        *,
        tp_plan: TensorParallelPlan | None = None,
    ) -> None:
        if type(accumulation_steps) is not int or accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be a positive integer")
        self.model = model
        if tp_plan is not None and tp_plan.model is not model:
            raise ValueError("TP plan belongs to a different model")
        if getattr(model, "_tp_plan", None) is not tp_plan:
            raise ValueError("A converted model must be trained with its own TP/SP plan")
        self.tp_plan = tp_plan
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.accumulation_steps = accumulation_steps
        self.step_id = 0

    def train_step(self, batch: Tensor) -> StepResult:
        if batch.ndim != 3 or batch.shape[0] == 0 or batch.shape[0] % self.accumulation_steps:
            raise ValueError("Batch must be [B,S,H] with B divisible by accumulation_steps")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), dtype=batch.dtype, device=batch.device)
        for microbatch in batch.chunk(self.accumulation_steps, dim=0):
            output = self.model(microbatch)
            loss = (
                self.tp_plan.backward_loss(output) if self.tp_plan else output.square().mean()
            ) / self.accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {self.step_id}")
            loss.backward()
            loss_sum += loss.detach()
        if self.tp_plan:
            self.tp_plan.finalize_gradients()
            loss_sum = self.tp_plan.reported_loss(loss_sum)
        # Keep gradients available for correctness checks until the next step.
        gradients = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                raise RuntimeError(f"Missing gradient for {name} at step {self.step_id}")
            if not torch.isfinite(param.grad).all():
                raise FloatingPointError(f"Non-finite gradient for {name} at step {self.step_id}")
            gradients.append(param.grad.detach().double().square().sum())
        grad_l2 = torch.stack(gradients).sum().sqrt().item()
        self.optimizer.step()
        with torch.no_grad():
            parameter_l2 = (
                torch.stack([p.double().square().sum() for p in self.model.parameters()])
                .sum()
                .sqrt()
                .item()
            )
        result = StepResult(self.step_id, loss_sum.item(), grad_l2, parameter_l2)
        self.step_id += 1
        return result


class ReferenceTrainer(Trainer):
    """Backward-compatible name for the unpartitioned reference training path."""
