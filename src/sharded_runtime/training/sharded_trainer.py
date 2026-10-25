"""Data/loss/optimizer owner; runtime buffers and materialization remain encapsulated."""

import torch

from sharded_runtime.training.trainer import StepResult


class ShardedTrainer:
    def __init__(self, model, runtime, lr):
        self.model, self.runtime = model, runtime
        self.optimizer = torch.optim.SGD(runtime.sharded_parameters(), lr=lr)
        runtime.bind_optimizer(self.optimizer)
        self.step_id = 0

    def train_step(self, batch, target):
        runtime = self.runtime
        with runtime.compute_context():
            try:
                runtime.begin_step(self.step_id)
                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(batch)
                loss = runtime.local_loss(output, target)
                loss.backward()
                runtime.finalize_backward()
                reported = runtime.reported_loss(loss)
                grad_l2 = (
                    sum(
                        p.grad.detach().double().square().sum()
                        for p in runtime.sharded_parameters()
                    )
                    .sqrt()
                    .item()
                )
                self.optimizer.step()
                parameter_l2 = (
                    sum(p.detach().double().square().sum() for p in runtime.sharded_parameters())
                    .sqrt()
                    .item()
                )
                runtime.end_step()
                result = StepResult(self.step_id, reported.item(), grad_l2, parameter_l2)
                self.step_id += 1
                return result
            except BaseException as exc:
                runtime.fail(exc)
                raise
