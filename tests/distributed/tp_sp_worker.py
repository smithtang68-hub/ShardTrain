"""Real torchrun worker for rank-sensitive mappings, Linear and Block correctness."""

import argparse
import runpy
from pathlib import Path

import torch
from torch import nn

from sharded_runtime.config import config_from_dict
from sharded_runtime.launch.bootstrap import WorkerSession
from sharded_runtime.models.reference import build_reference_model, synthetic_batch
from sharded_runtime.observability.metadata import write_json
from sharded_runtime.parallel.mappings import (
    TPGroup,
    copy_to_tp,
    gather_sequence,
    reduce_from_tp,
    reduce_scatter_sequence,
)
from sharded_runtime.parallel.model import parallelize_tp_sp
from sharded_runtime.parallel.tp_linear import ColumnParallelLinear, RowParallelLinear
from sharded_runtime.training.trainer import Trainer


def check(actual, expected, *, rtol=3e-5, atol=3e-6):
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def mappings_case(group):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).requires_grad_()
    before = x.detach().clone()
    (copy_to_tp(x, group) * (group.rank + 1)).sum().backward()
    check(x, before, rtol=0, atol=0)
    check(x.grad, torch.full_like(x, sum(range(1, group.degree + 1))), rtol=0, atol=0)
    x = (before + group.rank).requires_grad_()
    y = reduce_from_tp(x, group)
    check(y, before * group.degree + sum(range(group.degree)), rtol=0, atol=0)
    y.sum().backward()
    check(x.grad, torch.ones_like(x), rtol=0, atol=0)
    full = torch.arange(48, dtype=torch.float32).reshape(2, 6, 4)
    local = group.sequence_shard(full).detach().clone().requires_grad_()
    gathered = gather_sequence(local, group)
    check(gathered, full, rtol=0, atol=0)
    (gathered * (group.rank + 1)).sum().backward()
    check(local.grad, torch.full_like(local, sum(range(1, group.degree + 1))), rtol=0, atol=0)
    partial = (full + group.rank).requires_grad_()
    y = reduce_scatter_sequence(partial, group)
    expected = group.sequence_shard(full * group.degree + sum(range(group.degree)))
    check(y, expected, rtol=0, atol=0)
    (y * (group.rank + 1)).sum().backward()
    gradient = torch.cat([torch.full_like(y, rank + 1) for rank in range(group.degree)], dim=1)
    check(partial.grad, gradient, rtol=0, atol=0)
    check(partial, full + group.rank, rtol=0, atol=0)
    return {"case": "mappings", "status": "passed", "rank_sensitive_adjoint_checks": 4}


def linear_case(group, sp):
    torch.random.default_generator.manual_seed(1234)
    up = nn.Linear(8, 12, bias=False)
    down = nn.Linear(12, 8, bias=False)
    # Broadcast logical parameters; compare against the actual source weights.
    with torch.no_grad():
        group.broadcast(up.weight)
        group.broadcast(down.weight)
    column = ColumnParallelLinear(up, group, sp=sp)
    row = RowParallelLinear(down, group, sp=sp)
    batch = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8) / 64
    reference_x = batch.clone().requires_grad_()
    local_x = (group.sequence_shard(batch) if sp else batch).clone().requires_grad_()
    hidden = column(local_x)
    expected_hidden = up(reference_x)
    check(hidden, expected_hidden.chunk(group.degree, dim=-1)[group.rank])
    actual, expected = row(hidden), down(expected_hidden)
    check(actual, group.sequence_shard(expected) if sp else expected)
    (actual.square().mean() / (group.degree if sp else 1)).backward()
    expected.square().mean().backward()
    check(local_x.grad, group.sequence_shard(reference_x.grad) if sp else reference_x.grad)
    check(group.gather(column.weight.grad, 0), up.weight.grad)
    check(group.gather(row.weight.grad, 1), down.weight.grad)
    return {"case": "linear", "status": "passed", "sp": sp}


def block_case(session, sp, accumulation, layers):
    config = session.config.model
    group = session.context.tp_group
    reference = build_reference_model(config, 1234)
    parameters = {
        name: p.detach().clone().requires_grad_() for name, p in reference.named_parameters()
    }
    oracle = runpy.run_path(str(Path(__file__).parents[1] / "correctness/test_reference.py"))[
        "functional_oracle"
    ]
    # Deliberately different rank-1 initialization must be replaced by source rank broadcast.
    model = build_reference_model(config, 1234 + session.worker.rank)
    plan = parallelize_tp_sp(model, group, sp=sp)
    trainer = Trainer(
        model, lr=session.config.training.lr, accumulation_steps=accumulation, tp_plan=plan
    )
    identities = {name: id(p) for name, p in model.named_parameters()}
    for name, value in plan.global_parameters().items():
        check(value, parameters[name], rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        if plan.partitions[name].dim is not None:
            assert (
                parameter.untyped_storage().nbytes() == parameter.numel() * parameter.element_size()
            )
    results = []
    for step in range(3):
        model.zero_grad(set_to_none=True)
        batch = synthetic_batch(config, 4, 4321 + step + session.worker.rank)
        plan.group.broadcast(batch)
        reference_x = batch.clone().requires_grad_()
        local = plan.input_for_rank(batch).detach().clone().requires_grad_()
        expected = oracle(reference_x, parameters, config)
        actual = model(local)
        check(actual, plan.input_for_rank(expected), rtol=2e-5, atol=2e-6)
        if sp:
            check(plan.group.gather(actual.detach(), 1), expected, rtol=2e-5, atol=2e-6)
        plan.backward_loss(actual).backward()
        plan.finalize_gradients()
        reference_loss = expected.square().mean()
        reference_loss.backward()
        check(local.grad, plan.input_for_rank(reference_x.grad))
        for name, gradient in plan.global_parameters(gradients=True).items():
            check(gradient, parameters[name].grad)
        result = trainer.train_step(local.detach())
        check(torch.tensor(result.loss), reference_loss.detach(), rtol=2e-5, atol=2e-6)
        # Includes accumulation=2 and the once-per-step replicated gradient reduction.
        for name, gradient in plan.global_parameters(gradients=True).items():
            check(gradient, parameters[name].grad)
        with torch.no_grad():
            for parameter in parameters.values():
                parameter.sub_(session.config.training.lr * parameter.grad)
                parameter.grad = None
        for name, value in plan.global_parameters().items():
            check(value, parameters[name])
        assert {name: id(p) for name, p in model.named_parameters()} == identities
        assert {id(p) for group in trainer.optimizer.param_groups for p in group["params"]} == set(
            identities.values()
        )
        results.append({"step": step, "loss": result.loss})
    return {
        "case": "block",
        "status": "passed",
        "sp": sp,
        "accumulation": accumulation,
        "layers": layers,
        "steps": results,
        "parameters_compared": len(parameters),
        "model_seed": 1234,
        "input_seed": 4321,
        "global_batch_size": 4,
        "learning_rate": session.config.training.lr,
        "reduce_scatter_path": "all_reduce_plus_slice" if sp else "unused",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["mappings", "linear", "block"], required=True)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--mlp-hidden-size", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--debug.output_dir", dest="output_dir", required=True)
    args = parser.parse_args()
    config = config_from_dict(
        {
            "model": {
                "hidden_size": args.hidden_size,
                "num_heads": args.num_heads,
                "mlp_hidden_size": args.mlp_hidden_size,
                "seq_len": args.seq_len,
                "layers": args.layers,
            },
            "parallel": {"tp_degree": args.degree, "sp_enabled": args.sp},
            "training": {
                "accumulation_steps": args.accumulation,
                "local_batch_size": 4 // args.accumulation,
                "lr": 0.01,
            },
            "debug": {"output_dir": args.output_dir, "distributed_timeout_seconds": 20},
        }
    )
    with WorkerSession(config, "train") as session:
        group = TPGroup.from_process_group(session.context.tp_group)
        if args.case == "mappings":
            result = mappings_case(group)
        elif args.case == "linear":
            result = linear_case(group, args.sp)
        else:
            result = block_case(session, args.sp, args.accumulation, args.layers)
        session.logger.info("M1 correctness passed: %s", result)
        write_json(session.directory / f"correctness.rank{session.worker.rank}.json", result)


if __name__ == "__main__":
    main()
