"""Run reference/TP/SP training or an explicit process-group collective smoke test."""

import argparse
from dataclasses import asdict

from sharded_runtime.config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ShardTrain M1: FP32 reference, TP/SP training and topology smoke",
        epilog="Config overrides: --section.field value (e.g. --training.steps 3)",
        allow_abbrev=False,
    )
    parser.add_argument("--config")
    parser.add_argument("--smoke-groups", action="store_true", help="Check groups; do not train")
    parser.add_argument("--sharded-runtime", action="store_true", help="Use the v0.7 shard runtime")
    args, overrides = parser.parse_known_args(argv)
    try:
        config = load_config(args.config, overrides)
    except ConfigError as exc:
        parser.error(str(exc))
    if config.model.model_type == "qwen2" and not args.smoke_groups:
        # Qwen always uses the shard runtime, including the one-rank validation path.
        from sharded_runtime.training.qwen2_benchmark import run

        try:
            run(config, warmup=0, profile_mode="off")
        except ConfigError as exc:
            parser.error(str(exc))
        return 0
    # Defer torch imports until parsing has succeeded, keeping config diagnostics lightweight.
    from sharded_runtime.launch.bootstrap import WorkerSession
    from sharded_runtime.models.reference import build_reference_model, synthetic_batch
    from sharded_runtime.observability.metadata import write_json
    from sharded_runtime.parallel.model import parallelize_tp_sp
    from sharded_runtime.training.trainer import Trainer

    use_runtime = args.sharded_runtime or config.parallel.fsdp_degree > 1
    mode = "smoke" if args.smoke_groups else ("runtime" if use_runtime else "train")
    try:
        with WorkerSession(config, mode) as session:
            ctx = session.context
            if args.smoke_groups:
                results = ctx.smoke_collectives()
                write_json(session.directory / f"smoke.rank{session.worker.rank}.json", results)
            elif use_runtime:
                from sharded_runtime.runtime import ShardedRuntime
                from sharded_runtime.training.sharded_trainer import ShardedTrainer

                model = build_reference_model(config.model, config.training.seed)
                runtime = ShardedRuntime(
                    device=ctx.device,
                    reshard_after_forward=config.runtime.reshard_after_forward,
                    forward_prefetch=config.runtime.forward_prefetch,
                    backward_prefetch=config.runtime.backward_prefetch,
                    async_grad_rs=config.runtime.async_grad_rs,
                    communication_policy=config.runtime.communication_policy,
                    directory=session.directory,
                )
                runtime.register(model, ctx, sp=config.parallel.sp_enabled)
                trainer = ShardedTrainer(model, runtime, config.training.lr)
                results = []
                for step in range(config.training.steps):
                    seed = config.training.seed + 100 * (step + 1) + ctx.coordinate.fsdp
                    batch = synthetic_batch(
                        config.model, config.training.local_batch_size, seed, device=ctx.device
                    )
                    target = synthetic_batch(
                        config.model,
                        config.training.local_batch_size,
                        seed + 10000,
                        device=ctx.device,
                    )
                    result = trainer.train_step(
                        runtime.input_for_rank(batch), runtime.input_for_rank(target)
                    )
                    results.append(asdict(result))
                    session.logger.info("runtime step=%d loss=%.8f", result.step, result.loss)
                write_json(session.directory / f"steps.rank{session.worker.rank}.json", results)
            else:
                model = build_reference_model(config.model, config.training.seed).to(ctx.device)
                plan = (
                    parallelize_tp_sp(model, ctx.tp_group, sp=config.parallel.sp_enabled)
                    if config.parallel.tp_degree > 1
                    else None
                )
                trainer = Trainer(
                    model,
                    config.training.lr,
                    config.training.accumulation_steps,
                    tp_plan=plan,
                )
                results = []
                for step in range(config.training.steps):
                    # local_batch_size is per microbatch, effective B = B_local * accumulation.
                    batch = synthetic_batch(
                        config.model,
                        config.training.local_batch_size * config.training.accumulation_steps,
                        config.training.seed + step + 1,
                        device=ctx.device,
                    )
                    if plan:
                        plan.group.broadcast(batch)
                        batch = plan.input_for_rank(batch)
                    result = trainer.train_step(batch)
                    results.append(asdict(result))
                    session.logger.info(
                        "step=%d loss=%.8f gradient_l2=%.8f parameter_l2=%.8f",
                        result.step,
                        result.loss,
                        result.gradient_l2,
                        result.parameter_l2,
                    )
                write_json(session.directory / f"steps.rank{session.worker.rank}.json", results)
    except ConfigError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
