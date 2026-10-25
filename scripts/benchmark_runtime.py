"""E4 timing/profile worker. Run only after runner has verified the NCCL correctness gate."""

import argparse
import time

import torch

from sharded_runtime.config import load_config
from sharded_runtime.launch.bootstrap import WorkerSession
from sharded_runtime.models.reference import build_reference_model, synthetic_batch
from sharded_runtime.observability.metadata import write_json
from sharded_runtime.observability.profiling import ProfileCapture
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.training.sharded_trainer import ShardedTrainer


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", default="configs/runtime_cpu.yaml")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-mode", choices=["torch", "nsight", "off"], default="torch")
    args, overrides = parser.parse_known_args()
    config = load_config(args.config, overrides)
    with WorkerSession(config, "runtime") as session:
        mesh = session.context
        model = build_reference_model(config.model, config.training.seed)
        runtime = ShardedRuntime(
            device=mesh.device,
            reshard_after_forward=config.runtime.reshard_after_forward,
            forward_prefetch=config.runtime.forward_prefetch,
            backward_prefetch=config.runtime.backward_prefetch,
            async_grad_rs=config.runtime.async_grad_rs,
            communication_policy=config.runtime.communication_policy,
            directory=session.directory,
        )
        runtime.register(model, mesh, sp=config.parallel.sp_enabled)
        trainer = ShardedTrainer(model, runtime, config.training.lr)
        seed = config.training.seed + mesh.coordinate.fsdp
        batch = runtime.input_for_rank(
            synthetic_batch(
                config.model, config.training.local_batch_size, seed, device=mesh.device
            )
        )
        target = runtime.input_for_rank(
            synthetic_batch(
                config.model, config.training.local_batch_size, seed + 10000, device=mesh.device
            )
        )
        for _ in range(args.warmup):
            trainer.train_step(batch, target)
        if mesh.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        rows = []
        profiler = ProfileCapture(
            session.directory, session.worker.rank, mesh.device, mode=args.profile_mode
        )
        profiler.start()
        try:
            for step in range(config.training.steps):
                start = time.perf_counter()
                result = trainer.train_step(batch, target)
                if mesh.device.type == "cuda":
                    torch.cuda.synchronize()
                duration = time.perf_counter() - start
                tokens = (
                    config.training.local_batch_size
                    * config.model.seq_len
                    * config.parallel.fsdp_degree
                )
                rows.append(
                    dict(
                        step=step,
                        seconds=duration,
                        loss=result.loss,
                        tokens_per_second=tokens / duration,
                    )
                )
                profiler.step(step)
        finally:
            profiler.close()
        write_json(
            session.directory / f"benchmark.rank{session.worker.rank}.json",
            dict(
                steps=rows,
                max_allocated=torch.cuda.max_memory_allocated()
                if mesh.device.type == "cuda"
                else None,
                max_reserved=torch.cuda.max_memory_reserved()
                if mesh.device.type == "cuda"
                else None,
                profile_mode=args.profile_mode,
                warmup=args.warmup,
                communication_policy=config.runtime.communication_policy,
                pg_order_gpu_waits=runtime.comm.gpu_wait_timings() if runtime.comm.streams else [],
                note="No overlap/speedup claim without timeline inspection",
            ),
        )


if __name__ == "__main__":
    main()
