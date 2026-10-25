"""Reuse finalization training; inspect Adam step counters only after timing/cleanup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "experiments/fsdp_interview/scripts"))


def main():
    import train_workload as workload

    from experiments.finalization_4gpu import train as shared_train
    from experiments.q5_topology_revalidation_4gpu.contracts import save, sha

    original_trainer, original_run = workload.QwenTrainer, workload.run
    trainers = []

    class ObservedTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trainers.append(self)

    def run(args):
        payload = original_run(args)
        if len(trainers) != 1:
            raise ValueError("Expected one unchanged ShardTrain trainer")
        trainer = trainers[0]
        steps = [int(state["step"].item()) for state in trainer.optimizer.state.values()]
        expected = args.warmup + args.steps
        rows = payload["warmup_steps"] + payload["steps"]
        valid = bool(steps) and set(steps) == {expected} and trainer.step_id == expected
        valid = valid and len(rows) == expected and all(not row["overflow"] for row in rows)
        result = dict(
            status="PASS" if valid else "FAIL",
            expected_optimizer_updates=expected,
            observed_trainer_step_id=trainer.step_id,
            adam_state_leaf_count=len(steps),
            adam_leaf_step_min=min(steps) if steps else None,
            adam_leaf_step_max=max(steps) if steps else None,
            finite_clean_updates=sum(not row["overflow"] for row in rows),
            source_manifest_sha256=sha(os.environ["SHARDTRAIN_SOURCE_MANIFEST"]),
            scope="Actual final Adam leaf step counters; read once after all timing/cleanup, "
            "no optimizer math or timed training loop override",
        )
        save(Path(args.output_dir) / f"training_completion.rank{os.environ['RANK']}.json", result)
        if not valid:
            raise ValueError(
                "Full-model actual Adam update count differs from requested clean steps"
            )
        return payload

    workload.QwenTrainer, workload.run = ObservedTrainer, run
    try:
        shared_train.main()
    finally:
        workload.QwenTrainer, workload.run = original_trainer, original_run


if __name__ == "__main__":
    main()
