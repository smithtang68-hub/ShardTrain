"""Synthetic summary fixtures: global tokens once, profiles separate, missing data honest."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.fsdp_interview.scripts.analyze_communication import (
    nccl_test_rows,
    series_label,
    timing_aggregation,
    training_run,
)


def fixture_run(directory, *, profile=False, diagnostic=False):
    for rank in range(4):
        steps = [
            {
                "optimizer_step": step,
                "measured_step": step,
                "valid_tokens": 8176,
                "unique_input_tokens": 8192,
                "unique_training_tokens": 8176,
                "finite_gradients": True,
                "step_time_ms": milliseconds,
                "rank_step_time_ms": milliseconds - rank,
                "physical_free_bytes_endpoint": 8 * 2**30,
            }
            for step, milliseconds in enumerate((1000, 2000))
        ]
        payload = {
            "rank": rank,
            "world_size": 4,
            "status": "PASS",
            "steps": steps,
            "workload_hash": "synthetic-workload",
            "dataset": {"tokens_sha256": "fixture"},
            "source_manifest_sha256": "fixture",
            "implementation": "shardtrain",
            "policy": "C0",
            "guarded_baseline": True,
            "event_recorder": False,
            "profile": profile,
            "gemm_shape_diagnostic": diagnostic,
            "performance_claim_allowed": not diagnostic,
            "peak_allocated_bytes": 10 * 2**30,
            "peak_reserved_bytes": 12 * 2**30,
        }
        (Path(directory) / f"benchmark.rank{rank}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


class SummaryChecks(unittest.TestCase):
    def test_global_token_counts_are_not_multiplied_by_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_run(directory)
            summary, steps = training_run(directory)
            self.assertAlmostEqual(summary["training_tokens_per_s"], 8176 * 2 / 3)
            aggregate = timing_aggregation([summary], steps)
            self.assertEqual(aggregate[0]["optimizer_steps"], 2)
            self.assertAlmostEqual(aggregate[0]["input_tokens_per_s"], 8192 * 2 / 3)

    def test_profiled_run_never_enters_primary_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_run(directory, profile=True)
            summary, steps = training_run(directory)
            self.assertEqual(timing_aggregation([summary], steps), [])
            self.assertIsNone(summary["mfu_percent"])

    def test_unprofiled_shape_diagnostic_never_enters_primary_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_run(directory, diagnostic=True)
            summary, steps = training_run(directory)
            self.assertEqual(summary["status"], "PASS")
            self.assertTrue(summary["diagnostic_shapes"])
            self.assertFalse(summary["performance_claim_allowed"])
            self.assertEqual(timing_aggregation([summary], steps), [])
            self.assertIsNone(summary["mfu_percent"])

    def test_recorded_c0_is_separate_from_fast_control(self):
        self.assertEqual(
            series_label({"implementation": "shardtrain", "policy": "C0", "event_recorder": True}),
            "shardtrain/C0_recorded",
        )
        self.assertEqual(
            series_label(
                {"implementation": "shardtrain", "policy": "C0", "guarded_baseline": True}
            ),
            "shardtrain/C0_guarded",
        )

    def test_standard_nccl_tests_columns_and_wrong_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_stdout.log"
            path.write_text(
                "1024 256 float none -1 10.5 0.1 0.05 0 9.0 0.12 0.06 1\n", encoding="utf-8"
            )
            rows = nccl_test_rows(path)
            self.assertEqual(rows[0]["time_us"], 10.5)
            self.assertEqual(rows[0]["status"], "PASS")
            self.assertEqual(rows[1]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
