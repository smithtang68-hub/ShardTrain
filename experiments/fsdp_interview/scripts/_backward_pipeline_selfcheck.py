"""Synthetic owner-context association, independent of measured traces."""

from __future__ import annotations

import unittest

from experiments.fsdp_interview.scripts.analyze_backward_pipeline import (
    associate_current_compute,
    pipeline_rows,
)


class BackwardPipelineChecks(unittest.TestCase):
    def data(self):
        def event(opid, family, unit, start, end):
            return {
                "rank": 0,
                "runtime_microstep": 20,
                "phase": "BWD",
                "trace_status": "MATCHED",
                "family": family,
                "unit_id": unit,
                "runtime_op_id": opid,
                "gpu_start_timestamp": int(start * 1e6),
                "gpu_end_timestamp": int(end * 1e6),
                "group_role": "norm_gradient_bucket" if family == "tp_norm_allreduce" else None,
                "channel": "compute" if family == "tp_norm_allreduce" else family,
                "stream_ids": [16],
            }

        return {
            "step_windows": [{"rank": 0, "pid": 123}],
            "collectives": [
                event(0, "fsdp_parameter_ag", 2, 20, 40),
                event(1, "fsdp_gradient_rs", 4, 23, 38),
                event(2, "tp_norm_allreduce", 4, 25, 30),
            ],
            "kernels": [
                {
                    "pid": 123,
                    "is_nccl": False,
                    "stream_id": 16,
                    "kernel_rowid": 7,
                    "gpu_start_ns": 30_000_000,
                    "gpu_end_ns": 35_000_000,
                    "launch_apis": [{"globalTid": 456, "start": 15, "end": 16}],
                }
            ],
        }

    def test_exact_host_context_selects_nested_current_not_root(self):
        data = self.data()
        contexts = [
            {
                "pid": 123,
                "unit_id": 0,
                "runtime_microstep": 20,
                "start": 0,
                "end": 100,
                "globalTid": 456,
                "start_nvtx_rowid": 1,
                "end_nvtx_rowid": 2,
            },
            {
                "pid": 123,
                "unit_id": 3,
                "runtime_microstep": 20,
                "start": 10,
                "end": 20,
                "globalTid": 456,
                "start_nvtx_rowid": 3,
                "end_nvtx_rowid": 4,
            },
        ]
        kernels = associate_current_compute(data, contexts)
        self.assertEqual(kernels[0]["unit_id"], 3)
        # GPU starts far later than host context: correlation/context still valid.
        result = pipeline_rows(data, kernels)[0]
        self.assertEqual(
            (
                result["gradient_owner_unit"],
                result["current_bwd_unit"],
                result["target_prefetch_unit"],
            ),
            (4, 3, 2),
        )
        self.assertEqual(result["ag_rs_norm_owner_ms"], 5)
        self.assertEqual(result["ag_rs_compute_current_ms"], 5)
        self.assertEqual(result["norm_owner_compute_current_ms"], 0)

    def test_wrong_thread_cross_boundary_and_other_stream_rejected(self):
        for mutation in ("thread", "end", "stream"):
            data = self.data()
            if mutation == "thread":
                data["kernels"][0]["launch_apis"][0]["globalTid"] += 1
            elif mutation == "end":
                data["kernels"][0]["launch_apis"][0]["end"] = 21
            else:
                data["kernels"][0]["stream_id"] = 28
            contexts = [
                {
                    "pid": 123,
                    "unit_id": 3,
                    "runtime_microstep": 20,
                    "start": 10,
                    "end": 20,
                    "globalTid": 456,
                    "start_nvtx_rowid": 3,
                    "end_nvtx_rowid": 4,
                }
            ]
            self.assertEqual(associate_current_compute(data, contexts), [])


if __name__ == "__main__":
    unittest.main()
