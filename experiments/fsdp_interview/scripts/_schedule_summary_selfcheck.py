"""Synthetic metadata adjacency and AR-vs-RS role classification fixtures."""

from __future__ import annotations

import unittest

from experiments.fsdp_interview.scripts.summarize_schedule_trace import schedule_summary


def event(opid, family, unit, phase, start, end, role=None):
    return {
        "rank": 0,
        "trace_status": "MATCHED",
        "runtime_microstep": 20,
        "runtime_op_id": opid,
        "family": family,
        "unit_id": unit,
        "phase": phase,
        "gpu_start_timestamp": int(start * 1e6),
        "gpu_end_timestamp": int(end * 1e6),
        "group_role": role,
        "logical_stream_ptr": 123,
        "actual_submission_stream_ptr": 123,
        "stream_ids": [20],
        "channel": family,
    }


class ScheduleChecks(unittest.TestCase):
    def data(self):
        return {
            "step_windows": [{"rank": 0, "pid": 100}],
            "kernels": [],
            "group_skew": [],
            "gpu_idle_gaps": [],
            "collectives": [
                event(0, "fsdp_parameter_ag", 2, "FWD", 0, 10),
                event(1, "sp_activation_ag", 1, "FWD", 5, 15),
                event(2, "fsdp_parameter_ag", 2, "BWD", 20, 40),
                event(3, "fsdp_gradient_rs", 4, "BWD", 30, 50),
                event(4, "tp_norm_allreduce", 3, "BWD", 25, 35, "norm_gradient_bucket"),
                event(5, "fsdp_gradient_rs", 3, "BWD", 20, 40, "norm_gradient_bucket"),
            ],
        }

    def test_adjacent_units_and_norm_ar_not_norm_rs(self):
        result = schedule_summary(self.data())
        fwd, bwd = result["adjacent_unit_overlaps"]
        self.assertEqual(fwd["fsdp_ag_sp_ag_ms"], 5)
        self.assertEqual(bwd["fsdp_ag_rs_norm_ms"], 5)
        self.assertEqual(result["phase_overlaps"][1]["fsdp_ag_rs_norm_ms"], 10)

    def test_unrelated_unit_triple_does_not_prove_requested_pipeline(self):
        data = self.data()
        data["collectives"][4]["unit_id"] = 8
        result = schedule_summary(data)
        self.assertFalse(any(row["phase"] == "BWD" for row in result["adjacent_unit_overlaps"]))
        self.assertEqual(result["phase_overlaps"][1]["fsdp_ag_rs_norm_ms"], 10)

    def test_replicated_gradient_not_norm_and_wrong_launch_phase_not_bwd(self):
        data = self.data()
        data["collectives"][4]["group_role"] = "tp_replicated_gradient"
        result = schedule_summary(data)
        self.assertEqual(result["phase_overlaps"][1]["fsdp_ag_rs_norm_ms"], 0)
        data["collectives"][2]["phase"] = "FWD"
        self.assertFalse(
            any(row["phase"] == "BWD" for row in schedule_summary(data)["adjacent_unit_overlaps"])
        )


if __name__ == "__main__":
    unittest.main()
