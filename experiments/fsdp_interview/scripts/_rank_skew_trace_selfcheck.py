"""Synthetic exact-PG and injection-scope fixtures, without actual trace access."""

from __future__ import annotations

import copy
import unittest

from experiments.fsdp_interview.scripts.analyze_rank_skew_trace import analyze


def event(rank, seq, unit, api, start, end, selected=False):
    row = {
        "rank": rank,
        "process_group_label": "fsdp",
        "group_members": [0, 2],
        "collective_sequence": seq,
        "unit_id": unit,
        "runtime_microstep": 20,
        "runtime_op_id": seq,
        "optimizer_step": 10,
        "step": 10,
        "phase": "FWD",
        "collective_type": "fsdp_ag",
        "channel": "fsdp_ag",
        "trace_status": "MATCHED",
        "cpu_nsys_submit_ns": api,
        "gpu_start_timestamp": start,
        "gpu_end_timestamp": end,
        "gpu_kernel_residency_union_ms": (end - start) / 1e6,
        "stream_ids": [20],
    }
    if selected:
        row["rank_injection"] = {"mode": "cpu", "cpu_delay_ms": 10, "cpu_injection_wall_ms": 10.1}
        row["rank_injection_candidate"] = {"status": "SELECTED"}
    return row


class RankSkewChecks(unittest.TestCase):
    def data(self):
        return {
            "source_sqlite": "fixture.sqlite",
            "kernels": [],
            "auxiliary_nvtx_ranges": [],
            "step_windows": [
                {"rank": 0, "step": 10, "pid": 123, "start_ns": 0, "end_ns": 100000000}
            ],
            "collectives": [
                event(0, 10, 0, 1, 2, 30000000),
                event(2, 10, 0, 1, 2, 30000000),
                event(0, 11, 1, 11000000, 40000000, 50000000, True),
                event(2, 11, 1, 1000000, 30000000, 50000010),
            ],
        }

    def test_same_pg_signed_skew_and_preceding_queue(self):
        result = analyze(self.data())
        target = result["targets"][0]
        comparison = target["signed_comparisons"][0]
        self.assertEqual(result["matched_targets"], 1)
        self.assertEqual(comparison["target_minus_peer_api_ms"], 10)
        self.assertEqual(comparison["target_minus_peer_gpu_start_ms"], 10)
        self.assertEqual(comparison["target_minus_peer_gpu_end_ms"], -0.00001)
        self.assertEqual(target["preceding_same_channel"]["remaining_gpu_ms_at_target_api"], 19)

    def test_wrong_members_sequence_or_duplicate_never_match(self):
        for alteration in ("members", "seq", "duplicate"):
            data = self.data()
            if alteration == "members":
                data["collectives"][-1]["group_members"] = [0, 1]
            elif alteration == "seq":
                data["collectives"][-1]["collective_sequence"] += 1
            else:
                data["collectives"].append(copy.deepcopy(data["collectives"][-1]))
            result = analyze(data)
            self.assertEqual(result["matched_targets"], 0)
            self.assertEqual(result["targets"][0]["signed_comparisons"], [])

    def test_gpu_exact_same_thread_scope_not_nearest_activity(self):
        data = self.data()
        data["collectives"][2]["rank_injection"]["mode"] = "gpu"
        label = "interview/rank_gpu_injection/rank0/U1"
        scope = {"label": label, "pid": 123, "globalTid": 987, "start": 3, "end": 100}
        data["auxiliary_nvtx_ranges"] = [scope]
        kernel = {
            "pid": 123,
            "enclosing_nvtx_labels": [label],
            "is_nccl": False,
            "gpu_start_ns": 300,
            "gpu_end_ns": 500,
            "stream_id": 20,
            "launch_apis": [{"globalTid": 987, "start": 4, "end": 99}],
        }
        data["kernels"] = [kernel]
        self.assertEqual(analyze(data)["targets"][0]["gpu_injection"]["status"], "MATCHED")
        kernel["launch_apis"][0]["globalTid"] = 988
        self.assertEqual(analyze(data)["targets"][0]["gpu_injection"]["status"], "UNMATCHED")
        kernel["launch_apis"][0]["globalTid"] = 987
        kernel["launch_apis"][0]["end"] = 101
        self.assertEqual(analyze(data)["targets"][0]["gpu_injection"]["status"], "UNMATCHED")
        kernel["launch_apis"][0]["end"] = 99
        data["auxiliary_nvtx_ranges"].append(copy.deepcopy(scope))
        self.assertEqual(analyze(data)["targets"][0]["gpu_injection"]["status"], "UNMATCHED")


if __name__ == "__main__":
    unittest.main()
