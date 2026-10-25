"""Synthetic NSYS schema fixtures; never reported as measured GPU evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from experiments.fsdp_interview.scripts.parse_trace import (
    analyze,
    complement,
    duration_ms,
    host_allocation_apis,
    intersection,
    pinned_timelines,
)


class ParserChecks(unittest.TestCase):
    def test_host_api_requires_same_thread_enclosing_scope(self):
        data = {
            "scopes": [
                {
                    "scope_id": 1,
                    "globalTid": 7,
                    "start": 0,
                    "end": 1000,
                    "label": "q01q03/pinned_preallocate/bytes=4",
                }
            ],
            "apis": [
                {
                    "api_table": "CUPTI_ACTIVITY_KIND_RUNTIME",
                    "api_rowid": 1,
                    "api_name": "cudaHostAlloc_v3020",
                    "start": 100,
                    "end": 200,
                    "globalTid": 7,
                    "pid": 1,
                },
                {
                    "api_table": "CUPTI_ACTIVITY_KIND_RUNTIME",
                    "api_rowid": 2,
                    "api_name": "cudaHostRegister",
                    "start": 100,
                    "end": 200,
                    "globalTid": 8,
                    "pid": 1,
                },
            ],
        }
        result = host_allocation_apis(data)
        self.assertEqual(result[0]["scope_status"], "MATCHED")
        self.assertEqual(result[1]["scope_status"], "UNMATCHED")
        self.assertEqual(result[1]["enclosing_nvtx"], [])

    def test_pinned_overlap_uses_copy_api_and_gemm_submit_scope(self):
        label = "q01q03/P4/bytes=4/repeat=0/iteration=0/warmup=0"
        data = {
            "scopes": [
                {"pid": 1, "globalTid": 7, "start": 0, "end": 10_000_000, "label": label},
                {
                    "pid": 1,
                    "globalTid": 7,
                    "start": 2_000_000,
                    "end": 3_000_000,
                    "label": "q01q03/h2d_submit",
                },
            ],
            "apis": [
                {
                    "pid": 1,
                    "globalTid": 7,
                    "start": 2_100_000,
                    "end": 2_200_000,
                    "correlationId": 13,
                }
            ],
            "copies": [
                {"pid": 1, "correlationId": 13, "start": 4_000_000, "end": 8_000_000, "streamId": 2}
            ],
        }
        kernels = [
            {
                "pid": 1,
                "enclosing_nvtx_labels": [label, "q01q03/independent_gemm_submit"],
                "gpu_start_ns": 6_000_000,
                "gpu_end_ns": 9_000_000,
                "stream_id": 3,
            }
        ]
        result = pinned_timelines(data, kernels)
        self.assertEqual(result[0]["transfer_gemm_temporal_overlap_ms"], 2)
        data["apis"][0]["globalTid"] = 8
        self.assertEqual(pinned_timelines(data, kernels)[0]["status"], "UNMATCHED")

    def test_union_intersection_and_clipped_complement(self):
        self.assertEqual(intersection([(0, 10), (5, 15)], [(7, 12)]), [(7, 12)])
        self.assertEqual(complement([(-10, 3), (7, 12)], 0, 10), [(3, 7)])
        self.assertEqual(duration_ms([(0, 1_000_000), (500_000, 2_000_000)]), 2)

    def test_exact_thread_and_api_correlation_not_nearest_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.sqlite"
            events = []
            database = sqlite3.connect(path)
            database.executescript(
                "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);"
                "CREATE TABLE NVTX_EVENTS (start INTEGER,end INTEGER,"
                "globalTid INTEGER,text TEXT,textId INTEGER);"
                "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER,end INTEGER,"
                "globalTid INTEGER,nameId INTEGER,correlationId INTEGER);"
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER,"
                "globalPid INTEGER,streamId INTEGER,deviceId INTEGER,"
                "demangledName INTEGER,correlationId INTEGER);"
            )
            database.executemany(
                "INSERT INTO StringIds VALUES (?,?)",
                [
                    (1, "cudaLaunchKernel"),
                    (2, "ncclDevKernel_AllGather_RING_LL"),
                    (3, "gemm"),
                ],
            )
            for rank in range(2):
                pid = 100 + rank
                tid = (pid << 24) + pid
                label = f"interview/collective/test/rank{rank}/op0/pgfsdp/seq0/fsdp_ag/U1/FWD"
                database.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)",
                    (
                        0,
                        20_000_000,
                        tid,
                        f"interview/step/0/rank{rank}/shardtrain",
                        None,
                    ),
                )
                database.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)",
                    (
                        1_000_000,
                        4_000_000,
                        tid,
                        label,
                        None,
                    ),
                )
                database.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
                    (
                        2_000_000 + rank * 100_000,
                        2_100_000 + rank * 100_000,
                        tid,
                        1,
                        7,
                    ),
                )
                database.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
                    (
                        5_000_000,
                        10_000_000,
                        pid << 24,
                        20,
                        rank,
                        2,
                        7,
                    ),
                )
                database.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
                    (
                        8_000_000,
                        12_000_000,
                        pid << 24,
                        7,
                        rank,
                        3,
                        8,
                    ),
                )
                # Same correlation ID, wrong thread: it cannot match collective label.
                database.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
                    (
                        2_000_000,
                        2_100_000,
                        tid + 1,
                        1,
                        9,
                    ),
                )
                database.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
                    (
                        10_000_000,
                        11_000_000,
                        pid << 24,
                        24,
                        rank,
                        2,
                        9,
                    ),
                )
                events.append(
                    {
                        "rank": rank,
                        "runtime_op_id": 0,
                        "nvtx_label": label,
                        "family": "fsdp_parameter_ag",
                        "unit_id": 1,
                        "phase": "FWD",
                        "channel": "fsdp_ag",
                        "process_group_label": "fsdp",
                        "group_members": [0, 1],
                        "collective_sequence": 0,
                        "collective_type": "fsdp_ag",
                        "runtime_microstep": 0,
                        "measured_step": 0,
                    }
                )
            database.commit()
            database.close()
            event_path = Path(temporary) / "events.jsonl"
            event_path.write_text(
                "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
            )
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            result = analyze(path, [event_path])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            self.assertEqual(len(result["collectives"]), 2)
            self.assertTrue(all(row["trace_status"] == "MATCHED" for row in result["collectives"]))
            self.assertEqual(result["group_skew"][0]["nsys_cpu_launch_submit_skew_ms"], 0.1)
            self.assertEqual(result["step_metrics"][0]["compute_nccl_temporal_overlap_ms"], 3)
            self.assertEqual(
                sum(row["is_nccl"] and row["status"] == "UNMATCHED" for row in result["kernels"]), 2
            )


if __name__ == "__main__":
    unittest.main()
