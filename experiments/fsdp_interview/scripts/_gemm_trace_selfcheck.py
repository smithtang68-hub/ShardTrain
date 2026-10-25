"""Synthetic shape/NVTX/API/kernel fixtures; no measured GPU evidence."""

from __future__ import annotations

import copy
import unittest

from experiments.fsdp_interview.scripts.analyze_gemm_trace import (
    ScopeIndex,
    correlate_shapes,
    shape_summary,
)


def fixture():
    pid, native_tid = 123, 456
    tid = (pid << 24) | native_tid
    label = "interview/gemm/rank=0/ordinal=0/op=aten.mm.default/M=3/N=7/K=5/B=1/dtype=torch.float16"
    data = {
        "path": "synthetic-not-measured.sqlite",
        "scopes": [
            {
                "scope_id": 1,
                "pid": pid,
                "globalTid": tid,
                "start": 0,
                "end": 100,
                "label": "actual_backward_source_nvtx",
            },
            {"scope_id": 2, "pid": pid, "globalTid": tid, "start": 20, "end": 40, "label": label},
        ],
        "apis": [
            {
                "pid": pid,
                "globalTid": tid,
                "start": 25,
                "end": 30,
                "correlationId": 7,
                "api_table": "CUPTI_ACTIVITY_KIND_RUNTIME",
                "api_rowid": 1,
                "api_name": "cudaLaunchKernel",
            },
        ],
        "kernels": [
            {
                "pid": pid,
                "kernel_rowid": 1,
                "start": 200_000,
                "end": 1_200_000,
                "correlationId": 7,
                "streamId": 20,
                "deviceId": 0,
                "kernel_name": "actual_gemm",
                "is_nccl": False,
            },
            {
                "pid": pid + 1,
                "kernel_rowid": 2,
                "start": 1_200_000,
                "end": 2_200_000,
                "correlationId": 7,
                "streamId": 99,
                "deviceId": 1,
                "kernel_name": "different_process_same_correlation",
                "is_nccl": False,
            },
        ],
    }
    shape = {
        "rank": 0,
        "cpu_ordinal": 0,
        "cpu_native_thread_id": native_tid,
        "operator": "aten.mm.default",
        "M": 3,
        "N": 7,
        "K": 5,
        "batch_count": 1,
        "classification": "LOCAL",
        "shape_source": "PLAIN_LOCAL_ATEN",
        "operation_status": "RETURNED",
        "nvtx_label": label,
        "nvtx_emitted": True,
        "left": {"dtype": "torch.float16", "kind": "Tensor"},
        "right": {"dtype": "torch.float16", "kind": "Tensor"},
        "output": {"dtype": "torch.float16"},
        "phase": "train_step",
        "cpu_start_monotonic_ns": 1_000_000_000_000_000,
    }
    return data, shape


class GemmTraceChecks(unittest.TestCase):
    def test_exact_local_shape_and_cross_pid_reject(self):
        data, shape = fixture()
        rows, kernels = correlate_shapes(data, [shape])
        self.assertEqual(rows[0]["gpu_trace_status"], "MATCHED")
        self.assertEqual(rows[0]["mathematical_gemm_flops"], 210)
        self.assertEqual(rows[0]["gpu_kernel_residency_union_ms"], 1)
        self.assertEqual(rows[0]["kernel_rowids"], [1])
        self.assertEqual(len(kernels), 1)
        self.assertEqual(rows[0]["phase"], "train_step")
        self.assertEqual(
            rows[0]["enclosing_source_nvtx"][0]["label"], "actual_backward_source_nvtx"
        )
        self.assertEqual(shape_summary(rows)[0]["invocations"], 1)

    def test_native_thread_and_api_full_scope_boundaries(self):
        for mutation in ("metadata_thread", "api_thread", "api_end"):
            data, shape = fixture()
            if mutation == "metadata_thread":
                shape["cpu_native_thread_id"] += 1
            elif mutation == "api_thread":
                data["apis"][0]["globalTid"] += 1
            else:
                data["apis"][0]["end"] = 41
            rows, kernels = correlate_shapes(data, [shape])
            self.assertEqual(rows[0]["gpu_trace_status"], "UNMATCHED", mutation)
            self.assertEqual(kernels, [])

    def test_ambiguous_nvtx_and_duplicate_metadata_reject(self):
        data, shape = fixture()
        data["scopes"].append({**data["scopes"][1], "scope_id": 3})
        self.assertEqual(correlate_shapes(data, [shape])[0][0]["gpu_trace_status"], "UNMATCHED")
        data, shape = fixture()
        rows, _ = correlate_shapes(data, [shape, copy.deepcopy(shape)])
        self.assertTrue(all(row["gpu_trace_status"] == "UNMATCHED" for row in rows))

    def test_unknown_fake_global_and_unemitted_do_not_become_gpu_gemm(self):
        for classification in ("UNKNOWN", "LOCAL_LINEAR_COMPOSITE"):
            data, shape = fixture()
            shape.update(classification=classification, shape_source="LOGICAL_DTENSOR_NOT_LOCAL")
            rows, kernels = correlate_shapes(data, [shape])
            self.assertEqual(rows[0]["gpu_trace_status"], "EXCLUDED")
            self.assertIsNone(rows[0]["mathematical_gemm_flops"])
            self.assertEqual(kernels, [])
        data, shape = fixture()
        shape["nvtx_emitted"] = False
        self.assertEqual(correlate_shapes(data, [shape])[0][0]["gpu_trace_status"], "UNMATCHED")

    def test_inconsistent_local_source_or_fake_operands_rejected(self):
        for mutation in ("source", "FakeTensor", "DTensor"):
            data, shape = fixture()
            if mutation == "source":
                shape["shape_source"] = "LOGICAL_DTENSOR_NOT_LOCAL"
            else:
                shape["left"]["kind"] = mutation
            rows, kernels = correlate_shapes(data, [shape])
            self.assertEqual(rows[0]["gpu_trace_status"], "EXCLUDED")
            self.assertIsNone(rows[0]["mathematical_gemm_flops"])
            self.assertEqual(kernels, [])

    def test_auxiliary_kernels_remain_separate(self):
        data, shape = fixture()
        data["kernels"].append(
            {
                **data["kernels"][0],
                "kernel_rowid": 3,
                "kernel_name": "pack_unknown_auxiliary",
                "start": 500_000,
                "end": 1_000_000,
            }
        )
        rows, kernels = correlate_shapes(data, [shape])
        self.assertEqual(rows[0]["gpu_kernel_count"], 2)
        self.assertEqual(rows[0]["gpu_kernel_residency_union_ms"], 1)
        self.assertEqual(kernels[1]["kernel_name"], "pack_unknown_auxiliary")

    def test_interval_index_matches_independent_containment(self):
        rows = [{"id": index, "start": index % 7, "end": 20 + index % 11} for index in range(30)]
        index = ScopeIndex(rows)
        for start in range(32):
            for end in range(start, 32):
                expected = {
                    row["id"] for row in rows if row["start"] <= start and end <= row["end"]
                }
                observed = {row["id"] for row in index.containing(start, end)}
                self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
