"""Synthetic exact member/dtype/full-array reference checks."""

import json
import tempfile
import unittest
from pathlib import Path

from experiments.fsdp_interview.scripts.analyze_nccl_reference import compare_training


class NcclReferenceChecks(unittest.TestCase):
    def test_exact_full_output_dtype_and_members_only(self):
        reference = {
            "kind": "actual_training_message",
            "mode": "out_of_place",
            "status": "PASS",
            "exact_count_shape_supported": True,
            "exact_requested_size_reported": True,
            "physical_devices": "0,2",
            "collective": "all_gather",
            "dtype": "float",
            "reported_size_bytes": 2048,
            "time_us": 2000,
            "experiment_id": "synthetic",
            "stdout_path": "not-measured.log",
        }
        event = {
            "trace_status": "MATCHED",
            "family": "fsdp_parameter_ag",
            "rank": 0,
            "group_members": [0, 2],
            "dtype": "torch.float32",
            "output_bytes": 2048,
            "input_bytes": 1024,
            "phase": "FWD",
            "group_role": "parameter_materialization",
            "gpu_kernel_residency_union_ms": 8.0,
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.json"
            path.write_text(
                json.dumps({"source_sqlite": "not-measured.sqlite", "collectives": [event]})
            )
            result = compare_training([reference], [path])[0]
            self.assertEqual(result["isolated_exact_nccl_tests_mean_ms"], 2.0)
            self.assertEqual(result["training_p50_residency_ms"], 8.0)
            for mutation in ("members", "dtype", "size", "alignment"):
                wrong = dict(reference)
                if mutation == "members":
                    wrong["physical_devices"] = "1,3"
                elif mutation == "dtype":
                    wrong["dtype"] = "half"
                elif mutation == "size":
                    wrong["reported_size_bytes"] = 1024
                else:
                    wrong["exact_count_shape_supported"] = False
                self.assertEqual(
                    compare_training([wrong], [path])[0]["status"], "UNMATCHED_REFERENCE"
                )


if __name__ == "__main__":
    unittest.main()
