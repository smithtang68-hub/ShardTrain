"""CPU fixture checks for schedule legality guards and independent byte accounting.

These are parser/planner unit fixtures, not GPU experiment results.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.fsdp_interview.cases.q05_nccl_contention import (
    Collector,
    dependency_predecessors,
    signature,
    validate_schedule,
)
from experiments.fsdp_interview.cases.q06_comm_volume import (
    accounting_family,
    expected_vs_observed,
    launch_phase_prediction,
    modeled_send_bytes,
    volume_inventory,
)
from experiments.fsdp_interview.cases.q10_stream_sync import physical_stream_audit
from experiments.fsdp_interview.cases.q12_rank_skew import RankDelay, submit_skew


def event(rank, ordinal, *, kind="fsdp_ag", phase="FWD", unit=1, size=256):
    is_ag = kind == "fsdp_ag"
    members = [rank % 2, rank % 2 + 2]
    return {
        "world_size": 4,
        "rank": rank,
        "host_ordinal": ordinal,
        "process_group_label": "fsdp" if is_ag else "fsdp_grad",
        "collective_type": kind,
        "unit_id": unit,
        "phase": phase,
        "channel": "fsdp_ag" if is_ag else "fsdp_rs",
        "group_role": None,
        "input_shape": [size // 8 if is_ag else size // 4],
        "output_shapes": [[size // 4 if is_ag else size // 8]],
        "dtype": "torch.float32",
        "group_size": 2,
        "group_members": members,
        "collective_sequence": ordinal,
        "runtime_microstep": 0,
        "measured_step": 0,
        "input_bytes": size // 2 if is_ag else size,
        "output_bytes": size if is_ag else size // 2,
        "family": "fsdp_parameter_ag" if is_ag else "fsdp_gradient_rs",
        "cpu_collective_submit_ns": rank * 100_000,
        "runtime_op_id": ordinal,
    }


class CommunicationChecks(unittest.TestCase):
    def test_norm_kind_does_not_hide_large_replicated_gradient_traffic(self):
        row = {
            "collective_type": "tp_norm_allreduce",
            "family": "tp_norm_allreduce",
            "group_role": "norm_gradient_bucket",
        }
        self.assertEqual(accounting_family(row), "tp_norm_gradient_allreduce")
        row["group_role"] = "tp_replicated_gradient"
        self.assertEqual(accounting_family(row), "tp_replicated_gradient_allreduce")

    def test_guarded_baseline_checks_same_contract_without_extra_edges(self):
        with self.assertRaisesRegex(RuntimeError, "requires a cross-rank C0 contract"):
            Collector(policy="C0", record_events=False, guarded_baseline=True)
        contract = {"status": "QUALIFIED", "template": [signature(event(0, 0))]}
        baseline = Collector(
            policy="C0", record_events=False, guarded_baseline=True, contract=contract
        )
        coordinator = object()
        baseline._check_contract(coordinator, event(0, 0))
        with self.assertRaisesRegex(RuntimeError, "contract diverges"):
            baseline._check_contract(object(), event(0, 0, phase="BWD"))
        self.assertEqual(
            dependency_predecessors(
                "C0",
                "fsdp_parameter_ag",
                {
                    "sp_ag": {"ordinal": 1, "family": "sp_activation_ag"},
                },
            ),
            [],
        )

    def test_policy_edges_allow_ag_rs_only_for_c2(self):
        tails = {
            "fsdp_ag": {"ordinal": 2, "family": "fsdp_parameter_ag"},
            "fsdp_rs": {"ordinal": 3, "family": "fsdp_gradient_rs"},
        }
        self.assertEqual(dependency_predecessors("C2", "fsdp_parameter_ag", tails), [])
        self.assertEqual(
            [row["ordinal"] for row in dependency_predecessors("C2", "sp_activation_ag", tails)],
            [2, 3],
        )
        self.assertEqual(
            [row["ordinal"] for row in dependency_predecessors("C1", "sp_activation_ag", tails)],
            [3],
        )
        tails["compute"] = {"ordinal": 4, "family": "tp_norm_allreduce"}
        self.assertEqual(
            [row["ordinal"] for row in dependency_predecessors("C2", "fsdp_parameter_ag", tails)],
            [4],
        )

    def test_order_contract_rejects_per_rank_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for rank in range(4):
                rows = [event(rank, 0), event(rank, 1, phase="BWD")]
                path = Path(directory) / f"rank{rank}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                paths.append(path)
            self.assertEqual(validate_schedule(paths)["status"], "QUALIFIED")
            bad = [event(3, 0, phase="BWD"), event(3, 1)]
            paths[-1].write_text("".join(json.dumps(row) + "\n" for row in bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "template differs"):
                validate_schedule(paths)

    def test_independent_model_retained_units_and_split_gradient_buckets(self):
        unit0 = {
            "unit_id": 0,
            "expected_parameter_ag_fwd": 1,
            "expected_parameter_ag_bwd": 0,
            "expected_gradient_rs_buckets": 1,
            "padded_full_parameter_bytes": 256,
            "unpadded_tp_local_parameter_bytes": 252,
            "retains_forward_parameters": True,
        }
        unit1 = {
            **unit0,
            "unit_id": 1,
            "expected_parameter_ag_bwd": 1,
            "expected_gradient_rs_buckets": 2,
            "retains_forward_parameters": False,
        }
        rows = [
            event(0, 0, unit=0),
            event(0, 1, kind="fsdp_rs", phase="BWD", unit=0),
            event(0, 2),
            event(0, 3, phase="BWD"),
            event(0, 4, kind="fsdp_rs", phase="BWD", size=16),
            event(0, 5, kind="fsdp_rs", phase="BWD", size=240),
        ]
        manifest = {"rank": 0, "runtime_layouts": [{"fsdp_degree": 2, "units": [unit0, unit1]}]}
        checks = expected_vs_observed(rows, [manifest])
        self.assertEqual([row["status"] for row in checks], ["PASS", "PASS"])
        rows[-1]["input_bytes"] -= 4
        self.assertEqual(expected_vs_observed(rows, [manifest])[-1]["status"], "FAIL")

    def test_tied_second_use_is_predicted_from_source_config_not_observed_phase(self):
        unit = {
            "unit_id": 1,
            "expected_parameter_ag_fwd": 1,
            "expected_parameter_ag_bwd": 1,
            "expected_gradient_rs_buckets": 1,
            "padded_full_parameter_bytes": 256,
            "unpadded_tp_local_parameter_bytes": 256,
            "retains_forward_parameters": False,
        }
        layout = {
            "fsdp_degree": 2,
            "sp_enabled": True,
            "reshard_after_forward": True,
            "units": [{"unit_id": 0, "unit_class": "TiedEmbeddingUnit"}, unit],
        }
        switches = {"forward_prefetch": True, "communication_policy": "stream_ordered"}
        prediction = launch_phase_prediction(layout, unit, switches)
        self.assertEqual((prediction["launch_fwd"], prediction["launch_bwd"]), (2, 0))
        self.assertEqual((prediction["first_consume_fwd"], prediction["first_consume_bwd"]), (1, 1))
        switches["forward_prefetch"] = False
        self.assertEqual(launch_phase_prediction(layout, unit, switches)["launch_bwd"], 1)

    def test_ring_model_and_sp_inventory_are_distinct(self):
        ag = event(0, 0)
        rs = event(0, 1, kind="fsdp_rs", phase="BWD")
        sp = {**ag, "collective_type": "sp_ag", "family": "sp_activation_ag"}
        self.assertEqual(modeled_send_bytes(ag)[0], 128)
        self.assertEqual(modeled_send_bytes(rs)[0], 128)
        inventory, _ = volume_inventory([ag, rs, sp])
        self.assertEqual(
            {row["family"] for row in inventory},
            {"fsdp_parameter_ag", "fsdp_gradient_rs", "sp_activation_ag"},
        )

    def test_skew_requires_full_group_and_same_collective(self):
        rows = [event(0, 0), event(2, 0)]
        result = submit_skew(rows)
        self.assertEqual(result[0]["cpu_submit_skew_ms"], 0.2)
        self.assertEqual(submit_skew(rows[:1])[0]["status"], "UNMATCHED")

    def test_stream_mapping_requires_exact_nvtx_correlation(self):
        rows = [{**event(0, 0), "nvtx_label": "collective-A"}]
        kernels = [{"nvtx_label": "collective-B", "correlation_id": 1, "stream_id": 7}]
        self.assertEqual(physical_stream_audit(rows, kernels)[0]["status"], "UNMATCHED")
        kernels[0]["nvtx_label"] = "collective-A"
        self.assertEqual(physical_stream_audit(rows, kernels)[0]["nsys_stream_ids"], [7])

    def test_injection_refuses_unhit_target(self):
        delay = RankDelay(rank=0, unit_id=1)
        delay._rank_seen = 0
        with self.assertRaisesRegex(RuntimeError, "no selected collective was hit"):
            delay.validate_hits()

    def test_repeated_tied_fwd_ag_injects_only_selected_occurrence(self):
        delay = RankDelay(rank=0, unit_id=1, delay_ms=0)
        delay._prepared, delay._rank_seen = True, 0
        first = {**event(0, 0), "microstep_in_optimizer_step": 0}
        second = {**event(0, 1), "microstep_in_optimizer_step": 0}
        delay.before_collective(first, None)
        delay.before_collective(second, None)
        delay.validate_hits()
        self.assertEqual(delay.summary()["hit_count"], 1)
        self.assertEqual(delay.summary()["candidate_count"], 2)
        self.assertEqual(second["rank_injection_candidate"]["status"], "SKIPPED")
        self.assertNotIn("rank_injection", second)


if __name__ == "__main__":
    unittest.main()
