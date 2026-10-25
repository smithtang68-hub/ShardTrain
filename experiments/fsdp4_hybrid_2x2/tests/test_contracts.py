"""CPU checks are not four-GPU correctness results."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from experiments.fsdp4_hybrid_2x2.contracts import CASES, comparison_contract, groups, input_blocks


class ContractsTest(unittest.TestCase):
    def test_amp_fixture_rejects_constant_only_ten_step_expansion(self):
        from experiments.fsdp4_hybrid_2x2.gate_worker import validate_amp_fixture_steps

        validate_amp_fixture_steps(3)
        with self.assertRaises(ValueError):
            validate_amp_fixture_steps(10)

    def test_groups_and_no_cross_rank_tp_for_f4(self):
        for case in CASES:
            g = groups(case)
            self.assertEqual(sorted(v for group in g["fsdp"] for v in group), list(range(4)))
            self.assertEqual(sorted(v for group in g["tp"] for v in group), list(range(4)))
        self.assertEqual(groups("N-F")["fsdp"], [[0, 1, 2, 3]])
        self.assertEqual(groups("S-F")["tp"], [[0], [1], [2], [3]])

    def test_full_step_actual_sample_formula(self):
        for step in range(40):
            expected = list(range(step * 16, (step + 1) * 16))
            for case, c in CASES.items():
                selected = [b for f in range(c["F"]) for b in input_blocks(case, f * c["T"], step)]
                self.assertEqual(selected, expected)
                for rank in range(4):
                    self.assertEqual(
                        input_blocks(case, rank, step),
                        input_blocks(case, rank // c["T"] * c["T"], step),
                    )

    def test_microstep_difference_is_explicit(self):
        c = comparison_contract()["microstep_global_block_ids_step0"]
        self.assertEqual(c["S-H"], c["N-H"])
        self.assertEqual(c["S-F"], c["N-F"])
        self.assertNotEqual(c["S-H"], c["S-F"])
        self.assertEqual(c["S-F"][0], [0, 1, 4, 5, 8, 9, 12, 13])

    def test_config_four_cases(self):
        from sharded_runtime.config import load_config

        for case, c in CASES.items():
            config = load_config(Path(__file__).resolve().parents[1] / "configs" / (case + ".yaml"))
            config.validate(4, mode="runtime")
            self.assertEqual(
                config.parallel.fsdp_degree
                * config.training.local_batch_size
                * config.training.accumulation_steps,
                16,
            )
            self.assertEqual(config.parallel.sp_enabled, c["SP"])

    def test_native_tp1_skips_only_tp_plan(self):
        import sharded_runtime.training.qwen2_native_baseline as native
        from experiments.fsdp4_hybrid_2x2.worker import native_tp1_scope

        model = SimpleNamespace()
        mesh = SimpleNamespace(size=lambda: 1)
        with patch.object(native, "parallelize_native_qwen") as old:
            with native_tp1_scope():
                native.parallelize_native_qwen(model, mesh, sp_enabled=False)
                self.assertEqual((model.tp_degree, model.tp_rank, model.sp_enabled), (1, 0, False))
                self.assertTrue(model._p1_tp_plan_disabled)
                old.assert_not_called()
                with self.assertRaises(ValueError):
                    native.parallelize_native_qwen(model, mesh, sp_enabled=True)

    def test_actual_adam_counters_and_byte_audit(self):
        import torch

        from experiments.fsdp4_hybrid_2x2.worker import optimizer_end_audit

        parameters = [torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(2))]
        optimizer = torch.optim.AdamW(parameters, lr=1e-5, foreach=False)
        for p in parameters:
            p.grad = torch.ones_like(p)
        optimizer.step()
        audit = optimizer_end_audit(optimizer, 1)
        self.assertEqual((audit["step_min"], audit["step_max"], audit["leaf_count"]), (1, 1, 2))
        self.assertEqual(audit["parameter_bytes"], 20)
        self.assertEqual(audit["gradient_bytes"], 20)
        self.assertEqual(audit["optimizer_cpu_step_scalar_bytes"], 8)
        with self.assertRaises(AssertionError):
            optimizer_end_audit(optimizer, 2)

    def test_plan_fixed_matrix_and_diagnostic_isolation(self):
        import tempfile

        from experiments.fsdp4_hybrid_2x2.cli import plan

        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "input"
            record.write_bytes(b"fixture")
            args = SimpleNamespace(
                run_root=tmp,
                project="project",
                python="python",
                data=str(record),
                run_id="CPU_PLAN",
                timeout=1800,
                source_manifest=None,
                placement_manifest=str(record),
            )
            formal = plan(args, "benchmark")
            self.assertEqual(len(formal["stages"]), 12)
            self.assertEqual({s["case"] for s in formal["stages"]}, set(CASES))
            self.assertTrue(all(s["warmup"] == 10 and s["steps"] == 30 for s in formal["stages"]))
            self.assertTrue(all("--record-events" not in s["argv"] for s in formal["stages"]))
            self.assertTrue(all("--gemm-shapes" not in s["argv"] for s in formal["stages"]))
            self.assertEqual(len(plan(args, "correctness")["stages"]), 8)
            self.assertTrue(
                all("--communication-audit" in s["argv"] for s in plan(args, "capacity")["stages"])
            )

    def test_p0_review_rejects_unresolved_and_changed_evidence(self):
        import json
        import tempfile

        from experiments.fsdp4_hybrid_2x2.cli import validate_p0_review
        from experiments.fsdp4_hybrid_2x2.contracts import sha

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            placement = root / "placement"
            data = root / "data"
            for p in [source, placement, data]:
                p.write_bytes(p.name.encode())
            review = {
                "source_manifest_sha256": sha(source),
                "input_sha256": sha(data),
                "placement_manifest_sha256": sha(placement),
                "classification": "EXPECTED_NUMERICAL_DRIFT_SUPPORTED",
                "strict_independent_status": "FAIL",
                "signed_by": "root_agent",
                "scope": "bounded_performance_with_retained_strict_fail",
                "limits": [],
                "training_runtime_bug_confirmed": False,
                "continuation_authorized": True,
                "evidence": [
                    {"path": str(p.resolve()), "sha256": sha(p)} for p in [source, placement, data]
                ],
            }
            path = root / "review.json"
            path.write_text(json.dumps(review))
            validate_p0_review(path, source, placement, data)
            review["classification"] = "UNRESOLVED"
            path.write_text(json.dumps(review))
            with self.assertRaises(RuntimeError):
                validate_p0_review(path, source, placement, data)
            review["classification"] = "EXPECTED_NUMERICAL_DRIFT_SUPPORTED"
            path.write_text(json.dumps(review))
            source.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                validate_p0_review(path, source, placement, data)

    def test_name_stable_actual_cpu_source_hashes(self):
        from experiments.fsdp4_hybrid_2x2.worker import InitialStateAudit, tensor_record
        from sharded_runtime.config import ModelConfig
        from sharded_runtime.models.qwen2_factory import build_qwen_model
        from sharded_runtime.runtime.qwen2_adapter import _source_tensor
        from sharded_runtime.training.qwen2_native_baseline import initialize_same_source

        config = ModelConfig(
            model_type="qwen2",
            vocab_size=64,
            hidden_size=16,
            num_heads=4,
            num_key_value_heads=2,
            mlp_hidden_size=32,
            layers=2,
        )
        custom = build_qwen_model(config, seed=1234)
        native = build_qwen_model(config, seed=1234)
        audit = InitialStateAudit()
        for name, parameter in custom.named_parameters():
            audit.capture(name, _source_tensor(custom, name, parameter))
        initialize_same_source(native, 1234)
        actual = {n: tensor_record(p) for n, p in native.named_parameters()}
        self.assertEqual(audit.finish(actual)["status"], "PASS")
        bad = dict(actual)
        first = next(iter(bad))
        bad[first] = {**bad[first], "sha256": "bad"}
        with self.assertRaises(AssertionError):
            audit.finish(bad)


if __name__ == "__main__":
    unittest.main()
