"""Build acceptance commands, audit existing results, and run CPU-only fixtures.

The parent launcher owns subprocess timeouts and PID cleanup. This module never
connects remotely or silently substitutes CPU for a required four-GPU gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapping_worker import (  # noqa: E402
    PROJECT,
    RuntimeObserver,
    import_fixture,
    input_hashes,
    write_json,
)
from placement import load, make_manifest, validate_manifest, visible_order  # noqa: E402

CASES = (
    "startup",
    "fp32",
    "same_start",
    "overflow",
    "native_fp32",
    "native_fp16",
    "independent_rollout",
)


def _mapping_plan(placement_manifest, mapping_id, output, python=sys.executable):
    """Return stages for a single mapping; requires no Torch/CUDA initialization."""
    manifest = (
        load(placement_manifest)
        if isinstance(placement_manifest, (str, Path))
        else placement_manifest
    )
    validate_manifest(manifest)
    source_path = str(placement_manifest) if isinstance(placement_manifest, (str, Path)) else None
    if source_path is None:
        raise ValueError("Executable plans require a saved placement manifest path")
    rows = []
    for case in CASES:
        worker = (
            "mapping_worker.py"
            if case in {"startup", "fp32", "native_fp32", "native_fp16"}
            else "amp_worker.py"
        )
        destination = Path(output) / mapping_id / case
        argv = [
            str(python),
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=4",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=127.0.0.1:0",
            "--max_restarts=0",
            str(PROJECT / "experiments/finalization_4gpu" / worker),
            "--placement-manifest",
            source_path,
            "--mapping-id",
            mapping_id,
            "--output",
            str(destination),
            "--mode",
            case,
        ]
        if case == "fp32":
            argv += ["--steps", "10"]
        elif worker == "amp_worker.py":
            argv += ["--effective-updates", "10"]
        rows.append(
            {
                "stage_id": f"correctness_{mapping_id}_{case}",
                "name": f"correctness_{mapping_id}_{case}",
                "phase": "correctness",
                "case": case,
                "mapping_id": mapping_id,
                "argv": argv,
                "cwd": str(PROJECT),
                "env": {
                    "CUDA_VISIBLE_DEVICES": visible_order(manifest, mapping_id),
                    "OMP_NUM_THREADS": "1",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONPATH": str(PROJECT / "src"),
                    "NCCL_LAUNCH_ORDER_IMPLICIT": "1",
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                    "NCCL_DEBUG": "WARN",
                },
                "timeout_seconds": 1200,
                "timeout": 1200,
                "output": str(destination),
                "expected_result_pattern": ("mapping_" if worker == "mapping_worker.py" else "amp_")
                + case
                + ".rank{rank}.json",
                "status": "NOT_RUN_NO_GPU",
                "requires_fresh_output": True,
                "performance_claim_allowed": False,
            }
        )
    return rows


def build_plan(placement_manifest, run_root, python=sys.executable, mapping_id=None):
    root = Path(run_root)
    root = root if root.name == "correctness" else root / "correctness"
    rows = [
        stage
        for mapping in ([mapping_id] if mapping_id else ("M0", "M1"))
        for stage in _mapping_plan(placement_manifest, mapping, root, python)
    ]
    # Both mappings complete each required gate before independent-trajectory diagnostics.
    return sorted(rows, key=lambda stage: (CASES.index(stage["case"]), stage["mapping_id"]))


def summarize(output):
    """Require all four ranks and retain math failures; never infer PASS from exits."""
    output = Path(output)
    output = output if output.name == "correctness" else output / "correctness"
    cases, payloads = [], {}
    for mapping in ("M0", "M1"):
        for case in CASES:
            prefix = (
                "mapping_" if case in {"startup", "fp32", "native_fp32", "native_fp16"} else "amp_"
            )
            paths = [
                output / mapping / case / f"{prefix}{case}.rank{rank}.json" for rank in range(4)
            ]
            found = [path.exists() for path in paths]
            status = "NOT_RUN_NO_GPU"
            errors = []
            if any(found):
                if not all(found):
                    status = "FAIL"
                    errors.append("Missing rank result(s); not silently classified as no GPU")
                else:
                    records = [load(path) for path in paths]
                    payloads[mapping, case] = records
                    if any(
                        record.get("rank") != rank or record.get("mapping_id") != mapping
                        for rank, record in enumerate(records)
                    ):
                        status = "FAIL"
                        errors.append("Rank/mapping identity mismatch")
                    else:
                        status = (
                            "PASS"
                            if all(record.get("status") == "PASS" for record in records)
                            else "FAIL"
                        )
                    errors.extend(
                        {"rank": rank, "errors": record.get("errors", [])}
                        for rank, record in enumerate(records)
                        if record.get("status") != "PASS"
                    )
            cases.append(
                {
                    "mapping_id": mapping,
                    "case": case,
                    "status": status,
                    "raw_paths": [str(path) for path in paths],
                    "errors": errors,
                }
            )
    cross_mapping = []
    for case in CASES[1:]:
        if ("M0", case) not in payloads or ("M1", case) not in payloads:
            cross_mapping.append({"case": case, "status": "NOT_RUN_NO_GPU"})
            continue
        for rank in range(4):
            before, after = payloads["M0", case][rank], payloads["M1", case][rank]
            fields = [
                "initial_shard_fingerprints",
                "input_fingerprints",
                "source_manifest_sha256",
                "placement_manifest_sha256",
            ]
            if not case.startswith("native_") and case != "independent_rollout":
                fields.append("training_collective_contract")
            missing = [field for field in fields if field not in before or field not in after]
            equal = (
                not missing
                and all(before[field] == after[field] for field in fields)
                and bool(before.get("source_manifest_sha256"))
            )
            contract_prefix = None
            if case == "independent_rollout":
                first = before.get("training_collective_contract", [])
                second = after.get("training_collective_contract", [])
                count = min(len(first), len(second))
                contract_prefix = {
                    "M0_completed": len(first),
                    "M1_completed": len(second),
                    "common_microsteps_checked": count,
                }
                equal = equal and bool(count) and first[:count] == second[:count]
            cross_mapping.append(
                {
                    "case": case,
                    "rank": rank,
                    "status": "PASS" if equal else "FAIL",
                    "missing": missing,
                    "checked": fields,
                    "independent_contract_prefix": contract_prefix,
                    "scope": (
                        "Same logical-rank initial values, inputs and source; "
                        "not FP16 bitwise trajectory equality"
                    ),
                }
            )
    statuses = [row["status"] for row in cases + cross_mapping]
    source_hashes = {
        record.get("source_manifest_sha256") for records in payloads.values() for record in records
    }
    placement_hashes = {
        record.get("placement_manifest_sha256")
        for records in payloads.values()
        for record in records
    }
    result = {
        "status": "FAIL"
        if "FAIL" in statuses
        else "PASS"
        if set(statuses) == {"PASS"}
        else "NOT_RUN_NO_GPU",
        "cases": cases,
        "cross_mapping": cross_mapping,
        "source_manifest_sha256": next(iter(source_hashes)) if len(source_hashes) == 1 else None,
        "placement_manifest_sha256": next(iter(placement_hashes))
        if len(placement_hashes) == 1
        else None,
        "independent_rollout_failure_policy": (
            "Preserve first numerical failure; compare FP32, same-start "
            "and actual-gradient replay separately; never loosen "
            "tolerances or relabel as PASS"
        ),
    }
    required_statuses = [row["status"] for row in cases if row["case"] != "independent_rollout"] + [
        row["status"] for row in cross_mapping
    ]
    independent_statuses = [row["status"] for row in cases if row["case"] == "independent_rollout"]

    def aggregate(values):
        return (
            "FAIL" if "FAIL" in values else "PASS" if set(values) == {"PASS"} else "NOT_RUN_NO_GPU"
        )

    result["required_gate_status"] = aggregate(required_statuses)
    result["independent_status"] = aggregate(independent_statuses)
    write_json(output / "summary.json", result)
    return result


def validate_numeric_review(summary, review_path):
    """Validate an explicit human review's bindings, never convert a math FAIL to PASS.

    The root reviewer must inspect tensors and the same-machine control first. This
    function only checks the review evidence/schema, not the correctness of that
    scientific interpretation. Paths are existing raw files; all are SHA-bound.
    """
    review = load(review_path)
    if summary.get("required_gate_status") != "PASS" or summary.get("independent_status") != "FAIL":
        raise ValueError("Numerical limitation review cannot excuse a required gate failure")
    if review.get("classification") != "LIMITATION_KNOWN_FP16_NUMERIC_TRAJECTORY":
        raise ValueError("Unknown numerical review classification")
    for key in ("source_manifest_sha256", "placement_manifest_sha256"):
        if not summary.get(key) or review.get(key) != summary[key]:
            raise ValueError("Review differs from current source/placement: " + key)
    if not isinstance(review.get("scope"), str) or not review["scope"].strip():
        raise ValueError("An explicit review scope is required")
    if (
        not isinstance(review.get("limitations"), list)
        or not review["limitations"]
        or not all(isinstance(x, str) and x.strip() for x in review["limitations"])
    ):
        raise ValueError("Explicit nonempty limitations are required")
    failures = review.get("failure_raw_by_mapping")
    if not isinstance(failures, dict) or set(failures) != {"M0", "M1"}:
        raise ValueError("Review must account for both mapping outcomes")
    declared_raw = review.get("raw_paths")
    controls = review.get("pure_hf_control_paths")
    hashes = review.get("raw_sha256", {})
    if (
        not isinstance(declared_raw, list)
        or not declared_raw
        or not isinstance(controls, list)
        or not controls
    ):
        raise ValueError("Failure raw and same-machine pure-HF control files are required")
    all_paths = set(declared_raw + controls + [p for paths in failures.values() for p in paths])
    for path in all_paths:
        if not isinstance(path, str) or not Path(path).is_file():
            raise ValueError("Review raw path is missing: " + str(path))
        if hashes.get(path) != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError("Review raw hash mismatch: " + path)
    for case in summary.get("cases", []):
        if case["case"] != "independent_rollout":
            continue
        actual_failures = set()
        for path in case["raw_paths"]:
            value = load(path)
            if value.get("status") == "FAIL":
                actual_failures.add(path)
                numerical_rows = [
                    row for row in value.get("comparisons", []) if not row.get("passed", True)
                ]
                local_numeric = bool(numerical_rows) and all(
                    row.get("nonfinite_count", 0) == 0
                    and row.get("reference_nonfinite_count", 0) == 0
                    for row in numerical_rows
                )
                peer_numeric = (
                    value.get("numerical_failure", {}).get("peer_failure_propagated") is True
                )
                if value.get("plan", {}).get("mode") != "independent_rollout" or not (
                    local_numeric or peer_numeric
                ):
                    raise ValueError("Review cannot excuse non-numerical independent-case errors")
        if set(failures[case["mapping_id"]]) != actual_failures or not actual_failures <= set(
            declared_raw
        ):
            raise ValueError("Review does not bind every failed mapping/rank raw result")
    metadata_path = review.get("same_machine_metadata_path")
    if metadata_path not in all_paths:
        raise ValueError("Hash-bound current-run hardware metadata is required")
    metadata = load(metadata_path)
    for key in ("source_manifest_sha256", "placement_manifest_sha256"):
        if metadata.get(key) != summary[key]:
            raise ValueError("Control metadata source/placement differs")
    manifest = load(review["placement_manifest_path"])
    if (
        hashlib.sha256(Path(review["placement_manifest_path"]).read_bytes()).hexdigest()
        != summary["placement_manifest_sha256"]
    ):
        raise ValueError("Review placement manifest content differs")
    if (
        set(metadata.get("allowed_gpu_uuids", [])) != set(manifest["allowed_gpu_uuids"])
        or metadata.get("actual_uuid") not in manifest["allowed_gpu_uuids"]
    ):
        raise ValueError("Control hardware UUID is not from this four-GPU allocation")
    valid_controls = []
    for path in controls:
        value = load(path)
        if (
            value.get("cuda_executed") is True
            and value.get("actual_custom_runtime_executed") is False
            and value.get("diagnostic_execution_completed") is True
            and not value.get("errors")
            and value.get("compute_dtype") == "fp16"
            and value.get("seed") == 313
            and (value.get("atol"), value.get("rtol")) == (3e-3, 3e-3)
            and any(
                pair.get("parameter_failures")
                for step in value.get("steps_data", [])
                for pair in step.get("pairs", [])
            )
        ):
            valid_controls.append(path)
    if not valid_controls:
        raise ValueError(
            "No completed GPU pure-HF FP16 control reproduces a strict trajectory difference"
        )
    return {
        "status": "ACKNOWLEDGED_LIMITATION",
        "classification": review["classification"],
        "review_path": str(review_path),
        "review_sha256": hashlib.sha256(Path(review_path).read_bytes()).hexdigest(),
        "source_manifest_sha256": summary["source_manifest_sha256"],
        "placement_manifest_sha256": summary["placement_manifest_sha256"],
        "independent_status": "FAIL",
        "valid_control_paths": valid_controls,
        "scope": review["scope"],
        "limitations": review["limitations"],
    }


def cpu_selftest(output):
    """Meaningful CPU fixtures: logical seed invariance, immutable oracle, guards."""
    import torch

    amp = import_fixture("tests/distributed/qwen2_amp_worker.py", "_finalization_cpu_amp_helpers")

    def synthetic_environment(devices):
        allowed = [item["uuid"] for item in devices]
        return {
            "allowed_gpu_uuids": allowed,
            "devices": devices,
            "environment_source": "CPU_SYNTHETIC_FIXTURE_NOT_MEASURED",
            "topology": {a: {b: "X" if a == b else "NV1" for b in allowed} for a in allowed},
        }

    class Checks(unittest.TestCase):
        def test_same_logical_inputs_under_both_placements(self):
            devices = [
                {"uuid": "GPU-" + letter, "index": index, "pci_bus_id": str(index)}
                for index, letter in enumerate("ABCD")
            ]
            manifest = make_manifest(synthetic_environment(devices))
            frozen = copy.deepcopy(manifest)
            fingerprint = input_hashes(amp.token_batch, 12, 2)
            self.assertEqual(fingerprint, input_hashes(amp.token_batch, 12, 2))
            self.assertNotEqual(fingerprint[0]["tensors"], fingerprint[1]["tensors"])
            self.assertNotEqual(visible_order(manifest, "M0"), visible_order(manifest, "M1"))
            self.assertEqual(manifest, frozen)
            self.assertEqual(
                [int(amp.token_batch(f, 0)[1][:, 1:].ne(-100).sum()) for f in range(2)], [11, 9]
            )

        def test_oracle_setter_copies_and_does_not_mutate_reference(self):
            model = torch.nn.Linear(3, 2)
            model.config = SimpleNamespace(tie_word_embeddings=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
            weights = {
                name: parameter.detach().clone() for name, parameter in model.named_parameters()
            }
            states = {
                name: {
                    "step": torch.tensor(1.0),
                    "exp_avg": torch.ones_like(parameter),
                    "exp_avg_sq": torch.ones_like(parameter) * 2,
                }
                for name, parameter in model.named_parameters()
            }
            digest = amp.start_hash(weights, states)
            amp.copy_reference_start(model, optimizer, weights, states)
            with torch.no_grad():
                next(model.parameters()).add_(1)
            next(iter(optimizer.state.values()))["exp_avg"].add_(2)
            self.assertEqual(digest, amp.start_hash(weights, states))

        def test_decoder_input_gradient_scaling_fixture_on_cpu(self):
            from torch.nn import functional as functional
            from transformers import Qwen2Config, Qwen2ForCausalLM

            torch.set_num_threads(1)
            torch.manual_seed(313)
            config = Qwen2Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                tie_word_embeddings=True,
                max_position_embeddings=128,
                use_cache=False,
            )
            config._attn_implementation = "sdpa"
            dense = Qwen2ForCausalLM(config).train()
            replica = copy.deepcopy(dense)
            observed = {}

            def hook(key):
                def capture(_module, _args, output):
                    self.assertTrue(output.requires_grad)
                    output.retain_grad()
                    observed[key] = output

                return capture

            dense.model.embed_tokens.register_forward_hook(hook("dense"))
            replica.model.embed_tokens.register_forward_hook(hook("replica"))
            batches = [amp.token_batch(f, 0) for f in range(2)]
            full = [torch.cat([batch[i] for batch in batches]) for i in range(4)]
            denominator = full[1][:, 1:].ne(-100).sum()
            scale = 128.0
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                expected = dense(
                    input_ids=full[0],
                    labels=full[1],
                    attention_mask=full[2],
                    position_ids=full[3],
                    use_cache=False,
                )
                local = replica(
                    input_ids=batches[0][0],
                    attention_mask=batches[0][2],
                    position_ids=batches[0][3],
                    use_cache=False,
                )
            numerator = functional.cross_entropy(
                local.logits[:, :-1].float().reshape(-1, 64),
                batches[0][1][:, 1:].reshape(-1),
                reduction="sum",
                ignore_index=-100,
            )
            (expected.loss * scale).backward()
            (numerator * 2 / denominator * scale).backward()
            torch.testing.assert_close(
                observed["replica"].grad / scale,
                observed["dense"].grad[:2] * 2 / scale,
                atol=3e-6,
                rtol=3e-5,
            )

        def test_lifecycle_negative_guards(self):
            payload = {}
            observer = RuntimeObserver(payload)
            invalid = SimpleNamespace(optimizer_ready=False, comm=SimpleNamespace(live_buffers={}))
            with self.assertRaises(AssertionError):
                observer.finalized(invalid)
            invalid.optimizer_ready = True
            invalid.units = [SimpleNamespace(state="GRAD_READY", reductions=2)]
            with self.assertRaises(AssertionError):
                observer.finalized(invalid)

        def test_missing_results_and_partial_results_not_pass(self):
            import tempfile

            with tempfile.TemporaryDirectory(prefix="finalization-cpu-") as root:
                self.assertEqual(summarize(root)["status"], "NOT_RUN_NO_GPU")
                write_json(
                    Path(root) / "correctness/M0/startup/mapping_startup.rank0.json",
                    {"status": "PASS", "rank": 0, "mapping_id": "M0"},
                )
                self.assertEqual(summarize(root)["status"], "FAIL")

        def test_all_stages_have_the_same_logical_mapping_contract(self):
            import tempfile

            with tempfile.TemporaryDirectory(prefix="finalization-plan-") as root:
                devices = [{"uuid": "GPU-" + x, "index": i} for i, x in enumerate("ABCD")]
                manifest = make_manifest(synthetic_environment(devices))
                path = Path(root) / "placement.json"
                write_json(path, manifest)
                stages = build_plan(path, root)
                self.assertEqual(len(stages), 14)
                for stage in stages:
                    self.assertEqual(stage["phase"], "correctness")
                    self.assertEqual(
                        stage["env"]["CUDA_VISIBLE_DEVICES"],
                        visible_order(manifest, stage["mapping_id"]),
                    )
                    self.assertEqual(
                        stage["argv"][stage["argv"].index("--mode") + 1], stage["case"]
                    )
                    self.assertEqual(
                        "--effective-updates" in stage["argv"],
                        stage["case"] in {"same_start", "independent_rollout", "overflow"},
                    )

        def test_cross_mapping_corruption_is_a_failure(self):
            import tempfile

            with tempfile.TemporaryDirectory(prefix="finalization-summary-") as root:
                for mapping in ("M0", "M1"):
                    for case in CASES:
                        prefix = (
                            "mapping_"
                            if case in {"startup", "fp32", "native_fp32", "native_fp16"}
                            else "amp_"
                        )
                        for rank in range(4):
                            write_json(
                                Path(root)
                                / "correctness"
                                / mapping
                                / case
                                / f"{prefix}{case}.rank{rank}.json",
                                {
                                    "status": "PASS",
                                    "rank": rank,
                                    "mapping_id": mapping,
                                    "initial_shard_fingerprints": {"x": "initial"},
                                    "input_fingerprints": ["logical"],
                                    "training_collective_contract": [["unchanged"]],
                                    "source_manifest_sha256": "source",
                                    "placement_manifest_sha256": "placement",
                                },
                            )
                self.assertEqual(summarize(root)["status"], "PASS")
                # Independent trajectories may stop at different first-failure steps.
                # Initial inputs/source and the common model-contract prefix must agree.
                for mapping in ("M0", "M1"):
                    for rank in range(4):
                        independent_path = (
                            Path(root)
                            / "correctness"
                            / mapping
                            / "independent_rollout"
                            / f"amp_independent_rollout.rank{rank}.json"
                        )
                        independent = load(independent_path)
                        independent["status"] = "FAIL"
                        if mapping == "M1":
                            independent["training_collective_contract"].append(
                                ["later_before_failure"]
                            )
                        write_json(independent_path, independent)
                different_lengths = summarize(root)
                self.assertEqual(different_lengths["status"], "FAIL")
                self.assertEqual(different_lengths["independent_status"], "FAIL")
                self.assertEqual(different_lengths["required_gate_status"], "PASS")
                independent_path = (
                    Path(root)
                    / "correctness/M1/independent_rollout/amp_independent_rollout.rank0.json"
                )
                preserved = load(independent_path)
                for field, wrong in [
                    ("input_fingerprints", ["wrong_physical_seed"]),
                    ("training_collective_contract", [["different_common_op"]]),
                ]:
                    invalid = copy.deepcopy(preserved)
                    invalid[field] = wrong
                    write_json(independent_path, invalid)
                    self.assertEqual(summarize(root)["required_gate_status"], "FAIL")
                write_json(independent_path, preserved)
                path = Path(root) / "correctness/M1/fp32/mapping_fp32.rank2.json"
                value = load(path)
                value["input_fingerprints"] = ["physical_seed_was_wrong"]
                write_json(path, value)
                self.assertEqual(summarize(root)["status"], "FAIL")

        def test_numeric_review_requires_bound_raw_and_current_hardware(self):
            import tempfile

            with tempfile.TemporaryDirectory(prefix="finalization-review-") as root:
                root = Path(root)
                devices = [{"uuid": "GPU-" + x, "index": i} for i, x in enumerate("ABCD")]
                manifest_path = root / "placement.json"
                write_json(manifest_path, make_manifest(synthetic_environment(devices)))
                placement_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                metadata_path = root / "control_metadata.json"
                write_json(
                    metadata_path,
                    {
                        "source_manifest_sha256": "source",
                        "placement_manifest_sha256": placement_hash,
                        "allowed_gpu_uuids": [d["uuid"] for d in devices],
                        "actual_uuid": "GPU-A",
                    },
                )
                control_path = root / "pure_hf.json"
                control = {
                    "cuda_executed": True,
                    "actual_custom_runtime_executed": False,
                    "diagnostic_execution_completed": True,
                    "errors": [],
                    "compute_dtype": "fp16",
                    "seed": 313,
                    "atol": 3e-3,
                    "rtol": 3e-3,
                    "steps_data": [{"pairs": [{"parameter_failures": ["k_proj.bias"]}]}],
                }
                write_json(control_path, control)
                paths_by_mapping, cases = {}, []
                for mapping in ("M0", "M1"):
                    paths = []
                    for rank in range(4):
                        path = root / f"{mapping}.rank{rank}.json"
                        write_json(
                            path,
                            {
                                "status": "FAIL",
                                "plan": {"mode": "independent_rollout"},
                                "comparisons": [{"passed": False}],
                            },
                        )
                        paths.append(str(path))
                    paths_by_mapping[mapping] = paths
                    cases.append(
                        {"case": "independent_rollout", "mapping_id": mapping, "raw_paths": paths}
                    )
                raw_paths = [str(metadata_path)] + [
                    p for paths in paths_by_mapping.values() for p in paths
                ]
                review = {
                    "classification": "LIMITATION_KNOWN_FP16_NUMERIC_TRAJECTORY",
                    "source_manifest_sha256": "source",
                    "placement_manifest_sha256": placement_hash,
                    "scope": "CPU synthetic validation of review schema only",
                    "limitations": ["No GPU evidence in this unit fixture"],
                    "failure_raw_by_mapping": paths_by_mapping,
                    "raw_paths": raw_paths,
                    "pure_hf_control_paths": [str(control_path)],
                    "same_machine_metadata_path": str(metadata_path),
                    "placement_manifest_path": str(manifest_path),
                    "raw_sha256": {
                        p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for p in raw_paths + [str(control_path)]
                    },
                }
                summary = {
                    "status": "FAIL",
                    "required_gate_status": "PASS",
                    "independent_status": "FAIL",
                    "source_manifest_sha256": "source",
                    "placement_manifest_sha256": placement_hash,
                    "cases": cases,
                }
                path = root / "numeric_review.json"
                write_json(path, review)
                self.assertEqual(
                    validate_numeric_review(summary, path)["status"], "ACKNOWLEDGED_LIMITATION"
                )
                for field, wrong in [
                    ("scope", ""),
                    ("limitations", []),
                    ("source_manifest_sha256", "other"),
                    ("pure_hf_control_paths", []),
                    ("raw_sha256", {}),
                ]:
                    invalid = copy.deepcopy(review)
                    invalid[field] = wrong
                    write_json(path, invalid)
                    with self.assertRaises(ValueError, msg=field):
                        validate_numeric_review(summary, path)
                write_json(path, review)
                write_json(control_path, {**control, "cuda_executed": False})
                review["raw_sha256"][str(control_path)] = hashlib.sha256(
                    control_path.read_bytes()
                ).hexdigest()
                write_json(path, review)
                with self.assertRaises(ValueError):
                    validate_numeric_review(summary, path)

        def test_historical_thresholds_and_required_update_plan(self):
            from amp_worker import DeferredNumericalChecks, fixture_plan

            self.assertEqual((amp.ATOL, amp.RTOL), (3e-3, 3e-3))
            for mode in ("same_start", "independent_rollout"):
                plan = fixture_plan(
                    SimpleNamespace(mode=mode, mapping_id="M0", effective_updates=10)
                )
                self.assertEqual(plan["requested_effective_updates"], 10)
                self.assertEqual(plan["reference_synced_each_step"], mode == "same_start")
            with self.assertRaises(ValueError):
                fixture_plan(
                    SimpleNamespace(mode="same_start", mapping_id="M0", effective_updates=9)
                )
            comparisons = []
            deferred = DeferredNumericalChecks(amp.compare)
            deferred.compare(
                torch.ones(2), torch.zeros(2), "intentional_cpu_difference", comparisons
            )
            self.assertFalse(comparisons[0]["passed"])
            self.assertTrue(deferred.errors)
            self.assertEqual((comparisons[0]["atol"], comparisons[0]["rtol"]), (3e-3, 3e-3))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": [(str(test), error) for test, error in result.failures],
        "errors": [(str(test), error) for test, error in result.errors],
        "scope": (
            "CPU-only mapping/data/oracle/negative/review fixtures and HF input-gradient "
            "scaling; no CUDA/NCCL or new mapping GPU PASS claim"
        ),
        "torch": str(torch.__version__),
    }
    write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=["plan", "analyze", "cpu-test"])
    parser.add_argument("--placement-manifest")
    parser.add_argument("--mapping-id", choices=["M0", "M1"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--results")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.command == "cpu-test":
        payload = cpu_selftest(args.output)
    elif args.command == "plan":
        if not args.placement_manifest or not args.mapping_id or not args.results:
            parser.error("plan needs --placement-manifest --mapping-id --results")
        payload = {
            "status": "NOT_RUN_NO_GPU",
            "stages": build_plan(
                args.placement_manifest, args.results, args.python, args.mapping_id
            ),
        }
        write_json(args.output, payload)
    else:
        if not args.results:
            parser.error("analyze needs --results")
        payload = summarize(args.results)
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
