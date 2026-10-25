"""Read-only historical audit and pre-register P0 measurements before execution."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

import torch  # noqa: E402

from experiments.finalization_4gpu.mapping_worker import import_fixture, input_hashes  # noqa: E402
from experiments.fp16_numerics.common import (  # noqa: E402
    ATOL,
    BIAS,
    OPTIMIZER,
    RTOL,
    distribution,
    immutable_json,
    sha,
)


def build():
    base = PROJECT / "runs/finalization_4gpu/20261017_run01/server_return_v2"
    returned = json.loads((base / "final_return_v2_manifest.json").read_text())
    records = []
    for mapping in ("M0", "M1"):
        directory = base / "correctness" / mapping / "independent_rollout"
        path = directory / "amp_independent_rollout.rank3.json"
        tensor_path = directory / "failure_state.rank3.pt"
        for source in (path, tensor_path):
            entry = returned["files"][source.relative_to(base).as_posix()]
            if sha(source) != entry["sha256"] or source.stat().st_size != entry["bytes"]:
                raise ValueError("Historical raw differs from immutable return manifest")
        raw = json.loads(path.read_text())
        states = torch.load(tensor_path, map_location="cpu", weights_only=True)
        actual, expected = (
            states["runtime_shards"][BIAS],
            states["reference_parameters"][BIAS][12:16],
        )
        fail = distribution(actual, expected)
        if raw["status"] != "FAIL" or fail["passed"] or raw["actual_optimizer_updates"] != 2:
            raise ValueError("Original strict failure cannot be reconstructed")
        records.append(
            {
                "mapping_id": mapping,
                "source_class": "HISTORICAL_RAW",
                "rank": 3,
                "stage": raw["numerical_failure"]["stage"],
                "zero_based_step": 1,
                "effective_update": 2,
                "logical_parameter": BIAS,
                "global_indices": [12, 13, 14, 15],
                "comparison": fail,
                "runtime_gradient": states["runtime_gradients"][BIAS].tolist(),
                "dense_gradient": states["reference_gradients"][BIAS][12:16].tolist(),
                "runtime_parameters": actual.tolist(),
                "dense_parameters": expected.tolist(),
                "post_runtime_scaler": states["runtime_scaler"],
                "post_reference_scaler": states["reference_scaler"],
                "runtime_optimizer_options_actual_post_record": {
                    key: value
                    for key, value in states["runtime_optimizer"]["param_groups"][0].items()
                    if key != "params"
                },
                "reference_optimizer_options_actual_post_record": {
                    key: value
                    for key, value in states["reference_optimizer"]["param_groups"][0].items()
                    if key != "params"
                },
                "raw_json": path.relative_to(PROJECT).as_posix(),
                "raw_sha256": sha(path),
                "post_state_tensors": tensor_path.relative_to(PROJECT).as_posix(),
                "post_state_sha256": sha(tensor_path),
                "real_pre_optimizer_parameter_m_v_step": (
                    "NOT_AVAILABLE: failure_state saved after the second update; do "
                    "not invert post-state to manufacture exact pre-state"
                ),
                "pre_reduction_scaled_contributions": "NOT_AVAILABLE",
                "post_reduction_pre_unscale": "NOT_AVAILABLE",
            }
        )
    fixture = import_fixture("tests/distributed/qwen2_amp_worker.py", "_p0_contract_fixture")
    control_path = base / "correctness/pure_hf_control/diagnosis.json"
    controls = json.loads(control_path.read_text())
    history_control = {
        "source_class": "HISTORICAL_RAW",
        "path": control_path.relative_to(PROJECT).as_posix(),
        "sha256": sha(control_path),
        "actual_custom_runtime_executed": controls["actual_custom_runtime_executed"],
        "actual_distributed_collectives": controls["actual_distributed_collectives"],
        "actual_SP_execution": controls["actual_SP_execution"],
        "steps": controls["steps"],
        "initial_hashes": controls["initial_parameter_sha256"],
        "optimizer_pre_state": (
            "step0 explicitly empty; stepN can be read from the preceding real"
            " post-state since the independent models are not reset"
        ),
        "limitations": controls["limitations"],
    }
    git = ["git", "-c", f"safe.directory={PROJECT.as_posix()}"]
    head = (
        subprocess.run(git + ["rev-parse", "HEAD"], cwd=PROJECT, check=True, capture_output=True)
        .stdout.decode()
        .strip()
    )
    patch = subprocess.run(
        git + ["diff", "--binary"], cwd=PROJECT, check=True, capture_output=True
    ).stdout
    config, _ = __import__(
        "experiments.finalization_4gpu.legacy_fp16_partition_control", fromlist=["fixture_contract"]
    ).fixture_contract()
    return {
        "schema_version": 1,
        "contract_status": "PREREGISTERED_NOT_GPU_EXECUTED",
        "historical_strict_status": "FAIL",
        "original_tolerance": {
            "atol": ATOL,
            "rtol": RTOL,
            "definition": (
                "torch.allclose float32-cast tensors; finite required; original "
                "logits/loss/dX/dW/update/m/v/step comparisons retained"
            ),
        },
        "current_modes": {
            "independent": (
                "same initial global HF parameters and empty Adam state; never "
                "copy reference weights/moments/scaler between steps"
            ),
            "same_start": (
                "explicitly named separate fresh process; copy real Runtime pre-"
                "state to HF at every step; never accepted as independent"
            ),
        },
        "new_run_root": "runs/p0_p1_p2_4gpu/20261017_run01/p0",
        "steps": 10,
        "seed": 313,
        "model_config": config,
        "global_unique_parameters": 39456,
        "parallel": {
            "F": 2,
            "T": 2,
            "SP": True,
            "logical_mesh": [[0, 1], [2, 3]],
            "tp": [[0, 1], [2, 3]],
            "fsdp": [[0, 2], [1, 3]],
            "placement": (
                "M0 from current manifest; actual UUID/PCI/global/local/group "
                "ranks checked at runtime"
            ),
            "policy": "stream_ordered",
            "flags": "111",
        },
        "batch": {
            "local_batch_per_F": 2,
            "global_batch": 4,
            "sequence": 8,
            "accumulation": 1,
            "input_ids_labels_masks_positions": (
                "Original token_batch, including unequal valid counts and masked boundary labels"
            ),
            "fingerprints": input_hashes(fixture.token_batch, 10, 2),
        },
        "precision": {
            "persistent_parameters": "FP32",
            "effective_linear_operands": "FP16",
            "attention_math_BMM": "FP32",
            "norm_accumulation": "FP32",
            "gradient_reduce": "FP32",
            "master_adam_state": "FP32",
            "linear_dW_effective_operands": "FP16 GEMM result then FP32 master gradient",
            "projection_bias_gradient": (
                "upstream gradient.float().sum(batch,sequence), cast FP16 compute "
                "dtype, cast FP32 before FSDP RS; raw pre/post cast captured"
            ),
        },
        "optimizer": OPTIMIZER,
        "amp": {
            "initial_scale": 128.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2,
            "overflow": (
                "unexpected nonfinite/skip is a hard diagnostic blocker; use "
                "separate unchanged overflow gate for skip/recovery"
            ),
        },
        "historical_failures": records,
        "historical_pure_hf_control": history_control,
        "source_at_contract_creation": {
            "git_head": head,
            "tracked_dirty_patch_sha256": __import__("hashlib").sha256(patch).hexdigest(),
            "untracked_source_bound_by_future_root_freeze": True,
            "current_archive_sha256": "NOT_AVAILABLE_UNTIL_ROOT_FREEZE",
            "runtime_source_manifest_sha256": "REQUIRED_AT_ACTUAL_EXECUTION",
            "local_torch": torch.__version__,
            "current_GPU_CUDA_NCCL_hardware": (
                "NOT_RUN; captured per current worker, never copied from history"
            ),
            "source_sha256": {
                name: sha(PROJECT / name)
                for name in [
                    "tests/distributed/qwen2_amp_worker.py",
                    "experiments/finalization_4gpu/amp_worker.py",
                    "src/sharded_runtime/training/qwen2_trainer.py",
                    "src/sharded_runtime/runtime/materializer.py",
                    "src/sharded_runtime/runtime/qwen2_operators.py",
                ]
            },
        },
        "preregistered_measures": [
            "original strict allclose pass/fail separately from bitwise equality",
            "first nonbitwise transition and first strict failure, separately",
            (
                "per-parameter and valid local leaf concatenated max_abs, "
                "relative_L2, P50/P95/P99, cosine for finite nonzero norms, sign "
                "disagreements, nonfinite counts"
            ),
            (
                "loss trajectories, gradient/parameter norm ratios, top-k "
                "parameter names and global logical indices"
            ),
            (
                "K bias master/half view, reducer inputs per F rank, F-averaged "
                "scaled shard, unscale, pre/post Adam state and exact same-device "
                "replay"
            ),
            (
                "linear descriptive trend and first/last summaries; do not create "
                "new posthoc pass thresholds"
            ),
        ],
        "preregistered_risk_assessment": {
            "hard_blockers": [
                "nonfinite loss/gradient/parameter/state",
                "unexpected skipped effective update",
                "initial/input/ownership mismatch",
                "lifecycle/scaler/factor contract failure",
                "same-device actual-state optimizer replay mismatch",
            ],
            "strict_failures": (
                "retain every original .003 FAIL and continue finite independent "
                "diagnostic through10; no redefinition of pass"
            ),
            "broad_drift": (
                "report number/fraction of parameter tensors/elements failing "
                "original tolerance, relative norm and loss trends; descriptive, "
                "no invented normal threshold"
            ),
            "classification_requires": (
                "Root evaluates current same-start/FP32 gates, real-state replay, "
                "actual F2 average audit, pure controls and finite short "
                "trajectory. Replay match alone never certifies implementation."
            ),
            "long_term": (
                "ten updates cannot establish convergence, general stability, or "
                "absence of all runtime bugs"
            ),
        },
        "R0": "EXECUTED_OFFLINE_HISTORICAL_RAW",
        "R1": "NOT_RUN_CURRENT",
        "R2": "NOT_RUN_CURRENT",
        "R3": "NOT_RUN_CURRENT",
        "R4": "HISTORICAL_ONLY",
        "R5": "MAIN_HISTORICAL_PRE_STATE_MISSING; CURRENT_CAPTURE_REQUIRED",
        "gpu_test_pass": False,
        "production_changes": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(PROJECT / "experiments/fp16_numerics/contract.json")
    )
    args = parser.parse_args()
    immutable_json(args.output, build())
