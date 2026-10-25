"""CPU-only schema, actual-state replay, ownership and factor negative tests."""

import copy

import pytest
import torch

from experiments.fp16_numerics.audit_results import audit_payload
from experiments.fp16_numerics.common import OPTIMIZER, distribution, immutable_json, pack
from experiments.fp16_numerics.compare_trajectories import (
    reduction_audit,
    unique_global_distributions,
)
from experiments.fp16_numerics.replay_adamw import analyze_capture, replay_one, validate_state


def actual_step():
    parameter = torch.nn.Parameter(torch.tensor([0.002, -0.001, 0.0, 0.1]))
    optimizer = torch.optim.AdamW([parameter], **OPTIMIZER)
    parameter.grad = torch.tensor([1e-8, -5e-8, 0.0, 0.03])
    optimizer.step()
    parameter.grad = torch.tensor([6e-9, -6.5e-8, 1e-9, -0.01])
    pre = pack(parameter, optimizer)
    optimizer.step()
    return pre, pack(parameter, optimizer)


def test_real_pre_post_cpu_replay_bit_exact():
    pre, post = actual_step()
    result = replay_one(pre, post)
    assert result["status"] == "REPLAY_MATCHED"
    assert all(row["bitwise_equal"] for row in result["residuals"].values())


def test_missing_optimizer_pre_state_rejected():
    pre, post = actual_step()
    del pre["state"]["exp_avg"]
    with pytest.raises(ValueError, match="Missing actual"):
        replay_one(pre, post)


def test_no_post_state_inversion_or_missing_gradient():
    pre, _ = actual_step()
    pre["gradient"] = None
    with pytest.raises(ValueError, match="Real parameter"):
        validate_state(pre)


def test_counterfactual_does_not_mutate_real_inputs():
    pre, post = actual_step()
    before = pre["parameter"].clone()
    dense = copy.deepcopy(pre)
    dense["gradient"][0] *= -1
    result = analyze_capture(
        {
            "schema_version": 1,
            "runtime": {"pre": pre, "post": post},
            "reference": {"pre": dense, "post": post},
        }
    )
    assert torch.equal(pre["parameter"], before)
    assert result["counterfactual"]["gradient_sign_change"][0]
    assert result["counterfactual"]["gradient_only_parameter_delta"][0] != 0
    assert result["attribution_status"] == "DRIFT_ATTRIBUTION_INCOMPLETE"


def test_immutable_evidence(tmp_path):
    path = tmp_path / "raw.json"
    immutable_json(path, {"strict_status": "FAIL"})
    with pytest.raises(FileExistsError):
        immutable_json(path, {"strict_status": "PASS"})


def test_original_tolerance_is_not_bitwise_equality():
    row = distribution(torch.tensor([1e-7]), torch.tensor([0.0]))
    assert row["passed"] and not row["bitwise_equal"]
    assert row["cosine"] is None
    assert not distribution(torch.tensor([0.00327]), torch.tensor([0.0]))["passed"]


@pytest.mark.parametrize(
    "payload,mode",
    [
        ({"evidence_class": "OFFLINE_REPLAY", "mode": "independent"}, "independent"),
        ({"evidence_class": "CURRENT_GPU_REPRODUCTION", "mode": "same_start"}, "independent"),
        (
            {
                "evidence_class": "CURRENT_GPU_REPRODUCTION",
                "mode": "independent",
                "reference_synced_each_step": True,
            },
            "independent",
        ),
    ],
)
def test_offline_or_same_start_cannot_certify_independent(payload, mode, tmp_path):
    with pytest.raises(ValueError):
        audit_payload(payload, expected_mode=mode, directory=tmp_path)


def test_global_distribution_uses_unique_elements_and_checks_replicas():
    payloads, captures = [], []
    for rank in range(4):
        f, t = rank // 2, rank % 2
        layout = {
            "weight": {
                "global_shape": [2, 4],
                "tp_local_shape": [2, 2],
                "tp_dim": 1,
                "tp_rank": t,
                "fsdp_rank": f,
                "fsdp_shard_numel": 2,
                "valid_numel": 2,
            },
            "norm": {
                "global_shape": [2],
                "tp_local_shape": [2],
                "tp_dim": None,
                "tp_rank": t,
                "fsdp_rank": f,
                "fsdp_shard_numel": 1,
                "valid_numel": 1,
            },
        }
        payloads.append({"parameter_layout": layout})
        captures.append(
            {
                "distribution_shards": {
                    key: {
                        "weight": torch.tensor([f * 4 + t * 2, f * 4 + t * 2 + 1.0]),
                        "norm": torch.tensor([f + 1.0]),
                    }
                    for key in (
                        "parameters_actual",
                        "parameters_expected",
                        "gradients_actual",
                        "gradients_expected",
                    )
                }
            }
        )
    result = unique_global_distributions(captures, payloads)
    assert result["parameters"]["numel"] == 10 and result["parameters"]["max_abs"] == 0
    captures[1]["distribution_shards"]["parameters_actual"]["norm"][0] += 1
    with pytest.raises(AssertionError, match="TP replicated"):
        unique_global_distributions(captures, payloads)


def test_reduction_and_unscale_factors_have_negative_case():
    captures = []
    for rank in range(4):
        f, t = rank // 2, rank % 2
        full = torch.arange(8, dtype=torch.float32) + t
        pre = full + f * 8
        avg = full + 4
        reduced = avg[f * 4 : (f + 1) * 4].clone()
        captures.append(
            {
                "runtime": {"pre": {"parameter": torch.zeros(4), "gradient": reduced / 128}},
                "amp": {"scale_before": 128},
                "pre_forward": {"runtime_master": torch.zeros(4)},
                "gradient_transitions": {
                    "k_bias_pre_fsdp_scaled_tp_local": pre,
                    "k_bias_post_fsdp_average_scaled_shard": reduced,
                    "k_bias_reduced_scaled_shard": reduced.clone(),
                    "k_bias_reduced_unscaled_shard": reduced / 128,
                    "k_bias_sum_after_compute_cast": pre.clone(),
                    "k_bias_forward_materialized_master_tp_local": torch.zeros(8),
                    "k_bias_forward_effective_compute_tp_local": torch.zeros(
                        8, dtype=torch.float16
                    ),
                },
            }
        )
    assert reduction_audit(captures)["status"] == "PASS"
    captures[3]["gradient_transitions"]["k_bias_post_fsdp_average_scaled_shard"] *= 2
    assert reduction_audit(captures)["status"] == "FAIL"


def test_hidden_original_failure_is_rejected(tmp_path):
    from experiments.fp16_numerics.common import sha

    path = tmp_path / "tiny.pt"
    path.write_bytes(b"synthetic-schema-fixture-not-gpu-evidence")
    metric = {"passed": True}
    rows = [
        {
            "step": step,
            "effective_update": step + 1,
            "reference_synced_from_runtime": False,
            "overflow": False,
            "tensor_path": path.name,
            "tensor_sha256": sha(path),
            "failed_logical_ranks": [],
            "strict_status": "PASS",
            "logits": metric,
            "loss": metric,
            "decoder_input_gradient": metric,
            "parameter_gradients": {},
            "parameters_after": {},
            "optimizer_states": {},
            "strict_failures": [],
        }
        for step in range(10)
    ]
    payload = {
        "evidence_class": "CURRENT_GPU_REPRODUCTION",
        "mode": "independent",
        "reference_synced_each_step": False,
        "capture_status": "COMPLETED",
        "strict_status": "PASS",
        "completed_effective_updates": 10,
        "actual_distributed_collectives": True,
        "actual_custom_runtime_executed": True,
        "source_manifest_sha256": "synthetic-schema-only",
        "historical_failures_reclassified": False,
        "environment": {"world_size": 4},
        "placement": {"status": "PASS"},
        "rank": 0,
        "steps": rows,
    }
    rows[1]["parameters_after"] = {"key_bias": {"passed": False}}
    with pytest.raises(ValueError, match="hidden"):
        audit_payload(payload, expected_mode="independent", directory=tmp_path)
