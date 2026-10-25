"""Strict schema/provenance gates; collection completion never implies math PASS."""

from __future__ import annotations

from pathlib import Path

from experiments.fp16_numerics.common import sha
from experiments.fp16_numerics.replay_adamw import validate_state


def audit_payload(payload, *, expected_mode, directory):
    if payload.get("evidence_class") != "CURRENT_GPU_REPRODUCTION":
        raise ValueError("Offline/control evidence cannot become a current GPU test")
    if expected_mode not in {"independent", "same_start"} or payload.get("mode") != expected_mode:
        raise ValueError("A same-start result cannot certify independent rollout")
    if payload.get("reference_synced_each_step") != (expected_mode == "same_start"):
        raise ValueError("Reference synchronization semantics mislabelled")
    if (
        payload.get("capture_status") != "COMPLETED"
        or payload.get("completed_effective_updates") != 10
    ):
        raise ValueError("Missing effective updates or incomplete capture")
    if not payload.get("actual_distributed_collectives") or not payload.get(
        "actual_custom_runtime_executed"
    ):
        raise ValueError("Offline/control results cannot generate four-GPU test PASS")
    if payload.get("gpu_test_pass") is True and payload.get("strict_status") != "PASS":
        raise ValueError("Strict numerical FAIL cannot become GPU mathematical PASS")
    if (
        not payload.get("source_manifest_sha256")
        or payload.get("historical_failures_reclassified") is not False
    ):
        raise ValueError("Unbound source or rewritten historical failure")
    if (
        payload.get("environment", {}).get("world_size") != 4
        or payload.get("placement", {}).get("status") != "PASS"
    ):
        raise ValueError("Missing actual four-rank environment and placement verification")
    steps = payload.get("steps", [])
    if [row["step"] for row in steps] != list(range(10)) or [
        row["effective_update"] for row in steps
    ] != list(range(1, 11)):
        raise ValueError("Wrong attempt/update identity")
    for row in steps:
        if (
            row["reference_synced_from_runtime"] != (expected_mode == "same_start")
            or row["overflow"]
        ):
            raise ValueError("Wrong independence or skipped update")
        path = Path(directory) / row["tensor_path"]
        if not path.is_file() or sha(path) != row["tensor_sha256"]:
            raise ValueError("Missing/changed actual pre/post tensor capture")
        comparisons = [
            ("post_forward/logits", row["logits"]),
            ("post_forward/loss", row["loss"]),
            ("post_backward/decoder_input_gradient", row["decoder_input_gradient"]),
        ]
        comparisons += [
            ("post_unscale/" + name, stats) for name, stats in row["parameter_gradients"].items()
        ]
        comparisons += [
            ("post_optimizer/" + name, stats) for name, stats in row["parameters_after"].items()
        ]
        comparisons += [
            ("post_optimizer/" + name + "/" + key, stats)
            for name, states in row["optimizer_states"].items()
            for key, stats in states.items()
        ]
        local_failures = {name for name, stats in comparisons if not stats["passed"]}
        if {failure["transition"] for failure in row["strict_failures"]} != local_failures:
            raise ValueError("Original failing comparisons were hidden")
        if bool(local_failures) != (payload["rank"] in row["failed_logical_ranks"]):
            raise ValueError("Local failure missing from all-rank propagation")
        strict = bool(row["failed_logical_ranks"])
        if row["strict_status"] != ("FAIL" if strict else "PASS"):
            raise ValueError("Hidden original numerical failure")
    expected = "FAIL" if any(row["strict_status"] == "FAIL" for row in steps) else "PASS"
    if payload["strict_status"] != expected:
        raise ValueError("Hidden trajectory FAIL")
    return {"schema_status": "PASS", "strict_status": expected, "gpu_test_pass": expected == "PASS"}


def audit_tensor(capture, payload, step):
    if (
        capture.get("step") != step
        or capture.get("source_manifest_sha256") != payload["source_manifest_sha256"]
    ):
        raise ValueError("Wrong source/step tensor provenance")
    if capture.get("mode") != payload["mode"]:
        raise ValueError("Tensor mode differs from its run")
    for engine in ("runtime", "reference"):
        validate_state(capture[engine]["pre"], capture[engine]["post"])
    return True
