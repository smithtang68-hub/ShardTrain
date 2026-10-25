"""Replay real captured pre/post AdamW state; no invented or inverted pre-state."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from experiments.fp16_numerics.common import BIAS, OPTIMIZER, distribution, immutable_json, sha

CPU_ATOL, CPU_RTOL = 1e-9, 1e-6


def validate_state(pre, post=None):
    required = {"parameter", "gradient", "state_initialized", "state", "options"}
    if not required <= pre.keys() or pre["gradient"] is None:
        raise ValueError("Real parameter, gradient and explicit pre-optimizer state required")
    if pre["parameter"].dtype != torch.float32 or pre["gradient"].dtype != torch.float32:
        raise ValueError("This replay contract requires real FP32 master/gradient tensors")
    if pre["gradient"].shape != pre["parameter"].shape:
        raise ValueError("Gradient/parameter shape mismatch")
    if pre["state_initialized"]:
        if set(pre["state"]) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Missing actual initialized AdamW pre-state")
    elif pre["state"]:
        raise ValueError("Initial uninitialized state must be explicitly empty")
    if post is not None and (
        not post["state_initialized"] or set(post["state"]) != {"step", "exp_avg", "exp_avg_sq"}
    ):
        raise ValueError("Missing actual post-optimizer state")


def predict(pre, *, device="cpu", gradient=None):
    validate_state(pre)
    parameter = torch.nn.Parameter(pre["parameter"].to(device).clone())
    optimizer = torch.optim.AdamW([parameter], **pre["options"])
    parameter.grad = (pre["gradient"] if gradient is None else gradient).to(device).clone()
    if pre["state_initialized"]:
        optimizer.state[parameter] = {
            key: value.to(device if value.ndim else "cpu").clone()
            if torch.is_tensor(value)
            else copy.deepcopy(value)
            for key, value in pre["state"].items()
        }
    optimizer.step()
    return {
        "parameter": parameter.detach().cpu(),
        "state": {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in optimizer.state[parameter].items()
        },
    }


def terms(pre):
    """FP64 explanatory terms; not a claim of FP32 GPU bit identity."""
    validate_state(pre)
    options, state = pre["options"], pre["state"]
    theta, grad = pre["parameter"].double(), pre["gradient"].double()
    m = state["exp_avg"].double() if state else torch.zeros_like(theta)
    v = state["exp_avg_sq"].double() if state else torch.zeros_like(theta)
    step = int(state["step"]) + 1 if state else 1
    beta1, beta2 = options["betas"]
    new_m, new_v = beta1 * m + (1 - beta1) * grad, beta2 * v + (1 - beta2) * grad.square()
    mhat, vhat = new_m / (1 - beta1**step), new_v / (1 - beta2**step)
    denominator = vhat.sqrt() + options["eps"]
    ratio = mhat / denominator
    decay = -theta * options["lr"] * options["weight_decay"]
    adam = -options["lr"] * ratio
    return {
        "step_after": step,
        "gradient": grad.tolist(),
        "gradient_near_eps": (grad.abs() <= options["eps"]).tolist(),
        "m_before": m.tolist(),
        "v_before": v.tolist(),
        "m_after_fp64": new_m.tolist(),
        "v_after_fp64": new_v.tolist(),
        "bias_corrected_m": mhat.tolist(),
        "bias_corrected_v": vhat.tolist(),
        "sqrt_vhat_plus_eps": denominator.tolist(),
        "mhat_over_denominator": ratio.tolist(),
        "weight_decay_update_fp64": decay.tolist(),
        "adam_update_fp64": adam.tolist(),
        "predicted_parameter_fp64": (theta + decay + adam).tolist(),
    }


def replay_one(pre, post, *, device="cpu"):
    validate_state(pre, post)
    predicted = predict(pre, device=device)
    residuals = {
        "parameter": distribution(
            predicted["parameter"], post["parameter"], atol=CPU_ATOL, rtol=CPU_RTOL
        )
    }
    for key in ("step", "exp_avg", "exp_avg_sq"):
        residuals[key] = distribution(
            predicted["state"][key], post["state"][key], atol=CPU_ATOL, rtol=CPU_RTOL
        )
    return {
        "status": "REPLAY_MATCHED"
        if all(row["passed"] for row in residuals.values())
        else "REPLAY_MISMATCHED",
        "execution_device": device,
        "execution_torch": torch.__version__,
        "residual_tolerance": {"atol": CPU_ATOL, "rtol": CPU_RTOL},
        "residuals": residuals,
        "fp64_explanatory_terms": terms(pre),
    }


def analyze_capture(capture, *, device="cpu"):
    if capture.get("schema_version") != 1 or "runtime" not in capture or "reference" not in capture:
        raise ValueError("Not a real pre/post capture")
    a, b = capture["runtime"], capture["reference"]
    results = {
        "runtime": replay_one(a["pre"], a["post"], device=device),
        "reference": replay_one(b["pre"], b["post"], device=device),
    }
    # Both gradients must already represent the same logical elements.
    if a["pre"]["parameter"].shape != b["pre"]["parameter"].shape:
        raise ValueError("Counterfactual requires aligned ownership/index slice")
    shared = a["pre"]
    aa = predict(shared, device=device)["parameter"]
    ab = predict(shared, device=device, gradient=b["pre"]["gradient"])["parameter"]
    ba = predict(b["pre"], device=device, gradient=shared["gradient"])["parameter"]
    bb = predict(b["pre"], device=device)["parameter"]
    results["counterfactual"] = {
        "shared_runtime_state_runtime_vs_dense_grad": distribution(aa, ab),
        "shared_runtime_gradient_runtime_vs_dense_state": distribution(aa, ba),
        "shared_dense_gradient_runtime_vs_dense_state": distribution(ab, bb),
        "actual_input_prediction_total_difference": distribution(aa, bb),
        "gradient_only_parameter_delta": (aa - ab).tolist(),
        "state_only_at_runtime_gradient_parameter_delta": (aa - ba).tolist(),
        "actual_total_predicted_parameter_delta": (aa - bb).tolist(),
        "observed_parameter_delta": (a["post"]["parameter"] - b["post"]["parameter"]).tolist(),
        "gradient_sign_change": ((a["pre"]["gradient"] * b["pre"]["gradient"]) < 0).tolist(),
        "scope": (
            "Controlled FP32 replay; gradient/state effects are path-"
            "dependent, not additive causal percentages"
        ),
    }
    results["replay_status"] = (
        "REPLAY_MATCHED"
        if all(results[key]["status"] == "REPLAY_MATCHED" for key in ("runtime", "reference"))
        else "REPLAY_MISMATCHED"
    )
    results["attribution_status"] = "DRIFT_ATTRIBUTION_INCOMPLETE"
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--capture")
    sources.add_argument("--historical-control")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.capture:
        data = torch.load(args.capture, weights_only=True, map_location="cpu")
        result = analyze_capture(data, device=args.device)
        result.update(source_capture=str(Path(args.capture).resolve()), gpu_test_pass=False)
    else:
        result = historical_control(args.historical_control, device=args.device)
    immutable_json(args.output, result)


def historical_control(directory, *, device="cpu"):
    """Pre-state is real preceding post-state, never inverted from later state."""
    import json

    directory = Path(directory)
    diagnosis = json.loads((directory / "diagnosis.json").read_text())
    previous = {}
    records = []
    for step_row in diagnosis["steps_data"]:
        step = step_row["step"]
        path = directory / Path(step_row["tensor_file"]).name
        if sha(path) != step_row["tensor_sha256"]:
            raise ValueError("Historical tensor JSON hash mismatch")
        data = json.loads(path.read_text())
        for name, values in data.items():

            def tensor(field, values=values):
                return torch.tensor(values[field][BIAS]["values"], dtype=torch.float32)

            post_state = {
                key: torch.tensor(value["values"], dtype=torch.float32)
                for key, value in values["optimizer_state"][BIAS].items()
            }
            pre = {
                "parameter": tensor("parameters_before"),
                "gradient": tensor("gradients"),
                "state_initialized": step > 0,
                "state": previous.get(name, {}),
                "options": dict(OPTIMIZER),
            }
            post = {
                "parameter": tensor("parameters_after"),
                "gradient": tensor("gradients"),
                "state_initialized": True,
                "state": post_state,
                "options": dict(OPTIMIZER),
            }
            replay = replay_one(pre, post, device=device)
            records.append(
                {"step": step, "model": name, "tensor_sha256": sha(path), "replay": replay}
            )
            previous[name] = post_state
    return {
        "evidence_class": "HISTORICAL_RAW_OFFLINE_REPLAY",
        "gpu_test_pass": False,
        "historical_strict_failures_reclassified": False,
        "historical_diagnosis_sha256": sha(directory / "diagnosis.json"),
        "optimizer_options_source": (
            "INFERRED_FROM_FROZEN_LEGACY_CODE_AND_RECORDED_TORCH_VERSION: "
            "AdamW lr=.001 foreach=False other defaults; not an independently "
            "logged pre-options dict"
        ),
        "pre_state_source": (
            "step0 explicitly uninitialized; later steps use previous real "
            "post-state, source loop independent and never reset"
        ),
        "main_runtime_failure_exact_pre_state": (
            "NOT_AVAILABLE; no replay invented from the saved post-update failure_state"
        ),
        "records": records,
        "status": "REPLAY_MATCHED"
        if all(row["replay"]["status"] == "REPLAY_MATCHED" for row in records)
        else "REPLAY_MISMATCHED",
    }


if __name__ == "__main__":
    main()
