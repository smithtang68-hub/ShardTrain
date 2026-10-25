"""Actual-state replay, unique global distributions, reduction factors and trends."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from experiments.fp16_numerics.audit_results import audit_payload, audit_tensor
from experiments.fp16_numerics.common import distribution, immutable_json, sha
from experiments.fp16_numerics.replay_adamw import analyze_capture


def global_indices(layout):
    start = layout["fsdp_rank"] * layout["fsdp_shard_numel"]
    flat = torch.arange(start, start + layout["valid_numel"])
    coords = list(torch.unravel_index(flat, tuple(layout["tp_local_shape"])))
    if layout["tp_dim"] is not None:
        dim = layout["tp_dim"]
        coords[dim] = coords[dim] + layout["tp_rank"] * layout["tp_local_shape"][dim]
    result = torch.zeros_like(flat)
    for coordinate, width in zip(coords, layout["global_shape"], strict=True):
        result = result * width + coordinate
    return result


def unique_global_distributions(captures, payloads):
    """Fill every global logical element once; assert actual TP replicas agree."""
    summary, concatenated = (
        {},
        {
            key: []
            for key in (
                "parameters_actual",
                "parameters_expected",
                "gradients_actual",
                "gradients_expected",
            )
        },
    )
    for name in payloads[0]["parameter_layout"]:
        example = payloads[0]["parameter_layout"][name]
        count = __import__("math").prod(example["global_shape"])
        arrays = {key: torch.empty(count, dtype=torch.float32) for key in concatenated}
        filled = torch.zeros(count, dtype=torch.bool)
        for rank in range(4):
            layout = payloads[rank]["parameter_layout"][name]
            index = global_indices(layout)
            overlapping = filled[index]
            for key, target in arrays.items():
                values = captures[rank]["distribution_shards"][key][name][: len(index)]
                if overlapping.any() and not torch.equal(
                    target[index[overlapping]], values[overlapping]
                ):
                    raise AssertionError(
                        f"TP replicated actual/reference {key} differs: {name}/rank{rank}"
                    )
                target[index] = values
            filled[index] = True
        if not filled.all():
            raise AssertionError("Missing logical parameter elements")
        summary[name] = {
            "parameter": distribution(arrays["parameters_actual"], arrays["parameters_expected"]),
            "gradient": distribution(arrays["gradients_actual"], arrays["gradients_expected"]),
        }
        for key, value in arrays.items():
            concatenated[key].append(value)
    return {
        "parameters": distribution(
            torch.cat(concatenated["parameters_actual"]),
            torch.cat(concatenated["parameters_expected"]),
        ),
        "gradients": distribution(
            torch.cat(concatenated["gradients_actual"]),
            torch.cat(concatenated["gradients_expected"]),
        ),
        "parameter_count": len(summary),
        "per_parameter": summary,
        "parameter_tensors_failing_original_tolerance": [
            name for name, row in summary.items() if not row["parameter"]["passed"]
        ],
        "gradient_tensors_failing_original_tolerance": [
            name for name, row in summary.items() if not row["gradient"]["passed"]
        ],
        "top_parameter_errors": sorted(
            ({"name": name, **row["parameter"]} for name, row in summary.items()),
            key=lambda row: row["max_abs"] or 0,
            reverse=True,
        )[:5],
    }


def reduction_audit(captures):
    rows = []
    for t in range(2):
        ranks = [t, t + 2]
        values = [captures[rank]["gradient_transitions"] for rank in ranks]
        full = (
            values[0]["k_bias_pre_fsdp_scaled_tp_local"]
            + values[1]["k_bias_pre_fsdp_scaled_tp_local"]
        ) / 2
        for f, rank in enumerate(ranks):
            row = captures[rank]
            width = row["runtime"]["pre"]["parameter"].numel()
            expected = full.reshape(-1)[f * width : (f + 1) * width]
            actual = row["gradient_transitions"]["k_bias_post_fsdp_average_scaled_shard"]
            scaled = row["gradient_transitions"]["k_bias_reduced_scaled_shard"]
            unscaled = row["gradient_transitions"]["k_bias_reduced_unscaled_shard"]
            scale = row["amp"]["scale_before"]
            rows.append(
                {
                    "rank": rank,
                    "tp_coordinate": t,
                    "fsdp_coordinate": f,
                    "materialized_master_matches_owned_pre_forward_shard": distribution(
                        row["gradient_transitions"]["k_bias_forward_materialized_master_tp_local"][
                            f * width : (f + 1) * width
                        ],
                        row["pre_forward"]["runtime_master"],
                        atol=0,
                        rtol=0,
                    ),
                    "compute_view_matches_real_master_cast": distribution(
                        row["gradient_transitions"]["k_bias_forward_effective_compute_tp_local"],
                        row["gradient_transitions"][
                            "k_bias_forward_materialized_master_tp_local"
                        ].half(),
                        atol=0,
                        rtol=0,
                    ),
                    "fsdp_sum_then_divide_by_two": distribution(actual, expected, atol=0, rtol=0),
                    "post_reduction_reaches_pre_unscale": distribution(
                        scaled, actual, atol=0, rtol=0
                    ),
                    "amp_unscale_power_of_two": distribution(
                        unscaled, scaled / scale, atol=0, rtol=0
                    ),
                    "actual_optimizer_receives_unscaled_gradient": distribution(
                        row["runtime"]["pre"]["gradient"], unscaled, atol=0, rtol=0
                    ),
                    "sum_compute_cast_reaches_pre_fsdp": distribution(
                        row["gradient_transitions"]["k_bias_sum_after_compute_cast"],
                        row["gradient_transitions"]["k_bias_pre_fsdp_scaled_tp_local"],
                        atol=0,
                        rtol=0,
                    ),
                }
            )
    return {
        "rows": rows,
        "status": "PASS"
        if all(
            metric["passed"]
            for row in rows
            for key, metric in row.items()
            if isinstance(metric, dict)
        )
        else "FAIL",
        "scope": (
            "Actual target Kbias F2 input contribution -> SUM/F -> shard -> "
            "unscale -> optimizer; does not establish all parameter math"
        ),
    }


def trend(values):
    x = torch.arange(len(values), dtype=torch.float64)
    y = torch.tensor(values, dtype=torch.float64)
    slope = (
        float(((x - x.mean()) * (y - y.mean())).sum() / (x - x.mean()).square().sum())
        if len(values) > 1
        else None
    )
    return {
        "first": values[0],
        "last": values[-1],
        "first_three_mean": sum(values[:3]) / len(values[:3]),
        "last_three_mean": sum(values[-3:]) / len(values[-3:]),
        "linear_slope_per_update": slope,
        "monotonic_nondecreasing": all(
            b >= a for a, b in zip(values[:-1], values[1:], strict=True)
        ),
        "interpretation": "descriptive only; no posthoc pass threshold",
    }


def analyze(directory, *, device="cpu"):
    directory = Path(directory)
    payloads = [
        json.loads((directory / f"trajectory.rank{rank}.json").read_text()) for rank in range(4)
    ]
    source = payloads[0]["source_manifest_sha256"]
    mode = payloads[0]["mode"]
    checks = [
        audit_payload(payload, expected_mode=mode, directory=directory) for payload in payloads
    ]
    if (
        len({p["source_manifest_sha256"] for p in payloads}) != 1
        or len({p["placement_manifest_sha256"] for p in payloads}) != 1
    ):
        raise ValueError("Mixed source or placement")
    summaries, provenance = [], {}
    for step in range(10):
        captures = []
        for payload in payloads:
            path = directory / payload["steps"][step]["tensor_path"]
            capture = torch.load(path, map_location="cpu", weights_only=True)
            audit_tensor(capture, payload, step)
            captures.append(capture)
            provenance[str(path)] = sha(path)
        replay = [analyze_capture(capture, device=device) for capture in captures]
        summaries.append(
            {
                "step": step,
                "effective_update": step + 1,
                "strict_status": payloads[0]["steps"][step]["strict_status"],
                "first_nonbitwise_transition_by_rank": [
                    p["steps"][step]["first_nonbitwise_transition"] for p in payloads
                ],
                "loss_runtime": payloads[0]["steps"][step]["loss_runtime"],
                "loss_reference": payloads[0]["steps"][step]["loss_reference"],
                "loss_abs_error": abs(
                    payloads[0]["steps"][step]["loss_runtime"]
                    - payloads[0]["steps"][step]["loss_reference"]
                ),
                "global_unique": unique_global_distributions(captures, payloads),
                "target_reduction_audit": reduction_audit(captures),
                "target_adamw_replays": replay,
            }
        )
    result = {
        "schema_version": 1,
        "mode": mode,
        "classification": "UNRESOLVED_PENDING_CURRENT_CONTROLS_AND_REVIEW",
        "strict_independent_status": checks[0]["strict_status"]
        if mode == "independent"
        else "NOT_RUN_BY_THIS_MODE",
        "strict_status": checks[0]["strict_status"],
        "capture_status": "COMPLETED",
        "source_manifest_sha256": source,
        "placement_manifest_sha256": payloads[0]["placement_manifest_sha256"],
        "training_runtime_bug_confirmed": None,
        "continuation_authorized": False,
        "gpu_test_pass": False,
        "first_nonbitwise_transition": next(
            (
                {
                    "step": row["step"],
                    "transition_by_rank": row["first_nonbitwise_transition_by_rank"],
                }
                for row in summaries
                if any(row["first_nonbitwise_transition_by_rank"])
            ),
            None,
        ),
        "first_strict_failure": next(
            (
                {"step": row["step"], "effective_update": row["effective_update"]}
                for row in summaries
                if row["strict_status"] == "FAIL"
            ),
            None,
        ),
        "target_reduction_contract_status": "PASS"
        if all(row["target_reduction_audit"]["status"] == "PASS" for row in summaries)
        else "FAIL",
        "actual_state_replay_status": "REPLAY_MATCHED"
        if all(
            replay["replay_status"] == "REPLAY_MATCHED"
            for row in summaries
            for replay in row["target_adamw_replays"]
        )
        else "REPLAY_MISMATCHED",
        "all_local_leaf_same_device_bit_exact_replay": all(
            leaf["parameter_bit_exact"] and leaf["optimizer_state_bit_exact"]
            for p in payloads
            for row in p["steps"]
            for engine in ("runtime_optimizer_replay", "reference_optimizer_replay")
            for leaf in row[engine]
        ),
        "trends": {
            "global_unique_parameter_max_abs": trend(
                [row["global_unique"]["parameters"]["max_abs"] for row in summaries]
            ),
            "global_unique_parameter_relative_l2": trend(
                [row["global_unique"]["parameters"]["relative_l2"] for row in summaries]
            ),
            "loss_abs_error": trend([row["loss_abs_error"] for row in summaries]),
        },
        "steps": summaries,
        "capture_paths_sha256": provenance,
        "run_results_sha256": {
            str(directory / f"trajectory.rank{rank}.json"): sha(
                directory / f"trajectory.rank{rank}.json"
            )
            for rank in range(4)
        },
        "limits": [
            (
                "Ten small-model updates do not prove long-term convergence or "
                "rule out all runtime bugs."
            ),
            (
                "CPU replay has preregistered residual tolerance; every actual GPU"
                " optimizer update also receives same-device bit-exact all-leaf "
                "replay."
            ),
            (
                "Classification requires fresh FP32/same-start/control results and"
                " human review. Independent strict FAIL is never rewritten."
            ),
        ],
    }
    return result


def analyze_controls(directory, *, device="cpu"):
    directory = Path(directory)
    raw = json.loads((directory / "control.json").read_text())
    if (
        raw["capture_status"] != "COMPLETED"
        or raw["completed_effective_updates"] != 10
        or raw["reference_synced_each_step"]
    ):
        raise ValueError("Incomplete or non-independent pure HF control")
    if (
        raw["actual_custom_runtime_executed"]
        or raw["actual_distributed_collectives"]
        or raw["actual_SP_execution"]
    ):
        raise ValueError("Pure-control execution scope is inconsistent")
    rows = []
    for step in raw["steps"]:
        for name, pair in step["comparisons"].items():
            path = directory / pair["tensor_path"]
            if sha(path) != pair["tensor_sha256"]:
                raise ValueError("Changed pure HF capture")
            capture = torch.load(path, weights_only=True, map_location="cpu")
            if (
                capture["source_manifest_sha256"] != raw["source_manifest_sha256"]
                or capture["actual_custom_runtime_executed"]
            ):
                raise ValueError("Unbound or mislabeled pure-control tensors")
            rows.append(
                {
                    "step": step["step"],
                    "model": name,
                    "replay": analyze_capture(capture, device=device),
                    "tensor_path": str(path),
                    "tensor_sha256": sha(path),
                    "strict_parameter_failures": pair["strict_parameter_failures"],
                    "loss_runtime": step["models"][name]["loss"],
                    "loss_reference": step["models"]["hf_dense_b4"]["loss"],
                    "parameters": pair["all_parameters"],
                    "gradients": pair["all_gradients"],
                }
            )
    return {
        "schema_version": 1,
        "mode": "pure_hf_independent",
        "capture_status": "COMPLETED",
        "strict_status": raw["strict_status"],
        "source_manifest_sha256": raw["source_manifest_sha256"],
        "gpu_test_pass": False,
        "training_runtime_bug_confirmed": None,
        "classification": "ARITHMETIC_CONTROL_ONLY_NOT_A_RUNTIME_VERDICT",
        "control_path": str(directory / "control.json"),
        "control_sha256": sha(directory / "control.json"),
        "rows": rows,
        "limitations": raw["limitations"],
    }


def plot_result(result, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    if result["mode"] == "pure_hf_independent":
        for name in sorted({row["model"] for row in result["rows"]}):
            rows = [row for row in result["rows"] if row["model"] == name]
            x = [row["step"] + 1 for row in rows]
            axes[0, 0].plot(
                x, [row["parameters"]["max_abs"] for row in rows], marker="o", label=name
            )
            axes[0, 1].plot(
                x, [row["parameters"]["relative_l2"] for row in rows], marker="o", label=name
            )
            axes[1, 0].plot(x, [row["gradients"]["cosine"] for row in rows], marker="o", label=name)
            axes[1, 1].plot(
                x,
                [abs(row["loss_runtime"] - row["loss_reference"]) for row in rows],
                marker="o",
                label=name,
            )
    else:
        rows = result["steps"]
        x = [row["effective_update"] for row in rows]
        axes[0, 0].plot(
            x,
            [row["global_unique"]["parameters"]["max_abs"] for row in rows],
            marker="o",
            label="global unique",
        )
        axes[0, 1].plot(
            x, [row["global_unique"]["parameters"]["relative_l2"] for row in rows], marker="o"
        )
        axes[1, 0].plot(
            x, [row["global_unique"]["gradients"]["cosine"] for row in rows], marker="o"
        )
        axes[1, 1].plot(x, [row["loss_abs_error"] for row in rows], marker="o")
    for axis, title in zip(
        axes.flat,
        [
            "Parameter max absolute error",
            "Parameter relative L2 error",
            "Gradient cosine",
            "Loss absolute error",
        ],
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xlabel("Effective update")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        f"{result['mode']}: strict {result['strict_status']} retained; ten-update diagnostic"
    )
    figure.savefig(output / "trend.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--capture-dir")
    sources.add_argument("--control-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay-device", default="cpu")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    result = (
        analyze(args.capture_dir, device=args.replay_device)
        if args.capture_dir
        else analyze_controls(args.control_dir, device=args.replay_device)
    )
    output = Path(args.output_dir)
    immutable_json(output / "summary.json", result)
    if args.plots:
        plot_result(result, output)
    if args.control_dir:
        return
    with (output / "trend.csv").open("x", newline="", encoding="utf-8") as stream:
        fields = [
            "step",
            "effective_update",
            "strict_status",
            "loss_runtime",
            "loss_reference",
            "loss_abs_error",
            "parameter_max_abs",
            "parameter_relative_l2",
            "parameter_p50",
            "parameter_p95",
            "parameter_p99",
            "gradient_cosine",
            "failed_parameter_tensors",
            "failed_parameter_elements",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["steps"]:
            p, g = row["global_unique"]["parameters"], row["global_unique"]["gradients"]
            writer.writerow(
                {
                    **{key: row[key] for key in fields[:6]},
                    "parameter_max_abs": p["max_abs"],
                    "parameter_relative_l2": p["relative_l2"],
                    "parameter_p50": p["p50_abs"],
                    "parameter_p95": p["p95_abs"],
                    "parameter_p99": p["p99_abs"],
                    "gradient_cosine": g["cosine"],
                    "failed_parameter_tensors": len(
                        row["global_unique"]["parameter_tensors_failing_original_tolerance"]
                    ),
                    "failed_parameter_elements": p["tolerance_failed_elements"],
                }
            )


if __name__ == "__main__":
    main()
