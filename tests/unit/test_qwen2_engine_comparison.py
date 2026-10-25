"""Negative pairing checks; synthetic fixtures are not a model correctness oracle."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from sharded_runtime.training.performance import qwen_performance_summary

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/compare_qwen2_engines.py"
spec = importlib.util.spec_from_file_location("compare_qwen2_engines", SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def write_pair(tmp_path):
    directories = [tmp_path / "custom", tmp_path / "native"]
    config = {
        "model": {"model_type": "qwen2", "seq_len": 8},
        "parallel": {
            "fsdp_degree": 2,
            "tp_degree": 2,
            "sp_enabled": True,
            "device": "cuda",
            "backend": "nccl",
        },
        "training": {
            "dtype": "bf16",
            "optimizer": "adamw",
            "lr": 0.001,
            "steps": 2,
            "local_batch_size": 1,
            "accumulation_steps": 1,
            "seed": 1234,
        },
    }
    source = {
        "initialization": "random",
        "seed": 1234,
        "global_unique_parameter_numel": 8128,
        "tied_aliases": {"lm_head.weight": "model.embed_tokens.weight"},
        "config": {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 64,
            "tie_word_embeddings": True,
            "attention_dropout": 0.0,
        },
    }
    for index, directory in enumerate(directories):
        directory.mkdir()
        (directory / "gpu_topology.txt").write_text(
            " GPU0 GPU1 GPU2 GPU3 CPU Affinity\n"
            "GPU0 X NV2 PHB PHB 0-15\nGPU1 NV2 X PHB PHB 0-15\n"
            "GPU2 PHB PHB X NV2 16-31\nGPU3 PHB PHB NV2 X 16-31\n"
        )
        (directory / "gpu_topology_status.json").write_text('{"returncode":0}')
        factor = index + 1
        for rank in range(4):
            rows = []
            for step in range(4):
                row = {
                    "step": step,
                    "loss": 4.0 + index,
                    "valid_tokens": 14,
                    "loss_scale": 1.0,
                    "overflow": False,
                    "accumulation_steps": 1,
                }
                if step >= 2:
                    maximum = factor * (step - 1)
                    row.update(
                        measured_step=step - 2,
                        max_rank_seconds=maximum,
                        seconds=maximum * [0.91, 0.96, 1.0, 0.98][rank],
                        cuda_elapsed_ms=maximum * 800,
                    )
                rows.append(row)
            record = {
                "engine": ["shardtrain", "pytorch_fsdp2_dtensor"][index],
                "rank": rank,
                "world_size": 4,
                "coordinate": {"fsdp": rank // 2, "tp": rank % 2},
                "status": "completed",
                "errors": [],
                "oom": False,
                "nonfinite_loss": False,
                "nonfinite_detected": False,
                "profile_mode": "off",
                "runtime_events_enabled": False,
                "dtype": "bf16",
                "optimizer": "adamw",
                "micro_batch": 1,
                "global_batch": 2,
                "accumulation_steps": 1,
                "sequence_length": 8,
                "warmup": 2,
                "steady_steps_completed": 2,
                "overflow_steps": 0,
                "warmup_overflow_steps": 0,
                "optimizer_updates": 2,
                "model_source": source,
                "input_batch_digest": hashlib.sha256(f"replica{rank // 2}".encode()).hexdigest(),
                "precision_policy": {
                    "compute": "bf16",
                    "linear_backward": "bf16",
                    "parameters": "fp32",
                    "parameter_all_gather": "fp32",
                    "gradient_reduce": "fp32",
                    "optimizer_state": "fp32",
                    "norm_accumulation": "fp32",
                    "loss": "fp32",
                },
                "optimizer_hyperparameters": {
                    "lr": 0.001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                    "foreach": False,
                },
                "versions": {
                    "torch": "2.9.1+cu126",
                    "cuda_build": "12.6",
                    "nccl": [2, 27, 5],
                    "platform": "Linux",
                },
                "device": f"cuda:{rank}",
                "device_name": "Tesla V100-SXM2-32GB",
                "device_capability": [7, 0],
                "attention_backend": "PyTorch SDPA math; explicit grouped KV expansion",
                "steps": rows[2:],
                "warmup_steps": rows[:2],
                "max_allocated": factor * (1000 + rank),
                "max_reserved": factor * (2000 + rank),
                "allocated_end": factor * (700 + rank),
                "reserved_end": factor * (1500 + rank),
                "performance": qwen_performance_summary(
                    parameter_numel=8128,
                    layers=2,
                    hidden_size=16,
                    sequence_length=8,
                    global_batch=2,
                    steps=rows[2:],
                    world_size=4,
                    peak_tflops_per_gpu=125.0,
                ),
            }
            (directory / f"benchmark.rank{rank}.json").write_text(json.dumps(record))
            if index == 0:
                (directory / f"resolved_config.rank{rank}.yaml").write_text(yaml.safe_dump(config))
            else:
                (directory / f"environment.rank{rank}.json").write_text(
                    json.dumps({"config": config})
                )
    return directories


def modify(directory, rank, transform):
    path = directory / f"benchmark.rank{rank}.json"
    record = json.loads(path.read_text())
    transform(record)
    path.write_text(json.dumps(record))


def test_unprofiled_ratios_and_explicit_correctness_limit(tmp_path):
    pair = write_pair(tmp_path)
    originals = {path: path.read_bytes() for directory in pair for path in directory.iterdir()}
    result = comparison.compare_runs(*pair)
    assert result["status"] == "paired_conditions_match"
    assert result["shardtrain_over_native"]["mean_max_rank_seconds"] == 0.5
    assert result["shardtrain_speedup"] == 2
    for field in (
        "global_input_tokens_per_second",
        "global_valid_tokens_per_second",
        "mfu_percent",
    ):
        assert result["shardtrain_over_native"][field] == 2
    assert result["memory"]["max_allocated"]["shardtrain_over_native_by_rank"] == [0.5] * 4
    assert result["correctness"]["gradient_and_update_equivalence_proven"] is False
    assert result["correctness"]["requires_independent_oracle"] is True
    assert result["correctness"]["loss_similarity_used_as_evidence"] is False
    assert result["shardtrain_performance"]["global_input_tokens_per_step"] == 16
    # Deliberately different loss values pass pairing and never imply correctness.
    assert all(path.read_bytes() == contents for path, contents in originals.items())


@pytest.mark.parametrize("extra", [False, True])
def test_missing_or_extra_rank_rejected(tmp_path, extra):
    pair = write_pair(tmp_path)
    if extra:
        (pair[1] / "benchmark.rank4.json").write_text("{}")
    else:
        (pair[1] / "benchmark.rank3.json").unlink()
    with pytest.raises(comparison.ComparisonError, match="exactly four rank"):
        comparison.compare_runs(*pair)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("profile_mode", "nsight", "profile_mode"),
        ("runtime_events_enabled", True, "runtime event"),
        ("status", "failed", "status"),
        ("overflow_steps", 1, "overflow"),
        ("nonfinite_detected", True, "nonfinite"),
        ("device", "cuda:3", "device mapping"),
        ("steady_steps_completed", 1, "completed steps"),
    ],
)
def test_invalid_run_rejected(tmp_path, field, value, error):
    pair = write_pair(tmp_path)
    modify(pair[0], 0, lambda row: row.update({field: value}))
    with pytest.raises(comparison.ComparisonError, match=error):
        comparison.compare_runs(*pair)


@pytest.mark.parametrize(
    "field", ["input_batch_digest", "optimizer_hyperparameters", "device_name"]
)
def test_paired_rank_metadata_mismatch_rejected(tmp_path, field):
    pair = write_pair(tmp_path)
    for rank in range(4):

        def mutation(row):
            if field == "input_batch_digest":
                row[field] = "a" * 64
            elif field == "optimizer_hyperparameters":
                row[field]["eps"] = 1e-7
            else:
                row[field] = "different GPU"

        modify(pair[1], rank, mutation)
    with pytest.raises(comparison.ComparisonError, match=field):
        comparison.compare_runs(*pair)


def test_dtype_and_precision_policy_cannot_be_inferred_from_loss(tmp_path):
    pair = write_pair(tmp_path)
    for rank in range(4):
        modify(
            pair[1], rank, lambda row: row["precision_policy"].update(parameter_all_gather="bf16")
        )
    with pytest.raises(comparison.ComparisonError, match="parameter_all_gather"):
        comparison.compare_runs(*pair)


@pytest.mark.parametrize("peak", [None, 0, -1, float("nan")])
def test_missing_or_invalid_peak_rejected(tmp_path, peak):
    pair = write_pair(tmp_path)
    modify(pair[1], 0, lambda row: row["performance"].update(peak_tflops_per_gpu=peak))
    with pytest.raises(comparison.ComparisonError, match="peak"):
        comparison.compare_runs(*pair)


def test_different_explicit_peak_rejected_even_with_consistent_summary(tmp_path):
    pair = write_pair(tmp_path)
    for rank in range(4):

        def mutation(row):
            row["performance"] = qwen_performance_summary(
                parameter_numel=8128,
                layers=2,
                hidden_size=16,
                sequence_length=8,
                global_batch=2,
                steps=row["steps"],
                world_size=4,
                peak_tflops_per_gpu=112.0,
            )

        modify(pair[1], rank, mutation)
    with pytest.raises(comparison.ComparisonError, match="Explicit nominal peak"):
        comparison.compare_runs(*pair)


@pytest.mark.parametrize("kind", ["stale", "wrong_max", "nonfinite", "overflow"])
def test_bad_timing_and_step_records_rejected(tmp_path, kind):
    pair = write_pair(tmp_path)

    def mutation(row):
        if kind == "stale":
            row["performance"]["mfu_percent"] *= 10
        elif kind == "wrong_max":
            row["steps"][0]["seconds"] = 5
        elif kind == "nonfinite":
            row["steps"][0]["seconds"] = float("inf")
        else:
            row["warmup_steps"][0]["overflow"] = True

    modify(pair[0], 0, mutation)
    with pytest.raises(comparison.ComparisonError):
        comparison.compare_runs(*pair)


def test_topology_and_profiling_files_rejected(tmp_path):
    pair = write_pair(tmp_path)
    path = pair[1] / "gpu_topology.txt"
    original = path.read_text()
    path.write_text(original.replace("NV2", "PHB"))
    with pytest.raises(comparison.ComparisonError, match="topology"):
        comparison.compare_runs(*pair)
    path.write_text(original)
    (pair[0] / "profiling.rank0.json").write_text('{"mode":"torch","errors":[]}')
    with pytest.raises(comparison.ComparisonError, match="profiling mode"):
        comparison.compare_runs(*pair)


def test_off_profiling_metadata_is_legal_but_nsight_capture_is_rejected(tmp_path):
    pair = write_pair(tmp_path)
    for directory in pair:
        for rank in range(4):
            (directory / f"profiling.rank{rank}.json").write_text('{"mode":"off","errors":[]}')
    assert comparison.compare_runs(*pair)["status"] == "paired_conditions_match"
    (pair[1] / "timeline.sqlite").write_bytes(b"not parsed")
    with pytest.raises(comparison.ComparisonError, match="Nsight capture artifacts"):
        comparison.compare_runs(*pair)


def test_cli_writes_new_result_and_preserves_existing_file(tmp_path):
    pair = write_pair(tmp_path)
    output = tmp_path / "comparison.json"
    args = ["--shardtrain-run", str(pair[0]), "--native-run", str(pair[1]), "--output", str(output)]
    assert comparison.main(args) == 0
    assert json.loads(output.read_text())["shardtrain_speedup"] == 2
    original = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        comparison.main(args)
    assert exc.value.code == 2
    assert output.read_bytes() == original


def test_requested_mesh_and_accumulation_consistency_rejected(tmp_path):
    pair = write_pair(tmp_path)
    path = pair[1] / "environment.rank0.json"
    env = json.loads(path.read_text())
    changed = copy.deepcopy(env)
    changed["config"]["parallel"]["sp_enabled"] = False
    path.write_text(json.dumps(changed))
    with pytest.raises(comparison.ComparisonError, match="sp_enabled"):
        comparison.compare_runs(*pair)


@pytest.mark.parametrize(
    "field,value", [("torch", "2.8.0"), ("cuda_build", "12.4"), ("nccl", [2, 26, 2])]
)
def test_paired_library_versions_rejected(tmp_path, field, value):
    pair = write_pair(tmp_path)
    for rank in range(4):
        modify(pair[1], rank, lambda row: row["versions"].update({field: value}))
    with pytest.raises(comparison.ComparisonError, match=field):
        comparison.compare_runs(*pair)


def test_same_shape_but_different_source_seed_rejected(tmp_path):
    pair = write_pair(tmp_path)
    for rank in range(4):
        modify(pair[1], rank, lambda row: row["model_source"].update(seed=4321))
        path = pair[1] / f"environment.rank{rank}.json"
        env = json.loads(path.read_text())
        env["config"]["training"]["seed"] = 4321
        path.write_text(json.dumps(env))
    with pytest.raises(comparison.ComparisonError, match="seed"):
        comparison.compare_runs(*pair)


def test_different_model_count_rejected_with_recomputed_flops(tmp_path):
    pair = write_pair(tmp_path)
    for rank in range(4):

        def mutation(row):
            row["model_source"]["global_unique_parameter_numel"] = 8192
            row["performance"] = qwen_performance_summary(
                parameter_numel=8192,
                layers=2,
                hidden_size=16,
                sequence_length=8,
                global_batch=2,
                steps=row["steps"],
                world_size=4,
                peak_tflops_per_gpu=125.0,
            )

        modify(pair[1], rank, mutation)
    with pytest.raises(comparison.ComparisonError, match="global_unique_parameter_numel"):
        comparison.compare_runs(*pair)


def test_dtype_mismatch_rejected_with_consistent_local_metadata(tmp_path):
    pair = write_pair(tmp_path)
    for rank in range(4):

        def mutation(row):
            row["dtype"] = "fp16"
            row["precision_policy"].update(compute="fp16", linear_backward="fp16")

        modify(pair[1], rank, mutation)
        path = pair[1] / f"environment.rank{rank}.json"
        env = json.loads(path.read_text())
        env["config"]["training"]["dtype"] = "fp16"
        path.write_text(json.dumps(env))
    with pytest.raises(comparison.ComparisonError, match="dtype"):
        comparison.compare_runs(*pair)


def test_old_checkpoint_metadata_cannot_establish_same_start_weights(tmp_path):
    pair = write_pair(tmp_path)
    modify(pair[0], 0, lambda row: row["model_source"].update(initialization="local_safetensors"))
    with pytest.raises(comparison.ComparisonError, match="shared weight digest"):
        comparison.compare_runs(*pair)
