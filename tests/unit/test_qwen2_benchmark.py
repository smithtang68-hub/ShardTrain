"""Entry-point/artifact contract checks; model numerical tests live elsewhere."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sharded_runtime.models.qwen2 import Qwen2ModelConfig
from sharded_runtime.training import qwen2_benchmark
from sharded_runtime.training.qwen2_benchmark import main


def test_cpu_cli_global_tokens_and_profiling_artifacts(tmp_path, capsys):
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    for name in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        environment.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/benchmark_qwen2.py"),
            "--config",
            str(root / "configs/qwen25_tiny.yaml"),
            "--warmup",
            "0",
            "--profile-mode",
            "torch",
            "--model.vocab_size",
            "17",
            "--model.hidden_size",
            "8",
            "--model.layers",
            "1",
            "--model.num_heads",
            "2",
            "--model.num_key_value_heads",
            "1",
            "--model.mlp_hidden_size",
            "16",
            "--model.seq_len",
            "4",
            "--training.local_batch_size",
            "1",
            "--training.accumulation_steps",
            "2",
            "--training.steps",
            "1",
            "--debug.output_dir",
            str(tmp_path),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads((tmp_path / "benchmark.rank0.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["world_size"] == 1
    assert data["global_batch"] == 2
    assert data["steady_steps_completed"] == 1
    assert data["steps"][0]["valid_tokens"] == 6  # 2 samples * 3 shifted targets.
    assert data["steps"][0]["cuda_elapsed_ms"] is None
    assert data["max_allocated"] is data["max_reserved"] is None
    assert data["step_seconds_p50"] > 0
    assert data["tokens_per_second"] > 0
    assert not data["oom"] and not data["nonfinite_detected"]
    assert data["ownership"]["stable_shard_identity"]
    assert data["ownership"]["live_buffers"] == 0
    assert data["ownership"]["live_fullparam_leases"] == 0
    assert data["ownership"]["units_reduced_once_last_microbatch"]
    assert not data["nccl_correctness_verified"]
    assert not data["model_source"]["actual_checkpoint_logits_reference_verified"]
    profile = json.loads((tmp_path / "profiling.rank0.json").read_text())
    assert profile["mode"] == "torch" and profile["errors"] == []
    trace = json.loads((tmp_path / "profile.rank0.json").read_text())
    assert any(
        event.get("name") == "shardtrain/qwen/model_forward" for event in trace["traceEvents"]
    )
    assert (tmp_path / "resolved_config.rank0.yaml").is_file()
    assert (tmp_path / "environment.rank0.json").is_file()
    preserved = {
        name: (tmp_path / name).read_bytes()
        for name in ["benchmark.rank0.json", "profiling.rank0.json", "environment.rank0.json"]
    }
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--config",
                str(root / "configs/qwen25_tiny.yaml"),
                "--debug.output_dir",
                str(tmp_path),
            ]
        )
    assert error.value.code == 2
    assert "Choose a new --debug.output_dir" in capsys.readouterr().err
    for name, original in preserved.items():
        assert (tmp_path / name).read_bytes() == original


def test_bad_model_source_has_cli_diagnostic_and_failed_artifact(tmp_path, capsys):
    root = Path(__file__).resolve().parents[2]
    missing = tmp_path / "missing_checkpoint"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--config",
                str(root / "configs/qwen25_tiny.yaml"),
                "--model.checkpoint_path",
                str(missing),
                "--debug.output_dir",
                str(tmp_path),
                "--warmup",
                "0",
            ]
        )
    assert error.value.code == 2
    assert "Cannot construct Qwen model/checkpoint" in capsys.readouterr().err
    data = json.loads((tmp_path / "benchmark.rank0.json").read_text())
    assert data["status"] == "failed"
    assert data["steady_steps_completed"] == 0
    assert "does not exist" in data["errors"][0]


def test_cpu_dropout_seed_reproduces_loss_and_parameter_updates(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    for name in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        monkeypatch.delenv(name, raising=False)
    model_config = Qwen2ModelConfig(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_dropout=0.35,
        tie_word_embeddings=True,
    )
    config_path = tmp_path / "dropout_qwen_config.json"
    config_path.write_text(json.dumps(model_config.to_dict()), encoding="utf-8")
    observed_runs = []
    trainer_class = qwen2_benchmark.QwenTrainer

    class ObservedTrainer(trainer_class):
        """Observe real optimizer outputs without changing the training implementation."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observed = {"initial": self.snapshot(), "steps": []}
            observed_runs.append(self.observed)

        def snapshot(self):
            return [parameter.detach().clone() for parameter in self.runtime.sharded_parameters()]

        def train_step(self, *args, **kwargs):
            result = super().train_step(*args, **kwargs)
            self.observed["steps"].append(self.snapshot())
            return result

    monkeypatch.setattr(qwen2_benchmark, "QwenTrainer", ObservedTrainer)
    payloads = []
    for iteration in range(2):
        # Different caller RNG states must not affect the explicitly seeded workload.
        torch.manual_seed(900 + iteration)
        output_dir = tmp_path / f"experiment{iteration}"
        assert (
            main(
                [
                    "--config",
                    str(root / "configs/qwen25_tiny.yaml"),
                    "--model.hf_config_path",
                    str(config_path),
                    "--model.seq_len",
                    "4",
                    "--training.local_batch_size",
                    "1",
                    "--training.steps",
                    "2",
                    "--training.seed",
                    "41",
                    "--warmup",
                    "1",
                    "--debug.output_dir",
                    str(output_dir),
                ]
            )
            == 0
        )
        payloads.append(json.loads((output_dir / "benchmark.rank0.json").read_text()))
    assert all(payload["status"] == "completed" for payload in payloads)
    assert all(
        payload["model_source"]["config"]["attention_dropout"] == 0.35 for payload in payloads
    )
    losses = [
        [step["loss"] for step in payload["warmup_steps"] + payload["steps"]]
        for payload in payloads
    ]
    assert losses[0] == losses[1]
    assert payloads[0]["optimizer_updates"] == payloads[1]["optimizer_updates"] == 2
    first, second = observed_runs
    assert len(first["steps"]) == len(second["steps"]) == 3  # Warmup also updates parameters.
    for before, after in zip(first["initial"], second["initial"], strict=True):
        assert torch.equal(before, after)
    update_norms = []
    previous = first["initial"]
    for left, right in zip(first["steps"], second["steps"], strict=True):
        assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
        update_norms.append(
            sum(
                (a.double() - b.double()).square().sum().item()
                for a, b in zip(left, previous, strict=True)
            )
            ** 0.5
        )
        previous = left
    assert all(norm > 0 for norm in update_norms)
    (tmp_path / "dropout_reproducibility.json").write_text(
        json.dumps(
            {
                "seed": 41,
                "attention_dropout": 0.35,
                "losses": losses[0],
                "update_l2_norms": update_norms,
                "all_persistent_parameters_bitwise_equal": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
