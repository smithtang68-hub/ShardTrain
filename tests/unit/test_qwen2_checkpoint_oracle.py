"""Run the independent local-checkpoint acceptance CLI, including its real runtime."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")


@pytest.fixture
def checkpoint(tmp_path):
    if transformers.__version__ != "4.57.3":
        pytest.skip("Checkpoint oracle is pinned to transformers 4.57.3")
    config = transformers.Qwen2Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        attention_dropout=0.2,  # Both oracle models explicitly use eval mode.
    )
    torch.manual_seed(77)
    model = transformers.Qwen2ForCausalLM(config)
    directory = tmp_path / "checkpoint"
    model.save_pretrained(directory, safe_serialization=True, max_shard_size="1KB")
    assert (directory / "model.safetensors.index.json").is_file()
    return directory


def invoke(checkpoint, output, *, config=None):
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    for name in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        environment.pop(name, None)
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts/verify_qwen2_checkpoint.py"),
            "--hf-config",
            str(config or checkpoint / "config.json"),
            "--checkpoint",
            str(checkpoint),
            "--seq-len",
            "8",
            "--micro-batch",
            "1",
            "--seed",
            "77",
            "--profile-mode",
            "torch",
            "--output-dir",
            str(output),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_generated_sharded_checkpoint_cli_and_artifacts(checkpoint, tmp_path):
    output = tmp_path / "oracle"
    result = invoke(checkpoint, output)
    assert result.returncode == 0, result.stdout + result.stderr
    record = output / "checkpoint_oracle.rank0.json"
    data = json.loads(record.read_text())
    assert data["status"] == "passed" and data["rank"] == 0 and data["world_size"] == 1
    assert data["dtype"] == "fp32" and data["device"] == data["reference_device"] == "cpu"
    assert not data["actual_NCCL"] and data["optimizer_updates"] == 0
    assert data["global_valid_tokens"] == 7
    assert data["source"]["hf_config_sha256"]
    assert not data["source"]["official_weight_origin_verified"]
    assert data["source"]["validated_config"]["attention_dropout"] == 0.2
    logits, loss = data["comparisons"]
    assert logits["shape"] == [1, 8, 31]
    assert logits["passed"] and loss["passed"]
    assert logits["max_abs_error"] < 3e-6 and loss["max_abs_error"] < 3e-6
    assert data["ownership"]["weights_unchanged"]
    assert data["ownership"]["stable_shard_identity"]
    assert data["ownership"]["live_buffers"] == data["ownership"]["live_fullparam_leases"] == 0
    profile = json.loads((output / "profiling.rank0.json").read_text())
    assert profile["mode"] == "torch" and profile["errors"] == []
    assert (output / "profile.rank0.json").is_file()
    preserved = record.read_bytes()
    repeated = invoke(checkpoint, output)
    assert repeated.returncode == 2
    assert "choose a new --output-dir" in repeated.stderr
    assert record.read_bytes() == preserved


def test_checkpoint_config_mismatch_has_failed_record(checkpoint, tmp_path):
    config = json.loads((checkpoint / "config.json").read_text())
    config["num_hidden_layers"] = 1
    mismatched = tmp_path / "mismatched.json"
    mismatched.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "rejection"
    result = invoke(checkpoint, output, config=mismatched)
    assert result.returncode == 2
    assert "Unexpected keys" in result.stderr
    data = json.loads((output / "checkpoint_oracle.rank0.json").read_text())
    assert data["status"] == "failed" and data["optimizer_updates"] == 0
    assert data["comparisons"] == []
    assert "Checkpoint validation failed" in data["errors"][0]
