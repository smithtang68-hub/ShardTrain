from dataclasses import replace

import pytest

from sharded_runtime.config import Config, ConfigError, config_from_dict, load_config
from sharded_runtime.distributed.context import WorkerEnvironment


def test_default_reference_config():
    cfg = load_config("configs/correctness.yaml")
    cfg.validate(1)
    assert cfg.training.steps == 3
    assert not cfg.runtime.reshard_after_forward


def test_dotted_overrides_and_roundtrip(tmp_path):
    cfg = load_config(overrides=["--training.steps", "5", "--training.lr", "0.01"])
    assert cfg.training.steps == 5
    assert cfg.training.lr == 0.01
    assert config_from_dict(cfg.to_dict()) == cfg
    path = tmp_path / "bad.yaml"
    path.write_text("model: [broken", encoding="utf-8")
    with pytest.raises(ConfigError, match="Cannot read configuration"):
        load_config(path)


@pytest.mark.parametrize(
    "data",
    [
        {"unknown": {}},
        {"model": {"hidden_sze": 8}},
        {"training": {"steps": True}},
        {"parallel": {"sp_enabled": "false"}},
        {"model": []},
        [],
    ],
)
def test_schema_rejects_unknown_keys_and_wrong_types(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


@pytest.mark.parametrize(
    "overrides",
    [
        ["--training.steps"],
        ["--training.typo", "2"],
        ["steps", "2"],
        ["--training.steps", "2.5"],
        ["--parallel.sp_enabled", "yesplease"],
    ],
)
def test_bad_overrides(overrides):
    with pytest.raises(ConfigError):
        load_config(overrides=overrides)


@pytest.mark.parametrize(
    ("section", "values", "message"),
    [
        ("parallel", {"fsdp_degree": 0}, "positive integer"),
        ("parallel", {"tp_degree": -1}, "positive integer"),
        ("model", {"hidden_size": 7}, "num_heads"),
        ("model", {"num_heads": 3}, "num_heads"),
        ("model", {"dropout": 0.1}, "dropout"),
        ("training", {"lr": float("nan")}, "finite"),
        ("training", {"seed": -1}, "seed"),
        ("training", {"optimizer": "adamw"}, "SGD"),
        ("training", {"accumulation_steps": 0}, "positive integer"),
        ("parallel", {"sp_enabled": True}, "SP requires"),
        ("parallel", {"backend": "nccl"}, "CPU requires"),
        ("parallel", {"device": "cuda:0"}, "device"),
        ("runtime", {"reshard_after_forward": True}, "M3"),
        ("runtime", {"materialization": "async"}, "not implemented"),
        ("debug", {"output_dir": " "}, "empty"),
    ],
)
def test_invalid_semantics(section, values, message):
    cfg = config_from_dict({section: values})
    with pytest.raises(ConfigError, match=message):
        cfg.validate()


def test_world_size_and_milestone_rejection():
    cfg = config_from_dict({"parallel": {"fsdp_degree": 2}})
    with pytest.raises(ConfigError, match="WORLD_SIZE"):
        cfg.validate(1, mode="smoke")
    cfg.validate(2, mode="smoke")
    with pytest.raises(ConfigError, match="M0 reference training"):
        cfg.validate(2)


def test_sp_and_head_divisibility():
    cfg = config_from_dict(
        {
            "parallel": {"tp_degree": 2, "sp_enabled": True},
            "model": {"seq_len": 3},
        }
    )
    with pytest.raises(ConfigError, match="seq_len"):
        cfg.validate(2, mode="smoke")
    cfg = replace(cfg, model=replace(cfg.model, seq_len=4, num_heads=1))
    with pytest.raises(ConfigError, match="num_heads.*TP|num_heads.*tp"):
        cfg.validate(2, mode="smoke")


def test_worker_environment_distinguishes_global_local_rank():
    worker = WorkerEnvironment.read({"RANK": "5", "LOCAL_RANK": "1", "WORLD_SIZE": "8"})
    assert (worker.rank, worker.local_rank, worker.world_size, worker.launched) == (5, 1, 8, True)
    assert WorkerEnvironment.read({}) == WorkerEnvironment()


@pytest.mark.parametrize(
    "env",
    [
        {"RANK": "0"},
        {"RANK": "a", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
        {"RANK": "2", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
        {"RANK": "0", "LOCAL_RANK": "-1", "WORLD_SIZE": "2"},
    ],
)
def test_invalid_worker_environment(env):
    with pytest.raises(ConfigError):
        WorkerEnvironment.read(env)


def test_bad_execution_mode():
    with pytest.raises(ConfigError, match="execution mode"):
        Config().validate(mode="magic")
