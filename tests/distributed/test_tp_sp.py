import json

import pytest
from test_launch import ROOT, launch, require_gloo_transport, torchrun_args


def worker_args(degree, case):
    return torchrun_args(degree)[:-2] + [
        str(ROOT / "tests/distributed/tp_sp_worker.py"),
        "--degree",
        str(degree),
        "--case",
        case,
    ]


@pytest.mark.distributed
def test_rank_sensitive_autograd_mappings(tmp_path):
    require_gloo_transport()
    result = launch(worker_args(2, "mappings"), tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        data = json.loads((tmp_path / f"correctness.rank{rank}.json").read_text())
        assert data["rank_sensitive_adjoint_checks"] == 4


@pytest.mark.distributed
@pytest.mark.parametrize("sp", [False, True])
def test_column_row_outputs_and_backward(tmp_path, sp):
    require_gloo_transport()
    result = launch(worker_args(2, "linear") + (["--sp"] if sp else []), tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.distributed
@pytest.mark.parametrize(
    "degree,sp,accumulation,layers",
    [
        (1, False, 1, 1),
        (1, False, 2, 1),
        (2, False, 1, 1),
        (2, False, 2, 1),
        (2, True, 1, 1),
        (2, True, 2, 1),
        (2, False, 2, 2),
        (2, True, 2, 2),
    ],
)
def test_block_all_gradients_three_updates_and_accumulation(
    tmp_path,
    degree,
    sp,
    accumulation,
    layers,
):
    require_gloo_transport()
    command = worker_args(degree, "block") + [
        "--accumulation",
        str(accumulation),
        "--layers",
        str(layers),
    ]
    if sp:
        command.append("--sp")
    result = launch(command, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(degree):
        data = json.loads((tmp_path / f"correctness.rank{rank}.json").read_text())
        assert data["parameters_compared"] == 9 * layers
        assert [step["step"] for step in data["steps"]] == [0, 1, 2]
        assert data["accumulation"] == accumulation
        assert data["sp"] == sp
        assert "Context closed" in (tmp_path / f"rank{rank}.jsonl").read_text()


@pytest.mark.distributed
@pytest.mark.parametrize("sp", [False, True])
def test_tp_sp_training_cli_three_steps(tmp_path, sp):
    require_gloo_transport()
    result = launch(
        torchrun_args(2)
        + ["--config", "configs/tp_sp.yaml", "--parallel.sp_enabled", str(sp).lower()],
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        steps = json.loads((tmp_path / f"steps.rank{rank}.json").read_text())
        assert len(steps) == 3
        assert all(step["loss"] > 0 for step in steps)


@pytest.mark.distributed
@pytest.mark.parametrize("sp", [False, True])
def test_block_multiple_local_heads_and_sequence_layout(tmp_path, sp):
    require_gloo_transport()
    command = worker_args(2, "block") + [
        "--hidden-size",
        "16",
        "--num-heads",
        "4",
        "--mlp-hidden-size",
        "24",
        "--seq-len",
        "6",
        "--accumulation",
        "2",
    ]
    if sp:
        command.append("--sp")
    result = launch(command, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
