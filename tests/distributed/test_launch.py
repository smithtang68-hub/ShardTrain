import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]


def require_gloo_transport():
    if not dist.is_gloo_available():
        pytest.fail("The CPU multi-process gate requires a PyTorch build with Gloo")
    if sys.platform == "win32":
        try:
            dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")
        except RuntimeError:
            pytest.skip(
                "Windows wheel has no Gloo transport; full multi-process gate runs on Linux"
            )


def launch(command, directory, timeout=90):
    env = os.environ.copy()
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "LOCAL_WORLD_SIZE",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["OMP_NUM_THREADS"] = "1"
    env["USE_LIBUV"] = "0"
    if sys.platform == "win32":
        # FileStore rendezvous must also disable the launcher's shared TCPStore.
        env["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] = "1"
    argv = [sys.executable, *command, "--debug.output_dir", str(directory)]
    process_options = {}
    if sys.platform == "win32":
        # Torchrun cleanup broadcasts CTRL_C_EVENT within its console. Give the
        # worker tree a hidden console so a failing worker cannot interrupt pytest.
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        process_options.update(creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
    else:
        process_options["start_new_session"] = True
    process = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **process_options,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the entire launched test tree, including torchrun children on timeout.
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(f"Worker tree timed out after {timeout}s\n{stdout}\n{stderr}")
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def torchrun_args(nproc):
    # Windows torch 2.9.1 launcher TCPStore ignores USE_LIBUV; use its built-in FileStore.
    rendezvous = (
        ["--rdzv_conf=store_type=file", "--local_addr=127.0.0.1"]
        if sys.platform == "win32"
        else ["--rdzv_endpoint=127.0.0.1:0"]
    )
    return [
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        "--rdzv_backend=c10d",
        *rendezvous,
        "--max_restarts=0",
        "-m",
        "sharded_runtime.train",
    ]


@pytest.mark.distributed
@pytest.mark.parametrize("fsdp,tp", [(2, 1), (1, 2), (2, 2)])
def test_real_gloo_group_collectives_and_shutdown(tmp_path, fsdp, tp):
    require_gloo_transport()
    result = launch(
        torchrun_args(fsdp * tp)
        + [
            "--smoke-groups",
            "--parallel.fsdp_degree",
            str(fsdp),
            "--parallel.tp_degree",
            str(tp),
            "--debug.distributed_timeout_seconds",
            "20",
        ],
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(fsdp * tp):
        data = json.loads((tmp_path / f"smoke.rank{rank}.json").read_text())
        assert len(data["tp"]["ranks"]) == tp
        assert len(data["fsdp"]["ranks"]) == fsdp
        assert data["tp_grad"] == data["tp"]
        assert data["fsdp_grad"] == data["fsdp"]
        assert set(data["tp"]["ranks"]) & set(data["fsdp"]["ranks"]) == {rank}
        log = (tmp_path / f"rank{rank}.jsonl").read_text()
        assert "Context closed" in log
        assert "Collective smoke passed" in log
        metadata = json.loads((tmp_path / f"environment.rank{rank}.json").read_text())
        assert metadata["rank"] == rank
        assert metadata["world_size"] == fsdp * tp


@pytest.mark.distributed
def test_bad_degrees_fail_without_hanging_and_log_all_ranks(tmp_path):
    result = launch(torchrun_args(2) + ["--smoke-groups", "--parallel.fsdp_degree", "3"], tmp_path)
    assert result.returncode != 0
    for rank in range(2):
        log = (tmp_path / f"rank{rank}.jsonl").read_text()
        assert "WORLD_SIZE=2" in log
        assert "Worker bootstrap failed" in log


@pytest.mark.parametrize("use_torchrun", [False, True])
def test_reference_cli_three_steps_and_metadata(tmp_path, use_torchrun):
    if use_torchrun:
        require_gloo_transport()
    command = torchrun_args(1) if use_torchrun else ["-m", "sharded_runtime.train"]
    result = launch(command + ["--config", "configs/correctness.yaml"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    steps = json.loads((tmp_path / "steps.rank0.json").read_text())
    assert [step["step"] for step in steps] == [0, 1, 2]
    assert all(step["loss"] > 0 and step["gradient_l2"] > 0 for step in steps)
    assert (tmp_path / "resolved_config.rank0.yaml").is_file()
    assert (tmp_path / "environment.rank0.json").is_file()


def test_unknown_cli_field_is_clear_error(tmp_path):
    result = launch(["-m", "sharded_runtime.train", "--training.typo", "3"], tmp_path)
    assert result.returncode != 0
    assert "Unknown config field" in result.stderr


def test_invalid_log_level_is_reported_as_config_error(tmp_path):
    result = launch(["-m", "sharded_runtime.train", "--debug.log_level", "verbose"], tmp_path)
    assert result.returncode != 0
    assert "debug.log_level must be" in result.stderr
