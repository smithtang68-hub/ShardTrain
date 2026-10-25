"""Numerical gates for Qwen2 runtime, not a replacement for NCCL hardware gates."""

import importlib.metadata
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from test_launch import ROOT, require_gloo_transport, torchrun_args


def require_hf_reference():
    pytest.importorskip("transformers")
    version = importlib.metadata.version("transformers")
    if version != "4.57.3":
        pytest.skip(f"Qwen numerical oracle is pinned to transformers==4.57.3; found {version}")


def qwen_launch(argv, directory, timeout=240):
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
    # Keep caller's isolated HF dependency path; the legacy launch helper replaces it.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), env.get("PYTHONPATH"))))
    env["OMP_NUM_THREADS"] = "1"
    env["USE_LIBUV"] = "0"
    if sys.platform == "win32":
        env["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] = "1"
    command_line = [sys.executable, *argv, "--debug.output_dir", str(directory)]
    process = subprocess.Popen(
        command_line,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=sys.platform != "win32",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                check=False,
            )
        else:
            # Elastic workers can create their own process groups; killing only
            # torchrun's group can leave descendants holding the output pipes.
            def terminate_tree(pid):
                try:
                    children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
                except (FileNotFoundError, ProcessLookupError):
                    children = []
                for child in children:
                    terminate_tree(int(child))
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            terminate_tree(process.pid)
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(f"Qwen worker tree timed out after {timeout}s\n{stdout}\n{stderr}")
    (directory / "worker_stdout.log").write_text(stdout, encoding="utf-8")
    (directory / "worker_stderr.log").write_text(stderr, encoding="utf-8")
    return subprocess.CompletedProcess(command_line, process.returncode, stdout, stderr)


def command(f, t, sp, policy, flags, layers=4):
    argv = torchrun_args(f * t)[:-2] + ["tests/distributed/qwen2_worker.py"]
    if argv[-2] == "-m":
        del argv[-2]
    argv += [
        "--f",
        str(f),
        "--t",
        str(t),
        "--policy",
        policy,
        "--flags",
        flags,
        "--layers",
        str(layers),
    ]
    return argv + (["--sp"] if sp else [])


def check_result(directory, world_size, steps=3):
    per_rank = []
    for rank in range(world_size):
        result = json.loads((directory / f"qwen_correctness.rank{rank}.json").read_text())
        assert result["status"] == "passed" and result["steps"] == steps
        assert result["leaf_identities_stable"] and result["live_buffers"] == 0
        assert result["shared_parameter_owners"] == 1
        assert result["runtime_tied_parameter_alias"]
        assert not result["autograd_saved_full_parameters"]
        assert result["comparisons"]
        assert all(item["nan_count"] == 0 for item in result["comparisons"])
        if result["F"] > 1:
            assert all(
                len(set(row["valid_counts_by_replica"])) > 1 for row in result["step_metrics"]
            )
        rows = [
            json.loads(line)
            for line in (directory / f"runtime_events.rank{rank}.jsonl").read_text().splitlines()
        ]
        for step in range(steps):
            step_rows = [row for row in rows if row["step_id"] == step]
            materializations = [
                row for row in step_rows if row["event_type"] == "materialize_issue"
            ]
            assert len(materializations) == 2 * result["layers"] + 2
            assert not any(
                row["phase"] == "BWD" and row["unit_id"] in (0, result["layers"] + 1)
                for row in materializations
            )
            reductions = [row for row in step_rows if row["event_type"] == "unit_grads_ready"]
            assert len(reductions) == result["layers"] + 2
            assert len({row["unit_id"] for row in reductions}) == result["layers"] + 2
        # Every member of a particular PG must issue exactly the same operation sequence.
        per_rank.append(
            {
                pg: [
                    (row["pg_local_seq"], row["op_kind"])
                    for row in rows
                    if row["event_type"] == "collective_issue" and row["pg_id"] == pg
                ]
                for pg in ("world", "tp", "fsdp")
            }
        )
    f, t = result["F"], result["T"]
    assert all(item["world"] == per_rank[0]["world"] for item in per_rank)
    for replica in range(f):
        assert all(
            per_rank[replica * t + rank]["tp"] == per_rank[replica * t]["tp"] for rank in range(t)
        )
    for tp_rank in range(t):
        assert all(
            per_rank[replica * t + tp_rank]["fsdp"] == per_rank[tp_rank]["fsdp"]
            for replica in range(f)
        )


@pytest.mark.distributed
@pytest.mark.parametrize("f,t,sp", [(2, 1, False), (1, 2, True), (2, 2, True)])
@pytest.mark.parametrize("policy,flags", [("conservative", "000"), ("stream_ordered", "111")])
def test_qwen2_global_token_loss_gradients_tying_and_three_updates(
    tmp_path, f, t, sp, policy, flags
):
    require_hf_reference()
    require_gloo_transport()
    result = qwen_launch(command(f, t, sp, policy, flags), tmp_path, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    check_result(tmp_path, f * t)


@pytest.mark.distributed
@pytest.mark.parametrize("layers", [12, 28])
def test_qwen2_deep_unit_lifecycle(tmp_path, layers):
    require_hf_reference()
    require_gloo_transport()
    result = qwen_launch(command(2, 1, False, "conservative", "000", layers), tmp_path, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    check_result(tmp_path, 2)


@pytest.mark.distributed
@pytest.mark.parametrize("f,t", [(2, 1), (2, 2)])
def test_qwen2_initialization_mismatch_rejected_everywhere_without_model_mutation(tmp_path, f, t):
    require_hf_reference()
    require_gloo_transport()
    result = qwen_launch(
        command(f, t, t > 1, "conservative", "000") + ["--expect-init-mismatch"],
        tmp_path,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(f * t):
        rejection = json.loads((tmp_path / f"qwen_rejected_init.rank{rank}.json").read_text())
        assert rejection["status"] == "passed" and rejection["model_unchanged"]
        assert rejection["partial_shards"] == 0
