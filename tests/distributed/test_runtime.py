import itertools
import json

import pytest
from test_launch import launch, require_gloo_transport, torchrun_args


def runtime_command(
    f, t, sp, reshard, flags=(False, False, False), layers=4, hidden=8, policy="conservative"
):
    command = torchrun_args(f * t)[:-2] + ["tests/distributed/runtime_worker.py"]
    # Remove torchrun's module marker: this worker is an explicit script path.
    if command[-2] == "-m":
        del command[-2]
    return command + [
        "--input-grad",
        "--model.layers",
        str(layers),
        "--model.hidden_size",
        str(hidden),
        "--model.num_heads",
        "4",
        "--model.mlp_hidden_size",
        "20",
        "--model.seq_len",
        "4",
        "--parallel.fsdp_degree",
        str(f),
        "--parallel.tp_degree",
        str(t),
        "--parallel.sp_enabled",
        str(sp).lower(),
        "--runtime.reshard_after_forward",
        str(reshard).lower(),
        "--runtime.forward_prefetch",
        str(flags[0]).lower(),
        "--runtime.backward_prefetch",
        str(flags[1]).lower(),
        "--runtime.async_grad_rs",
        str(flags[2]).lower(),
        "--runtime.communication_policy",
        policy,
        "--training.lr",
        "0.01",
        "--debug.distributed_timeout_seconds",
        "40",
    ]


def check_run(directory, world_size, reshard, layers=4):
    for rank in range(world_size):
        result = json.loads((directory / f"correctness.rank{rank}.json").read_text())
        assert result["status"] == "passed" and result["steps"] == 3
        assert result["leaf_identities_stable"] and result["live_buffers"] == 0
        assert result["comparisons"]
        assert all(r["nan_count"] == 0 for r in result["comparisons"])
        events = [
            json.loads(line)
            for line in (directory / f"runtime_events.rank{rank}.jsonl").read_text().splitlines()
        ]
        for step in range(3):
            rows = [r for r in events if r["step_id"] == step]
            ags = [r for r in rows if r["event_type"] == "materialize_issue"]
            assert len(ags) == layers + ((layers - 1) if reshard else 0)
            assert not any(r["unit_id"] == layers - 1 and r["phase"] == "BWD" for r in ags)
            ready = [r for r in rows if r["event_type"] == "unit_grads_ready"]
            assert len(ready) == layers and len({r["unit_id"] for r in ready}) == layers
            if result["SP"] and result["T"] > 1:
                issues = [row for row in rows if row["event_type"] == "collective_issue"]
                for phase, ags_count in (("FWD", 2), ("BWD", 4)):
                    ags = [
                        row for row in issues if row["kind"] == "sp_ag" and row["phase"] == phase
                    ]
                    assert len(ags) == layers * ags_count
                    rss = [
                        row
                        for row in issues
                        if row["kind"] in {"sp_rs", "sp_rs_math_allreduce"}
                        and row["phase"] == phase
                    ]
                    assert len(rss) == layers * 2
                for role in ("qkv", "gate_up", "qkv_wgrad", "gate_up_wgrad"):
                    assert (
                        sum(row["kind"] == "sp_ag" and row["group_role"] == role for row in issues)
                        == layers
                    )
                assert all(
                    row["group_members"] and row["input_buffer_ids"] and row["output_buffer_ids"]
                    for row in issues
                    if row["kind"] == "sp_ag"
                )
        assert "Context closed" in (directory / f"rank{rank}.jsonl").read_text()


@pytest.mark.distributed
@pytest.mark.parametrize("policy", ["conservative", "stream_ordered"])
@pytest.mark.parametrize("input_grad", [False, True])
def test_shared_single_block_sublayers_and_gradients(tmp_path, policy, input_grad):
    require_gloo_transport()
    command = runtime_command(1, 2, True, True, layers=1, hidden=16, policy=policy)
    command += ["--model.seq_len", "8"]
    if not input_grad:
        command.remove("--input-grad")
    result = launch(command, tmp_path, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    check_run(tmp_path, 2, True, layers=1)


@pytest.mark.distributed
@pytest.mark.parametrize(
    "f,t,sp,reshard,hidden",
    [
        (2, 1, False, True, 8),
        (1, 2, True, True, 8),
        (2, 2, True, True, 8),
        (2, 2, True, False, 8),
        (2, 2, False, True, 8),
        (4, 2, True, True, 12),
        (2, 4, True, True, 8),
        (8, 1, False, True, 12),
    ],
)
def test_four_block_sync_three_steps_all_outputs_gradients_and_updates(
    tmp_path, f, t, sp, reshard, hidden
):
    require_gloo_transport()
    result = launch(runtime_command(f, t, sp, reshard, hidden=hidden), tmp_path, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    check_run(tmp_path, f * t, reshard)


@pytest.mark.distributed
@pytest.mark.parametrize("flags", list(itertools.product((False, True), repeat=3)))
def test_all_independent_async_switch_combinations_against_dense(tmp_path, flags):
    require_gloo_transport()
    result = launch(runtime_command(2, 2, True, True, flags), tmp_path, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    check_run(tmp_path, 4, True)


@pytest.mark.distributed
@pytest.mark.parametrize(
    "f,t,sp,reshard,flags",
    [(2, 2, True, True, flags) for flags in itertools.product((False, True), repeat=3)]
    + [
        (2, 2, True, False, (True, True, True)),
        (2, 2, False, True, (True, True, True)),
        (4, 2, True, True, (True, True, True)),
    ],
)
def test_stream_ordered_prefetch_boundaries_reference_and_pg_sequence(
    tmp_path, f, t, sp, reshard, flags
):
    from scripts.run_matrix import compare_sequences

    require_gloo_transport()
    result = launch(
        runtime_command(f, t, sp, reshard, flags, policy="stream_ordered"), tmp_path, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    check_run(tmp_path, f * t, reshard)
    compare_sequences(tmp_path, f, t)
    for rank in range(f * t):
        events = [
            json.loads(line)
            for line in (tmp_path / f"runtime_events.rank{rank}.jsonl").read_text().splitlines()
        ]
        for step in range(3):
            rows = [row for row in events if row["step_id"] == step]
            boundaries = [row for row in rows if row["event_type"] == "prefetch_boundary"]
            assert len(boundaries) == 3 * (int(flags[0]) + int(flags[1])) if sp else not boundaries
            for boundary in boundaries:
                phase = boundary["phase"]
                suffix = "attention.v_proj.weight" if phase == "FWD" else "mlp.down_proj.weight"
                assert boundary["param_id"].endswith(suffix)
                index = rows.index(boundary)
                previous = [
                    row
                    for row in rows[:index]
                    if row["event_type"] == "collective_issue"
                    and row["unit_id"] == boundary["unit_id"]
                ]
                assert previous[-1]["kind"] == "sp_ag"
                target = rows[index + 1]
                assert target["unit_id"] == boundary["target_unit"]
                assert target["event_type"] in {"buffer_allocate", "prefetch_skipped_retained"}


def test_single_process_runtime_cli_and_training_metadata(tmp_path):
    result = launch(
        [
            "-m",
            "sharded_runtime.train",
            "--sharded-runtime",
            "--config",
            "configs/runtime_cpu.yaml",
            "--parallel.fsdp_degree",
            "1",
            "--parallel.tp_degree",
            "1",
            "--parallel.sp_enabled",
            "false",
            "--model.hidden_size",
            "8",
            "--model.num_heads",
            "2",
            "--model.mlp_hidden_size",
            "12",
            "--model.seq_len",
            "4",
            "--runtime.forward_prefetch",
            "true",
            "--runtime.backward_prefetch",
            "true",
            "--runtime.async_grad_rs",
            "true",
        ],
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(json.loads((tmp_path / "steps.rank0.json").read_text())) == 3
    events = (tmp_path / "runtime_events.rank0.jsonl").read_text()
    assert "optimizer_ready" in events and "step_end" in events
