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


def assert_prefetch_boundaries(rows, step, sp, reshard, flags, layers=4):
    def indices(event_type, **fields):
        return [
            index
            for index, row in enumerate(rows)
            if row["event_type"] == event_type
            and all(row.get(key) == value for key, value in fields.items())
        ]

    def one(event_type, **fields):
        found = indices(event_type, **fields)
        assert len(found) == 1, (event_type, fields, found)
        return found[0]

    def generation(unit):
        phase = "BWD" if reshard and unit < layers - 1 else "FWD"
        return rows[one("materialize_issue", unit_id=unit, phase=phase)][
            "materialization_generation"
        ]

    boundaries = indices("prefetch_boundary")
    forward = indices("prefetch_boundary", phase="FWD")
    backward = indices("prefetch_boundary", phase="BWD")
    assert len(forward) == (layers - 1) * int(sp and flags[0])
    assert len(backward) == (layers - 1) * int(flags[1])
    assert len(boundaries) == len(forward) + len(backward)
    assert [
        (rows[index]["unit_id"], rows[index]["target_unit"], rows[index]["param_id"])
        for index in forward
    ] == (
        [(unit, unit + 1, f"{unit}.attention.v_proj.weight") for unit in range(layers - 1)]
        if sp and flags[0]
        else []
    )

    # FSDP-only prefetch starts at unit entry, without an SP-input boundary event.
    if not sp and flags[0]:
        for unit in range(layers - 1):
            assert (
                one("unit_forward", unit_id=unit)
                < one("materialize_issue", unit_id=unit + 1, phase="FWD")
                < one("unit_forward", unit_id=unit + 1)
            )

    prefetch_end = {}
    for index in boundaries:
        boundary = rows[index]
        phase, target = boundary["phase"], boundary["target_unit"]
        following = rows[index + 1]
        assert (following["unit_id"], following["phase"]) == (target, phase)
        if phase == "BWD" and not reshard:
            assert following["event_type"] == "prefetch_skipped_retained"
            prefetch_end[index] = one("prefetch_skipped_retained", unit_id=target, phase=phase)
            assert not indices("materialize_issue", unit_id=target, phase=phase)
            assert not indices("collective_issue", unit_id=target, phase=phase, kind="fsdp_ag")
        else:
            assert following["event_type"] == "buffer_allocate"
            assert following["owner"] == "fsdp_ag"
            ag = one("collective_issue", unit_id=target, phase=phase, kind="fsdp_ag")
            issued = one("materialize_issue", unit_id=target, phase=phase)
            assert index < ag < issued
            assert not indices("prefetch_skipped_retained", unit_id=target, phase=phase)
            prefetch_end[index] = issued
        entry = "unit_forward" if phase == "FWD" else "unit_pre_backward"
        assert prefetch_end[index] < one(entry, unit_id=target)

        if phase == "FWD" or (sp and not flags[2]):
            previous = [
                row
                for row in rows[:index]
                if row["event_type"] == "collective_issue"
                and row["unit_id"] == boundary["unit_id"]
                and row["phase"] == phase
            ]
            assert previous[-1]["kind"] == "sp_ag"
            assert previous[-1]["group_role"] == ("qkv" if phase == "FWD" else "down_proj_row_dx")

    dispatch = sp and flags[1] and flags[2]
    tokens = indices("backward_prefetch_token")
    if dispatch:
        assert [
            (
                rows[index]["unit_id"],
                rows[index]["current_unit"],
                rows[index]["target_unit"],
                rows[index]["gradient_owner"],
                rows[index]["scheduling_boundary"],
            )
            for index in backward
        ] == [
            (layers - 1, layers - 1, layers - 2, None, "backward_gradient_dispatch_seed"),
            *[
                (owner, owner - 1, owner - 2, owner, "previous_gradient_dispatch")
                for owner in range(layers - 1, 1, -1)
            ],
        ]
        assert len(tokens) == layers - 1
        for index in backward:
            boundary = rows[index]
            current, target = boundary["current_unit"], boundary["target_unit"]
            assert boundary["token_step"] == step
            token_index = one("backward_prefetch_token", current_unit=current)
            token = rows[token_index]
            for key in (
                "unit_id",
                "current_unit",
                "target_unit",
                "gradient_owner",
                "scheduling_boundary",
            ):
                assert token[key] == boundary[key]
            assert token["current_generation"] == generation(current)
            assert token["target_generation"] == generation(target)
            assert token["parameter_version"] == step
            assert token["status"] == ("issued" if reshard else "retained")
            assert prefetch_end[index] < token_index
            if boundary["gradient_owner"] is None:
                assert one("unit_pre_backward", unit_id=current) < index
                current_ags = indices(
                    "collective_issue", unit_id=current, phase="BWD", kind="sp_ag"
                )
                assert token_index < min(current_ags)
            else:
                owner = boundary["gradient_owner"]
                assert owner == current + 1 == target + 2
                issued = indices("gradient_bucket_issued", unit_id=owner)
                consumed = indices("gradient_bucket_consumed", unit_id=owner)
                assert [rows[i]["bucket"] for i in issued] == ["norm", "main"]
                assert [rows[i]["bucket"] for i in consumed] == ["norm", "main"]
                assert max(issued) < index < token_index < min(consumed)
                assert token_index < one("unit_pre_backward", unit_id=current)
    else:
        assert not tokens
        assert [(rows[i]["unit_id"], rows[i]["target_unit"]) for i in backward] == (
            [(unit, unit - 1) for unit in range(layers - 1, 0, -1)] if flags[1] else []
        )
        for index in backward:
            boundary = rows[index]
            unit = boundary["unit_id"]
            assert one("unit_pre_backward", unit_id=unit) < index
            if sp:
                assert boundary["param_id"] == f"{unit}.mlp.down_proj.weight"
            else:
                assert boundary["scheduling_boundary"] == "backward_unit_entry"


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
            assert_prefetch_boundaries(rows, step, sp, reshard, flags)


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
