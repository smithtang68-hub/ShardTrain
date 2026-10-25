"""Audit completed runtime rank logs and freeze reproducible shared-SP repair evidence."""

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from run_matrix import compare_sequences


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def directories(roots):
    for path in roots:
        for root, dirs, files in os.walk(path, followlinks=False):
            dirs[:] = [name for name in dirs if not name.endswith("current")]
            if "correctness.rank0.json" in files:
                yield Path(root)


def audit(directory):
    first = json.loads((directory / "correctness.rank0.json").read_text())
    if "F" not in first:
        return None  # Historical M1 acceptance uses a different result schema.
    f, t = first["F"], first["T"]
    compare_sequences(directory, f, t)
    ranks = []
    for rank in range(f * t):
        result = json.loads((directory / f"correctness.rank{rank}.json").read_text())
        assert result["status"] == "passed" and result["steps"] == 3
        assert result["leaf_identities_stable"] and result["live_buffers"] == 0
        rows = [
            json.loads(line)
            for line in (directory / f"runtime_events.rank{rank}.jsonl").read_text().splitlines()
        ]
        counts = []
        for step in range(result["steps"]):
            events = [row for row in rows if row["step_id"] == step]
            layers = sum(row["event_type"] == "unit_forward" for row in events)
            issues = [row for row in events if row["event_type"] == "collective_issue"]
            count = Counter((row["phase"], row["kind"], row["group_role"]) for row in issues)
            if result["SP"] and t > 1:
                for role in ("qkv", "gate_up"):
                    assert count["FWD", "sp_ag", role] == layers
                    assert count["BWD", "sp_ag", role + "_wgrad"] == layers
                for phase, ags in (("FWD", 2), ("BWD", 4)):
                    assert (
                        sum(
                            value
                            for (p, kind, role), value in count.items()
                            if p == phase and kind == "sp_ag"
                        )
                        == layers * ags
                    )
                    assert (
                        sum(
                            value
                            for (p, kind, role), value in count.items()
                            if p == phase and kind in {"sp_rs", "sp_rs_math_allreduce"}
                        )
                        == layers * 2
                    )
                for row in issues:
                    if row["kind"] == "sp_ag":
                        assert row["input_buffer_ids"] and row["output_buffer_ids"]
                        assert row["group_members"] and row["pg_local_seq"] is not None
                saved = [
                    row for row in events if row["event_type"] == "projection_group_saved_local"
                ]
                assert len(saved) == layers * 2
                assert all(row["storage_ptr"] != row["full_storage_ptr"] for row in saved)
            # Copy-out enters the FIFO only at its own Unit's entry, after the previous
            # Unit finishes its forward submission; retained U3 never gets a BWD AG.
            materializations = [row for row in events if row["event_type"] == "materialize_issue"]
            assert len(materializations) == layers + (layers - 1 if result["reshard"] else 0)
            assert not any(
                row["unit_id"] == layers - 1 and row["phase"] == "BWD" for row in materializations
            )
            for index, row in enumerate(events):
                if (
                    row["event_type"] == "copy_out_start"
                    and row["phase"] == "FWD"
                    and row["unit_id"]
                ):
                    assert any(
                        previous["unit_id"] == row["unit_id"] - 1
                        and previous["event_type"] in {"RESHARDED", "FWD_RETAINED"}
                        for previous in events[:index]
                    )
                if row["dependency_type"] == "DATA_DEP" and row.get("producer_kind") == "fsdp_ag":
                    assert row.get("consumer") not in {"sp_rs", "sp_rs_math_allreduce"}
            counts.append(
                dict(
                    step=step,
                    layers=layers,
                    collectives=[
                        dict(phase=phase, kind=kind, group_role=role, count=value)
                        for (phase, kind, role), value in sorted(
                            count.items(), key=lambda pair: str(pair[0])
                        )
                    ],
                )
            )
        comparisons = result["comparisons"]
        ranks.append(
            dict(
                rank=rank,
                comparisons=len(comparisons),
                max_abs_error=max(row["max_abs_error"] for row in comparisons),
                nan_count=sum(row["nan_count"] for row in comparisons),
                counts=counts,
                result_file=(directory / f"correctness.rank{rank}.json").as_posix(),
                trace_file=(directory / f"runtime_events.rank{rank}.jsonl").as_posix(),
            )
        )
    return dict(
        directory=directory.as_posix(),
        F=f,
        T=t,
        SP=first["SP"],
        communication_policy=first["communication_policy"],
        input_grad=first["input_grad"],
        reshard=first["reshard"],
        flags=[first[key] for key in ("forward_prefetch", "backward_prefetch", "async_grad_rs")],
        ranks=ranks,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    files = {
        path: digest(Path(path))
        for path in paths
        if (
            path.startswith(("src/", "tests/", "configs/", "scripts/", ".github/"))
            or path in {"pyproject.toml", ".gitignore"}
        )
    }
    for path, sha in files.items():
        blob = subprocess.check_output(["git", "show", f"{source_commit}:{path}"])
        assert hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() == sha, path
    experiments = [
        result
        for directory in directories(args.directories)
        if (result := audit(directory)) is not None
    ]
    assert experiments, "No completed runtime evidence"
    all_ranks = [rank for result in experiments for rank in result["ranks"]]
    output = dict(
        source_commit=source_commit,
        actual_NCCL=False,
        manifest=dict(algorithm="SHA256 with CRLF normalized to LF", files=files),
        runtime_experiments=len(experiments),
        rank_runs=len(all_ranks),
        tensor_comparisons=sum(rank["comparisons"] for rank in all_ranks),
        max_abs_error=max(rank["max_abs_error"] for rank in all_ranks),
        nan_count=sum(rank["nan_count"] for rank in all_ranks),
        experiments=experiments,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in output.items() if key not in {"experiments", "manifest"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
