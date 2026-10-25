"""Pure/read-only P2 contracts; never substitute metadata for numerical gates."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "experiments/fsdp_interview/cases"))
from q05_nccl_contention import (  # noqa: E402
    dependency_predecessors,
    read_events,
    signature,
    validate_schedule,
)

POLICIES = ("C0_guarded", "C1", "C2")
MODES = ("fp32", "same_start", "overflow")
TOKEN_SHA = "9bc98fb7e97dc261eabf597e550bb9f403a9ab71d8d97956bb31ffa7def9f5ea"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(data)
    return digest.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write("\n")


def assert_m0(manifest):
    from experiments.finalization_4gpu.placement import validate_manifest

    validate_manifest(manifest)
    placement = manifest["placements"]["M0"]
    if [row["topology"] for row in placement["groups"]["fsdp"]] != ["NV2", "NV2"]:
        raise ValueError("BLOCKED_ENVIRONMENT: P2 requires the current two-NV2 M0")
    return placement


def assert_workload(path):
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "model": dict(
            model_type="qwen2",
            layers=28,
            hidden_size=1536,
            vocab_size=151936,
            seq_len=512,
            num_heads=12,
            num_key_value_heads=2,
            mlp_hidden_size=8960,
            tie_word_embeddings=True,
            checkpoint_path="",
        ),
        "parallel": dict(
            fsdp_degree=2, tp_degree=2, sp_enabled=True, device="cuda", backend="nccl"
        ),
        "runtime": dict(
            reshard_after_forward=True,
            forward_prefetch=True,
            backward_prefetch=True,
            async_grad_rs=True,
            communication_policy="stream_ordered",
        ),
        "training": dict(
            dtype="fp16",
            optimizer="adamw",
            lr=0.00001,
            local_batch_size=4,
            accumulation_steps=2,
            seed=1234,
        ),
    }
    for section, values in required.items():
        for key, value in values.items():
            if config.get(section, {}).get(key) != value:
                raise ValueError(f"BLOCKED_WORKLOAD: fixed P2 {section}.{key} changed")
    return config


def event_paths(directory):
    return [Path(directory) / f"communication_events.rank{rank}.jsonl" for rank in range(4)]


def qualify(directory, output, *, source_manifest, placement_manifest, scope):
    paths = event_paths(directory)
    value = validate_schedule(paths)
    value.update(
        source_manifest_sha256=sha(source_manifest),
        placement_manifest_sha256=sha(placement_manifest),
        source_paths=[dict(path=str(path), sha256=sha(path)) for path in paths],
        architecture_scope=scope,
    )
    save(output, value)
    return value


def normalized_rows(paths):
    result = {}
    for path in paths:
        rows = [row for row in read_events([path]) if row.get("phase") in {"FWD", "BWD"}]
        if not rows:
            raise ValueError("Missing recorded FWD/BWD model operations")
        first = {}
        normalized = []
        for index, row in enumerate(rows):
            key = row["process_group_label"], tuple(row["group_members"])
            first.setdefault(key, row["collective_sequence"])
            if row["status"] != "SUBMITTED":
                raise ValueError("A recorded communication failed")
            normalized.append(
                dict(
                    signature(row),
                    group_members=row["group_members"],
                    input_bytes=row["input_bytes"],
                    output_bytes=row["output_bytes"],
                    relative_host_ordinal=index,
                    relative_pg_sequence=row["collective_sequence"] - first[key],
                )
            )
        rank = rows[0]["rank"]
        if rank in result or {row["rank"] for row in rows} != {rank}:
            raise ValueError("Duplicate/mixed recorded rank")
        result[rank] = normalized
    if set(result) != set(range(4)):
        raise ValueError("Missing four-rank communication evidence")
    return result


def audit_waits(paths, policy):
    """Check every actual new edge against the unchanged pure Q5 planner."""
    counts = {"training_ops": 0, "added_waits": 0, "cpu_synchronize": 0}
    base = "C0" if policy == "C0_guarded" else policy
    for path in paths:
        tails = {}
        for row in read_events([path]):
            if row.get("phase") not in {"FWD", "BWD"}:
                continue
            expected = dependency_predecessors(base, row["family"], tails)
            actual = [
                edge
                for edge in row["dependencies"]
                if edge.get("type", "").startswith("EXPERIMENT_")
            ]
            if any(edge["type"] != "EXPERIMENT_COMM_ORDER" for edge in actual):
                raise ValueError("Unexpected compute/global synchronization intervention")
            if [edge["producer_host_ordinal"] for edge in actual] != [
                edge["ordinal"] for edge in expected
            ]:
                raise ValueError("Actual injected waits differ from original Q5 edge planner")
            if any(
                edge["producer_host_ordinal"] >= row["host_ordinal"]
                or edge.get("cpu_synchronize") is not False
                for edge in actual
            ):
                raise ValueError("Wait points to future work or introduces CPU synchronization")
            tails[row["channel"]] = dict(ordinal=row["host_ordinal"], family=row["family"])
            counts["training_ops"] += 1
            counts["added_waits"] += len(actual)
    return dict(
        status="PASS",
        policy=policy,
        **counts,
        scope="Actual issue metadata and monotonic planner edges; not complete CUDA DAG",
    )


def validate_p0_review(path, source_manifest, placement_manifest=None):
    if not path:
        raise ValueError("BLOCKED_CORRECTNESS: root's current P0 continuation review is required")
    value = load(path)
    if (
        value.get("source_manifest_sha256") != sha(source_manifest)
        or value.get("classification") != "EXPECTED_NUMERICAL_DRIFT_SUPPORTED"
        or value.get("training_runtime_bug_confirmed") is not False
        or value.get("continuation_authorized") is not True
        or value.get("strict_independent_status") not in {"FAIL", "PASS"}
        or value.get("signed_by") != "root_agent"
        or value.get("scope") != "bounded_performance_with_retained_strict_fail"
        or value.get("input_sha256") != TOKEN_SHA
        or not value.get("limits")
        or (
            placement_manifest and value.get("placement_manifest_sha256") != sha(placement_manifest)
        )
    ):
        raise ValueError("BLOCKED_CORRECTNESS: no current-source bounded P0 review")
    evidence = value.get("evidence", [])
    if len(evidence) < 3 or any(
        not Path(row["path"]).is_file() or row.get("sha256") != sha(row["path"]) for row in evidence
    ):
        raise ValueError("P0 review lacks existing immutable raw evidence")
    return value
