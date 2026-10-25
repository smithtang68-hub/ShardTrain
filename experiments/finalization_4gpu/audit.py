"""Candidate/source and M0/M1 contracts checked before formal measurements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .cli import load, save, sha


def verify_source(project, manifest_path):
    project = Path(project).resolve()
    manifest = load(manifest_path)
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Source manifest needs a nonempty relative-path dictionary")
    checked, skipped = 0, []
    for relative, record in files.items():
        if Path(relative).suffix.lower() not in {
            ".py",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".cu",
            ".c",
            ".cpp",
            ".h",
            ".sh",
        }:
            skipped.append(relative)
            continue
        path = (project / relative).resolve()
        if not path.is_relative_to(project) or not path.is_file() or sha(path) != record["sha256"]:
            raise ValueError(f"Source snapshot mismatch: {relative}")
        checked += 1
    return {
        "status": "PASS",
        "execution_files_checked": checked,
        "non_execution_documents_not_requalified": skipped,
        "manifest_sha256": sha(manifest_path),
    }


def audit_capacity(root):
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "experiments/fsdp_interview/cases")
    )
    from q05_nccl_contention import read_events, validate_schedule
    from q06_comm_volume import first_consumer_phases

    root = Path(root)
    results, states, inputs, sources = {}, {}, {}, set()
    for mapping, case in (("M0", "S0"), ("M1", "S1")):
        directory = root / "capacity" / f"capacity_{case}_r0"
        paths = [directory / f"communication_events.rank{rank}.jsonl" for rank in range(4)]
        contract = validate_schedule(paths)
        per_rank = {}
        for rank in range(4):
            consumers = first_consumer_phases(
                read_events([directory / f"runtime_events.rank{rank}.jsonl"])
            )
            rows = [row for row in read_events([paths[rank]]) if row.get("phase") in {"FWD", "BWD"}]
            first_sequences = {}
            normalized = []
            for ordinal, row in enumerate(rows):
                key = (row["process_group_label"], tuple(row["group_members"]))
                first_sequences.setdefault(key, row["collective_sequence"])
                normalized.append(
                    {
                        **{
                            field: row[field]
                            for field in (
                                "process_group_label",
                                "group_members",
                                "collective_type",
                                "unit_id",
                                "phase",
                                "channel",
                                "input_shape",
                                "output_shapes",
                                "input_bytes",
                                "output_bytes",
                                "dtype",
                                "group_role",
                            )
                        },
                        "relative_ordinal": ordinal,
                        "relative_pg_sequence": row["collective_sequence"] - first_sequences[key],
                        "first_copy_out_phase": consumers.get((rank, row["runtime_op_id"])),
                    }
                )
                if (
                    row["collective_type"] == "fsdp_ag"
                    and (rank, row["runtime_op_id"]) not in consumers
                ):
                    raise ValueError("A parameter AG lacks exact first-copy-out consumer evidence")
            per_rank[str(rank)] = {
                "operations": len(normalized),
                "normalized_sha256": hashlib.sha256(
                    json.dumps(normalized, sort_keys=True).encode()
                ).hexdigest(),
            }
            if mapping == "M1" and per_rank[str(rank)] != results["M0"]["per_rank"][str(rank)]:
                raise ValueError(
                    "Mapping changed ordered logical collective/payload/first-consume contract"
                )
        results[mapping] = {"contract": contract, "per_rank": per_rank}
    for case, engine in (
        ("S0", "shardtrain"),
        ("S1", "shardtrain"),
        ("N0", "native"),
        ("N1", "native"),
    ):
        directory = root / "capacity" / f"capacity_{case}_r0"
        for rank in range(4):
            startup = load(directory / f"placement.rank{rank}.json")
            benchmark = load(directory / f"benchmark.rank{rank}.json")
            state = startup["initial_local_state"]["sha256"]
            key = (engine, rank)
            if key in states and states[key] != state:
                raise ValueError("Mapping changed same-logical-rank initial shard values")
            states[key] = state
            batch = [
                (
                    row["optimizer_step"],
                    row["input_block_start"],
                    row["input_batch_sha256"],
                    row["valid_tokens"],
                )
                for row in benchmark["warmup_steps"] + benchmark["steps"]
            ]
            if rank in inputs and inputs[rank] != batch:
                raise ValueError("Mapping or engine changed logical-rank input assignment")
            inputs[rank] = batch
            sources.add(benchmark["source_manifest_sha256"])
    if len(sources) != 1:
        raise ValueError("Capacity gates used different snapshots")
    result = {
        "status": "PASS",
        "source_manifest_sha256": next(iter(sources)),
        "collective_comparison": results,
        "same_engine_initial_shards": "PASS",
        "logical_input_assignment": "PASS",
        "scope": "Relative PG sequences and exact producer-ticket first-copy-out phases; "
        "no numerical proof inferred from metadata",
    }
    save(root / "capacity_contract.json", result)
    return result
