"""Current-run small-oracle, observer, payload and first-consumer gate audit."""

from __future__ import annotations

from pathlib import Path

from .contracts import (
    MODES,
    POLICIES,
    audit_waits,
    event_paths,
    load,
    normalized_rows,
    read_events,
    save,
    sha,
    stable,
    validate_schedule,
)


def audit_gates(root, source_manifest, placement_manifest, *, policies=POLICIES):
    root = Path(root)
    comparisons, records = [], []
    for mode in MODES:
        baseline = {}
        reference = None
        for policy in policies:
            directory = root / "p2" / "correctness" / policy / mode
            paths = event_paths(directory / "observer")
            validate_schedule(paths)
            normalized = normalized_rows(paths)
            if reference is None:
                reference = normalized
            if normalized != reference:
                raise ValueError("Policy changed the tiny oracle PG/shape/payload/sequence")
            wait = audit_waits(paths, policy)
            for rank in range(4):
                file = directory / (
                    f"mapping_fp32.rank{rank}.json"
                    if mode == "fp32"
                    else f"amp_{mode}.rank{rank}.json"
                )
                value = load(file)
                observer = load(directory / f"q5_observer.rank{rank}.json")
                if (
                    value["status"] != "PASS"
                    or not value.get("lifecycle")
                    or not value.get("training_collective_contract")
                    or value.get("source_manifest_sha256") != sha(source_manifest)
                    or value.get("placement_manifest_sha256") != sha(placement_manifest)
                    or observer["instrumentation"]["status"] != "RECORDED"
                    or not observer["instrumentation"]["record_events"]
                    or observer["instrumentation"]["gpu_timing"]
                ):
                    raise ValueError("Missing strict numerical/lifecycle/observer/source gate")
                contract = {
                    field: value[field]
                    for field in (
                        "initial_shard_fingerprints",
                        "input_fingerprints",
                        "training_collective_contract",
                    )
                }
                if rank in baseline and contract != baseline[rank]:
                    raise ValueError(
                        "Tiny policy changed initialization/input/first-consumer contract"
                    )
                baseline[rank] = contract
                comparisons.append(
                    dict(
                        mode=mode,
                        policy=policy,
                        rank=rank,
                        status="PASS",
                        raw_path=str(file),
                        raw_sha256=sha(file),
                        operations=len(normalized[rank]),
                        contract_sha256=stable(contract),
                    )
                )
            records.append(wait)
    result = dict(
        status="PASS",
        policies=list(policies),
        comparisons=comparisons,
        wait_edges=records,
        source_manifest_sha256=sha(source_manifest),
        placement_manifest_sha256=sha(placement_manifest),
        independent_fp16_scope="Not rerun here; P0 strict FAIL and classification remain separate",
        numerical_scope="4L FP32 independent SGD; FP16 same-start AdamW/replay and overflow; "
        "unchanged existing strict thresholds; not full 1.5B dense equivalence",
    )
    save(root / "p2" / "contracts" / "small_oracle_gate.json", result)
    return result


def audit_capacity(root, source_manifest, placement_manifest, *, policies=POLICIES):
    from q06_comm_volume import first_consumer_phases

    root = Path(root)
    baseline = None
    records = []
    for policy in policies:
        directory = root / "p2" / "capacity" / policy
        paths = event_paths(directory)
        validate_schedule(paths)
        normalized = normalized_rows(paths)
        wait = audit_waits(paths, policy)
        actual = {}
        for rank in range(4):
            consumers = first_consumer_phases(
                read_events([directory / f"runtime_events.rank{rank}.jsonl"])
            )
            issues = [
                row for row in read_events([paths[rank]]) if row.get("phase") in {"FWD", "BWD"}
            ]
            for row, issue in zip(normalized[rank], issues, strict=True):
                row["first_copy_out_phase"] = consumers.get((rank, issue["runtime_op_id"]))
                if issue["collective_type"] == "fsdp_ag" and row["first_copy_out_phase"] is None:
                    raise ValueError("A parameter AG has no exact first-copy-out evidence")
            placement = load(directory / f"placement.rank{rank}.json")
            benchmark = load(directory / f"benchmark.rank{rank}.json")
            if (
                placement["status"] != "PASS"
                or benchmark["status"] != "PASS"
                or benchmark["source_manifest_sha256"] != sha(source_manifest)
                or placement["placement_manifest_sha256"] != sha(placement_manifest)
                or benchmark.get("steady_steps_completed") != 2
                or benchmark.get("event_recorder") is not True
            ):
                raise ValueError("Full-model finite/capacity/initialization evidence missing")
            actual[rank] = dict(
                operations=normalized[rank],
                actual_uuid=placement["actual_uuid"],
                actual_pci_bus_id=placement["actual_pci_bus_id"],
                ordered_pg_members=placement["actual_pg_members"],
                initial_local_state=placement["initial_local_state"],
                inputs=[
                    (
                        row["optimizer_step"],
                        row["input_block_start"],
                        row["input_batch_sha256"],
                        row["valid_tokens"],
                    )
                    for row in benchmark["warmup_steps"] + benchmark["steps"]
                ],
            )
        if baseline is None:
            baseline = actual
        if actual != baseline:
            raise ValueError("Policy changed full-model math/PG/first-consume/input/state contract")
        records.append(
            dict(
                policy=policy,
                status="PASS",
                wait_edges=wait,
                per_rank={
                    rank: dict(operations=len(value["operations"]), normalized_sha256=stable(value))
                    for rank, value in actual.items()
                },
            )
        )
    result = dict(
        status="PASS",
        policies=list(policies),
        records=records,
        source_manifest_sha256=sha(source_manifest),
        placement_manifest_sha256=sha(placement_manifest),
        scope="Separate recorded full-model gate; profiler-off timing uses online original guard "
        "without per-op recording; metadata equivalence is not a dense mathematical proof",
    )
    save(root / "p2" / "contracts" / "collective_contract_audit.json", result)
    return result
