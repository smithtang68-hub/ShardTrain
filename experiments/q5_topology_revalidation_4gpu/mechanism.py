"""Small exact-parser residency audit; never treat non-NCCL kernels as owned GEMM."""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

from .contracts import load, save, sha


def union(values):
    result = []
    for start, end in sorted(values):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersection(left, right):
    left, right = union(left), union(right)
    a = b = 0
    total = 0
    while a < len(left) and b < len(right):
        start, end = max(left[a][0], right[b][0]), min(left[a][1], right[b][1])
        total += max(0, end - start)
        if left[a][1] <= right[b][1]:
            a += 1
        else:
            b += 1
    return total / 1e6


def trace_facts(parsed, identities=None):
    all_rows = parsed["collectives"]
    if not all(row["trace_status"] == "MATCHED" for row in all_rows):
        raise ValueError("Some captured collectives lack exact NVTX/API/correlation linkage")
    collectives = [row for row in all_rows if row["phase"] in {"FWD", "BWD"}]
    selected = {(row["rank"], row["runtime_op_id"]): row for row in collectives}
    groups = defaultdict(list)
    streams = defaultdict(set)
    for kernel in parsed["kernels"]:
        key = kernel.get("rank"), kernel.get("runtime_op_id")
        if kernel.get("is_nccl") and kernel.get("status") == "MATCHED" and key in selected:
            row = selected[key]
            groups[row["rank"], row["family"]].append(
                (kernel["gpu_start_ns"], kernel["gpu_end_ns"])
            )
            native = identities.get(kernel["kernel_rowid"]) if identities else None
            if identities is not None and native is None:
                raise ValueError("Matched parsed kernel rowid is absent from raw SQLite")
            if native and (
                native["globalPid"] >> 24 != kernel["pid"]
                or native["correlationId"] != kernel["correlation_id"]
                or native["start"] != kernel["gpu_start_ns"]
                or native["end"] != kernel["gpu_end_ns"]
                or native["deviceId"] != kernel["device_id"]
                or native["streamId"] != kernel["stream_id"]
            ):
                raise ValueError("Parsed kernel differs from raw SQLite identity")
            streams[row["rank"], row["channel"]].add(
                (
                    kernel["pid"],
                    native["contextId"] if native else None,
                    kernel["device_id"],
                    kernel["stream_id"],
                )
            )
    ranks = []
    for rank in range(4):
        ag = groups[rank, "fsdp_parameter_ag"]
        rs = groups[rank, "fsdp_gradient_rs"]
        sp = groups[rank, "sp_activation_ag"]
        non_fsdp = [
            interval
            for (owner, family), values in groups.items()
            if owner == rank and not family.startswith("fsdp_")
            for interval in values
        ]
        all_intervals = [
            interval
            for (owner, _), values in groups.items()
            if owner == rank
            for interval in values
        ]
        if not ag or not rs or not sp:
            raise ValueError("Missing parameter/gradient/SP branches")
        depth, maximum = 0, 0
        for _, change in sorted(
            [(start, 1) for start, _ in all_intervals] + [(end, -1) for _, end in all_intervals]
        ):
            depth += change
            maximum = max(maximum, depth)
        ranks.append(
            dict(
                rank=rank,
                fsdp_ag_sp_ag_residency_ms=intersection(ag, sp),
                fsdp_ag_rs_residency_ms=intersection(ag, rs),
                fsdp_nonfsdp_residency_ms=intersection(ag + rs, non_fsdp),
                maximum_nccl_residency_depth=maximum,
            )
        )
    peer_groups = defaultdict(list)
    for row in collectives:
        peer_groups[
            row["process_group_label"], tuple(row["group_members"]), row["collective_sequence"]
        ].append(row)
    peers = []
    for key, values in peer_groups.items():
        if len(values) != len(key[1]) or {row["rank"] for row in values} != set(key[1]):
            raise ValueError("Incomplete actual PG peer evidence")
        peers.append(
            dict(
                pg_label=key[0],
                members=key[1],
                sequence=key[2],
                kind=values[0]["collective_type"],
                unit_id=values[0]["unit_id"],
                phase=values[0]["phase"],
                api_submit_skew_ms=(
                    max(row["cpu_nsys_submit_ns"] for row in values)
                    - min(row["cpu_nsys_submit_ns"] for row in values)
                )
                / 1e6,
                gpu_begin_skew_ms=(
                    max(row["gpu_start_timestamp"] for row in values)
                    - min(row["gpu_start_timestamp"] for row in values)
                )
                / 1e6,
                gpu_end_skew_ms=(
                    max(row["gpu_end_timestamp"] for row in values)
                    - min(row["gpu_end_timestamp"] for row in values)
                )
                / 1e6,
            )
        )
    return dict(
        captured=len(all_rows),
        training=len(collectives),
        per_rank=ranks,
        peer_skew=peers,
        actual_streams=[
            dict(rank=key[0], channel=key[1], actual_namespaces=sorted(values))
            for key, values in sorted(streams.items())
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--trace",
        nargs=3,
        action="append",
        required=True,
        metavar=("POLICY", "PARSED_JSON", "SQLITE"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {}
    source_hashes = set()
    for policy, path, sqlite_path in args.trace:
        if policy not in {"C0_guarded", "C1", "C2"} or policy in result:
            raise ValueError("Only fresh explicit C0_guarded/C1/C2 traces are accepted")
        capture = Path(args.run_root) / "p2/profiles" / policy
        parsed = load(path)
        audits = []
        for rank in range(4):
            audit = load(capture / f"placement.rank{rank}.json")
            benchmark = load(capture / f"benchmark.rank{rank}.json")
            if (
                audit["status"] != "PASS"
                or not benchmark["profile"]
                or benchmark["policy"] != ("C0" if policy == "C0_guarded" else policy)
                or benchmark["guarded_baseline"] != (policy == "C0_guarded")
            ):
                raise ValueError(
                    "Selected trace is not the current explicitly labelled guarded policy"
                )
            audits.append(audit)
            source_hashes.add(audit["source_manifest_sha256"])
        database_path = Path(sqlite_path).resolve()
        if parsed["source_size_bytes"] != database_path.stat().st_size:
            raise ValueError("Parsed result does not match this raw SQLite size")
        database = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
        database.row_factory = sqlite3.Row
        try:
            database.execute("PRAGMA query_only=ON")
            identities = {
                row["rowid"]: dict(row)
                for row in database.execute(
                    "SELECT rowid,globalPid,contextId,deviceId,streamId,correlationId,start,end "
                    "FROM CUPTI_ACTIVITY_KIND_KERNEL"
                )
            }
            processes = {
                row["globalPid"] >> 24: row["pid"]
                for row in database.execute("SELECT globalPid,pid FROM PROCESSES")
            }
        finally:
            database.close()
        facts = trace_facts(parsed, identities)
        for row in facts["actual_streams"]:
            audit = audits[row["rank"]]
            row.update(
                actual_uuid=audit["actual_uuid"],
                actual_pci_bus_id=audit["actual_pci_bus_id"],
                uuid_scope="Actual rank startup UUID/PCI plus exact rank NVTX/API/kernel; "
                "CUPTI deviceId is not joined to global hardware inventory",
                os_pids=sorted({processes[identity[0]] for identity in row["actual_namespaces"]}),
            )
        if policy == "C1" and any(
            row["maximum_nccl_residency_depth"] > 1 for row in facts["per_rank"]
        ):
            raise ValueError("C1 physical NCCL residency was not serialized")
        if policy == "C2" and any(
            row["fsdp_nonfsdp_residency_ms"] > 0 for row in facts["per_rank"]
        ):
            raise ValueError("C2 physical FSDP/non-FSDP residency was not isolated")
        result[policy] = dict(
            parsed_path=path,
            parsed_sha256=sha(path),
            sqlite_path=sqlite_path,
            sqlite_sha256=sha(sqlite_path),
            **facts,
        )
    if len(source_hashes) != 1:
        raise ValueError("Profiles do not bind one current source snapshot")
    value = dict(
        status="PASS" if {"C0_guarded", "C1"}.issubset(result) else "PARTIAL",
        traces=result,
        source_script_sha256=sha(__file__),
        source_manifest_sha256=next(iter(source_hashes)),
        limits=[
            "Only newly announced exact-parser results; rank namespace/stream IDs are trace scoped",
            "NCCL kernel residency includes peer waits; not wire or saved time",
            "Current-unit GEMM and full CUDA event DAG UNKNOWN",
            "C2 edges permit AG/RS concurrency but do not guarantee actual positive overlap",
        ],
    )
    save(args.output, value)
    print(value["status"])


if __name__ == "__main__":
    main()
