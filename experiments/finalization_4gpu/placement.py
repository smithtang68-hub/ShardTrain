"""UUID placement only: the logical F2/T2 mesh is deliberately invariant.

This module imports neither torch nor CUDA and can validate all permutations on CPU.
An environment file must describe the *currently allocated* four physical UUIDs.
"""

from __future__ import annotations

import json
from pathlib import Path

GROUPS = {"tp": [[0, 1], [2, 3]], "fsdp": [[0, 2], [1, 3]]}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def link(environment, left, right):
    graph = environment.get("topology", {})
    return graph.get(left, {}).get(right, graph.get(right, {}).get(left, "UNKNOWN"))


def _placement(uuids, devices, environment, mapping_id):
    records = {item["uuid"]: item for item in devices}
    ranks = [
        {
            "rank": rank,
            "local_rank": rank,
            "cuda_ordinal": rank,
            "f": rank // 2,
            "t": rank % 2,
            "uuid": uuid,
            "pci_bus_id": records[uuid].get("pci_bus_id"),
        }
        for rank, uuid in enumerate(uuids)
    ]
    groups = {
        role: [
            {
                "ordered_logical_members": list(members),
                "ordered_physical_uuids": [uuids[rank] for rank in members],
                "topology": link(environment, *[uuids[rank] for rank in members]),
            }
            for members in pairs
        ]
        for role, pairs in GROUPS.items()
    }
    return {
        "mapping_id": mapping_id,
        "status": "PASS",
        "ordered_gpu_uuids": list(uuids),
        "ranks": ranks,
        "groups": groups,
    }


def make_manifest(environment):
    devices = environment.get("devices", [])
    allowed = environment.get("allowed_gpu_uuids")
    if not allowed:
        raise ValueError("Explicit allocated allowed_gpu_uuids are required")
    if len(allowed) != 4 or len(set(allowed)) != 4:
        raise ValueError("Exactly four unique allocated GPU UUIDs are required")
    known = {item.get("uuid") for item in devices}
    if not set(allowed) <= known or any(not x.startswith("GPU-") for x in allowed):
        raise ValueError("Every allocated device must have a full physical GPU UUID")
    for left in allowed:
        for right in allowed:
            if left != right:
                forward = environment.get("topology", {}).get(left, {}).get(right)
                reverse = environment.get("topology", {}).get(right, {}).get(left)
                if not forward or forward in {"X", "UNKNOWN"} or forward != reverse:
                    raise ValueError("A complete symmetric measured topology is required")
    first = _placement(allowed, devices, environment, "M0")
    candidate = _placement(
        [allowed[0], allowed[1], allowed[3], allowed[2]], devices, environment, "M1"
    )
    old_links = [item["topology"] for item in first["groups"]["fsdp"]]
    new_links = [item["topology"] for item in candidate["groups"]["fsdp"]]
    # No generic topology search or intentionally degraded baseline. Preserve the
    # TP sets and test this one hypothesis only when the actual graph supports it.
    applicable = all(value == "NV1" for value in new_links) and "PHB" in old_links
    candidate.update(
        name="M1_fsdp_nv1_pairs" if applicable else "M1_swap23_tradeoff",
        hypothesis_applicability="PASS" if applicable else "NOT_APPLICABLE_TOPOLOGY",
        hypothesis_reason=(
            "Two candidate NV1 pairs replace a baseline PHB pair"
            if applicable
            else "Original NV1-improvement hypothesis is inapplicable; "
            "this is a bidirectional placement tradeoff comparison"
        ),
    )
    result = {
        "schema_version": 1,
        "status": "PASS",
        "hypothesis_applicability": "PASS" if applicable else "NOT_APPLICABLE_TOPOLOGY",
        "allowed_gpu_uuids": list(allowed),
        "logical_mesh": [[0, 1], [2, 3]],
        "logical_groups": {role: [list(pair) for pair in pairs] for role, pairs in GROUPS.items()},
        "environment_source": environment.get("environment_source"),
        "placements": {"M0": first, "M1": candidate},
        "note": "M1 reverses ordered TP members D,C; "
        "only their unordered physical set is preserved",
    }
    validate_manifest(result)
    return result


def validate_manifest(manifest):
    allowed = manifest["allowed_gpu_uuids"]
    if len(allowed) != 4 or len(set(allowed)) != 4:
        raise ValueError("Allocation must be a four-UUID bijection")
    if manifest.get("logical_mesh") != [[0, 1], [2, 3]] or manifest.get("logical_groups") != GROUPS:
        raise ValueError("Logical mesh/group permutation is prohibited")
    expected = {"M0": allowed, "M1": [allowed[0], allowed[1], allowed[3], allowed[2]]}
    for mapping_id, placement in manifest["placements"].items():
        if mapping_id not in expected or placement["ordered_gpu_uuids"] != expected[mapping_id]:
            raise ValueError("Unexpected physical permutation")
        if len(placement["ranks"]) != 4:
            raise ValueError("All four logical ranks are required")
        if set(placement["groups"]) != set(GROUPS):
            raise ValueError("Missing or extra logical PG axes")
        for rank, row in enumerate(placement["ranks"]):
            if (
                row["rank"],
                row["local_rank"],
                row["cuda_ordinal"],
                row["f"],
                row["t"],
                row["uuid"],
            ) != (rank, rank, rank, rank // 2, rank % 2, expected[mapping_id][rank]):
                raise ValueError("Rank/coordinate/device assignment mismatch")
        for role, pairs in GROUPS.items():
            if len(placement["groups"][role]) != 2:
                raise ValueError("Each axis needs two disjoint pairs")
            for pair, row in zip(pairs, placement["groups"][role], strict=True):
                if row["ordered_logical_members"] != pair or row["ordered_physical_uuids"] != [
                    expected[mapping_id][rank] for rank in pair
                ]:
                    raise ValueError("Ordered PG membership mismatch")
    if set(manifest["placements"]) != {"M0", "M1"}:
        raise ValueError("Both mappings must be recorded, including inapplicable M1")
    return manifest


def placement_for(manifest, mapping_id):
    validate_manifest(manifest)
    result = manifest["placements"][mapping_id]
    if result["status"] != "PASS":
        raise ValueError(f"{mapping_id}: {result['status']}: {result.get('reason', '')}")
    return result


def visible_order(manifest, mapping_id):
    return ",".join(placement_for(manifest, mapping_id)["ordered_gpu_uuids"])


def worker_expectation(manifest, mapping_id, rank):
    if rank not in range(4):
        raise ValueError("Single-node four-rank worker expected")
    return placement_for(manifest, mapping_id)["ranks"][rank]
