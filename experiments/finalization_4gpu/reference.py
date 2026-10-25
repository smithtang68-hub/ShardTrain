"""Small actual-payload pair matrix, reusing the pinned NCCL-tests parser."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .placement import placement_for


def inventory_from_events(path):
    """Take representative messages from this candidate, never old constants."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("phase") in {"FWD", "BWD"}]

    def pick(kind, role=None, largest=False, unit=None):
        choices = [
            row
            for row in rows
            if row.get("collective_type") == kind
            and (role is None or row.get("group_role") == role)
            and (unit is None or row.get("unit_id") == unit)
        ]
        if not choices:
            raise ValueError(f"Actual payload missing: {kind}/{role}/{unit}")
        return (max if largest else min)(choices, key=lambda row: row["input_bytes"])

    selected = [
        ("decoder_parameter_ag", "fsdp", "all_gather", pick("fsdp_ag", unit=2)),
        ("embedding_parameter_ag", "fsdp", "all_gather", pick("fsdp_ag", largest=True)),
        (
            "decoder_gradient_rs",
            "fsdp",
            "reduce_scatter",
            pick("fsdp_rs", role="main_gradient_bucket", unit=2),
        ),
        ("activation_ag", "tp", "all_gather", pick("sp_ag", largest=True)),
        ("activation_rs", "tp", "reduce_scatter", pick("sp_rs", largest=True)),
        (
            "embedding_gradient_ar",
            "tp",
            "all_reduce",
            pick("tp_norm_allreduce", role="tp_replicated_gradient", largest=True),
        ),
    ]
    points = []
    for label, axis, collective, row in selected:
        size = row["output_bytes"] if collective == "all_gather" else row["input_bytes"]
        dtype = {"torch.float32": "float", "torch.float16": "half"}.get(row["dtype"])
        if dtype is None or size <= 0 or size % 32:
            raise ValueError("Actual payload unsupported by pinned two-GPU NCCL-tests alignment")
        points.append(
            {
                "label": label,
                "axis": axis,
                "collective": collective,
                "dtype": dtype,
                "full_bytes": size,
                "input_bytes_per_rank": size // 2 if collective == "all_gather" else size,
                "output_bytes_per_rank": size // 2 if collective == "reduce_scatter" else size,
                "actual_event": row,
            }
        )
    return {
        "status": "PASS",
        "source": str(path),
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "points": points,
    }


def make_reference_plan(manifest, inventory, build, repeats=3):
    if inventory.get("status") != "PASS" or build.get("status") != "PASS":
        raise ValueError("Current inventory and locked NCCL build must pass")
    bins = build["binaries"]
    result = []
    for point in inventory["points"]:
        seen = set()
        for mapping in ("M0", "M1"):
            for group in placement_for(manifest, mapping)["groups"][point["axis"]]:
                pair = tuple(group["ordered_physical_uuids"])
                if pair in seen:
                    continue
                seen.add(pair)
                name = point["collective"] + "_perf"
                binary = bins.get(name)
                if binary is None and name == "all_reduce_perf":
                    # Older manifests only index AG/RS. An independently built,
                    # same-source AR sibling must exist and gets its own hash.
                    sibling = Path(bins["all_gather_perf"]["path"]).with_name(name)
                    if not sibling.is_file():
                        raise FileNotFoundError(
                            "Actual all_reduce_perf is required for TP embedding reference"
                        )
                    binary = {
                        "path": str(sibling),
                        "sha256": hashlib.sha256(sibling.read_bytes()).hexdigest(),
                    }
                if binary is None:
                    raise ValueError(f"NCCL build has no {name}")
                for repeat in range(repeats):
                    entry = {key: value for key, value in point.items() if key != "actual_event"}
                    entry.update(
                        mapping_id=mapping,
                        ordered_gpu_uuids=list(pair),
                        topology=group["topology"],
                        repetition=repeat,
                        binary=binary,
                        tests_commit=build["commit"],
                        nccl_library_sha256=build["nccl"]["library_sha256"],
                    )
                    entry["argv"] = [
                        binary["path"],
                        "-b",
                        str(point["full_bytes"]),
                        "-e",
                        str(point["full_bytes"]),
                        "-g",
                        "2",
                        "-t",
                        "1",
                        "-w",
                        "10",
                        "-n",
                        "20",
                        "-d",
                        point["dtype"],
                        "-c",
                        "1",
                    ]
                    result.append(entry)
    return result
