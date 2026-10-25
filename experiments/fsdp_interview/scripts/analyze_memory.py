#!/usr/bin/env python3
"""Read bounded allocator snapshots without importing Torch or executing pickle.

Only primitive pickle opcodes and containers are accepted. This module never
opens timeline traces. ``device_traces`` are allocator event histories, not GPU
execution timelines. All hashes reported here are LOCAL hashes; the delivery
manifest must be validated separately before claiming an end-to-end match.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import pickle
import pickletools
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ALLOWED_OPCODES = frozenset(
    "PROTO FRAME STOP MARK POP POP_MARK DUP NONE NEWTRUE NEWFALSE "
    "INT BININT BININT1 BININT2 LONG LONG1 LONG4 FLOAT BINFLOAT "
    "STRING BINSTRING SHORT_BINSTRING UNICODE BINUNICODE SHORT_BINUNICODE "
    "BINUNICODE8 BINBYTES SHORT_BINBYTES BINBYTES8 BYTEARRAY8 "
    "EMPTY_LIST LIST APPEND APPENDS EMPTY_TUPLE TUPLE TUPLE1 TUPLE2 TUPLE3 "
    "EMPTY_DICT DICT SETITEM SETITEMS EMPTY_SET ADDITEMS FROZENSET "
    "PUT BINPUT LONG_BINPUT GET BINGET LONG_BINGET MEMOIZE".split()
)
ACTIVE_STATES = frozenset(("active_allocated", "active_awaiting_free"))
BLOCK_STATES = ACTIVE_STATES | {"inactive"}
TRACE_ACTIONS = frozenset(
    (
        "alloc",
        "free_requested",
        "free_completed",
        "segment_alloc",
        "segment_free",
        "segment_map",
        "segment_unmap",
        "snapshot",
        "oom",
    )
)
GIB = 1024**3
ALLOCATOR_SOURCE = {
    "torch_git_version": "5811a8d7da873dd699ff6687092c225caffcf1bb",
    "url": "https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/c10/cuda/CUDACachingAllocator.cpp",
    "rules": {
        "round_size": "minimum and multiple of 512 B with power2 divisions disabled",
        "split": "small remainder >=512 B; large remainder >1 MiB with unlimited max_split_size",
        "alloc_free_event_size": "requested_size, not block storage size",
        "snapshot_event_size": "total active block storage",
        "segment_event_size": "segment storage size",
    },
    "scope": "independent snapshot replay; no allocator or runtime implementation is copied",
}


class SnapshotError(ValueError):
    """Malformed, unsafe, incomplete, or unsupported snapshot data."""


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise SnapshotError(f"pickle class lookup rejected: {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        raise SnapshotError("pickle persistent references rejected")


def primitive_loads(payload: bytes) -> Any:
    """Prescan before unpickling; reject all object construction/execution ops."""
    if not payload:
        raise SnapshotError("empty pickle")
    final_position = -1
    final_name = ""
    try:
        for opcode, _argument, position in pickletools.genops(payload):
            if opcode.name not in ALLOWED_OPCODES:
                raise SnapshotError(f"pickle opcode rejected: {opcode.name} at {position}")
            final_position, final_name = position, opcode.name
    except (ValueError, UnicodeError) as exc:
        raise SnapshotError(f"invalid pickle opcode stream: {exc}") from exc
    if final_name != "STOP" or final_position != len(payload) - 1:
        raise SnapshotError("pickle must end at its only STOP; trailing data rejected")
    try:
        value = PrimitiveUnpickler(io.BytesIO(payload)).load()
    except (pickle.UnpicklingError, EOFError, IndexError, KeyError, TypeError) as exc:
        raise SnapshotError(f"invalid primitive pickle: {exc}") from exc
    seen: set[int] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) in (type(None), bool, int, float, str, bytes, bytearray):
            continue
        if type(item) not in (list, tuple, dict, set, frozenset):
            raise SnapshotError(f"nonprimitive value rejected: {type(item).__name__}")
        if id(item) in seen:
            continue
        seen.add(id(item))
        if len(seen) > 2_000_000:
            raise SnapshotError("too many primitive containers")
        if type(item) is dict:
            pending.extend(item.keys())
            pending.extend(item.values())
        else:
            pending.extend(item)
    return value


def integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SnapshotError(f"{label}: expected integer >= {minimum}, got {value!r}")
    return value


def mapping(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise SnapshotError(f"{label}: expected dictionary with string keys")
    return value


def array(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        raise SnapshotError(f"{label}: expected list")
    return value


def frames(value: Any, label: str) -> list[dict[str, Any]]:
    result = array(value, label)
    for index, row in enumerate(result):
        row = mapping(row, f"{label}[{index}]")
        if type(row.get("filename")) is not str or type(row.get("name")) is not str:
            raise SnapshotError(f"{label}[{index}]: frame filename/name must be strings")
        integer(row.get("line"), f"{label}[{index}].line")
    return result


def callsite(block_or_event: dict[str, Any]) -> dict[str, str | int]:
    """A captured callsite is not proof of tensor ownership."""
    for row in block_or_event.get("frames", []):
        filename = row["filename"].replace("\\", "/")
        if "/project/" in filename:
            return {
                "status": "PROJECT_CALLSITE_CAPTURED",
                "owner": "UNKNOWN",
                "filename": filename.split("/project/", 1)[1],
                "line": row["line"],
                "function": row["name"],
            }
    return {"status": "UNKNOWN", "owner": "UNKNOWN"}


def load_snapshot(path: Path, maximum_bytes: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink():
        raise SnapshotError(f"snapshot file must not be a symlink: {path}")
    if not path.is_file():
        raise SnapshotError(f"missing snapshot: {path}")
    length = path.stat().st_size
    if not 0 < length <= maximum_bytes:
        raise SnapshotError(f"snapshot size outside allowed bounds: {length}")
    payload = path.read_bytes()
    if len(payload) != length:
        raise SnapshotError("snapshot changed while reading")
    provenance = {
        "path": str(path),
        "bytes": len(payload),
        "sha256_local": hashlib.sha256(payload).hexdigest(),
        "remote_hash_match": "NOT_VERIFIED_BY_THIS_ANALYSIS",
    }
    return mapping(primitive_loads(payload), "snapshot"), provenance


def validate_snapshot(value: dict[str, Any]) -> dict[str, Any]:
    """Audit schema, address coverage and the allocator's segment counters."""
    segments = array(value.get("segments"), "segments")
    traces = array(value.get("device_traces"), "device_traces")
    by_device: dict[int, list[dict[str, Any]]] = {}
    block_counts: Counter[str] = Counter()
    unknown_allocated_bytes = 0
    for index, segment in enumerate(segments):
        label = f"segments[{index}]"
        segment = mapping(segment, label)
        device = integer(segment.get("device"), f"{label}.device")
        address = integer(segment.get("address"), f"{label}.address", 1)
        size = integer(segment.get("total_size"), f"{label}.total_size", 1)
        integer(segment.get("stream"), f"{label}.stream")
        if segment.get("segment_type") not in ("small", "large"):
            raise SnapshotError(f"{label}: unrecognized segment_type")
        blocks = array(segment.get("blocks"), f"{label}.blocks")
        cursor = address
        allocated = active = requested = 0
        for block_index, block in enumerate(blocks):
            block_label = f"{label}.blocks[{block_index}]"
            block = mapping(block, block_label)
            block_address = integer(block.get("address", cursor), f"{block_label}.address", 1)
            block_size = integer(block.get("size"), f"{block_label}.size", 1)
            block_requested = integer(block.get("requested_size"), f"{block_label}.requested_size")
            state = block.get("state")
            if state not in BLOCK_STATES:
                raise SnapshotError(f"{block_label}: invalid state {state!r}")
            if block_address != cursor:
                raise SnapshotError(f"{block_label}: block address coverage is not contiguous")
            if state in ACTIVE_STATES and block_requested > block_size:
                raise SnapshotError(f"{block_label}: active request exceeds block storage")
            frames(block.get("frames", []), f"{block_label}.frames")
            block_counts[state] += 1
            if state == "active_allocated":
                allocated += block_size
                if callsite(block)["status"] == "UNKNOWN":
                    unknown_allocated_bytes += block_size
            if state in ACTIVE_STATES:
                active += block_size
                requested += block_requested
            cursor += block_size
        if cursor != address + size:
            raise SnapshotError(f"{label}: total_size does not equal block coverage")
        for name, recomputed in (
            ("allocated_size", allocated),
            ("active_size", active),
            ("requested_size", requested),
        ):
            if integer(segment.get(name), f"{label}.{name}") != recomputed:
                raise SnapshotError(f"{label}: {name} disagrees with blocks")
        by_device.setdefault(device, []).append(segment)
    for device, device_segments in by_device.items():
        previous_end = 0
        for segment in sorted(device_segments, key=lambda row: row["address"]):
            if segment["address"] < previous_end:
                raise SnapshotError(f"device {device}: overlapping segments")
            previous_end = segment["address"] + segment["total_size"]
    trace_summaries = []
    for device, history in enumerate(traces):
        history = array(history, f"device_traces[{device}]")
        action_counts: Counter[str] = Counter()
        samples: dict[str, list[dict[str, Any]]] = {}
        for index, event in enumerate(history):
            label = f"device_traces[{device}][{index}]"
            event = mapping(event, label)
            action = event.get("action")
            if action not in TRACE_ACTIONS:
                raise SnapshotError(f"{label}: unsupported action {action!r}")
            for name in ("addr", "size", "stream"):
                if name in event:
                    integer(event[name], f"{label}.{name}")
            if action != "snapshot" and ("addr" not in event or "size" not in event):
                raise SnapshotError(f"{label}: missing address/size")
            if "time_us" in event and (
                type(event["time_us"]) not in (int, float) or not math.isfinite(event["time_us"])
            ):
                raise SnapshotError(f"{label}: invalid time_us")
            frames(event.get("frames", []), f"{label}.frames")
            action_counts[action] += 1
            if len(samples.setdefault(action, [])) < 3:
                samples[action].append(
                    {
                        **{
                            key: event[key]
                            for key in ("action", "addr", "size", "stream", "time_us")
                            if key in event
                        },
                        "event_index": index,
                        "frame_count": len(event.get("frames", [])),
                        "callsite": callsite(event),
                    }
                )
        trace_summaries.append(
            {
                "device": device,
                "event_count": len(history),
                "action_counts": dict(action_counts),
                "first_samples": samples,
            }
        )
    totals = {
        "reserved_bytes": sum(row["total_size"] for row in segments),
        "allocated_bytes": sum(row["allocated_size"] for row in segments),
        "active_bytes": sum(row["active_size"] for row in segments),
        "requested_bytes": sum(row["requested_size"] for row in segments),
    }
    totals["awaiting_free_bytes"] = totals["active_bytes"] - totals["allocated_bytes"]
    totals["inactive_cache_bytes"] = totals["reserved_bytes"] - totals["active_bytes"]
    return {
        "root_keys": sorted(value),
        "segment_count": len(segments),
        "block_count": sum(block_counts.values()),
        "block_states": dict(block_counts),
        "devices_with_segments": sorted(by_device),
        "counters": totals,
        "unknown_callsite_allocated_bytes": unknown_allocated_bytes,
        "allocator_settings": value.get("allocator_settings", {}),
        "device_traces": trace_summaries,
    }


def inspect_rank(
    capture_dir: Path, rank: int, maximum_bytes: int
) -> tuple[dict[str, Any], tuple[Any, Any]]:
    profile_path = capture_dir / f"profiling.rank{rank}.json"
    profiling = mapping(json.loads(profile_path.read_text(encoding="utf-8")), "profiling")
    profiling_bytes = profile_path.read_bytes()
    environment_path = capture_dir / f"environment.rank{rank}.json"
    environment_bytes = environment_path.read_bytes()
    environment = mapping(json.loads(environment_bytes), "environment")
    if profiling.get("rank") != rank or environment.get("rank") != rank:
        raise SnapshotError(f"rank {rank}: profile/environment identity mismatch")
    capacity = integer(profiling.get("memory_history_max_entries"), "memory_history_max_entries", 1)
    start, start_provenance = load_snapshot(
        capture_dir / f"memory.start.rank{rank}.pickle", maximum_bytes
    )
    end, end_provenance = load_snapshot(
        capture_dir / f"memory.end.rank{rank}.pickle", maximum_bytes
    )
    start_audit, end_audit = validate_snapshot(start), validate_snapshot(end)
    if start_audit["devices_with_segments"] != [environment.get("local_rank")]:
        raise SnapshotError(f"rank {rank}: populated allocator device differs from environment")
    if len(start["device_traces"]) != len(end["device_traces"]):
        raise SnapshotError("start/end trace device counts differ")
    history_audit = []
    for device, (before, after) in enumerate(
        zip(start["device_traces"], end["device_traces"], strict=True)
    ):
        prefix_match = len(before) <= len(after) and before == after[: len(before)]
        history_audit.append(
            {
                "device": device,
                "start_count": len(before),
                "end_count": len(after),
                "max_entries": capacity,
                "start_exact_prefix_of_end": prefix_match,
                "capacity_reached": len(before) >= capacity or len(after) >= capacity,
                "window_event_count": len(after) - len(before) if prefix_match else None,
                "replay_candidate": prefix_match
                and len(before) < capacity
                and len(after) < capacity,
                "caveat": "A full buffer cannot prove absence of older overwritten events.",
            }
        )
    return (
        {
            "rank": rank,
            "capture_dir": str(capture_dir),
            "raw_provenance": {"start": start_provenance, "end": end_provenance},
            "profiling_provenance": {
                "path": str(profile_path),
                "bytes": len(profiling_bytes),
                "sha256_local": hashlib.sha256(profiling_bytes).hexdigest(),
            },
            "environment": environment,
            "environment_provenance": {
                "path": str(environment_path),
                "bytes": len(environment_bytes),
                "sha256_local": hashlib.sha256(environment_bytes).hexdigest(),
            },
            "start": start_audit,
            "end": end_audit,
            "history": history_audit,
            "profiling": profiling,
            "peak_replay_status": "NOT_RUN",
            "boundaries": [
                "Allocator history is not a GPU timeline or stream dependency proof.",
                "Reserved bytes include inactive allocator cache; "
                "they are not live parameter bytes.",
                "Snapshots exclude directly allocated NCCL/CUDA memory.",
                "Captured allocation callsites do not establish tensor ownership; "
                "otherwise UNKNOWN.",
                "Only local SHA is recorded; remote delivery hash verification remains separate.",
            ],
        },
        (start, end),
    )


def block_rows(snapshot: dict[str, Any], device: int) -> list[dict[str, Any]]:
    result = []
    for segment in snapshot["segments"]:
        if segment["device"] != device:
            continue
        address = segment["address"]
        for block in segment["blocks"]:
            if block["state"] in ACTIVE_STATES:
                result.append(
                    {
                        "address": block.get("address", address),
                        "segment_address": segment["address"],
                        "stream": segment["stream"],
                        "pool": segment["segment_type"],
                        "state": block["state"],
                        "storage_bytes": block["size"],
                        "requested_bytes": block["requested_size"],
                        **{f"callsite_{key}": value for key, value in callsite(block).items()},
                    }
                )
            address += block["size"]
    return sorted(result, key=lambda row: row["address"])


class NativeReplay:
    """Default native allocator block accounting, starting from known live blocks.

    Address selection is supplied by recorded events. We only reconstruct the
    storage size/splits and lifecycle needed to audit counters. Custom rounding,
    expandable segments and finite max_split_size are deliberately unsupported.
    A replay is trusted only if its complete final block map and reported peaks
    match the independently saved endpoint/allocator counters.
    """

    def __init__(self, snapshot: dict[str, Any], device: int) -> None:
        self.device = device
        self.segments: dict[int, dict[str, Any]] = {}
        self.blocks: dict[int, dict[str, Any]] = {}
        self.start_live_addresses: set[int] = set()
        self.serial = 0
        self.counters = dict.fromkeys(
            ("reserved_bytes", "allocated_bytes", "active_bytes", "requested_bytes"), 0
        )
        for original in snapshot["segments"]:
            if original["device"] != device:
                continue
            segment = {
                "address": original["address"],
                "size": original["total_size"],
                "stream": original["stream"],
                "pool": original["segment_type"],
                "blocks": [],
            }
            self.segments[segment["address"]] = segment
            cursor = segment["address"]
            for original_block in original["blocks"]:
                block = {
                    "address": cursor,
                    "size": original_block["size"],
                    "requested": original_block["requested_size"],
                    "state": original_block["state"],
                    "frames": original_block.get("frames", []),
                    "segment": segment["address"],
                    "allocation_origin": "START_HELD"
                    if original_block["state"] in ACTIVE_STATES
                    else "INACTIVE",
                    "allocation_event_index": None,
                    "generation": 0,
                }
                segment["blocks"].append(cursor)
                self.blocks[cursor] = block
                if block["state"] in ACTIVE_STATES:
                    self.start_live_addresses.add(cursor)
                    self.counters["active_bytes"] += block["size"]
                    self.counters["requested_bytes"] += block["requested"]
                if block["state"] == "active_allocated":
                    self.counters["allocated_bytes"] += block["size"]
                cursor += block["size"]
            self.counters["reserved_bytes"] += segment["size"]
        self.points = [self.point(-1, None, "start")]
        self.peaks = {
            key: {"bytes": value, "event_index": -1, "time_us": None}
            for key, value in self.counters.items()
        }
        self.peak_allocated_blocks = self.live_rows()
        self.max_awaiting = {
            "bytes": self.counters["active_bytes"] - self.counters["allocated_bytes"],
            "event_index": -1,
        }

    def point(self, index: int, timestamp: Any, action: str) -> dict[str, Any]:
        return {
            "event_index": index,
            "time_us": timestamp,
            "action": action,
            **self.counters,
            "awaiting_free_bytes": self.counters["active_bytes"] - self.counters["allocated_bytes"],
            "inactive_cache_bytes": self.counters["reserved_bytes"] - self.counters["active_bytes"],
        }

    def live_rows(self) -> list[dict[str, Any]]:
        result = []
        for address, block in sorted(self.blocks.items()):
            if block["state"] not in ACTIVE_STATES:
                continue
            segment = self.segments[block["segment"]]
            result.append(
                {
                    "address": address,
                    "segment_address": block["segment"],
                    "stream": segment["stream"],
                    "pool": segment["pool"],
                    "state": block["state"],
                    "storage_bytes": block["size"],
                    "requested_bytes": block["requested"],
                    "allocation_origin": block["allocation_origin"],
                    "allocation_event_index": block["allocation_event_index"],
                    "generation": block["generation"],
                    **{f"callsite_{key}": value for key, value in callsite(block).items()},
                }
            )
        return result

    def consume(self, event: dict[str, Any], index: int) -> None:
        action, address, requested = event["action"], event.get("addr", 0), event.get("size", 0)
        if action == "alloc":
            block = self.blocks.get(address)
            if block is None or block["state"] != "inactive":
                raise SnapshotError(f"event {index}: alloc address is not an inactive block start")
            segment = self.segments[block["segment"]]
            rounded = max(512, ((requested + 511) // 512) * 512)
            if rounded > block["size"] or segment["stream"] != event["stream"]:
                raise SnapshotError(
                    f"event {index}: alloc exceeds block or changes allocation stream"
                )
            remainder = block["size"] - rounded
            split = remainder >= 512 if segment["pool"] == "small" else remainder > 1024**2
            if split:
                remainder_address = address + rounded
                remainder_block = {**block, "address": remainder_address, "size": remainder}
                self.blocks[remainder_address] = remainder_block
                segment["blocks"].insert(
                    bisect.bisect_right(segment["blocks"], address), remainder_address
                )
                block["size"] = rounded
            self.serial += 1
            block.update(
                requested=requested,
                state="active_allocated",
                frames=event.get("frames", []),
                allocation_origin="WINDOW_ALLOC",
                allocation_event_index=index,
                generation=self.serial,
            )
            self.counters["allocated_bytes"] += block["size"]
            self.counters["active_bytes"] += block["size"]
            self.counters["requested_bytes"] += requested
        elif action in ("free_requested", "free_completed"):
            block = self.blocks.get(address)
            expected_state = (
                "active_allocated" if action == "free_requested" else "active_awaiting_free"
            )
            if block is None or block["state"] != expected_state or block["requested"] != requested:
                raise SnapshotError(
                    f"event {index}: {action} does not match address/state/requested size"
                )
            segment = self.segments[block["segment"]]
            if segment["stream"] != event["stream"]:
                raise SnapshotError(f"event {index}: free changes allocation stream")
            if action == "free_requested":
                block["state"] = "active_awaiting_free"
                self.counters["allocated_bytes"] -= block["size"]
            else:
                self.counters["active_bytes"] -= block["size"]
                self.counters["requested_bytes"] -= requested
                block.update(
                    state="inactive",
                    frames=[],
                    allocation_origin="INACTIVE",
                    allocation_event_index=None,
                )
                position = bisect.bisect_left(segment["blocks"], address)
                # Adjacent inactive native blocks coalesce on free completion.
                if position > 0:
                    previous_address = segment["blocks"][position - 1]
                    previous = self.blocks[previous_address]
                    if previous["state"] == "inactive":
                        if previous_address + previous["size"] != address:
                            raise SnapshotError(f"event {index}: noncontiguous previous block")
                        self.blocks.pop(address)
                        segment["blocks"].pop(position)
                        block["address"] = previous_address
                        block["size"] += previous["size"]
                        self.blocks[previous_address] = block
                        address, position = previous_address, position - 1
                if position + 1 < len(segment["blocks"]):
                    next_address = segment["blocks"][position + 1]
                    following = self.blocks[next_address]
                    if following["state"] == "inactive":
                        if address + block["size"] != next_address:
                            raise SnapshotError(f"event {index}: noncontiguous next block")
                        block["size"] += following["size"]
                        self.blocks.pop(next_address)
                        segment["blocks"].pop(position + 1)
        elif action == "segment_alloc":
            if address in self.segments or requested < 2 * 1024**2:
                raise SnapshotError(f"event {index}: unsupported or duplicate native segment")
            if any(
                start < address + requested and address < start + row["size"]
                for start, row in self.segments.items()
            ):
                raise SnapshotError(
                    f"event {index}: newly allocated segment overlaps existing storage"
                )
            self.segments[address] = {
                "address": address,
                "size": requested,
                "stream": event["stream"],
                "pool": "small" if requested == 2 * 1024**2 else "large",
                "blocks": [address],
            }
            self.blocks[address] = {
                "address": address,
                "size": requested,
                "requested": 0,
                "state": "inactive",
                "frames": [],
                "segment": address,
                "allocation_origin": "INACTIVE",
                "allocation_event_index": None,
                "generation": 0,
            }
            self.counters["reserved_bytes"] += requested
        elif action == "segment_free":
            segment = self.segments.get(address)
            if (
                segment is None
                or segment["size"] != requested
                or segment["stream"] != event["stream"]
            ):
                raise SnapshotError(f"event {index}: segment free does not match existing segment")
            if any(self.blocks[item]["state"] != "inactive" for item in segment["blocks"]):
                raise SnapshotError(f"event {index}: freed segment contains active storage")
            for item in segment["blocks"]:
                self.blocks.pop(item)
            self.segments.pop(address)
            self.counters["reserved_bytes"] -= requested
        elif action == "snapshot":
            if requested != self.counters["active_bytes"]:
                raise SnapshotError(
                    f"event {index}: snapshot active-size marker differs from replay"
                )
        elif action == "oom":
            pass  # OOM does not create a live allocation; recorded separately.
        else:
            raise SnapshotError(f"event {index}: unsupported native action {action}")
        if (
            not 0
            <= self.counters["allocated_bytes"]
            <= self.counters["active_bytes"]
            <= self.counters["reserved_bytes"]
        ):
            raise SnapshotError(f"event {index}: impossible allocator storage counters")
        if not 0 <= self.counters["requested_bytes"] <= self.counters["active_bytes"]:
            raise SnapshotError(f"event {index}: impossible requested byte counter")
        self.points.append(self.point(index, event.get("time_us"), action))
        for key, value in self.counters.items():
            if value > self.peaks[key]["bytes"]:
                self.peaks[key] = {
                    "bytes": value,
                    "event_index": index,
                    "time_us": event.get("time_us"),
                }
                if key == "allocated_bytes":
                    self.peak_allocated_blocks = self.live_rows()
        pending = self.counters["active_bytes"] - self.counters["allocated_bytes"]
        if pending > self.max_awaiting["bytes"]:
            self.max_awaiting = {"bytes": pending, "event_index": index}

    def compare_end(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        expected = NativeReplay(snapshot, self.device)
        expected_map = {
            address: (
                row["size"],
                row["state"],
                row["segment"],
                row["requested"] if row["state"] in ACTIVE_STATES else None,
            )
            for address, row in expected.blocks.items()
        }
        observed_map = {
            address: (
                row["size"],
                row["state"],
                row["segment"],
                row["requested"] if row["state"] in ACTIVE_STATES else None,
            )
            for address, row in self.blocks.items()
        }
        expected_segments = {
            address: (row["size"], row["stream"], row["pool"])
            for address, row in expected.segments.items()
        }
        observed_segments = {
            address: (row["size"], row["stream"], row["pool"])
            for address, row in self.segments.items()
        }
        mismatched = sorted(
            address
            for address in expected_map.keys() | observed_map.keys()
            if expected_map.get(address) != observed_map.get(address)
        )
        return {
            "all_block_addresses_sizes_states_live_requests_match": not mismatched,
            "segment_addresses_sizes_streams_pools_match": expected_segments == observed_segments,
            "counter_match": self.counters == expected.counters,
            "expected_block_count": len(expected_map),
            "replayed_block_count": len(observed_map),
            "mismatch_count": len(mismatched),
            "first_mismatch_addresses": mismatched[:20],
            "inactive_requested_metadata": (
                "Not compared; stale request metadata is not live storage."
            ),
        }


def replay_rank(
    row: dict[str, Any], start: dict[str, Any], end: dict[str, Any]
) -> tuple[dict[str, Any], NativeReplay | None]:
    if row.get("environment", {}).get("torch", "").split("+", 1)[0] != "2.9.1":
        return {"status": "NOT_RUN", "reason": "source semantics are pinned to Torch 2.9.1"}, None
    devices = row["start"]["devices_with_segments"]
    if len(devices) != 1:
        return {
            "status": "NOT_RUN",
            "reason": "requires one populated allocator device per rank",
        }, None
    device = devices[0]
    guard = row["history"][device]
    if not guard["replay_candidate"]:
        return {
            "status": "NOT_RUN",
            "reason": "history prefix mismatch or bounded buffer capacity reached",
        }, None
    settings = row["start"]["allocator_settings"]
    unsupported = (
        settings != row["end"]["allocator_settings"]
        or settings.get("expandable_segments") is not False
        or settings.get("max_split_size") != -1
        or not settings.get("roundup_power2_divisions")
        or any(value != 0 for value in settings.get("roundup_power2_divisions", {}).values())
        or "cudaMallocAsync" in settings.get("PYTORCH_CUDA_ALLOC_CONF", "")
    )
    if unsupported:
        return {
            "status": "NOT_RUN",
            "reason": "only stable default native allocation rounding/splitting is supported",
        }, None
    events = end["device_traces"][device][len(start["device_traces"][device]) :]
    if any(event["action"] in ("segment_map", "segment_unmap") for event in events):
        return {"status": "NOT_RUN", "reason": "expandable segment events are unsupported"}, None
    replay = NativeReplay(start, device)
    try:
        for index, event in enumerate(events, len(start["device_traces"][device])):
            replay.consume(event, index)
    except SnapshotError as exc:
        return {
            "status": "FAILED",
            "reason": str(exc),
            "consumed_event_count": len(replay.points) - 1,
        }, None
    endpoint = replay.compare_end(end)
    memory = row["profiling"].get("memory", [])
    comparison = []
    if len(memory) == 1:
        for field, recomputed in (
            ("allocated", replay.counters["allocated_bytes"]),
            ("reserved", replay.counters["reserved_bytes"]),
            ("max_allocated", replay.peaks["allocated_bytes"]["bytes"]),
            ("max_reserved", replay.peaks["reserved_bytes"]["bytes"]),
        ):
            comparison.append(
                {
                    "field": field,
                    "reported": memory[0][field],
                    "replayed": recomputed,
                    "match": memory[0][field] == recomputed,
                }
            )
    checks = [
        endpoint["all_block_addresses_sizes_states_live_requests_match"],
        endpoint["segment_addresses_sizes_streams_pools_match"],
        endpoint["counter_match"],
        bool(comparison),
        *(item["match"] for item in comparison),
    ]
    live = replay.live_rows()
    surviving_start = [item for item in live if item["allocation_origin"] == "START_HELD"]
    new_live = [item for item in live if item["allocation_origin"] == "WINDOW_ALLOC"]
    return {
        "status": "VERIFIED" if all(checks) else "UNVERIFIED",
        "device": device,
        "consumed_event_count": len(events),
        "source_semantics": ALLOCATOR_SOURCE,
        "endpoint": endpoint,
        "profiling_counter_comparison": comparison,
        "peaks": replay.peaks,
        "max_awaiting_free": replay.max_awaiting,
        "oom_events": sum(event["action"] == "oom" for event in events),
        "endpoint_live_generation": {
            "start_held_surviving_block_count": len(surviving_start),
            "start_held_surviving_storage_bytes": sum(
                item["storage_bytes"] for item in surviving_start
            ),
            "window_alloc_still_live_block_count": len(new_live),
            "window_alloc_still_live_storage_bytes": sum(
                item["storage_bytes"] for item in new_live
            ),
            "scope": "Allocator allocation lifetimes; an address can be freed "
            "and reused for a different Tensor.",
        },
    }, replay


def held_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, Any, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["state"],
            row.get("callsite_filename", "UNKNOWN"),
            row.get("callsite_line"),
            row.get("callsite_function", "UNKNOWN"),
        )
        group = groups.setdefault(
            key,
            {
                "state": key[0],
                "filename": key[1],
                "line": key[2],
                "function": key[3],
                "tensor_owner": "UNKNOWN",
                "block_count": 0,
                "storage_bytes": 0,
                "requested_bytes": 0,
            },
        )
        group["block_count"] += 1
        group["storage_bytes"] += row["storage_bytes"]
        group["requested_bytes"] += row["requested_bytes"]
    return {
        "block_count": len(rows),
        "allocated_bytes": sum(
            row["storage_bytes"] for row in rows if row["state"] == "active_allocated"
        ),
        "awaiting_free_bytes": sum(
            row["storage_bytes"] for row in rows if row["state"] == "active_awaiting_free"
        ),
        "unknown_callsite_bytes": sum(
            row["storage_bytes"] for row in rows if row["callsite_status"] == "UNKNOWN"
        ),
        "groups_by_captured_callsite": sorted(
            groups.values(), key=lambda group: group["storage_bytes"], reverse=True
        ),
        "interpretation": "Callsite groups are captured evidence, not inferred tensor ownership.",
    }


def sample_points(replay: NativeReplay) -> list[dict[str, Any]]:
    step = max(1, math.ceil(len(replay.points) / 3000))
    indices = {0, len(replay.points) - 1, *range(0, len(replay.points), step)}
    indices.update(peak["event_index"] + 1 for peak in replay.peaks.values())
    indices.add(replay.max_awaiting["event_index"] + 1)
    return [replay.points[index] for index in sorted(indices)]


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        replay = row.get("replay", {})
        device = replay.get("device")
        history = row["history"][device] if device is not None else None
        end = row["end"]["counters"]
        result.append(
            {
                "rank": row["rank"],
                "replay_status": row["peak_replay_status"],
                "start_pickle_bytes": row["raw_provenance"]["start"]["bytes"],
                "end_pickle_bytes": row["raw_provenance"]["end"]["bytes"],
                "start_segments": row["start"]["segment_count"],
                "end_segments": row["end"]["segment_count"],
                "start_blocks": row["start"]["block_count"],
                "end_blocks": row["end"]["block_count"],
                "history_events": history["end_count"] if history else None,
                "history_capacity": history["max_entries"] if history else None,
                "history_truncated_possible": history["capacity_reached"] if history else None,
                "endpoint_allocated_bytes": end["allocated_bytes"],
                "endpoint_active_bytes": end["active_bytes"],
                "endpoint_reserved_bytes": end["reserved_bytes"],
                "endpoint_inactive_cache_bytes": end["inactive_cache_bytes"],
                "allocated_delta_bytes": end["allocated_bytes"]
                - row["start"]["counters"]["allocated_bytes"],
                "reserved_delta_bytes": end["reserved_bytes"]
                - row["start"]["counters"]["reserved_bytes"],
                "peak_allocated_bytes": replay.get("peaks", {})
                .get("allocated_bytes", {})
                .get("bytes"),
                "peak_active_bytes": replay.get("peaks", {}).get("active_bytes", {}).get("bytes"),
                "max_awaiting_free_bytes": replay.get("max_awaiting_free", {}).get("bytes"),
                "endpoint_unknown_callsite_bytes": row["end"]["unknown_callsite_allocated_bytes"],
                "oom_events": replay.get("oom_events"),
                "remote_hash_match": "NOT_VERIFIED_BY_THIS_ANALYSIS",
            }
        )
    return result


def write_report(path: Path, rows: list[dict[str, Any]], capture_dir: Path) -> None:
    summary = summary_rows(rows)
    lines = [
        "# C0_guarded allocator snapshot audit",
        "",
        f"分析对象：`{capture_dir.as_posix()}` 的 start/end 四 rank memory snapshots。",
        "未读取 Nsight timeline 或其他实验原始 trace。单位 GiB = 2^30 B。",
        "",
        "| rank | 回放 | 历史事件/容量 | 实占峰值 GiB | 起/止实占 GiB "
        "| reserved GiB | 结束缓存 GiB | 等待释放峰值 GiB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def number(row: dict[str, Any], key: str) -> str:
        return f"{row[key] / GIB:.6f}" if row[key] is not None else "NOT_RUN"

    for row in summary:
        lines.append(
            f"| {row['rank']} | {row['replay_status']} "
            f"| {row['history_events']}/{row['history_capacity']} "
            f"| {number(row, 'peak_allocated_bytes')} | {number(row, 'endpoint_allocated_bytes')} "
            f"| {number(row, 'endpoint_reserved_bytes')} "
            f"| {number(row, 'endpoint_inactive_cache_bytes')} "
            f"| {number(row, 'max_awaiting_free_bytes')} |"
        )
    lines.extend(
        [
            "",
            "`VERIFIED` 表示：完整分配历史从 start 的已知 live blocks 回放；",
            "结束时全部 block 的地址、大小、状态及活跃 request 大小与 end 一致；",
            "segment 地址、大小、stream、pool 以及 allocated/reserved "
            "结束值和峰值与独立 profiling 统计一致。",
            "inactive block 的 requested_size 可能是旧 request 元数据，"
            "因此不当作 live bytes，也不要求该字段相同。",
            "",
            "历史完整性依据是 start 精确为 end 的前缀、两者均未达到配置容量。",
            "这仅覆盖捕获窗口；不排除窗口以前发生的分配或长期其他步骤中的问题。",
            "历史 alloc/free size 是 requested bytes，"
            "回放按已核验的 native allocator 默认规则重建 storage bytes。",
            f"规则依据：[实际 Torch 2.9.1 对应提交的 allocator]({ALLOCATOR_SOURCE['url']})。",
            "自定义 rounding、finite max_split_size、expandable segments 会停止该类回放。",
            "",
            "reserved 是缓存池总容量，包含 live 和 inactive storage，不能用于推断活跃参数数量。",
            "active_awaiting_free 的 free_requested 只表示 Python/allocator 请求释放；",
            "free_completed 才表明跨流用途已完成并可重用。"
            "它本身不证明 GPU stream 的等待事件或正确 overlap。",
            "",
            "`held.start/end.rankN.csv` 为快照中实际持有的 block；",
            "`held.replayed_end.rankN.csv` 还记录分配代际，"
            "区分 start 持有和窗口中新建后仍持有的分配。",
            "地址可能重用，相同地址不意味着相同 Tensor。"
            "`held.peak_allocated.rankN.csv` 保存实占峰值处 live/pending blocks。",
            "未捕获源 frame 的归因明确为 `UNKNOWN`。"
            "捕获的 callsite 也只证明分配调用位置，不自动证明 Tensor 类型。",
            "",
            "所有 raw SHA 为本地读取计算值；本模块没有验证远端完整回传 manifest。",
            "直接由 NCCL/CUDA 分配的内存不在这些 allocator snapshots 中，"
            "故这些值不是整张卡的总物理显存使用。",
            "",
            "端点保持相同 allocated/reserved 可以支持当前一个窗口无实占增长；"
            "不能独立证明全部分片逻辑或长期无泄漏。",
            "性能、stream 与 overlap 结论应使用另行公告的 timeline 分析。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(output_dir: Path, curves: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, AttributeError) as exc:
        return {"status": "NOT_RUN", "reason": f"matplotlib import is unavailable: {exc}"}
    if not curves:
        return {"status": "NOT_RUN", "reason": "no VERIFIED allocator replays"}
    count = len(curves)
    columns = min(2, count)
    plot_rows = math.ceil(count / columns)
    figure, axes = plt.subplots(plot_rows, columns, figsize=(12, 4 * plot_rows), squeeze=False)
    colors = {"allocated_bytes": "#1766a8", "active_bytes": "#d1752a", "reserved_bytes": "#777777"}
    for axis, (rank, points) in zip(axes.flat, curves.items(), strict=False):
        x = [point["event_index"] for point in points]
        for key, label in (
            ("reserved_bytes", "reserved incl. cache"),
            ("active_bytes", "active incl. pending"),
            ("allocated_bytes", "allocated"),
        ):
            axis.plot(
                x,
                [point[key] / GIB for point in points],
                label=label,
                color=colors[key],
                linewidth=1.2,
                linestyle="--" if key == "active_bytes" else "-",
            )
        axis.set_title(f"rank {rank}")
        axis.set_xlabel("Allocator event index (not GPU time)")
        axis.set_ylabel("GiB")
        axis.grid(alpha=0.2)
        axis.set_ylim(bottom=0)
    for axis in list(axes.flat)[count:]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.96))
    figure.suptitle("C0_guarded: verified native CUDA allocator replay", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_dir / "allocator_replay.png", dpi=160)
    figure.savefig(output_dir / "allocator_replay.svg")
    plt.close(figure)
    return {
        "status": "CREATED",
        "files": ["allocator_replay.png", "allocator_replay.svg"],
        "axis": "allocator event index; not GPU execution time",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--capture-dir", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    result.add_argument("--max-snapshot-mib", type=int, default=256)
    result.add_argument("--inspect-only", action="store_true")
    result.add_argument("--plots", action="store_true")
    result.add_argument(
        "--version-evidence", type=Path, help="Installed source-audit JSON with torch_git_commit"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.ranks or len(set(args.ranks)) != len(args.ranks) or min(args.ranks) < 0:
        raise SnapshotError("ranks must be unique nonnegative integers")
    if args.max_snapshot_mib < 1:
        raise SnapshotError("maximum snapshot size must be positive")
    if args.output_dir.exists():
        raise SnapshotError(f"refusing existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    version_evidence = None
    if args.version_evidence:
        source_bytes = args.version_evidence.read_bytes()
        source = json.loads(source_bytes)
        if source.get("torch_git_commit") != ALLOCATOR_SOURCE["torch_git_version"]:
            raise SnapshotError("source-audit commit does not match pinned allocator semantics")
        version_evidence = {
            "path": str(args.version_evidence),
            "bytes": len(source_bytes),
            "sha256_local": hashlib.sha256(source_bytes).hexdigest(),
            "torch_version": source.get("torch_version"),
            "torch_git_commit": source["torch_git_commit"],
            "scope": "same installed Torch environment source audit; not a remote file hash check",
        }
    rows = []
    curves = {}
    for rank in args.ranks:
        row, _snapshots = inspect_rank(args.capture_dir, rank, args.max_snapshot_mib * 1024**2)
        if (
            version_evidence
            and row["environment"].get("torch") != version_evidence["torch_version"]
        ):
            raise SnapshotError(f"rank {rank}: source-audit Torch version differs from capture")
        if not args.inspect_only:
            row["replay"], replay = replay_rank(row, *_snapshots)
            row["peak_replay_status"] = row["replay"]["status"]
            device = (
                row["start"]["devices_with_segments"][0]
                if len(row["start"]["devices_with_segments"]) == 1
                else None
            )
            if device is not None:
                write_csv(
                    args.output_dir / f"held.start.rank{rank}.csv",
                    block_rows(_snapshots[0], device),
                )
                write_csv(
                    args.output_dir / f"held.end.rank{rank}.csv", block_rows(_snapshots[1], device)
                )
            if replay is not None:
                write_csv(args.output_dir / f"allocator_events.rank{rank}.csv", replay.points)
                write_csv(args.output_dir / f"held.replayed_end.rank{rank}.csv", replay.live_rows())
                write_csv(
                    args.output_dir / f"held.peak_allocated.rank{rank}.csv",
                    replay.peak_allocated_blocks,
                )
                row["replay"]["held_peak_summary"] = held_summary(replay.peak_allocated_blocks)
                row["replay"]["held_replayed_end_summary"] = held_summary(replay.live_rows())
                if row["peak_replay_status"] == "VERIFIED":
                    curves[rank] = sample_points(replay)
        rows.append(row)
        write_json(args.output_dir / f"memory.rank{rank}.json", row)
        del _snapshots
    manifest = {
        "status": "INSPECTED"
        if args.inspect_only
        else "VERIFIED"
        if all(row["peak_replay_status"] == "VERIFIED" for row in rows)
        else "UNVERIFIED",
        "capture_dir": str(args.capture_dir),
        "ranks": args.ranks,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "argv": sys.argv,
        "rank_files": [f"memory.rank{rank}.json" for rank in args.ranks],
        "remote_hash_match": "NOT_VERIFIED_BY_THIS_ANALYSIS",
        "source_semantics": ALLOCATOR_SOURCE,
        "installed_version_evidence": version_evidence or "NOT_PROVIDED",
        "plots": write_plots(args.output_dir, curves) if args.plots else {"status": "NOT_RUN"},
    }
    write_csv(args.output_dir / "summary.csv", summary_rows(rows))
    write_report(args.output_dir / "README.md", rows, args.capture_dir)
    write_json(args.output_dir / "analysis_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(args.output_dir),
                "ranks": [
                    {
                        "rank": row["rank"],
                        "status": row["peak_replay_status"],
                        "peaks": row.get("replay", {}).get("peaks"),
                        "endpoint": row["end"]["counters"],
                    }
                    for row in rows
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if manifest["status"] in ("INSPECTED", "VERIFIED") else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SnapshotError, OSError, json.JSONDecodeError) as exc:
        print(f"memory analysis failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
