"""CPU checks of the snapshot loader and allocator lifecycle audit safeguards."""

from __future__ import annotations

import copy
import json
import pickle
import tempfile
from pathlib import Path

import analyze_memory as audit


def rejects(operation) -> None:
    try:
        operation()
    except audit.SnapshotError:
        return
    raise AssertionError("unsafe or inconsistent input was accepted")


def fixture() -> dict:
    return {
        "segments": [
            {
                "device": 0,
                "address": 4096,
                "stream": 7,
                "total_size": 2 * 1024**2,
                "allocated_size": 512,
                "active_size": 512,
                "requested_size": 1,
                "segment_type": "small",
                "blocks": [
                    {
                        "address": 4096,
                        "size": 512,
                        "requested_size": 1,
                        "state": "active_allocated",
                        "frames": [],
                    },
                    {
                        "address": 4608,
                        "size": 2 * 1024**2 - 512,
                        "requested_size": 0,
                        "state": "inactive",
                        "frames": [],
                    },
                ],
            }
        ],
        "device_traces": [[]],
    }


def event(action: str, address: int, size: int, stream: int = 7) -> dict:
    return {"action": action, "addr": address, "size": size, "stream": stream, "frames": []}


def main() -> int:
    primitives = {"a": [None, True, False, -1, 3.5, "test"], "b": (1, 2)}
    for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
        payload = pickle.dumps(primitives, protocol=protocol)
        assert audit.primitive_loads(payload) == primitives
        rejects(lambda payload=payload: audit.primitive_loads(payload + b"."))
    assert audit.primitive_loads(pickle.dumps(b"bytes", protocol=4)) == b"bytes"
    # These payloads are rejected by prescan before any Python object lookup,
    # reduction or persistent reference could execute. No command is embedded.
    for payload in (
        b"cbuiltins\nstr\n.",
        b"\x80\x04\x93.",
        b"\x80\x04)R.",
        b"\x80\x04)\x81.",
        b"Punsafe-reference\n.",
    ):
        rejects(lambda payload=payload: audit.primitive_loads(payload))
    start = fixture()
    assert audit.validate_snapshot(start)["counters"]["allocated_bytes"] == 512
    malformed = copy.deepcopy(start)
    malformed["segments"][0]["allocated_size"] += 1
    rejects(lambda: audit.validate_snapshot(malformed))
    malformed = copy.deepcopy(start)
    malformed["segments"][0]["blocks"][0]["state"] = "unknown"
    rejects(lambda: audit.validate_snapshot(malformed))
    replay = audit.NativeReplay(start, 0)
    replay.consume(event("snapshot", 0, 512, 0), 0)
    replay.consume(event("alloc", 4608, 4), 1)
    assert replay.counters["allocated_bytes"] == 1024
    replay.consume(event("free_requested", 4608, 4), 2)
    assert replay.counters["allocated_bytes"] == 512
    assert replay.counters["active_bytes"] == 1024
    rejects(lambda: replay.consume(event("alloc", 4608, 4), 3))
    replay.consume(event("free_completed", 4608, 4), 3)
    endpoint = replay.compare_end(start)
    assert endpoint["all_block_addresses_sizes_states_live_requests_match"]
    assert endpoint["counter_match"]
    assert replay.peaks["allocated_bytes"]["bytes"] == 1024
    assert replay.max_awaiting["bytes"] == 512
    # A second allocation at the same address gets a new generation. Address
    # reuse cannot make a window allocation appear to survive from start.
    replay.consume(event("alloc", 4608, 1), 4)
    assert replay.live_rows()[-1]["allocation_origin"] == "WINDOW_ALLOC"
    replay.consume(event("free_requested", 4608, 1), 5)
    replay.consume(event("free_completed", 4608, 1), 6)
    rejects(lambda: replay.consume(event("free_completed", 4608, 1), 7))
    new_address = 4 * 1024**2
    replay.consume(event("segment_alloc", new_address, 2 * 1024**2, 9), 7)
    assert replay.counters["reserved_bytes"] == 4 * 1024**2
    replay.consume(event("alloc", new_address, 1, 9), 8)
    rejects(lambda: replay.consume(event("segment_free", new_address, 2 * 1024**2, 9), 9))
    replay.consume(event("free_requested", new_address, 1, 9), 9)
    replay.consume(event("free_completed", new_address, 1, 9), 10)
    replay.consume(event("segment_free", new_address, 2 * 1024**2, 9), 11)
    assert replay.compare_end(start)["counter_match"]
    with tempfile.TemporaryDirectory(prefix="shardtrain_memory_audit_") as directory:
        directory = Path(directory)
        profile = {
            "rank": 0,
            "memory_history_max_entries": 10,
            "memory": [
                {
                    "allocated": 512,
                    "reserved": 2 * 1024**2,
                    "max_allocated": 512,
                    "max_reserved": 2 * 1024**2,
                }
            ],
        }
        (directory / "profiling.rank0.json").write_text(json.dumps(profile), encoding="utf-8")
        (directory / "environment.rank0.json").write_text(
            json.dumps({"rank": 0, "local_rank": 0, "torch": "2.9.1+cu126"}),
            encoding="utf-8",
        )
        before, after = fixture(), fixture()
        before["device_traces"][0] = [event("snapshot", 0, 512, 0)]
        after["device_traces"][0] = [event("snapshot", 0, 512, 0)] * 10
        for label, value in (("start", before), ("end", after)):
            (directory / f"memory.{label}.rank0.pickle").write_bytes(pickle.dumps(value))
        row, snapshots = audit.inspect_rank(directory, 0, 1024**2)
        assert row["history"][0]["capacity_reached"]
        assert not row["history"][0]["replay_candidate"]
        assert audit.replay_rank(row, *snapshots)[0]["status"] == "NOT_RUN"
        after["device_traces"][0] = [event("snapshot", 0, 1024, 0)]
        (directory / "memory.end.rank0.pickle").write_bytes(pickle.dumps(after))
        row, snapshots = audit.inspect_rank(directory, 0, 1024**2)
        assert not row["history"][0]["start_exact_prefix_of_end"]
        assert not row["history"][0]["replay_candidate"]
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "CPU synthetic audit guards; no GPU execution",
                "checks": [
                    "primitive pickle protocols and execution opcodes",
                    "snapshot counters/states",
                    "history capacity and exact-prefix replay guards",
                    "storage rounding, pending release, merge, generation reuse "
                    "and segment lifecycle",
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
