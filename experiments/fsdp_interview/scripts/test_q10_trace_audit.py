"""CPU negative checks of Q10 attribution; no actual trace or GPU is read."""

from __future__ import annotations

import copy
import json

import analyze_q10_trace as audit


def main() -> int:
    namespace = (1 << 24) + 1234
    kernel = {
        "pid": namespace,
        "launch_apis": [{"start": 15, "end": 16}],
        "kernel_name": "ncclDevKernel_ReduceScatter_Sum_f32_RING_LL",
        "kernel_rowid": 1,
        "stream_id": 27,
        "device_id": 0,
        "correlation_id": 1,
        "gpu_start_ns": 30,
        "gpu_end_ns": 40,
        "duration_ms": 1e-5,
        "enclosing_nvtx_labels": [],
    }
    scopes = [
        {
            "pid": namespace,
            "scope_id": 1,
            "start": 10,
            "end": 20,
            "label": "q10/microstep_0/backward",
        },
        {"pid": 1234, "scope_id": 2, "start": 10, "end": 20, "label": "q10/microstep_1/backward"},
    ]
    step = {"pid": namespace, "scope_id": 3, "step": 0, "start_ns": 0, "end_ns": 50}
    data = {"kernels": [kernel], "auxiliary_nvtx_ranges": scopes, "step_windows": [step]}
    rows = audit.collect_operations(data, {namespace: 0})
    assert rows[0]["microstep"] == 0 and rows[0]["phase"] == "backward"
    assert rows[0]["pg_role"] == "UNKNOWN" and rows[0]["target_parameter_unit"] == "UNKNOWN"
    assert not audit.summarize(rows, [])["steps"][0]["accumulation_trace_check"]["pass"]
    ambiguous = copy.deepcopy(data)
    ambiguous["auxiliary_nvtx_ranges"].append({**scopes[0], "scope_id": 4})
    assert audit.collect_operations(ambiguous, {namespace: 0})[0]["microstep"] is None
    unknown = copy.deepcopy(data)
    unknown["kernels"][0]["launch_apis"] = []
    assert audit.collect_operations(unknown, {namespace: 0})[0]["microstep"] is None
    records = [
        {"api": {"start": 1, "end": 2}, "streamId": 17},
        {"api": {"start": 4, "end": 5}, "streamId": 16},
    ]
    assert audit.known_record_predecessor(records, 3)[0]["streamId"] == 17
    assert audit.known_record_predecessor(records, 6)[0]["streamId"] == 16
    assert audit.known_record_predecessor(records, 0)[1].startswith("UNKNOWN")
    assert audit.known_record_predecessor(records, 4)[1] == "AMBIGUOUS_CONCURRENT_RECORD"
    overlap = [
        {"api": {"start": 0, "end": 10}, "streamId": 17},
        {"api": {"start": 1, "end": 2}, "streamId": 16},
        {"api": {"start": 4, "end": 5}, "streamId": 20},
    ]
    assert audit.known_record_predecessor(overlap, 6)[1] == "AMBIGUOUS_CONCURRENT_RECORD"
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "synthetic CPU trace attribution only",
                "checks": [
                    "VM namespace separation",
                    "unique API scope containment",
                    "unknown role preserved",
                    "first-micro RS violation detected",
                    "record generation reuse/missing/inflight "
                    "and older overlapping record rejected",
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
