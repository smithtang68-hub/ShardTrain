"""CPU safeguards for Q10 history capacity and genuinely reported peaks."""

from __future__ import annotations

import copy
import json

import analyze_memory as memory
import analyze_q10_memory as audit
from test_memory_audit import fixture


def main() -> int:
    start = fixture()
    end = copy.deepcopy(start)
    end["device_traces"][0] = [{"action": "snapshot"}]
    assert audit.history_guards(start, end, 2)[0]["replay_candidate"]
    assert not audit.history_guards(start, end, 1)[0]["replay_candidate"]
    changed_start = copy.deepcopy(start)
    changed_start["device_traces"][0] = [{"action": "different"}]
    assert not audit.history_guards(changed_start, end, 2)[0]["replay_candidate"]
    replay = memory.NativeReplay(start, 0)
    steps = [
        {"phase": "measured", "peak_allocated_bytes": 512, "peak_reserved_bytes": 2 * 1024**2}
        for _ in range(2)
    ]
    summary = {
        "measured_steps": 2,
        "max_allocated_bytes": 512,
        "max_reserved_bytes": 2 * 1024**2,
    }
    checks = audit.reported_peak_checks(steps, summary, replay)
    assert all(row["summary_matches_steps"] and row["summary_matches_replay"] for row in checks)
    bad_summary = {**summary, "max_allocated_bytes": 513}
    checks = audit.reported_peak_checks(steps, bad_summary, replay)
    assert not checks[0]["summary_matches_steps"]
    assert not checks[0]["summary_matches_replay"]
    bad_steps = copy.deepcopy(steps)
    bad_steps[0]["peak_allocated_bytes"] = 1024
    checks = audit.reported_peak_checks(bad_steps, summary, replay)
    assert not checks[0]["summary_matches_steps"]
    settings = {
        "expandable_segments": False,
        "max_split_size": -1,
        "roundup_power2_divisions": {"1": 0},
        "PYTORCH_CUDA_ALLOC_CONF": "",
    }
    assert audit.supported_settings(settings, settings)
    assert not audit.supported_settings(settings, {**settings, "expandable_segments": True})
    assert not audit.supported_settings(
        settings, {**settings, "roundup_power2_divisions": {"1": 2}}
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "checks": [
                    "exact history prefix required",
                    "full bounded history cannot prove no truncation",
                    "real summary/step peaks must independently equal replay",
                    "unsupported allocator semantics rejected",
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
