"""Equivalent exact trace parser with indexed same-thread NVTX containment.

This is offline acceleration only. The frozen parser and GPU experiment source
are unchanged. A temporary Python function wrapper is restored after analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.fsdp_interview.scripts import parse_trace  # noqa: E402
from experiments.fsdp_interview.scripts.analyze_gemm_trace import ScopeIndex  # noqa: E402


class IndexedScopes:
    def __init__(self):
        self.cache = {}

    def __call__(self, scopes_by_tid, api, *, communication_only=False):
        key = id(scopes_by_tid)
        if key not in self.cache:
            # Retain the dictionary identity to prevent id reuse across maps.
            self.cache[key] = (
                scopes_by_tid,
                {tid: ScopeIndex(scopes) for tid, scopes in scopes_by_tid.items()},
                {
                    id(row): ordinal
                    for scopes in scopes_by_tid.values()
                    for ordinal, row in enumerate(scopes)
                },
            )
        index = self.cache[key][1].get(api.get("globalTid"))
        if index is None:
            return []
        values = index.containing(api["start"], api["end"])
        values.sort(key=lambda row: self.cache[key][2][id(row)])
        return [
            row
            for row in values
            if not communication_only or row["label"].startswith(parse_trace.COMM_PREFIXES)
        ]


def analyze(path, event_paths=None, rank_pids=None):
    original = parse_trace._enclosing_scopes
    try:
        parse_trace._enclosing_scopes = IndexedScopes()
        result = parse_trace.analyze(path, event_paths, rank_pids)
    finally:
        parse_trace._enclosing_scopes = original
    result["parser_provenance"] = {
        "base_parser_sha256": hashlib.sha256(Path(parse_trace.__file__).read_bytes()).hexdigest(),
        "offline_acceleration_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interval_index_sha256": hashlib.sha256(
            Path(sys.modules[ScopeIndex.__module__].__file__).read_bytes()
        ).hexdigest(),
        "semantics": "same-thread complete API containment; original scope_id order retained",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--events", nargs="*")
    parser.add_argument("--rank-pids", help="JSON object mapping rank to PID")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve previous trace analysis; choose a fresh output directory")
    rank_pids = json.loads(Path(args.rank_pids).read_text()) if args.rank_pids else None
    result = analyze(args.sqlite, args.events, rank_pids)
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "step_metrics",
        "collectives",
        "group_skew",
        "stream_distribution",
        "gpu_idle_gaps",
        "pinned_transfer_timelines",
        "host_allocation_apis",
    ):
        parse_trace._csv(output / f"{name}.csv", result[name])
    (output / "trace_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_sqlite": result["source_sqlite"],
                "steps": len(result["step_windows"]),
                "collectives": len(result["collectives"]),
                "matched_collectives": sum(
                    row["trace_status"] == "MATCHED" for row in result["collectives"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
