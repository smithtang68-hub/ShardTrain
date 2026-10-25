"""Read-only OSRT evidence inside exact same-thread primary pinned setup NVTX."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("preserve old derived report; choose fresh output directory")
    report = json.loads(args.trace_analysis.read_text())
    primary = []
    for api in report["host_allocation_apis"]:
        scopes = [
            scope
            for scope in api.get("enclosing_nvtx", [])
            if scope["label"].startswith("q01q03/pinned_preallocate/bytes=")
        ]
        if len(scopes) == 1:
            primary.append((api, scopes[0]))
    if len(primary) != 1:
        raise ValueError("primary same-thread setup API missing/ambiguous; do not infer costs")
    api, scope = primary[0]
    sqlite_path = Path(report["source_sqlite"]).resolve()
    database = sqlite3.connect(sqlite_path.as_uri() + "?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA query_only=ON")
    try:
        tables = {
            row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not {"OSRT_API", "OSRT_CALLCHAINS", "StringIds"} <= tables:
            raise ValueError("OSRT evidence not captured; no startup attribution")
        strings = {
            row["id"]: row["value"] for row in database.execute("SELECT id,value FROM StringIds")
        }
        rows = [
            dict(row)
            for row in database.execute(
                "SELECT rowid AS rowid,* FROM OSRT_API WHERE globalTid=? AND start>=? AND end<=?",
                (api["globalTid"], scope["start"], scope["end"]),
            )
        ]
        for row in rows:
            row["name"] = strings.get(row["nameId"], "UNKNOWN")
            row["duration_ms"] = (row["end"] - row["start"]) / 1e6
            row["enclosing_setup_scope"] = scope
            row["callchain"] = [
                {
                    **dict(frame),
                    "symbol_name": strings.get(frame["symbol"], "UNKNOWN"),
                    "module_name": strings.get(frame["module"], "UNKNOWN"),
                }
                for frame in database.execute(
                    "SELECT * FROM OSRT_CALLCHAINS WHERE id=? ORDER BY stackDepth",
                    (row["callchainId"],),
                )
            ]
    finally:
        database.close()
    payload = {
        "source_sqlite": str(sqlite_path),
        "source_trace_analysis": str(args.trace_analysis),
        "primary_cuda_api": api,
        "primary_nvtx": scope,
        "main_thread_osrt_calls": rows,
        "counts": dict(Counter(row["name"] for row in rows)),
        "time_before_primary_cuda_api_ms": (api["start"] - scope["start"]) / 1e6,
        "limits": [
            "Only main-thread OSRT API fully enclosed by exact pinned setup NVTX is attributed.",
            "Callchain symbols support code path attribution, not unobserved CPU instruction time.",
            "CUDA API duration excludes malloc, threadpool startup and page prefault costs.",
            "Single cold instrumented allocation is not steady allocator speed ranking.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "setup_osrt_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    fields = [
        "rowid",
        "name",
        "start",
        "end",
        "duration_ms",
        "globalTid",
        "callchainId",
        "callchain",
    ]
    with (args.output_dir / "setup_osrt_calls.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key]) if isinstance(row[key], list) else row[key]
                    for key in fields
                }
            )
    print(
        json.dumps(
            {
                "source_sqlite": str(sqlite_path),
                "counts": payload["counts"],
                "time_before_primary_cuda_api_ms": payload["time_before_primary_cuda_api_ms"],
            }
        )
    )


if __name__ == "__main__":
    main()
