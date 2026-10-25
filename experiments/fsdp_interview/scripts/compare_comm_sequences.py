"""Compare captured ordered mathematical communication metadata across traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SIGNATURE_FIELDS = (
    "rank",
    "process_group_label",
    "group_members",
    "collective_sequence",
    "collective_type",
    "family",
    "unit_id",
    "phase",
    "runtime_microstep",
    "input_bytes",
    "output_bytes",
    "input_shape",
    "output_shapes",
    "dtype",
    "group_role",
)


def signature(data):
    by_rank = {}
    for row in sorted(
        data["collectives"], key=lambda value: (value["rank"], value["host_ordinal"])
    ):
        by_rank.setdefault(row["rank"], []).append({key: row.get(key) for key in SIGNATURE_FIELDS})
    return by_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Fresh output directory required: {output}")
    reference = None
    reports = []
    for path in args.trace_json:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        own = signature(data)
        if reference is None:
            reference = own
        ranks = []
        for rank in sorted(set(own) | set(reference)):
            rows, expected = own.get(rank, []), reference.get(rank, [])
            mismatch = [
                {"ordinal_index": index, "actual": row, "reference": baseline}
                for index, (row, baseline) in enumerate(zip(rows, expected, strict=False))
                if row != baseline
            ]
            ranks.append(
                {
                    "rank": rank,
                    "actual_count": len(rows),
                    "reference_count": len(expected),
                    "status": "PASS" if rows == expected else "FAIL",
                    "signature_sha256": hashlib.sha256(
                        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "differences": mismatch,
                }
            )
        reports.append(
            {
                "source_analysis": str(Path(path).resolve()),
                "source_sqlite": data["source_sqlite"],
                "status": "PASS" if all(row["status"] == "PASS" for row in ranks) else "FAIL",
                "rank_results": ranks,
            }
        )
    result = {
        "status": "PASS" if reports and all(row["status"] == "PASS" for row in reports) else "FAIL",
        "signature_fields": SIGNATURE_FIELDS,
        "reference_analysis": str(Path(args.trace_json[0]).resolve()),
        "traces": reports,
        "scope_limit": "Metadata/count/shape/order equality; no numerical or GPU execution proof",
    }
    output.mkdir(parents=True)
    (output / "communication_sequence_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "traces": len(reports)}))


if __name__ == "__main__":
    main()
