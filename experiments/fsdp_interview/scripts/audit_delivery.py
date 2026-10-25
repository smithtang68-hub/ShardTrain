"""Connect used Windows raw artifacts to the verified immutable server package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(8 * 2**20):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--working-root", type=Path, required=True)
    parser.add_argument("--source-before", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    canonical, working = args.canonical_root.resolve(), args.working_root.resolve()
    remote = json.loads((canonical / "remote_artifact_manifest.json").read_text())
    returned = json.loads((canonical / "return_download_manifest.json").read_text())
    assert returned["remote_hash_verification"]["status"] == "passed"
    files = {row["path"]: row for row in remote["files"]}
    local_files = {row["path"]: row for row in returned["files"]}
    assert all(
        local_files[name]["sha256"] == row["sha256"] and local_files[name]["bytes"] == row["bytes"]
        for name, row in files.items()
    )
    inventory = json.loads((canonical / "suite_final_inventory.json").read_text())
    assert inventory["counts"] == {"nsys_rep": 21, "sqlite": 21, "memory_pickle": 150}
    verified = []
    for suffix in ("*.nsys-rep", "*.sqlite", "*.pickle"):
        for path in sorted(working.rglob(suffix)):
            relative = path.relative_to(working)
            if relative.parts[0] == "analysis":
                continue
            name = "results/" + relative.as_posix()
            assert name in files, f"Working raw artifact lacks remote identity: {path}"
            row = files[name]
            assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], path
            verified.append({"working_path": str(path), "remote_path": name, "status": "PASS"})
    before = json.loads(args.source_before.read_text())["files"]
    code_rows = []
    for name, row in before.items():
        if name.split("/", 1)[0] not in {"src", "scripts", "configs", "tests"}:
            continue
        observed = digest(args.project_root / name)
        assert observed == row["sha256"], f"Production code changed: {name}"
        code_rows.append(
            {
                "path": name,
                "sha256_before": row["sha256"],
                "sha256_after": observed,
                "status": "PASS",
            }
        )
    assert len(code_rows) == 114
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "delivery_audit.json",
        "production_code_scope_audit_final.json",
        "trace_inventory.csv",
    ):
        assert not (out / name).exists(), f"Preserve prior audit: {name}"
    payload = {
        "status": "PASS",
        "canonical_root": str(canonical),
        "verified_regular_files": len(local_files),
        "remote_manifest_files": len(files),
        "original_symlinks": remote["original_symlink_count"],
        "counts": inventory["counts"],
        "working_raw_artifacts_verified": verified,
        "note": "Archive members checked by unpacker; working raw copies independently rehashed. "
        "Derived analysis is separate from immutable raw server artifacts.",
    }
    (out / "delivery_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "production_code_scope_audit_final.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "checks": len(code_rows),
                "rows": code_rows,
                "scope": "Compared with task-start v2 dirty source manifest; "
                "prior changes retained.",
            },
            indent=2,
        )
        + "\n"
    )
    with (out / "trace_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "remote_path",
                "canonical_path",
                "working_path",
                "bytes",
                "sha256",
                "status",
            ],
        )
        writer.writeheader()
        for name in inventory["profiles"]:
            alias = working / Path(name).relative_to("results")
            row = files[name]
            writer.writerow(
                {
                    "remote_path": name,
                    "canonical_path": str(canonical / name),
                    "working_path": str(alias) if alias.is_file() else "",
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                    "status": "PASS",
                }
            )
    print(
        json.dumps(
            {
                "status": "PASS",
                "verified_regular_files": len(local_files),
                "working_raw_verified": len(verified),
                "production_files_unchanged": 114,
            }
        )
    )


if __name__ == "__main__":
    main()
