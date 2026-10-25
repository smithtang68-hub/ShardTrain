"""Freeze dirty sources for the four-GPU finalization, without copying raw artifacts.

Example: python scripts/freeze_finalization.py --output-dir runs/finalization_4gpu/RUN --version v1
The source archive and all referenced bytes are captured before hashing again to
detect concurrent edits. This never commits, resets, pushes or removes files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

FOLDERS = (
    "src",
    "scripts",
    "configs",
    "tests",
    "docs",
    "reports",
    "experiments/fsdp_interview",
    "experiments/finalization_4gpu",
    ".github",
)
SKIP_DIRS = {
    "results",
    "runs",
    "data",
    "node_modules",
    "__pycache__",
    ".git",
    ".idea",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "artifacts",
}
SOURCE_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".csv", ".txt", ".json"}


def source_paths(root: Path) -> list[Path]:
    result = set()
    for folder in FOLDERS:
        start = root / folder
        if not start.exists():
            continue
        for base, directories, files in os.walk(start, followlinks=False):
            directories[:] = sorted(
                d
                for d in directories
                if d not in SKIP_DIRS
                and d not in {"build", "dist"}
                and not d.endswith((".egg-info", ".dist-info"))
            )
            for name in sorted(files):
                path = Path(base) / name
                if path.is_symlink() or path.suffix not in SOURCE_SUFFIXES:
                    continue
                # Reports carry human evidence, not large historical derived JSON.
                if folder == "reports" and path.suffix not in {".md", ".csv"}:
                    continue
                if path.stat().st_size > 4 * 1024 * 1024:
                    raise ValueError(f"Unexpected large source file: {path}")
                result.add(path)
    result.update(root.glob("*.md"))
    result.update(root / item for item in ("pyproject.toml", ".gitignore"))
    return sorted(result)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", *args],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout


def freeze(root: Path, output: Path, version: str, *, snapshot_namespace: str = "") -> dict:
    if not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError("Version must be v1, v2, ...")
    if snapshot_namespace and not re.fullmatch(r"[a-z0-9_]{1,64}", snapshot_namespace):
        raise ValueError("Snapshot namespace must use lowercase letters, numbers or underscores")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"source_manifest_{version}.json"
    archive_path = output / f"project_{version}.tar.gz"
    receipt_path = output / f"freeze_receipt_{version}.json"
    diff_path = output / f"tracked_diff_{version}.patch"
    for path in (manifest_path, archive_path, receipt_path, diff_path):
        if path.exists():
            raise FileExistsError(path)
    paths = source_paths(root)
    content = {p.relative_to(root).as_posix(): p.read_bytes() for p in paths}
    rows = {rel: {"bytes": len(value), "sha256": sha(value)} for rel, value in content.items()}
    head = git(root, "rev-parse", "HEAD").decode().strip()
    status = git(root, "status", "--short").decode(errors="replace")
    dirty_diff = git(root, "diff", "--binary", "HEAD")
    if paths != source_paths(root) or any(
        sha((root / rel).read_bytes()) != row["sha256"] for rel, row in rows.items()
    ):
        raise RuntimeError("Source changed during freeze; no snapshot was published")
    manifest = {
        "schema_version": 1,
        "version": version,
        "snapshot_id": (snapshot_namespace + "_" if snapshot_namespace else "")
        + output.name
        + "_"
        + version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": head,
        "git_branch": git(root, "branch", "--show-current").decode().strip(),
        "git_status": status,
        "tracked_diff_sha256": sha(dirty_diff),
        "scope": "Dirty source bytes; raw artifacts, credentials and input payloads excluded",
        "files": rows,
    }
    raw_manifest = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    with archive_path.open("xb") as file:
        with tarfile.open(fileobj=file, mode="w:gz") as bundle:
            for rel, raw in content.items():
                entry = tarfile.TarInfo("project/" + rel)
                entry.mode = 0o644
                entry.size = len(raw)
                bundle.addfile(entry, io.BytesIO(raw))
            for name, raw in ((manifest_path.name, raw_manifest), (diff_path.name, dirty_diff)):
                entry = tarfile.TarInfo(name)
                entry.size = len(raw)
                bundle.addfile(entry, io.BytesIO(raw))
    manifest_path.write_bytes(raw_manifest)
    diff_path.write_bytes(dirty_diff)
    receipt = {
        "status": "PASS",
        "snapshot_id": manifest["snapshot_id"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha(raw_manifest),
        "source_files": len(rows),
        "archive": str(archive_path),
        "archive_sha256": sha(archive_path.read_bytes()),
        "archive_bytes": archive_path.stat().st_size,
        "git_commit": head,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(freeze(root, Path(args.output_dir).resolve(), args.version), indent=2))


if __name__ == "__main__":
    main()
