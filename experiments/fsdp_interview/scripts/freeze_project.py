"""Freeze the current dirty working tree with exact hashes; no credentials or old data."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--version", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for folder in ["src", "scripts", "configs", "tests", "docs", "experiments/fsdp_interview"]:
        paths.extend(
            x
            for x in (root / folder).rglob("*")
            if x.is_file()
            and "__pycache__" not in x.parts
            and "results" not in x.parts
            and x.suffix not in {".pyc", ".pickle", ".nsys-rep", ".sqlite", ".npy"}
        )
    paths.extend(root.glob("*.md"))
    paths.extend(root / x for x in ["pyproject.toml", ".gitignore"])
    snapshot = {x.relative_to(root).as_posix(): x.read_bytes() for x in sorted(set(paths))}
    manifest = {
        rel: {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for rel, raw in snapshot.items()
    }
    commit = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    record = out / f"source_manifest_{args.version}.json"
    if record.exists():
        raise FileExistsError("Do not overwrite an earlier source freeze")
    record.write_text(
        json.dumps(
            {
                "git_commit": commit,
                "git_status": status,
                "version": args.version,
                "files": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    archive = out / f"project_{args.version}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for rel in manifest:
            entry = tarfile.TarInfo("project/" + rel)
            entry.size = len(snapshot[rel])
            bundle.addfile(entry, io.BytesIO(snapshot[rel]))
        bundle.add(record, arcname=f"source_manifest_{args.version}.json", recursive=False)
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "source_files": len(manifest),
                "git_commit": commit,
            }
        )
    )


if __name__ == "__main__":
    main()
