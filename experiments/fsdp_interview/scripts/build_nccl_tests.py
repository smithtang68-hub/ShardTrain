"""Build only official nccl-tests AG/RS against the exact pip NCCL library.

All external stages use owned_stage.run_stage. This script creates a NEW build
directory under the current run root and never modifies an existing installation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path

import owned_stage

OFFICIAL_REPO = "https://github.com/NVIDIA/nccl-tests.git"
DEFAULT_NCCL_HOME = (
    "/root/shardtrain_qwen/20261011_run01/venv/lib/python3.12/site-packages/nvidia/nccl"
)


def digest(path: Path) -> str:
    hash_value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            hash_value.update(block)
    return hash_value.hexdigest()


def configure_owner(run_root: str, project_dir: str | None) -> Path:
    run = Path(run_root).resolve()
    project = Path(project_dir).resolve() if project_dir else run / "project"
    if not project.is_dir() or not project.is_relative_to(run):
        raise ValueError("Project must exist inside this run root")
    owned_stage.BASE, owned_stage.PROJECT = run, project
    return run


def inside(path: str | Path, run: Path) -> Path:
    value = Path(path).resolve()
    if not value.is_relative_to(run) or value == run:
        raise ValueError(f"Artifact path must be inside current run root: {value}")
    if any(char in str(value) for char in (" ", "\n", "\r", "$", "`", ";")):
        raise ValueError("Build/link paths must not contain make/shell metacharacters or spaces")
    return value


def version_probe(nccl_home: Path, expected: str) -> dict:
    header = nccl_home / "include" / "nccl.h"
    library = (nccl_home / "lib" / "libnccl.so.2").resolve(strict=True)
    text = header.read_text(encoding="utf-8")
    parts = []
    for field in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(r"^\s*#\s*define\s+NCCL_" + field + r"\s+(\d+)", text, re.M)
        if not match:
            raise RuntimeError(f"NCCL header missing version {field}")
        parts.append(int(match.group(1)))
    expected_parts = list(map(int, expected.split(".")))
    if parts != expected_parts or len(parts) != 3:
        raise RuntimeError(f"NCCL headers {parts} != required {expected_parts}")
    loaded = ctypes.CDLL(str(library))
    loaded.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    loaded.ncclGetVersion.restype = ctypes.c_int
    value = ctypes.c_int()
    code = loaded.ncclGetVersion(ctypes.byref(value))
    wanted = parts[0] * 10000 + parts[1] * 100 + parts[2]
    if code != 0 or value.value != wanted:
        raise RuntimeError(f"Loaded NCCL version {value.value}, error {code}, wanted {wanted}")
    return {
        "header": str(header),
        "header_sha256": digest(header),
        "header_version": parts,
        "library": str(library),
        "library_sha256": digest(library),
        "library_bytes": library.stat().st_size,
        "nccl_get_version": value.value,
    }


def nccl_link_path(text: str) -> Path:
    matches = re.findall(r"^\s*libnccl\.so(?:\.\d+)*\s+=>\s+(\S+)", text, re.M)
    if len(matches) != 1 or matches[0] == "not":
        raise RuntimeError(f"Cannot verify exactly one NCCL shared library in ldd output: {text}")
    return Path(matches[0]).resolve(strict=True)


def stage(name: str, command: list, timeout: int, environment: dict | None = None):
    record = owned_stage.run_stage(name, command, timeout=timeout, extra_env=environment)
    if record["status"] != "passed":
        raise RuntimeError(f"Owned stage failed: {name}")
    return record, owned_stage.BASE / "logs" / name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--project-dir")
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--nccl-home", default=DEFAULT_NCCL_HOME)
    parser.add_argument("--cuda-home", default="/usr/local/cuda-12.4")
    parser.add_argument("--tag", default="v2.15.0")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-nccl", default="2.27.5")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--stage-prefix", default="nccl_tests_build")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("Owned CUDA build stages require Linux")
    if not re.fullmatch(r"v\d+\.\d+\.\d+", args.tag):
        raise ValueError("Tag must be a fixed released version")
    if args.expected_commit and not re.fullmatch(r"[a-fA-F0-9]{40}", args.expected_commit):
        raise ValueError("Expected commit must be a full SHA1")
    run = configure_owner(args.run_root, args.project_dir)
    root = inside(args.build_root, run)
    if root.exists():
        raise FileExistsError(f"Refusing to replace existing build directory: {root}")
    root.mkdir(parents=True, exist_ok=False)
    manifest_path = root / "build_manifest.json"
    manifest = {"status": "BUILDING", "argv": sys.argv, "config": vars(args), "stages": []}
    owned_stage.write_json(manifest_path, manifest)
    try:
        nccl = Path(args.nccl_home).resolve(strict=True)
        cuda = Path(args.cuda_home).resolve(strict=True)
        for value in (nccl, cuda):
            if any(char in str(value) for char in (" ", "\n", "\r", "$", "`", ";")):
                raise ValueError("CUDA/NCCL paths contain unsupported make/shell metacharacters")
        nvcc = cuda / "bin" / "nvcc"
        if not nvcc.is_file():
            raise FileNotFoundError(nvcc)
        if os.environ.get("LD_PRELOAD"):
            raise RuntimeError("LD_PRELOAD is set; cannot guarantee the requested NCCL library")
        manifest["nccl"] = version_probe(nccl, args.expected_nccl)
        library_dir = str(Path(manifest["nccl"]["library"]).parent)
        environment = {
            "LD_LIBRARY_PATH": os.pathsep.join(
                filter(
                    None, (library_dir, str(cuda / "lib64"), os.environ.get("LD_LIBRARY_PATH", ""))
                )
            ),
            "LIBRARY_PATH": library_dir,
            "NCCL_DEBUG": "WARN",
            "GIT_TERMINAL_PROMPT": "0",
        }
        manifest["link_environment"] = environment
        record, log = stage(args.stage_prefix + "_nvcc", [nvcc, "--version"], 60, environment)
        manifest["stages"].append(record)
        manifest["nvcc_version"] = (log / "stdout.log").read_text()
        if not re.search(r"release\s+12\.4(?:,|\s)", manifest["nvcc_version"]):
            raise RuntimeError("This build fixture requires an actual CUDA 12.4 nvcc")
        record, log = stage(
            args.stage_prefix + "_resolve_tag",
            [
                "git",
                "ls-remote",
                "--tags",
                OFFICIAL_REPO,
                f"refs/tags/{args.tag}",
                f"refs/tags/{args.tag}^{{}}",
            ],
            180,
            environment,
        )
        manifest["stages"].append(record)
        refs = dict(line.split()[::-1] for line in (log / "stdout.log").read_text().splitlines())
        commit = refs.get(f"refs/tags/{args.tag}^{{}}", refs.get(f"refs/tags/{args.tag}"))
        if not commit or (args.expected_commit and commit.lower() != args.expected_commit.lower()):
            raise RuntimeError(f"Released tag resolution mismatch: {commit}")
        source = root / "source"
        record, _ = stage(
            args.stage_prefix + "_clone",
            ["git", "clone", "--depth", "1", "--branch", args.tag, OFFICIAL_REPO, source],
            300,
            environment,
        )
        manifest["stages"].append(record)
        record, log = stage(
            args.stage_prefix + "_head", ["git", "-C", source, "rev-parse", "HEAD"], 60
        )
        manifest["stages"].append(record)
        head = (log / "stdout.log").read_text().strip()
        if head != commit:
            raise RuntimeError(f"Cloned HEAD {head} != pinned tag commit {commit}")
        record, log = stage(
            args.stage_prefix + "_tracked", ["git", "-C", source, "ls-files", "-z"], 60
        )
        manifest["stages"].append(record)
        files = [name for name in (log / "stdout.log").read_bytes().decode().split("\0") if name]
        source_hashes = []
        for name in files:
            path = (source / name).resolve(strict=True)
            if not path.is_relative_to(source):
                raise RuntimeError(f"Tracked source escapes checkout: {name}")
            source_hashes.append(
                {"path": name, "sha256": digest(path), "bytes": path.stat().st_size}
            )
        owned_stage.write_json(root / "source_manifest.json", source_hashes)
        manifest.update(
            official_repository=OFFICIAL_REPO,
            tag=args.tag,
            commit=head,
            source_manifest_sha256=digest(root / "source_manifest.json"),
            source_dir=str(source),
        )
        prefix = root / "nccl_prefix"
        prefix.mkdir()
        (prefix / "include").symlink_to(nccl / "include", target_is_directory=True)
        (prefix / "lib").mkdir()
        for name in ("libnccl.so", "libnccl.so.2"):
            (prefix / "lib" / name).symlink_to(manifest["nccl"]["library"])
        build = root / "bin"
        binaries = [build / f"{name}_perf" for name in ("all_gather", "reduce_scatter")]
        linkflags = (
            f"-L{cuda / 'lib64'} -lcudart -lrt -L{prefix / 'lib'} -lnccl "
            f"-Xlinker -rpath -Xlinker {library_dir} "
            f"-Xlinker -rpath -Xlinker {cuda / 'lib64'}"
        )
        command = [
            "make",
            "-C",
            source / "src",
            f"-j{args.jobs}",
            "MPI=0",
            "MPI_IBM=0",
            f"CUDA_HOME={cuda}",
            f"NCCL_HOME={prefix}",
            f"BUILDDIR={build}",
            "NVCC_GENCODE=-gencode=arch=compute_70,code=sm_70",
            f"NVLDFLAGS={linkflags}",
            *binaries,
        ]
        record, _ = stage(args.stage_prefix + "_make", command, args.timeout_seconds, environment)
        manifest["stages"].append(record)
        manifest["build_command"] = list(map(str, command))
        manifest["binaries"] = {}
        for binary in binaries:
            record, log = stage(
                args.stage_prefix + "_ldd_" + binary.stem, ["ldd", binary], 60, environment
            )
            manifest["stages"].append(record)
            text = (log / "stdout.log").read_text()
            actual = nccl_link_path(text)
            if str(actual) != manifest["nccl"]["library"]:
                raise RuntimeError(f"Binary loads unexpected NCCL: {actual}")
            manifest["binaries"][binary.name] = {
                "path": str(binary),
                "sha256": digest(binary),
                "ldd": text,
                "resolved_nccl_library": str(actual),
            }
        manifest["status"] = "PASS"
    except Exception as exc:
        manifest.update(status="FAIL", reason=str(exc), exception_type=type(exc).__name__)
        raise
    finally:
        owned_stage.write_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "build_manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
