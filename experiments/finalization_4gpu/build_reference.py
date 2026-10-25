"""Extend the existing locked AG/RS build with its same-source AllReduce target.

The historical build manifest is immutable. A fresh manifest records the added
target, exact binary hash and actual ldd selection for the embedding-gradient test.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

from .cli import configure_owner, load, save, sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--build-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError("Choose a fresh immutable extended build manifest")
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "experiments/fsdp_interview/scripts")
    )
    from build_nccl_tests import nccl_link_path, version_probe

    build = copy.deepcopy(load(args.build_manifest))
    if build["status"] != "PASS" or os.environ.get("LD_PRELOAD"):
        raise RuntimeError("A passed locked build without LD_PRELOAD is required")
    actual = version_probe(Path(build["config"]["nccl_home"]), build["config"]["expected_nccl"])
    if actual["library_sha256"] != build["nccl"]["library_sha256"]:
        raise RuntimeError("NCCL library changed")
    owner = configure_owner(args)
    binary = Path(build["binaries"]["all_gather_perf"]["path"]).with_name("all_reduce_perf")
    old_targets = {item["path"] for item in build["binaries"].values()}
    command = [value for value in build["build_command"] if value not in old_targets] + [
        str(binary)
    ]
    environment = {"LD_LIBRARY_PATH": build["link_environment"]["LD_LIBRARY_PATH"]}
    result = owner.run_stage(
        "nccl_build_all_reduce", command, timeout=args.timeout, extra_env=environment
    )
    if result["status"] != "passed" or not binary.is_file():
        raise RuntimeError("AllReduce target did not build")
    linked = owner.run_stage(
        "nccl_build_all_reduce_ldd", ["ldd", str(binary)], timeout=60, extra_env=environment
    )
    text = (Path(args.run_root) / "logs/nccl_build_all_reduce_ldd/stdout.log").read_text()
    if linked["status"] != "passed" or str(nccl_link_path(text)) != actual["library"]:
        raise RuntimeError("AllReduce links a different NCCL library")
    build["binaries"][binary.name] = {
        "path": str(binary),
        "sha256": sha(binary),
        "ldd": text,
        "resolved_nccl_library": actual["library"],
    }
    build.update(
        parent_build_manifest=str(args.build_manifest),
        parent_build_manifest_sha256=sha(args.build_manifest),
        extra_build_command=command,
        extra_stages=[result, linked],
    )
    save(args.output, build)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
