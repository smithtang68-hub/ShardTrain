"""Portable thin launchers for the fixed four-GPU finalization matrix.

GPU commands are plan-only unless --execute is explicit. No shell strings, global
process kills, history deletion, device tuning, or logical mesh changes are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

from .placement import load, make_manifest, placement_for, validate_manifest, visible_order

PROJECT = Path(__file__).resolve().parents[2]
OLD = PROJECT / "experiments/fsdp_interview/scripts"
sys.path.insert(0, str(OLD))
MATRIX = {
    "S0": ("shardtrain", "M0"),
    "S1": ("shardtrain", "M1"),
    "N0": ("native", "M0"),
    "N1": ("native", "M1"),
}
ORDERS = [["S0", "N1", "S1", "N0"], ["N0", "S1", "N1", "S0"], ["S1", "N0", "S0", "N1"]]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 2**20), b""):
            value.update(part)
    return value.hexdigest()


def save(path, payload, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)
        stream.write("\n")


def write_ledger(root):
    """Rebuild a mutable CSV index from recorded stage status and owned logs."""
    root = Path(root)
    rows = []
    for path in sorted((root / "stages").glob("*.json")):
        record = load(path)
        execution = record.get("execution", {})
        rows.append(
            {
                "stage": record["name"],
                "status": record["status"],
                "started_unix": record.get("started_unix"),
                "ended_unix": record.get("ended_unix"),
                "exit_code": execution.get("exit_code"),
                "timed_out": execution.get("timed_out"),
                "status_path": str(path),
                "command_path": str(root / "logs" / record["name"] / "command.json"),
                "ownership_path": str(root / "logs" / record["name"] / "process_ownership.json"),
                "telemetry_path": str(root / "logs" / record["name"] / "nvml_memory.jsonl"),
            }
        )
    if rows:
        with (root / "stage_ledger.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def parse_topology(text, by_index):
    """Strip display-only ANSI codes while preserving the saved original output."""
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    lines = clean.splitlines()
    header = next(
        (
            re.findall(r"\bGPU\d+\b", line)
            for line in lines
            if len(re.findall(r"\bGPU\d+\b", line)) >= 4
        ),
        [],
    )
    if len(header) < 4:
        raise ValueError("BLOCKED_ENVIRONMENT: complete topology header was not captured")
    graph = {}
    for line in lines:
        fields = line.split()
        if fields and re.fullmatch(r"GPU\d+", fields[0]) and fields[0][3:] in by_index:
            if len(fields) < len(header) + 1:
                raise ValueError("Truncated GPU topology row")
            graph[by_index[fields[0][3:]]] = {
                by_index[label[3:]]: value
                for label, value in zip(header, fields[1 : 1 + len(header)], strict=True)
            }
    return graph


def audit_environment(args):
    if args.environment_json:
        environment = load(args.environment_json)
        environment["environment_source"] = str(Path(args.environment_json).resolve())
    else:
        commands = {
            "devices": [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            "topology": ["nvidia-smi", "topo", "-m"],
            "p2p_nvlink": ["nvidia-smi", "topo", "-p2p", "n"],
            "p2p_pci": ["nvidia-smi", "topo", "-p2p", "p"],
            "processes": ["nvidia-smi"],
            "nvlink_status": ["nvidia-smi", "nvlink", "-s"],
        }
        observations = {}
        for name, argv in commands.items():
            try:
                result = subprocess.run(
                    argv, capture_output=True, text=True, timeout=30, check=False
                )
                observations[name] = {
                    "argv": argv,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except (OSError, subprocess.TimeoutExpired) as error:
                observations[name] = {"argv": argv, "exit_code": None, "error": repr(error)}
        if observations["devices"].get("exit_code") != 0:
            raise RuntimeError("NOT_RUN_NO_GPU: nvidia-smi device inventory unavailable")
        devices = []
        for row in csv.reader(observations["devices"]["stdout"].splitlines()):
            if row:
                devices.append(
                    dict(
                        zip(
                            [
                                "index",
                                "uuid",
                                "pci_bus_id",
                                "name",
                                "memory_total_mib",
                                "memory_free_mib",
                                "driver",
                            ],
                            [value.strip() for value in row],
                            strict=True,
                        )
                    )
                )
        by_index = {str(item["index"]): item["uuid"] for item in devices}
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        requested = args.allowed_uuids or (inherited.split(",") if inherited else None)
        if not requested:
            raise ValueError("Pass --allowed-uuids or retain the allocator's CUDA_VISIBLE_DEVICES")

        def resolve(value):
            if value in by_index:
                return by_index[value]
            candidates = [item["uuid"] for item in devices if item["uuid"].startswith(value)]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous/unavailable assigned device: {value}")
            return candidates[0]

        allowed = [resolve(value) for value in requested]
        if inherited is not None and not set(allowed) <= {
            resolve(value) for value in inherited.split(",")
        }:
            raise ValueError("Cannot expose a GPU outside the inherited allocation")
        graph = parse_topology(observations["topology"].get("stdout", ""), by_index)
        environment = {
            "schema_version": 1,
            "timestamp_unix": time.time(),
            "platform": platform.platform(),
            "allowed_gpu_uuids": allowed,
            "devices": devices,
            "topology": graph,
            "observations": observations,
            "environment": {
                key: value
                for key, value in os.environ.items()
                if key.startswith(("CUDA_", "NCCL_", "OMP_", "PYTORCH_", "TORCH_"))
            },
        }
    manifest = make_manifest(environment)
    root = Path(args.run_root)
    save(root / "environment/environment.json", environment)
    save(root / "placement.json", manifest)
    return {
        "status": "PASS",
        "hypothesis_applicability": manifest["hypothesis_applicability"],
        "placement": str(root / "placement.json"),
    }


def training_spec(args, case, repeat, phase):
    engine, mapping = MATRIX[case]
    name = f"{phase}_{case}_r{repeat}"
    output = Path(args.run_root) / ("profiles" if phase == "profile" else phase) / name
    warmup, steps = (
        (10, 30) if phase == "benchmark" else ((10, 1) if phase == "profile" else (2, 2))
    )
    argv = [
        args.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "-m",
        "experiments.finalization_4gpu.train",
        "--implementation",
        engine,
        "--config",
        args.config,
        "--data",
        args.data or "INPUT_NOT_SET",
        "--output-dir",
        str(output),
        "--run-id",
        args.run_id,
        "--experiment-id",
        name,
        "--warmup",
        str(warmup),
        "--steps",
        str(steps),
        "--policy",
        "C0",
    ]
    if phase == "capacity" and engine == "shardtrain":
        argv.append("--record-events")
    if phase == "profile":
        argv.append("--profile")
        argv = [
            args.nsys or "NSYS_NOT_SET",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=false",
            "--output",
            str(output / "timeline"),
        ] + argv
    return {
        "name": name,
        "phase": phase,
        "case": case,
        "engine": engine,
        "mapping_id": mapping,
        "repeat": repeat,
        "warmup": warmup,
        "steps": steps,
        "output": str(output),
        "argv": argv,
        "timeout": args.timeout,
        "profile": phase == "profile",
        "source_manifest_sha256": sha(args.source_manifest) if args.source_manifest else None,
    }


def make_plan(args, phase):
    manifest = load(args.placement_manifest)
    validate_manifest(manifest)
    if phase in {"benchmark", "profile", "capacity"}:
        stages = [
            training_spec(args, case, repeat, phase)
            for repeat, order in enumerate(
                ORDERS
                if phase == "benchmark"
                else [["S0", "S1"] if phase == "profile" else list(MATRIX)]
            )
            for case in order
        ]
    elif phase == "correctness":
        from . import correctness

        stages = correctness.build_plan(args.placement_manifest, args.run_root, python=args.python)
    elif phase == "nccl-reference":
        from .reference import make_reference_plan

        points = make_reference_plan(
            manifest,
            load(Path(args.run_root) / "communication_inventory.json"),
            load(args.build_manifest),
            args.repeats,
        )
        stages = [
            {
                **entry,
                "name": f"nccl_reference_{index:03d}",
                "phase": phase,
                "timeout": args.timeout,
                "output": str(Path(args.run_root) / "nccl_reference" / f"point{index:03d}"),
            }
            for index, entry in enumerate(points)
        ]
    else:
        raise ValueError(phase)
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": phase,
        "status": "NOT_RUN",
        "logical_mesh": [[0, 1], [2, 3]],
        "placement_manifest": str(args.placement_manifest),
        "placement_manifest_sha256": sha(args.placement_manifest),
        "source_manifest": args.source_manifest,
        "source_manifest_sha256": sha(args.source_manifest) if args.source_manifest else None,
        "data": args.data,
        "input_sha256": sha(args.data) if args.data else None,
        "stages": stages,
    }


def configure_owner(args):
    import owned_stage

    owned_stage.BASE, owned_stage.PROJECT = (
        Path(args.run_root).resolve(),
        Path(args.project).resolve(),
    )

    def environment():
        value = os.environ.copy()
        value.update(
            PYTHONPATH=str(Path(args.project) / "src") + os.pathsep + str(Path(args.project)),
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="1",
            NCCL_LAUNCH_ORDER_IMPLICIT="1",
            NCCL_DEBUG="WARN",
            TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
            TOKENIZERS_PARALLELISM="false",
        )
        if args.source_manifest:
            value.update(
                SHARDTRAIN_SOURCE_MANIFEST=str(Path(args.source_manifest).resolve()),
                SHARDTRAIN_SOURCE_MANIFEST_SHA256=sha(args.source_manifest),
            )
        return value

    owned_stage.common.base_env = environment
    return owned_stage


def run_stage(args, stage, manifest):
    owner = configure_owner(args)
    name = stage["name"]
    state_path = Path(args.run_root) / "stages" / (name + ".json")
    fingerprint = hashlib.sha256(json.dumps(stage, sort_keys=True).encode()).hexdigest()
    if state_path.exists():
        previous = load(state_path)
        if (
            args.resume
            and previous.get("status") == "PASS"
            and previous.get("stage_fingerprint") == fingerprint
        ):
            return previous
        raise FileExistsError(
            f"Stage exists; preserve failed/interrupted data and use a new run: {name}"
        )
    record = {
        "name": name,
        "stage_fingerprint": fingerprint,
        "status": "RUNNING",
        "started_unix": time.time(),
        "stage": stage,
    }
    save(state_path, record)
    mapping = stage.get("mapping_id", "M0")
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(stage["ordered_gpu_uuids"])
        if "ordered_gpu_uuids" in stage
        else visible_order(manifest, mapping),
        "SHARDTRAIN_PLACEMENT_MANIFEST": str(Path(args.placement_manifest).resolve()),
        "SHARDTRAIN_MAPPING_ID": mapping,
    }
    if stage["phase"] in {"profile", "correctness", "nccl-reference"}:
        env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,ENV")
    try:
        if stage["phase"] == "nccl-reference":
            build = load(args.build_manifest)
            env["LD_LIBRARY_PATH"] = build["link_environment"]["LD_LIBRARY_PATH"]
            if sha(stage["binary"]["path"]) != stage["binary"]["sha256"]:
                raise RuntimeError("NCCL-tests binary changed after build freeze")
            if os.environ.get("LD_PRELOAD"):
                raise RuntimeError("LD_PRELOAD prevents a guaranteed NCCL library selection")
            from build_nccl_tests import nccl_link_path, version_probe

            actual = version_probe(
                Path(build["config"]["nccl_home"]), build["config"]["expected_nccl"]
            )
            if actual["library_sha256"] != build["nccl"]["library_sha256"]:
                raise RuntimeError("NCCL library changed after build freeze")
            linked = subprocess.run(
                ["ldd", stage["binary"]["path"]],
                env={**os.environ, **env},
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            record["ldd"] = {
                "exit_code": linked.returncode,
                "stdout": linked.stdout,
                "stderr": linked.stderr,
            }
            if linked.returncode != 0 or str(nccl_link_path(linked.stdout)) != actual["library"]:
                raise RuntimeError("NCCL-tests would load a different library")
        actual_environment = {**owner.common.base_env(), **env}
        selected_keys = {
            key
            for key in actual_environment
            if key.startswith(
                ("CUDA_", "NCCL_", "OMP_", "TORCH_", "PYTORCH_", "PYTHON", "SHARDTRAIN_")
            )
        }
        selected_keys.update(
            {
                "CUDA_DEVICE_ORDER",
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_ALLOC_CONF",
                "PYTORCH_CUDA_ALLOC_CONF",
                "LD_LIBRARY_PATH",
            }
        )
        save(
            Path(args.run_root) / "commands" / f"{name}.json",
            {
                "argv": stage["argv"],
                "cwd": str(owner.PROJECT),
                "environment": {key: actual_environment.get(key) for key in sorted(selected_keys)},
                "null_means": "unset in the actual launch environment",
                "source_manifest_sha256": sha(args.source_manifest)
                if args.source_manifest
                else None,
                "placement_manifest_sha256": sha(args.placement_manifest),
            },
        )
        result = owner.run_stage(
            name, stage["argv"], timeout=stage.get("timeout", args.timeout), extra_env=env
        )
        record.update(
            status="PASS"
            if result["status"] == "passed"
            else ("TIMEOUT" if result.get("timed_out") else "FAIL"),
            execution=result,
        )
        if record["status"] != "PASS":
            raise RuntimeError(f"Owned stage failed: {name}")
        if stage["phase"] in {"benchmark", "capacity", "profile"}:
            from run_matrix import check_training

            payloads = check_training(
                Path(stage["output"]), stage["steps"], safety=stage["phase"] == "capacity"
            )
            for rank, payload in enumerate(payloads):
                if (
                    payload["unique_training_tokens_per_step"] != 8176
                    or payload["unique_input_tokens_per_step"] != 8192
                ):
                    raise RuntimeError("Actual label/input counts differ from fixed workload")
                audit = load(Path(stage["output"]) / f"placement.rank{rank}.json")
                if (
                    audit["status"] != "PASS"
                    or audit["actual_uuid"]
                    != placement_for(manifest, mapping)["ordered_gpu_uuids"][rank]
                ):
                    raise RuntimeError("Rank placement audit failed")
            if stage["phase"] == "profile":
                export = {
                    "name": name + "_export",
                    "phase": "export",
                    "mapping_id": mapping,
                    "argv": [
                        args.nsys,
                        "export",
                        "--type=sqlite",
                        "--force-overwrite=false",
                        "--output",
                        str(Path(stage["output"]) / "timeline.sqlite"),
                        str(Path(stage["output"]) / "timeline.nsys-rep"),
                    ],
                }
                run_stage(args, export, manifest)
        elif stage["phase"] == "nccl-reference":
            from run_nccl_tests import parse_output

            logs = Path(args.run_root) / "logs" / name
            rows, check = parse_output((logs / "stdout.log").read_text())
            if check["status"] != "PASS" or any(
                row["reported_size_bytes"] != stage["full_bytes"] for row in rows
            ):
                raise RuntimeError("NCCL-tests correctness or exact requested bytes failed")
            record.update(metrics=rows, check=check)
    except BaseException as error:
        record.update(
            status="FAIL" if record["status"] in {"RUNNING", "PASS"} else record["status"],
            error=repr(error),
        )
        raise
    finally:
        record["ended_unix"] = time.time()
        save(state_path, record, exclusive=False)
        write_ledger(args.run_root)
    return record


def validate_before_execute(args, phase, plan):
    if platform.system() != "Linux":
        raise RuntimeError("NOT_RUN_NO_GPU: owned CUDA execution requires Linux")
    if not args.source_manifest or not Path(args.source_manifest).is_file():
        raise ValueError("Freeze the source and pass --source-manifest before execution")
    from .audit import verify_source

    verify_source(args.project, args.source_manifest)
    if phase in {"benchmark", "capacity", "profile"}:
        if not args.data or not Path(args.data).is_file():
            raise FileNotFoundError("A real frozen token file is required")
        dataset = load(Path(args.data).with_name("dataset.json"))
        if dataset["tokens_sha256"] != plan["input_sha256"]:
            raise ValueError("Actual token hash differs from dataset manifest")
    if phase in {"capacity", "benchmark", "profile", "nccl-reference"}:
        gate = load(Path(args.run_root) / "correctness/summary.json")
        if gate.get("status") != "PASS":
            from .correctness import validate_numeric_review

            review_path = Path(args.run_root) / "correctness/numeric_review.json"
            validate_numeric_review(gate, review_path)
        if gate.get("source_manifest_sha256") != plan["source_manifest_sha256"]:
            raise RuntimeError("Correctness gate and current candidate snapshots differ")
        if gate.get("placement_manifest_sha256") != plan["placement_manifest_sha256"]:
            raise RuntimeError("Correctness gate and current physical placement manifests differ")
    if phase in {"benchmark", "profile", "nccl-reference"}:
        for case in MATRIX:
            gate = load(Path(args.run_root) / "stages" / f"capacity_{case}_r0.json")
            if gate.get("status") != "PASS":
                raise RuntimeError(
                    "Both engines and mappings must pass safe capacity before this phase"
                )
        contract = load(Path(args.run_root) / "capacity_contract.json")
        if (
            contract.get("status") != "PASS"
            or contract.get("source_manifest_sha256") != plan["source_manifest_sha256"]
        ):
            raise RuntimeError(
                "Current mapping/state/input/communication contract is not qualified"
            )
    if phase == "profile" and not args.nsys:
        raise ValueError("Pass the installed, version-recorded Nsight executable with --nsys")
    if phase in {"benchmark", "profile"}:
        reference_plan = load(Path(args.run_root) / "plans/nccl-reference.json")
        if not reference_plan["stages"] or any(
            load(Path(args.run_root) / "stages" / (item["name"] + ".json")).get("status") != "PASS"
            for item in reference_plan["stages"]
        ):
            raise RuntimeError("Current actual-payload pair references are incomplete")


def execute_correctness(args, plan, manifest):
    """Preserve independent failures, finish their peer mapping, always summarize.

    Only independent numerical stages may continue after a *completed* owned
    cleanup. Their FAIL is retained and returned as a nonzero CLI result.
    """
    from . import correctness

    try:
        for stage in plan["stages"]:
            try:
                run_stage(args, stage, manifest)
            except Exception:
                path = Path(args.run_root) / "stages" / (stage["name"] + ".json")
                record = load(path) if path.is_file() else {}
                execution = record.get("execution", {})
                clean = (
                    execution.get("status") == "failed"
                    and execution.get("timed_out") is False
                    and execution.get("remaining_active_owned_pids") == []
                    and execution.get("ownership_errors") == []
                    and execution.get("driver_exception") is None
                    and execution.get("nvml", {}).get("errors") == []
                )
                if stage.get("case") != "independent_rollout" or not clean:
                    raise
    finally:
        summary = correctness.summarize(args.run_root)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "inspect",
            "dry-run",
            "correctness",
            "capacity",
            "benchmark",
            "profile",
            "nccl-reference",
            "analyze",
        ],
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", default=str(PROJECT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--placement-manifest")
    parser.add_argument("--source-manifest")
    parser.add_argument("--data")
    parser.add_argument("--config", default="experiments/fsdp_interview/configs/primary.yaml")
    parser.add_argument("--nsys")
    parser.add_argument("--build-manifest")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--phase",
        choices=["correctness", "capacity", "benchmark", "profile", "nccl-reference"],
        default="benchmark",
    )
    parser.add_argument("--environment-json")
    parser.add_argument("--allowed-uuids", nargs=4)
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.repeats <= 0 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("Positive timeout/repeats and a simple unique run ID are required")
    args.placement_manifest = args.placement_manifest or str(Path(args.run_root) / "placement.json")
    try:
        if args.command == "inspect":
            result = audit_environment(args)
        elif args.command == "analyze":
            from .analyze import analyze

            result = analyze(
                Path(args.run_root), Path(args.output or (Path(args.run_root) / "analysis"))
            )
        else:
            phase = args.phase if args.command == "dry-run" else args.command
            plan = make_plan(args, phase)
            plan_path = Path(args.run_root) / "plans" / f"{phase}.json"
            if plan_path.exists():
                if not args.resume or load(plan_path) != plan:
                    raise FileExistsError(
                        "Plan exists; use --resume only with identical frozen arguments"
                    )
            else:
                save(plan_path, plan)
            if args.command == "dry-run" or not args.execute:
                result = {
                    "status": "NOT_RUN",
                    "phase": phase,
                    "plan": str(plan_path),
                    "stages": len(plan["stages"]),
                }
            else:
                validate_before_execute(args, phase, plan)
                manifest = load(args.placement_manifest)
                if phase == "profile":
                    helper = {
                        "name": "profile_tool_help",
                        "phase": "tool_help",
                        "mapping_id": "M0",
                        "argv": [args.nsys, "profile", "--help"],
                        "timeout": 60,
                    }
                    run_stage(args, helper, manifest)
                    help_text = (
                        Path(args.run_root) / "logs/profile_tool_help/stdout.log"
                    ).read_text()
                    for option in (
                        "--trace",
                        "--sample",
                        "--cpuctxsw",
                        "--capture-range",
                        "--capture-range-end",
                        "--force-overwrite",
                    ):
                        if option not in help_text:
                            raise RuntimeError(
                                f"Installed NSYS does not advertise required option {option}"
                            )
                if phase == "correctness":
                    summary = execute_correctness(args, plan, manifest)
                    if summary.get("status") != "PASS":
                        print(
                            json.dumps(
                                {
                                    "status": summary.get("status"),
                                    "phase": phase,
                                    "required_gate_status": summary.get("required_gate_status"),
                                    "independent_status": summary.get("independent_status"),
                                    "exit_code": 1,
                                }
                            ),
                            flush=True,
                        )
                        return 1
                else:
                    for stage in plan["stages"]:
                        run_stage(args, stage, manifest)
                if phase == "capacity":
                    from .audit import audit_capacity
                    from .reference import inventory_from_events

                    audit_capacity(args.run_root)
                    save(
                        Path(args.run_root) / "communication_inventory.json",
                        inventory_from_events(
                            Path(args.run_root)
                            / "capacity/capacity_S0_r0/communication_events.rank0.jsonl"
                        ),
                    )
                result = {"status": "PASS", "phase": phase, "stages": len(plan["stages"])}
        print(json.dumps(result, default=str), flush=True)
        return 0
    except FileExistsError as error:
        print(
            json.dumps({"status": "BLOCKED_ENVIRONMENT", "error": str(error), "exit_code": 3}),
            file=sys.stderr,
        )
        return 3
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": repr(error), "exit_code": 1}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
