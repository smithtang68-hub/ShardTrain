"""Thin M0-only Q5 plans; GPU execution is explicit and owned by the root runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.finalization_4gpu import cli as shared
from experiments.finalization_4gpu.audit import verify_source

from .contracts import (
    MODES,
    POLICIES,
    TOKEN_SHA,
    assert_m0,
    assert_workload,
    load,
    qualify,
    save,
    sha,
    validate_p0_review,
)

PROJECT = Path(__file__).resolve().parents[2]
ORDER = (POLICIES, ("C2", "C0_guarded", "C1"), ("C1", "C2", "C0_guarded"))


def torchrun(args, module):
    return [
        args.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "-m",
        module,
    ]


def gate_spec(args, policy, mode, *, bootstrap=False):
    label = "bootstrap_" + mode if bootstrap else policy + "_" + mode
    output = Path(args.run_root) / "p2" / ("bootstrap" if bootstrap else "correctness")
    output = output / mode if bootstrap else output / policy / mode
    argv = torchrun(args, "experiments.q5_topology_revalidation_4gpu.gate_worker") + [
        "--mode",
        mode,
        "--policy",
        "C0_recorded" if bootstrap else policy,
        "--placement-manifest",
        args.placement_manifest,
        "--output",
        str(output),
        "--run-id",
        args.run_id,
    ]
    if not bootstrap:
        argv += ["--contract", str(Path(args.run_root) / "p2/contracts" / (mode + ".json"))]
    return dict(
        name="p2_gate_" + label,
        phase="correctness",
        mapping_id="M0",
        policy=policy,
        mode=mode,
        output=str(output),
        argv=argv,
        timeout=args.timeout,
    )


def training_spec(args, policy, repeat, phase, *, bootstrap=False):
    warmup, steps = (10, 30) if phase == "benchmark" else (10, 1) if phase == "profile" else (2, 2)
    if bootstrap:
        warmup, steps = 1, 1
    label = "C0_recorded" if bootstrap else policy
    name = f"p2_{phase}_{label}_r{repeat}"
    output = Path(args.run_root) / "p2" / ("profiles" if phase == "profile" else phase) / label
    if phase == "benchmark":
        output = output / ("r" + str(repeat))
    argv = torchrun(args, "experiments.q5_topology_revalidation_4gpu.train") + [
        "--implementation",
        "shardtrain",
        "--config",
        args.config,
        "--data",
        args.data,
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
        "C0" if label.startswith("C0_") else label,
    ]
    if not bootstrap:
        argv += ["--contract", str(Path(args.run_root) / "p2/contracts/full.json")]
        if policy == "C0_guarded":
            argv += ["--guarded-baseline"]
    if phase == "capacity":
        argv += ["--record-events"]
    if phase == "profile":
        argv += ["--profile"]
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
    return dict(
        name=name,
        phase=phase,
        mapping_id="M0",
        policy=label,
        repeat=repeat,
        warmup=warmup,
        steps=steps,
        output=str(output),
        argv=argv,
        timeout=args.timeout,
    )


def plan(args, phase):
    if phase == "bootstrap":
        return [gate_spec(args, "C0_recorded", mode, bootstrap=True) for mode in MODES] + [
            training_spec(args, "C0_recorded", 0, "capacity", bootstrap=True)
        ]
    if phase == "gates":
        return [gate_spec(args, policy, mode) for mode in MODES for policy in POLICIES]
    if phase == "capacity":
        return [training_spec(args, policy, 0, phase) for policy in POLICIES]
    if phase == "benchmark":
        return [
            training_spec(args, policy, repeat, phase)
            for repeat, order in enumerate(ORDER)
            for policy in order
        ]
    if phase == "profile":
        return [training_spec(args, policy, 0, phase) for policy in ("C0_guarded", "C1")]
    raise ValueError(phase)


def validate_before(args, phase):
    verify_source(args.project, args.source_manifest)
    frozen = load(args.source_manifest)["files"]
    for filename in Path(__file__).parent.glob("*.py"):
        if str(filename.relative_to(PROJECT)).replace("\\", "/") not in frozen:
            raise ValueError("P2 execution file absent from source snapshot: " + filename.name)
    manifest = load(args.placement_manifest)
    assert_m0(manifest)
    assert_workload(args.config)
    if sha(args.data) != TOKEN_SHA:
        raise ValueError("BLOCKED_INPUT: real fixed token version differs")
    registration = dict(
        status="FROZEN_PROTOCOL_NOT_A_GPU_RESULT",
        source_manifest_sha256=sha(args.source_manifest),
        placement_manifest_sha256=sha(args.placement_manifest),
        input_sha256=sha(args.data),
        config_sha256=sha(args.config),
        ordered_m0_uuids=manifest["placements"]["M0"]["ordered_gpu_uuids"],
        policies=list(POLICIES),
        logical_mesh=manifest["logical_mesh"],
        formal=dict(
            fresh_runs_per_policy=3,
            warmup=10,
            measured=30,
            record_events=False,
            gpu_timing=False,
            common_online_guard=True,
        ),
        C3="NOT_RUN_OPTIONAL",
        Q12="NOT_RUN_OPTIONAL",
        observer_source_sha256=sha(
            PROJECT / "experiments/fsdp_interview/cases/q05_nccl_contention.py"
        ),
    )
    registration_path = Path(args.run_root) / "p2/contracts/q5_m0_contract.json"
    if registration_path.exists():
        if load(registration_path) != registration:
            raise ValueError("Frozen P2 protocol changed; preserve output and use a fresh run")
    else:
        save(registration_path, registration)
    if phase != "bootstrap":
        scopes = MODES if phase == "gates" else ("full",)
        for scope in scopes:
            contract = load(Path(args.run_root) / "p2/contracts" / (scope + ".json"))
            if (
                contract.get("status") != "QUALIFIED"
                or contract.get("source_manifest_sha256") != sha(args.source_manifest)
                or contract.get("placement_manifest_sha256") != sha(args.placement_manifest)
                or contract.get("architecture_scope") != scope
            ):
                raise ValueError("Bootstrap architecture/source/placement contract differs")
    if phase not in {"bootstrap", "gates"}:
        from .audit import audit_gates

        gate_path = Path(args.run_root) / "p2/contracts/small_oracle_gate.json"
        if not gate_path.exists():
            audit_gates(args.run_root, args.source_manifest, args.placement_manifest)
        gate = load(gate_path)
        if (
            gate["status"] != "PASS"
            or gate["source_manifest_sha256"] != sha(args.source_manifest)
            or gate["placement_manifest_sha256"] != sha(args.placement_manifest)
        ):
            raise ValueError("BLOCKED_CORRECTNESS: policy oracle gates differ from current source")
        validate_p0_review(args.p0_review, args.source_manifest, args.placement_manifest)
    if phase in {"benchmark", "profile"}:
        value = load(Path(args.run_root) / "p2/contracts/collective_contract_audit.json")
        if (
            value["status"] != "PASS"
            or value["source_manifest_sha256"] != sha(args.source_manifest)
            or value["placement_manifest_sha256"] != sha(args.placement_manifest)
        ):
            raise ValueError("BLOCKED_CAPACITY: missing current-source three-policy contract")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "command",
        choices=("plan", "bootstrap", "gates", "capacity", "benchmark", "profile", "analyze"),
    )
    parser.add_argument(
        "--phase",
        choices=("bootstrap", "gates", "capacity", "benchmark", "profile"),
        default="benchmark",
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", default="20261017_run01")
    parser.add_argument("--project", default=str(PROJECT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--placement-manifest")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="experiments/fsdp_interview/configs/primary.yaml")
    parser.add_argument("--nsys")
    parser.add_argument("--p0-review")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    args.placement_manifest = args.placement_manifest or str(Path(args.run_root) / "placement.json")
    phase = args.phase if args.command == "plan" else args.command
    if args.command == "analyze":
        from .analysis import analyze

        result = analyze(args.run_root, args.output)
        print(result["status"])
        return 0 if result["status"] == "Q5_M0_THREE_POLICY_PASS" else 1
    stages = plan(args, phase)
    if not args.execute or args.command == "plan":
        target = (
            Path(args.output)
            if args.output
            else Path(args.run_root) / "p2/plans" / (phase + ".json")
        )
        save(
            target,
            dict(
                status="NOT_RUN_PLAN_ONLY",
                stages=stages,
                policies=list(POLICIES),
                optional_C3="NOT_RUN_OPTIONAL",
                optional_Q12="NOT_RUN_OPTIONAL",
            ),
        )
        print(str(target))
        return 0
    manifest = validate_before(args, phase)
    for stage in stages:
        shared.run_stage(args, stage, manifest)
        if phase == "bootstrap":
            scope = stage.get("mode", "full")
            directory = (
                Path(stage["output"]) / "observer" if scope != "full" else Path(stage["output"])
            )
            target = Path(args.run_root) / "p2/contracts" / (scope + ".json")
            if not (args.resume and target.exists()):
                qualify(
                    directory,
                    target,
                    source_manifest=args.source_manifest,
                    placement_manifest=args.placement_manifest,
                    scope=scope,
                )
    if phase == "gates":
        from .audit import audit_gates

        audit_gates(args.run_root, args.source_manifest, args.placement_manifest)
    if phase == "capacity":
        from .audit import audit_capacity

        audit_capacity(args.run_root, args.source_manifest, args.placement_manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileExistsError as error:
        print("REFUSED_OVERWRITE:", error, file=sys.stderr)
        raise SystemExit(3) from error
    except Exception as error:
        print("FAILED_OR_BLOCKED:", repr(error), file=sys.stderr)
        raise SystemExit(1) from error
