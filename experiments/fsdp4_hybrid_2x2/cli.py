"""Fixed P1 phases; --execute is required for GPU work, using owned_stage."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from experiments.finalization_4gpu import cli as shared
from experiments.finalization_4gpu.audit import verify_source

from .contracts import CASES, ORDERS, TOKEN_SHA, comparison_contract, save, sha

PROJECT = Path(__file__).resolve().parents[2]


def validate_p0_review(path, source, placement, data):
    review = shared.load(path) if path else {}
    if (
        review.get("source_manifest_sha256") != sha(source)
        or review.get("classification") != "EXPECTED_NUMERICAL_DRIFT_SUPPORTED"
        or review.get("strict_independent_status") not in {"PASS", "FAIL"}
        or review.get("signed_by") != "root_agent"
        or review.get("scope") != "bounded_performance_with_retained_strict_fail"
        or review.get("input_sha256") != sha(data)
        or review.get("placement_manifest_sha256") != sha(placement)
        or not isinstance(review.get("limits"), list)
        or review.get("training_runtime_bug_confirmed") is not False
        or review.get("continuation_authorized") is not True
    ):
        raise RuntimeError(
            "BLOCKED_CORRECTNESS: root's source/input/placement-bound P0 review required"
        )
    evidence = review.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < 3:
        raise ValueError("P0 review needs actual current capture, replay and control evidence")
    for record in evidence:
        evidence_path = Path(record["path"])
        if (
            not evidence_path.is_absolute()
            or not evidence_path.is_file()
            or sha(evidence_path) != record["sha256"]
        ):
            raise ValueError("P0 continuation evidence missing or SHA changed")
    return review


def plan(args, phase):
    stages = []
    order = [c for r in ORDERS for c in r] if phase == "benchmark" else list(CASES)
    for index, case in enumerate(order):
        repeat = index // 4 if phase == "benchmark" else 0
        c = CASES[case]
        kinds = ["fp32", "fp16"] if phase == "correctness" else [None]
        for kind in kinds:
            name = f"p1_{phase}_{case}_{kind or 'train'}_r{repeat}"
            output = Path(args.run_root) / phase / name
            argv = [
                args.python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=4",
                "--module",
            ]
            if phase == "correctness":
                argv += [
                    "experiments.fsdp4_hybrid_2x2.gate_worker",
                    "--case",
                    case,
                    "--kind",
                    kind,
                    "--output",
                    str(output),
                ]
                warmup, steps = 0, None
            else:
                warmup, steps = (10, 30) if phase == "benchmark" else (1, 2)
                argv += [
                    "experiments.fsdp4_hybrid_2x2.worker",
                    "--case",
                    case,
                    "--implementation",
                    c["engine"],
                    "--config",
                    str(
                        Path(args.project)
                        / "experiments/fsdp4_hybrid_2x2/configs"
                        / (case + ".yaml")
                    ),
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
                ]
                if phase == "capacity":
                    argv += ["--record-events", "--communication-audit", "--gemm-shapes"]
            stages.append(
                dict(
                    name=name,
                    phase=phase,
                    case=case,
                    engine=c["engine"],
                    mapping_id="M0",
                    repeat=repeat,
                    warmup=warmup,
                    steps=steps,
                    kind=kind,
                    output=str(output),
                    argv=argv,
                    timeout=args.timeout,
                    profile=False,
                    source_manifest_sha256=sha(args.source_manifest)
                    if args.source_manifest
                    else None,
                )
            )
    return dict(
        status="NOT_RUN",
        run_id=args.run_id,
        phase=phase,
        stages=stages,
        source_manifest_sha256=sha(args.source_manifest) if args.source_manifest else None,
        placement_manifest_sha256=sha(args.placement_manifest),
        input_sha256=sha(args.data),
    )


def check_correctness(root, expected_source=None):
    rows, sources = [], set()
    for case in CASES:
        for kind in ["fp32", "fp16"]:
            directory = Path(root) / "correctness" / f"p1_correctness_{case}_{kind}_r0"
            for rank in range(4):
                path = directory / f"gate_audit.rank{rank}.json"
                value = shared.load(path)
                if value["status"] != "PASS" or value["placement"]["status"] != "PASS":
                    raise ValueError(f"BLOCKED_CORRECTNESS: {path}")
                sources.add(value["placement"]["source_manifest_sha256"])
                rows.append(dict(path=str(path), sha256=sha(path), case=case, kind=kind, rank=rank))
    if (
        len(sources) != 1
        or None in sources
        or expected_source is not None
        and sources != {expected_source}
    ):
        raise ValueError("Correctness evidence belongs to a different tested source")
    return dict(
        status="PASS",
        rank_cases=len(rows),
        cases=8,
        evidence=rows,
        source_manifest_sha256=next(iter(sources)),
        scope="Current independent tiny gates; no independent long FP16 rollout claim",
    )


def audit_training(root, data, phase, *, repeats):
    import numpy as np

    from .contracts import input_blocks

    blocks = np.load(data, allow_pickle=False)
    if sha(data) != TOKEN_SHA:
        raise ValueError("Fixed real token SHA mismatch")
    state_hashes, source_hashes, rows, assignments = set(), set(), [], {}
    for case, c in CASES.items():
        for repeat in range(repeats):
            directory = Path(root) / phase / f"p1_{phase}_{case}_train_r{repeat}"
            for rank in range(4):
                payload = shared.load(directory / f"benchmark.rank{rank}.json")
                audit = shared.load(directory / f"placement.rank{rank}.json")
                if payload["status"] != "PASS" or audit["status"] != "PASS":
                    raise ValueError("Failed or missing training/placement")
                if audit["source_manifest_sha256"] != payload["source_manifest_sha256"]:
                    raise ValueError("Training and global initialization audit source differs")
                counters = audit["optimizer_end"]
                if (
                    counters["status"] != "PASS"
                    or counters["step_min"] != payload["warmup"] + len(payload["steps"])
                    or counters["step_max"] != counters["step_min"]
                ):
                    raise ValueError("Actual Adam leaf update counters differ")
                actual = (
                    payload["fsdp_group_size"],
                    payload["tp_group_size"],
                    payload["config"]["parallel"]["sp_enabled"],
                    payload["microbatch"],
                )
                if actual != (c["F"], c["T"], c["SP"], c["microbatch"]):
                    raise ValueError("Case layout identity mismatch")
                if payload["implementation"] != c["engine"] or payload["profile"]:
                    raise ValueError("Engine/profile identity mismatch")
                if payload["accumulation_steps"] != 2 or payload["seq_len"] != 512:
                    raise ValueError("Workload shape mismatch")
                if phase == "benchmark" and (
                    payload["event_recorder"]
                    or payload["guarded_baseline"]
                    or payload["warmup"] != 10
                    or len(payload["steps"]) != 30
                ):
                    raise ValueError("Formal C0_fast timing contract mismatch")
                state_hashes.add(audit["initial_global_state"]["global_state_sha256"])
                source_hashes.add(payload["source_manifest_sha256"])
                for step in payload["warmup_steps"] + payload["steps"]:
                    selected = input_blocks(case, rank, step["optimizer_step"], len(blocks))
                    expected = (
                        __import__("hashlib")
                        .sha256(blocks[selected[0] : selected[-1] + 1].tobytes())
                        .hexdigest()
                    )
                    if (
                        step["input_block_start"] != selected[0]
                        or step["input_batch_sha256"] != expected
                    ):
                        raise ValueError("Actual logical sample IDs/hash mismatch")
                    if (
                        step["valid_tokens"] != 8176
                        or step["unique_input_tokens"] != 8192
                        or step["overflow"]
                        or not step["finite_gradients"]
                    ):
                        raise ValueError(
                            "Nonfinite, skipped update or global sample-count mismatch"
                        )
                    if rank % c["T"] == 0:
                        key = (case, repeat, step["optimizer_step"])
                        parts = assignments.setdefault(key, [[], []])
                        for micro in range(2):
                            parts[micro] += selected[
                                micro * c["microbatch"] : (micro + 1) * c["microbatch"]
                            ]
                rows.append(
                    dict(case=case, repeat=repeat, rank=rank, status="PASS", path=str(directory))
                )
    if len(state_hashes) != 1 or len(source_hashes) != 1 or None in source_hashes:
        raise ValueError("NON_MATCHED: four groups differ in actual global initialization/source")
    assignment_rows = []
    for (case, repeat, step), parts in sorted(assignments.items()):
        actual = sorted(parts[0] + parts[1])
        expected = list(range(step * 16 % len(blocks), step * 16 % len(blocks) + 16))
        if actual != expected or any(len(p) != 8 or len(set(p)) != 8 for p in parts):
            raise ValueError("Actual global step/microstep sample partition mismatch")
        partner = ("N" if case.startswith("S") else "S") + case[1:]
        if parts != assignments[(partner, repeat, step)]:
            raise ValueError("Same-layout engines differ in actual microstep samples")
        assignment_rows.append(
            dict(
                case=case,
                repeat=repeat,
                optimizer_step=step,
                microstep_global_block_ids=parts,
                full_global_block_ids=actual,
            )
        )
    return dict(
        status="PASS",
        phase=phase,
        cases=4,
        rank_runs=len(rows),
        source_manifest_sha256=next(iter(source_hashes)),
        global_state_sha256=next(iter(state_hashes)),
        token_sha256=TOKEN_SHA,
        actual_parameter_initialization="full per-parameter reconstruction/hash",
        evidence=rows,
        actual_input_assignments=assignment_rows,
        limits=[
            "Initialization proof does not establish full1.5B gradient equivalence",
            "Across strategies H/F, microstep subsets differ but full-step16-block set matches; "
            "same-layout self/native identical",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "phase", choices=["contracts", "correctness", "capacity", "benchmark", "analyze"]
    )
    p.add_argument("--run-root", required=True)
    p.add_argument("--run-id", default="20261017_run01")
    p.add_argument("--project", default=str(PROJECT))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--placement-manifest", required=True)
    p.add_argument("--source-manifest")
    p.add_argument("--data", required=True)
    p.add_argument("--p0-review")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output")
    args = p.parse_args()
    root = Path(args.run_root)
    root.mkdir(parents=True, exist_ok=True)
    args.nsys, args.build_manifest = None, None
    manifest = shared.load(args.placement_manifest)
    shared.validate_manifest(manifest)
    if sha(args.data) != TOKEN_SHA:
        raise ValueError("Unexpected token version; do not substitute synthetic")
    if args.phase == "contracts":
        contract = comparison_contract()
        contract.update(
            placement_manifest_sha256=sha(args.placement_manifest),
            source_manifest_sha256=sha(args.source_manifest) if args.source_manifest else None,
        )
        save(root / "contracts/comparison_contract.json", contract)
        for case in CASES:
            save(root / "contracts" / (case.replace("-", "_") + ".json"), contract["cases"][case])
        return
    if args.phase == "analyze":
        from .analyze import analyze

        analyze(root, Path(args.output or root / "analysis"), args.data)
        return
    proposed = plan(args, args.phase)
    plan_path = root / "plans" / (args.phase + ("_execute" if args.execute else "_dry") + ".json")
    if not plan_path.exists():
        save(plan_path, proposed)
    if not args.execute:
        print(json.dumps(proposed, indent=2))
        return
    if platform.system() != "Linux":
        raise RuntimeError("GPU execution only on root's authorized Linux allocation")
    verify_source(args.project, args.source_manifest)
    if args.phase in {"capacity", "benchmark"}:
        check_correctness(root, sha(args.source_manifest))
    if args.phase == "benchmark":
        capacity = audit_training(root, args.data, "capacity", repeats=1)
        if capacity["source_manifest_sha256"] != sha(args.source_manifest):
            raise ValueError("Capacity belongs to a different candidate source")
        validate_p0_review(args.p0_review, args.source_manifest, args.placement_manifest, args.data)
    for stage in proposed["stages"]:
        shared.run_stage(args, stage, manifest)
        if args.phase == "correctness":
            for rank in range(4):
                value = shared.load(Path(stage["output"]) / f"gate_audit.rank{rank}.json")
                if value["status"] != "PASS":
                    raise ValueError("Tiny correctness audit failed")
    if args.phase == "correctness":
        summary = check_correctness(root, sha(args.source_manifest))
    else:
        summary = audit_training(
            root, args.data, args.phase, repeats=3 if args.phase == "benchmark" else 1
        )
    save(root / args.phase / "contract_audit.json", summary)


if __name__ == "__main__":
    main()
