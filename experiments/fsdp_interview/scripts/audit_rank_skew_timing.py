"""Validate Q12 timing injection hits and the fixed per-rank input sequence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def audit(runs):
    reports = []
    reference_inputs = {}
    common = None
    for run in runs:
        directory = Path(run["directory"])
        payloads = [
            json.loads((directory / f"benchmark.rank{rank}.json").read_text(encoding="utf-8"))
            for rank in range(run["world_size"])
        ]
        first = payloads[0]
        fingerprint = {
            key: first.get(key)
            for key in (
                "workload_hash",
                "seq_len",
                "microbatch",
                "accumulation_steps",
                "precision",
                "optimizer",
                "seed",
                "implementation",
                "policy",
                "guarded_baseline",
            )
        }
        fingerprint["tokens_sha256"] = first["dataset"]["tokens_sha256"]
        if common is None:
            common = fingerprint
        issues = []
        if fingerprint != common:
            issues.append("cross-condition workload/config divergence")
        injection = first["injection"]
        target_rank = injection["rank"]
        rank_reports = []
        for rank, payload in enumerate(payloads):
            steps = payload["steps"]
            warmup = payload["warmup_steps"]
            target = payload["instrumentation"]["injection"]
            hits = target["hits"]
            expected_hits = len(steps) if rank == target_rank else 0
            # Source-derived tied unit1 first/second forward use, not an event-fit.
            expected_candidates = 2 * len(steps) if rank == target_rank else 0
            if target["hit_count"] != expected_hits or len(hits) != expected_hits:
                issues.append(f"rank{rank} injection hit-count mismatch")
            if target["candidate_count"] != expected_candidates:
                issues.append(f"rank{rank} tied-use candidate-count mismatch")
            for hit in hits:
                selected = [row for row in steps if row["optimizer_step"] == hit["optimizer_step"]]
                if len(selected) != 1:
                    issues.append(f"rank{rank} injection outside unique measured optimizer step")
                    continue
                step = selected[0]
                correct = (
                    hit["rank"] == target_rank
                    and hit["unit_id"] == injection["unit_id"]
                    and hit["phase"] == injection["phase"]
                    and hit["collective_type"] == injection["kind"]
                    and hit["collective_occurrence"] == 0
                    and hit["measured_step"] == step["measured_step"]
                    and hit["runtime_microstep"]
                    == step["optimizer_step"] * payload["accumulation_steps"]
                )
                if not correct:
                    issues.append(f"rank{rank} injection selection mismatch")
            for step in steps:
                key = rank, step["measured_step"]
                shape_input = (
                    step["input_batch_sha256"],
                    step["input_block_start"],
                    step["unique_input_tokens"],
                    step["unique_training_tokens"],
                    step["valid_tokens"],
                    step["accumulation_steps"],
                )
                if key not in reference_inputs:
                    reference_inputs[key] = shape_input
                elif reference_inputs[key] != shape_input:
                    issues.append(f"rank{rank} measured input/shape/token sequence divergence")
                if not step["finite_gradients"] or step["overflow"]:
                    issues.append(f"rank{rank} finite/update check failed")
            ownership = payload["ownership_after"]
            walls = [hit["cpu_injection_wall_ms"] for hit in hits]
            rank_reports.append(
                {
                    "rank": rank,
                    "warmup_steps": len(warmup),
                    "measured_steps": len(steps),
                    "actual_candidates": target["candidate_count"],
                    "expected_candidates": expected_candidates,
                    "actual_hits": target["hit_count"],
                    "expected_hits": expected_hits,
                    "parameter_shard_bytes": ownership["parameter_shard_bytes"],
                    "gradient_shard_bytes": ownership["gradient_shard_bytes"],
                    "injection_cpu_wall_mean_ms": statistics.mean(walls) if walls else None,
                    "injection_cpu_wall_min_ms": min(walls) if walls else None,
                    "injection_cpu_wall_max_ms": max(walls) if walls else None,
                }
            )
        reports.append(
            {
                "directory": str(directory),
                "experiment_id": first["experiment_id"],
                "injection": injection,
                "status": "FAIL" if issues else "PASS",
                "issues": issues,
                "rank_results": rank_reports,
            }
        )
    return {
        "status": "PASS" if reports and all(row["status"] == "PASS" for row in reports) else "FAIL",
        "runs": reports,
        "common_workload": common,
        "rank_optimizer_steps": sum(
            row["measured_steps"] for run in reports for row in run["rank_results"]
        ),
        "selected_hits": sum(row["actual_hits"] for run in reports for row in run["rank_results"]),
        "candidate_count": sum(
            row["actual_candidates"] for run in reports for row in run["rank_results"]
        ),
        "source_limit": "Full-run finite/update checks; does not replace the separate dense oracle",
        "injection_limit": "GPU mode CPU wall includes enqueue only, not exact GPU kernel duration",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.timing_summary)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Fresh output directory required: {output}")
    data = json.loads(source.read_text(encoding="utf-8"))
    result = audit(data["training_runs"])
    result["source_timing_summary"] = str(source.resolve())
    output.mkdir(parents=True)
    (output / "timing_injection_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "rank_optimizer_steps", "selected_hits", "candidate_count")
            }
        )
    )


if __name__ == "__main__":
    main()
