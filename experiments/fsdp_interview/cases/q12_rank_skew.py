"""Causal CPU-submit/GPU-work injection without changing collective order."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path


class RankDelay:
    """Inject once at an explicitly valid collective in a selected rank/microstep.

    CPU mode sleeps *before the NCCL API*. GPU mode runs preallocated GEMMs on
    the actual collective submission stream before the NCCL call. The latter is
    GPU work-before-collective skew, not CPU delay or nominal delay_ms. No barrier
    or synchronize is introduced. select microstep_index=0 to avoid injecting
    twice when accumulation_steps=2.
    """

    def __init__(
        self,
        *,
        rank=0,
        unit_id=1,
        phase="FWD",
        kind="fsdp_ag",
        mode="cpu",
        delay_ms=0,
        microstep_index=0,
        measured_only=True,
        collective_occurrence=0,
        gemm_size=1024,
        gemm_repeats=4,
    ):
        if mode not in {"cpu", "gpu"}:
            raise ValueError("Injection mode must be cpu or gpu")
        if delay_ms not in {0, 2, 10, 50}:
            raise ValueError("CPU sweep delay_ms must be one of 0, 2, 10, 50")
        if unit_id < 0 or phase not in {"FWD", "BWD"}:
            raise ValueError("A real nonnegative unit and FWD/BWD phase are required")
        if not isinstance(collective_occurrence, int) or collective_occurrence < 0:
            raise ValueError("collective_occurrence must be a nonnegative integer")
        if not 16 <= gemm_size <= 4096 or not 1 <= gemm_repeats <= 256:
            raise ValueError("GEMM injection bounds: size 16..4096; repeats 1..256")
        self.rank, self.unit_id, self.phase, self.kind = rank, unit_id, phase, kind
        self.mode, self.delay_ms = mode, delay_ms
        self.microstep_index, self.measured_only = microstep_index, measured_only
        self.collective_occurrence = collective_occurrence
        self.gemm_size, self.gemm_repeats = gemm_size, gemm_repeats
        self.hits, self._prepared, self._injected, self._rank_seen = [], False, set(), None
        self._candidate_counts = defaultdict(int)
        self._gpu_buffers = None

    def prepare(self, runtime):
        if self._prepared:
            return
        import torch
        import torch.distributed as dist

        rank = runtime.comm.rank
        self._rank_seen = rank
        world = dist.get_world_size() if dist.is_initialized() else 1
        if not 0 <= self.rank < world:
            raise ValueError(f"Target rank {self.rank} is outside world size {world}")
        valid = {unit.unit_id: unit for unit in runtime.units}
        if self.unit_id not in valid:
            raise ValueError(f"Target unit {self.unit_id} is absent; actual units {sorted(valid)}")
        if self.kind == "fsdp_ag":
            if runtime.materializer.degree <= 1:
                raise ValueError("No FSDP AG exists with degree one")
            unit = valid[self.unit_id]
            if self.phase == "BWD" and (
                type(unit).__name__ == "TiedEmbeddingUnit"
                or unit is runtime.units[-1]
                or not runtime.reshard_after_forward
            ):
                raise ValueError(
                    "Target BWD AG cannot occur because its forward weights are retained"
                )
        if self.mode == "gpu" and rank == self.rank:
            if runtime.device.type != "cuda":
                raise ValueError("GPU injection needs CUDA")
            # This allocation/fill occurs at an untimed begin_step boundary.
            with torch.cuda.stream(runtime.comm.streams["compute"]):
                left = torch.full(
                    (self.gemm_size, self.gemm_size),
                    0.01,
                    dtype=torch.float16,
                    device=runtime.device,
                )
                right = torch.full_like(left, 0.01)
                result = torch.empty_like(left)
                prepared = torch.cuda.Event()
                prepared.record(runtime.comm.streams["compute"])
            self._gpu_buffers = left, right, result, prepared
        self._prepared = True

    def matches(self, row):
        return (
            row["rank"] == self.rank
            and row["unit_id"] == self.unit_id
            and row["phase"] == self.phase
            and row["collective_type"] == self.kind
            and (not self.measured_only or row.get("measured_step") is not None)
            and (
                self.microstep_index is None
                or row.get("microstep_in_optimizer_step") == self.microstep_index
            )
        )

    def before_collective(self, row, comm):
        if not self.matches(row):
            return
        if not self._prepared:
            raise RuntimeError("RankDelay.prepare(runtime) must run at an untimed boundary")
        key = row["runtime_microstep"], row["unit_id"], row["phase"], row["collective_type"]
        occurrence = self._candidate_counts[key]
        self._candidate_counts[key] += 1
        row["rank_injection_candidate"] = {
            "occurrence": occurrence,
            "selected_occurrence": self.collective_occurrence,
            "status": "SELECTED" if occurrence == self.collective_occurrence else "SKIPPED",
        }
        if occurrence != self.collective_occurrence:
            return
        if key in self._injected:
            raise RuntimeError(f"Target collective matched twice in microstep {key}")
        self._injected.add(key)
        before = time.monotonic_ns()
        hit = {
            name: row.get(name)
            for name in (
                "rank",
                "optimizer_step",
                "measured_step",
                "runtime_microstep",
                "unit_id",
                "layer",
                "process_group_label",
                "collective_sequence",
                "phase",
                "collective_type",
            )
        }
        hit.update(
            mode=self.mode,
            cpu_delay_ms=self.delay_ms if self.mode == "cpu" else None,
            location="immediately before NCCL API on the actual collective submission stream",
            injected_begin_ns=before,
            collective_occurrence=occurrence,
        )
        if self.mode == "cpu":
            if self.delay_ms:
                time.sleep(self.delay_ms / 1000)
        else:
            import torch

            left, right, result, prepared = self._gpu_buffers
            stream = torch.cuda.current_stream(comm.device)
            stream.wait_event(prepared)
            for tensor in (left, right, result):
                tensor.record_stream(stream)
            torch.cuda.nvtx.range_push(
                f"interview/rank_gpu_injection/rank{self.rank}/U{self.unit_id}"
            )
            try:
                for _ in range(self.gemm_repeats):
                    torch.mm(left, right, out=result)
            finally:
                torch.cuda.nvtx.range_pop()
            hit.update(
                gemm_size=self.gemm_size,
                gemm_repeats=self.gemm_repeats,
                actual_stream_ptr=stream.cuda_stream,
                expected_ms=None,
                gpu_timing="measure exact GEMM kernels in NSYS; not calibrated delay_ms",
            )
        hit["injected_end_ns"] = time.monotonic_ns()
        hit["cpu_injection_wall_ms"] = (hit["injected_end_ns"] - before) / 1e6
        row["rank_injection"] = hit
        self.hits.append(hit)

    def validate_hits(self):
        if self._rank_seen == self.rank and not self.hits:
            raise RuntimeError(
                "Injection NOT_RUN: target existed but no selected collective was hit"
            )

    def summary(self):
        return {
            "target_rank": self.rank,
            "target_unit": self.unit_id,
            "phase": self.phase,
            "kind": self.kind,
            "mode": self.mode,
            "delay_ms": self.delay_ms,
            "microstep_index": self.microstep_index,
            "collective_occurrence": self.collective_occurrence,
            "candidate_count": sum(self._candidate_counts.values()),
            "hit_count": len(self.hits),
            "hits": self.hits,
        }


def submit_skew(rows):
    """Single-host monotonic CPU timestamps only; never mix these with GPU times."""
    groups = defaultdict(list)
    for row in rows:
        if row.get("cpu_collective_submit_ns") is None or row.get("measured_step") is None:
            continue
        key = row["process_group_label"], tuple(row["group_members"]), row["collective_sequence"]
        groups[key].append(row)
    result = []
    for (label, members, sequence), values in sorted(groups.items()):
        matched = {value["rank"] for value in values} == set(members) and len(values) == len(
            members
        )
        timestamps = [value["cpu_collective_submit_ns"] for value in values]
        result.append(
            {
                "process_group_label": label,
                "group_members": list(members),
                "collective_sequence": sequence,
                "measured_step": values[0]["measured_step"],
                "runtime_microstep": values[0]["runtime_microstep"],
                "unit_id": values[0]["unit_id"],
                "phase": values[0]["phase"],
                "kind": values[0]["collective_type"],
                "status": "MATCHED" if matched else "UNMATCHED",
                "cpu_submit_skew_ms": (max(timestamps) - min(timestamps)) / 1e6
                if matched
                else None,
                "rank_submit_ns": {
                    str(value["rank"]): value["cpu_collective_submit_ns"] for value in values
                },
                "clock_assumption": (
                    "one Linux host CLOCK_MONOTONIC; no cross-machine clock comparison"
                ),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    from experiments.fsdp_interview.cases.q05_nccl_contention import read_events

    result = submit_skew(read_events(args.events))
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "rank_submit_skew.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (destination / "rank_submit_skew.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(result[0]) if result else ["status"])
        writer.writeheader()
        writer.writerows(result)
    valid = [row["cpu_submit_skew_ms"] for row in result if row["cpu_submit_skew_ms"] is not None]
    print(
        json.dumps(
            {
                "matched": len(valid),
                "unmatched": len(result) - len(valid),
                "median_submit_skew_ms": statistics.median(valid) if valid else None,
            }
        )
    )


if __name__ == "__main__":
    main()
