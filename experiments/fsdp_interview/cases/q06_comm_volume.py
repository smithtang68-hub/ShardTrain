"""Tensor-payload inventory and per-rank ideal-ring model, never wire-byte claims."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def modeled_send_bytes(row):
    """One rank's one-direction sends in a hypothetical ring (not HBM traffic)."""
    size = row["group_size"]
    factor = (size - 1) / size
    kind = row["collective_type"]
    if kind in {"fsdp_ag", "sp_ag"}:
        return factor * row["output_bytes"], "ideal_ring_allgather"
    if kind in {"fsdp_rs", "sp_rs"}:
        return factor * row["input_bytes"], "ideal_ring_reduce_scatter"
    if "allreduce" in kind:
        return 2 * factor * row["input_bytes"], "ideal_ring_allreduce"
    return None, "not_modeled"


def payload_checks(row):
    kind, size = row["collective_type"], row["group_size"]
    if kind in {"fsdp_ag", "sp_ag"}:
        return row["output_bytes"] == size * row["input_bytes"]
    if kind in {"fsdp_rs", "sp_rs"}:
        return row["input_bytes"] == size * row["output_bytes"]
    return None


def accounting_family(row):
    """The runtime norm kind also carries large replicated embedding gradients."""
    if row["collective_type"] == "tp_norm_allreduce":
        if row.get("group_role") == "norm_gradient_bucket":
            return "tp_norm_gradient_allreduce"
        if row.get("group_role") in {"tp_replicated_gradient", "replicated_gradient"}:
            return "tp_replicated_gradient_allreduce"
        return "tp_norm_allreduce_unclassified"
    return row["family"]


def volume_inventory(rows, *, measured_only=True):
    inventory = defaultdict(
        lambda: {
            "collective_count": 0,
            "input_bytes": 0,
            "output_bytes": 0,
            "modeled_ring_send_bytes": 0.0,
            "unmodeled_count": 0,
            "payload_ratio_failures": 0,
        }
    )
    enriched = []
    for row in rows:
        if row["phase"] not in {"FWD", "BWD"}:
            continue
        if measured_only and row.get("measured_step") is None:
            continue
        send, model = modeled_send_bytes(row)
        checked = payload_checks(row)
        event = {
            **row,
            "runtime_family": row["family"],
            "family": accounting_family(row),
            "modeled_ring_send_bytes": send,
            "ring_model": model,
            "payload_ratio_valid": checked,
        }
        enriched.append(event)
        key = (
            row["rank"],
            row["measured_step"],
            row["runtime_microstep"],
            row["unit_id"],
            row["phase"],
            accounting_family(row),
            row["process_group_label"],
            tuple(row["group_members"]),
            row["dtype"],
            row.get("group_role"),
        )
        totals = inventory[key]
        totals["collective_count"] += 1
        totals["input_bytes"] += row["input_bytes"]
        totals["output_bytes"] += row["output_bytes"]
        totals["modeled_ring_send_bytes"] += send or 0
        totals["unmodeled_count"] += int(send is None)
        totals["payload_ratio_failures"] += int(checked is False)
    output = []
    labels = (
        "rank",
        "measured_step",
        "runtime_microstep",
        "unit_id",
        "phase",
        "family",
        "process_group_label",
        "group_members",
        "dtype",
        "group_role",
    )
    for key, values in sorted(inventory.items(), key=lambda item: str(item[0])):
        output.append(dict(zip(labels, key, strict=True), **values))
    return output, enriched


def launch_phase_prediction(layout, unit, runtime_switches):
    """Source-derived issue-vs-first-consume phase correction for a tied root.

    RuntimeQwenLinear(column=False, tied=True) → QwenRowLinear.forward calls
    the tied embedding handle's prefetch_after_input(FWD) again at LM head use.
    Unit 1 was already resharded; its new AG lease stays until BWD first-copy-out.
    This predicts 2/0 issue labels and 1/1 first-copy-out uses independently of
    observed events. No mutation or relabeling of the live schedule is performed.
    """
    use_fwd, use_bwd = unit["expected_parameter_ag_fwd"], unit["expected_parameter_ag_bwd"]
    tied = any(
        item.get("unit_class") == "TiedEmbeddingUnit" and item["unit_id"] == 0
        for item in layout["units"]
    )
    late_tied_prefetch = (
        tied
        and unit["unit_id"] == 1
        and use_bwd > 0
        and runtime_switches.get("forward_prefetch") is True
        and runtime_switches.get("communication_policy") == "stream_ordered"
        and layout.get("sp_enabled") is True
        and layout.get("reshard_after_forward") is True
    )
    return {
        "launch_fwd": use_fwd + use_bwd if late_tied_prefetch else use_fwd,
        "launch_bwd": 0 if late_tied_prefetch else use_bwd,
        "first_consume_fwd": use_fwd,
        "first_consume_bwd": use_bwd,
        "late_tied_root_prefetch": late_tied_prefetch,
        "reason": (
            "Tied LM head QwenRowLinear.forward repeats unit0 handle's FWD prefetch "
            "to resharded unit1; lease first consumed by BWD copy-out"
            if late_tied_prefetch
            else "ordinary materialization schedule"
        ),
    }


def first_consumer_phases(runtime_rows):
    """Exact producer_ticket→copy_out_start phase; no nearest-time inference."""
    result = {}
    for row in runtime_rows:
        if row.get("event_type") != "copy_out_start" or row.get("producer_ticket") is None:
            continue
        key = row["rank"], row["producer_ticket"]
        value = row["phase"]
        if key in result and result[key] != value:
            raise ValueError(f"Conflicting first-copy-out phase for ticket {key}")
        result[key] = value
    return result


def expected_vs_observed(rows, manifests, consumer_phases=None):
    """Independently predict FSDP counts/bytes using registered parameter specs.

    Special cases: tied embedding retained FWD/BWD; last parameter unit retained;
    norm/main gradient buckets split but their payload sums to one full gradient.
    Accumulation has one real AG/RS schedule per microstep in this runtime.
    """
    layouts = {}
    for manifest in manifests:
        rank = manifest.get("rank")
        if rank is None:
            # Event files supply rank if a manifest omitted a direct rank field.
            rank = manifest.get("stream_rank")
        candidates = manifest["runtime_layouts"]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one runtime layout per rank run")
        layouts[rank] = candidates[0], manifest.get("runtime_switches", {})
    grouped = defaultdict(list)
    for row in rows:
        if row.get("measured_step") is None or row["phase"] not in {"FWD", "BWD"}:
            continue
        grouped[(row["rank"], row["measured_step"], row["runtime_microstep"])].append(row)
    result = []
    for (rank, step, microstep), events in sorted(grouped.items()):
        if rank not in layouts:
            raise ValueError(f"No independent parameter layout for rank {rank}")
        layout, switches = layouts[rank]
        for unit in layout["units"]:
            subset = [row for row in events if row["unit_id"] == unit["unit_id"]]
            forward = [
                row
                for row in subset
                if row["collective_type"] == "fsdp_ag" and row["phase"] == "FWD"
            ]
            backward = [
                row
                for row in subset
                if row["collective_type"] == "fsdp_ag" and row["phase"] == "BWD"
            ]
            reductions = [row for row in subset if row["collective_type"] == "fsdp_rs"]
            if any(row["collective_type"] == "fsdp_rs_math_allreduce" for row in subset):
                raise ValueError("Gloo math RS is not a NCCL reduce-scatter byte comparison")
            prediction = launch_phase_prediction(layout, unit, switches)
            expected_fwd = prediction["launch_fwd"]
            expected_bwd = prediction["launch_bwd"]
            expected_rs = unit["expected_gradient_rs_buckets"]
            size = unit["padded_full_parameter_bytes"]
            expected_ag_bytes = (expected_fwd + expected_bwd) * size
            expected_rs_bytes = size if layout["fsdp_degree"] > 1 else 0
            observed_ag_bytes = sum(row["output_bytes"] for row in forward + backward)
            observed_rs_bytes = sum(row["input_bytes"] for row in reductions)
            passed = (
                len(forward) == expected_fwd
                and len(backward) == expected_bwd
                and len(reductions) == expected_rs
                and observed_ag_bytes == expected_ag_bytes
                and observed_rs_bytes == expected_rs_bytes
            )
            consumers = (
                [consumer_phases.get((rank, row["runtime_op_id"])) for row in forward + backward]
                if consumer_phases is not None
                else []
            )
            consume_complete = len(consumers) == len(forward + backward) and all(
                phase in {"FWD", "BWD"} for phase in consumers
            )
            observed_use_fwd = consumers.count("FWD") if consume_complete else None
            observed_use_bwd = consumers.count("BWD") if consume_complete else None
            result.append(
                {
                    "rank": rank,
                    "measured_step": step,
                    "runtime_microstep": microstep,
                    "unit_id": unit["unit_id"],
                    "status": "PASS" if passed else "FAIL",
                    "expected_fwd_ag_count": expected_fwd,
                    "observed_fwd_ag_count": len(forward),
                    "expected_bwd_ag_count": expected_bwd,
                    "observed_bwd_ag_count": len(backward),
                    "count_phase_scope": "AG issue/launch label; not first consumer phase",
                    "expected_first_consume_fwd_ag_count": prediction["first_consume_fwd"],
                    "expected_first_consume_bwd_ag_count": prediction["first_consume_bwd"],
                    "observed_first_consume_fwd_ag_count": observed_use_fwd,
                    "observed_first_consume_bwd_ag_count": observed_use_bwd,
                    "first_consume_phase_status": (
                        "PASS"
                        if observed_use_fwd == prediction["first_consume_fwd"]
                        and observed_use_bwd == prediction["first_consume_bwd"]
                        else "FAIL"
                    )
                    if consume_complete
                    else "NOT_CAPTURED",
                    "late_tied_root_prefetch": prediction["late_tied_root_prefetch"],
                    "phase_prediction_reason": prediction["reason"],
                    "expected_rs_bucket_count": expected_rs,
                    "observed_rs_bucket_count": len(reductions),
                    "expected_full_ag_bytes": expected_ag_bytes,
                    "observed_full_ag_bytes": observed_ag_bytes,
                    "expected_full_gradient_bytes": expected_rs_bytes,
                    "observed_full_gradient_bytes": observed_rs_bytes,
                    "padding_bytes_per_full_unit": size - unit["unpadded_tp_local_parameter_bytes"],
                    "unit_retains_forward": unit["retains_forward_parameters"],
                    "note": (
                        "Counts predicted from registered parameter units; SP/norm traffic excluded"
                    ),
                }
            )
    return result


def message_sizes(rows):
    """NCCL-tests -b/-e size is full AG output / full RS input.

    AllGatherGetCollByteCount and ReduceScatterGetCollByteCount split count by
    nranks (with 16-byte alignment); base BW uses paramcount*eltSize*nranks.
    The pinned build source still needs a checksum in the experiment evidence.
    """
    sizes = defaultdict(set)
    for row in rows:
        if row["collective_type"] in {"fsdp_ag", "fsdp_rs"}:
            sizes[row["collective_type"]].add(
                (row["input_bytes"], row["output_bytes"], row["dtype"])
            )
    return {
        key: [
            {
                "input_bytes_per_rank": value[0],
                "output_bytes_per_rank": value[1],
                "nccl_tests_size_bytes": value[1] if key == "fsdp_ag" else value[0],
                "dtype": value[2],
            }
            for value in sorted(values)
        ]
        for key, values in sizes.items()
    }


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--runtime-events", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    from experiments.fsdp_interview.cases.q05_nccl_contention import read_events

    events = list(read_events(args.events))
    manifests = []
    for path in args.manifests:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        benchmark_path = Path(path).with_name(
            Path(path).name.replace("communication_manifest", "benchmark")
        )
        if benchmark_path.is_file():
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            manifest["runtime_switches"] = benchmark["config"]["runtime"]
            manifest["independent_runtime_config_source"] = str(benchmark_path)
        if "rank" not in manifest:
            import re

            match = re.search(r"rank(\d+)", Path(path).name)
            if not match:
                raise ValueError(f"Cannot identify manifest rank from {path}")
            manifest["rank"] = int(match.group(1))
        manifests.append(manifest)
    consumer_phases = (
        first_consumer_phases(read_events(args.runtime_events)) if args.runtime_events else None
    )
    inventory, enriched = volume_inventory(events)
    for event in enriched:
        event["launch_phase"] = event["phase"]
        event["first_consume_phase"] = (
            consumer_phases.get((event["rank"], event["runtime_op_id"]))
            if consumer_phases is not None
            else None
        )
    checks = expected_vs_observed(events, manifests, consumer_phases)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "collective_inventory.csv", inventory)
    write_csv(destination / "expected_vs_observed.csv", checks)
    (destination / "collective_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )
    with (destination / "collective_events_enriched.jsonl").open("w", encoding="utf-8") as target:
        for row in enriched:
            target.write(json.dumps(row) + "\n")
    summary = {
        "status": "PASS" if checks and all(row["status"] == "PASS" for row in checks) else "FAIL",
        "event_count": len(enriched),
        "unit_checks": len(checks),
        "failed_unit_checks": [row for row in checks if row["status"] != "PASS"],
        "first_consume_phase_checks": {
            key: sum(row["first_consume_phase_status"] == key for row in checks)
            for key in ("PASS", "FAIL", "NOT_CAPTURED")
        },
        "tied_root_second_use_quirk": {
            "units": sorted({row["unit_id"] for row in checks if row["late_tied_root_prefetch"]}),
            "source_prediction": (
                "QwenRowLinear.forward at tied LM head invokes unit0 FWD boundary again; "
                "resharded unit1 issues FWD-labelled AG then first consumes it in BWD"
            ),
            "total_ag_bytes_effect": "no additional AG; phase labels shift 1/1 to 2/0 for unit1",
        },
        "nccl_test_message_sizes": message_sizes(enriched),
        "traffic_scope": "per-rank, one-direction ideal Ring send model; not link/HBM measurement",
        "accumulation_scope": (
            "sum actual microsteps per optimizer step; runtime RS happens each microstep"
        ),
    }
    (destination / "communication_volume_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
