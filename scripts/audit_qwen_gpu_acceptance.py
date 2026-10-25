"""Offline Qwen rank-log audit; recorded edges are not physical GPU dependency proof.

No CUDA execution, SSH, or pickle deserialization is performed. Optional trace
inspection reports observed kernel overlap, never infers correctness from it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def close(actual, expected):
    return finite(actual) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-9)


def discover(root):
    for directory, children, files in os.walk(root, followlinks=False):
        children[:] = [name for name in children if not name.endswith("current")]
        for kind, filename in (
            ("worker", "qwen_correctness.rank0.json"),
            ("benchmark", "benchmark.rank0.json"),
        ):
            if filename in files:
                yield Path(directory), kind


def expected_shards(config, f, t):
    h, m = config["hidden_size"], config["intermediate_size"]
    kv = config["num_key_value_heads"] * h // config["num_attention_heads"]
    n, v = config["num_hidden_layers"], config["vocab_size"]
    require(h % t == kv % t == m % t == 0, "config dimensions do not divide TP")
    per_layer = [
        h * h // t,
        h // t,
        h * kv // t,
        kv // t,
        h * kv // t,
        kv // t,
        h * h // t,
        h * m // t,
        h * m // t,
        h * m // t,
        h,
        h,
    ]
    sizes = [v * h, h] + per_layer * n
    if not config["tie_word_embeddings"]:
        sizes.append(v * h)
    return len(sizes), sum((size + f - 1) // f for size in sizes)


def audit_worker(data, rank, f, t):
    require(data["status"] == "passed" and data["rank"] == rank, "worker status/rank")
    require((data["F"], data["T"]) == (f, t), "worker mesh mismatch")
    require(data["transformers"] == "4.57.3", "HF oracle must be exactly 4.57.3")
    require(data["leaf_identities_stable"] and data["live_buffers"] == 0, "leaf/buffer invariant")
    require(
        data["runtime_tied_parameter_alias"] and data["shared_parameter_owners"] == 1,
        "tied alias/owner invariant",
    )
    require(not data["autograd_saved_full_parameters"], "autograd saved FullParam")
    comparisons = data["comparisons"]
    require(
        comparisons and len({row["name"] for row in comparisons}) == len(comparisons),
        "empty/duplicate comparisons",
    )
    for row in comparisons:
        require(
            row["nan_count"] == 0 and finite(row["max_abs_error"]) and finite(row["max_rel_error"]),
            f"nonfinite comparison {row['name']}",
        )
        require(row["atol"] > 0 and row["rtol"] > 0, "missing comparison tolerance")
    for step in range(data["steps"]):
        groups = defaultdict(dict)
        for row in comparisons:
            if row["name"].startswith(f"step{step}/"):
                name, kind = row["name"].rsplit("/", 1)
                groups[kind][name] = row
        require(
            set(groups["gradient"]) == set(groups["gradient_shard"]) == set(groups["update"]),
            f"step {step}: parameter comparison sets differ",
        )
        require(
            len(groups["gradient"]) == 12 * data["layers"] + 2,
            f"step {step}: incomplete canonical parameter comparisons",
        )
        for name, gradient in groups["gradient"].items():
            shard = groups["gradient_shard"][name]
            require(
                math.prod(shard["shape"]) == (math.prod(gradient["shape"]) + f - 1) // f,
                f"wrong FSDP shard shape: {name}",
            )
            require(
                groups["update"][name]["shape"] == gradient["shape"],
                f"update/gradient shape mismatch: {name}",
            )
        require(
            all(kind in groups for kind in ("logits", "global_loss", "decoder_input_gradient")),
            f"step {step}: missing activation/loss comparisons",
        )
    require(len(data["step_metrics"]) == data["steps"], "missing step metrics")
    for row in data["step_metrics"]:
        require(row["live_buffers"] == 0, "live buffer across steps")
        require(finite(row["reported_global_loss"]), "nonfinite reported global loss")
        if f > 1:
            require(
                len(row["valid_counts_by_replica"]) == f
                and len(set(row["valid_counts_by_replica"])) > 1,
                "unequal token-count reference case missing",
            )
    first = [
        row
        for row in comparisons
        if row["name"].startswith("step0/") and row["name"].endswith("/gradient_shard")
    ]
    shard_numel = sum(math.prod(row["shape"]) for row in first)
    if "persistent_shard_numel" in data:
        require(data["persistent_shard_numel"] == shard_numel, "reported shard numel differs")
        require(data["persistent_shard_bytes"] == 4 * shard_numel, "FP32 shard bytes differ")
    return {
        "tensor_comparisons": len(comparisons),
        "max_abs_error": max(row["max_abs_error"] for row in comparisons),
        "max_abs_error_by_kind": {
            kind: max(
                row["max_abs_error"]
                for row in comparisons
                if row["name"].rsplit("/", 1)[-1] == kind
            )
            for kind in {row["name"].rsplit("/", 1)[-1] for row in comparisons}
        },
        "persistent_shard_numel": shard_numel,
        "persistent_shard_bytes": shard_numel * 4,
        "hf_reference_gate_passed": True,
        "saved_fullparam_check": "worker hook assertion passed; raw tensors not reloaded",
    }


def audit_events(path, rank, f, t, policy, cuda):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    issues, ready, work, protection, active, relations = {}, {}, {}, {}, {}, defaultdict(set)
    input_edges, wait_edges, pg_edges, copy_waits = set(), set(), set(), set()
    launch_edges, launch_contract = set(), None
    completion_sources = Counter()
    orders, counts = defaultdict(list), Counter()
    current_bytes = peak_bytes = 0
    for index, row in enumerate(rows):
        require(row["rank"] == rank, "event rank mismatch")
        event, op, buffer = row["event_type"], row.get("op_id"), row.get("buffer_id")
        counts[event] += 1
        require(event != "collective_failed", "recorded collective failure")
        if event == "collective_issue":
            require(op not in issues, "duplicate communication ticket")
            pg, members = row["pg_id"], tuple(row["group_members"])
            expected = {
                "world": tuple(range(f * t)),
                "tp": tuple((rank // t) * t + i for i in range(t)),
                "fsdp": tuple(i * t + rank % t for i in range(f)),
                "tp_grad": tuple((rank // t) * t + i for i in range(t)),
                "fsdp_grad": tuple(i * t + rank % t for i in range(f)),
            }
            require(pg in expected and members == expected[pg], f"wrong {pg} membership")
            require(row["pg_local_seq"] == len(orders[pg]), f"{pg} sequence gap")
            issues[op] = (index, row)
            orders[pg].append(
                (
                    row["pg_local_seq"],
                    row["op_kind"],
                    row["numel"],
                    row["dtype"],
                    row["unit_id"],
                    row["phase"],
                    row["step_id"],
                    tuple(row["input_shape"]),
                    tuple(row["output_shape"]),
                )
            )
            for identifier in row["input_buffer_ids"] + row["output_buffer_ids"]:
                require(identifier in active, f"ticket uses nonlive buffer {identifier}")
                relations[identifier].add(op)
        elif event == "work_issued":
            require(op in issues and op not in work, "work before issue/duplicate work")
            work[op] = index
        elif event == "nccl_launch_order_contract":
            require(row["NCCL_LAUNCH_ORDER_IMPLICIT"] == "1", "implicit ordering not enabled")
            require(tuple(row["nccl_version"]) >= (2, 26, 0), "unsupported NCCL launch ordering")
            require(
                row["cuda_runtime_version"] >= 12030 and row["cuda_driver_version"] >= 12030,
                "unsupported CUDA launch completion ordering",
            )
            require(not row["gpu_completion_chain"], "implicit order mislabeled as completion")
            launch_contract = row
        elif event == "collective_launch_order":
            previous = row["previous_op_id"]
            require(
                previous in work and op in issues and previous + 1 == op,
                "broken host communication launch order",
            )
            require(
                row["ordering_scope"] == "host_launch" and not row["gpu_completion_dependency"],
                "host launch order must not require previous GPU completion",
            )
            launch_edges.add((previous, op))
        elif event == "collective_consumer_ready":
            require(op in work and index > work[op] and row["consumer_ready"], "ready before work")
            ready.setdefault(op, index)
        elif event == "buffer_allocate":
            require(buffer not in active, "duplicate live buffer ID")
            active[buffer] = row["bytes"]
            current_bytes += row["bytes"]
            peak_bytes = max(peak_bytes, current_bytes)
        elif event == "buffer_retire":
            require(buffer in active and buffer in protection, "unprotected/duplicate retirement")
            require(
                all(ticket in ready for ticket in relations[buffer]),
                "buffer retired before its collective became consumable",
            )
            current_bytes -= active.pop(buffer)
        if event == "dependency_edge":
            dependency = row["dependency_type"]
            require(dependency in {"DATA_DEP", "PG_ORDER", "BUFFER_LIFETIME"}, "unknown edge type")
            if row.get("consumer") == "collective_input":
                input_edges.add(row["consumer_ticket"])
            if row.get("consumer") == "collective_ready_event":
                source = row.get("completion_source", "work_stream_join")
                if source == "current_stream_event":
                    require(
                        row.get("execution_mode") == "nccl_current_stream_v291"
                        and row.get("api_async_op") is False
                        and row.get("work_returned") is False
                        and row.get("work_wait_called") is False
                        and row.get("logical_cuda_stream") is not None
                        and row.get("collective_cuda_stream") == row["logical_cuda_stream"],
                        "invalid current-stream collective completion contract",
                    )
                    issue = issues[row["producer_ticket"]][1]
                    require(
                        issue.get("backend") == "nccl"
                        and issue.get("current_stream_version_qualified") is True
                        and issue.get("torch_build_version", "").split("+", 1)[0] == "2.9.1",
                        "unqualified current-stream NCCL backend",
                    )
                else:
                    require(
                        source == "work_stream_join" and row.get("work_wait_called"),
                        "collective missing backend Work readiness join",
                    )
                completion_sources[source] += 1
                wait_edges.add((row["producer_ticket"], row["wait_stream"]))
            if dependency == "DATA_DEP" and str(row.get("consumer", "")).startswith("copy_out/U"):
                copy_waits.add((row["producer_ticket"], row["consumer"]))
            if dependency == "BUFFER_LIFETIME" and buffer is not None:
                require(buffer in active, "lifetime protection for nonlive buffer")
                require(
                    row.get("retirement") == ("record_stream" if cuda else "last_read_complete"),
                    "retirement protection does not match device",
                )
                protection[buffer] = index
            if dependency == "PG_ORDER":
                previous, target = row["producer_ticket"], row["consumer_ticket"]
                pg_edges.add((previous, target))
                require(
                    previous in work and target in issues and previous + 1 == target,
                    "broken total communication order",
                )
                if policy == "stream_ordered":
                    require(
                        row["wait_stream"] == issues[target][1]["issue_stream"],
                        "ordered PG wait landed on a different stream",
                    )
            if dependency == "DATA_DEP" and row.get("producer_kind") == "fsdp_ag":
                require(
                    row.get("consumer") not in {"sp_rs", "sp_rs_math_allreduce"},
                    "unrelated FSDP AG data dependency on SP RS",
                )
        if event == "copy_out_start" and row.get("producer_ticket") is not None:
            ticket = row["producer_ticket"]
            require(ticket in ready, "copy-out before AG consumer readiness")
            if cuda:
                require(
                    (ticket, f"copy_out/U{row['unit_id']}") in copy_waits,
                    "copy-out missing its AG data wait",
                )
        if event in {"optimizer_ready", "step_end"}:
            require(not active, f"live buffers at {event}")
    require(not active and len(issues) == len(work) == len(ready), "unfinished buffers/tickets")
    if cuda:
        if launch_edges:
            require(policy == "stream_ordered" and not pg_edges, "mixed launch/completion chain")
            require(
                all(
                    issue.get("backend") in {"nccl", "gloo", "local"}
                    for _, issue in issues.values()
                ),
                "host launch ordering evidence missing a supported backend",
            )
            if any(
                issue.get("backend") == "nccl" and len(issue["group_members"]) > 1
                for _, issue in issues.values()
            ):
                require(launch_contract is not None, "missing NCCL implicit ordering contract")
        for op, (_, issue) in issues.items():
            require(op in input_edges, "collective missing input producer event")
            require(
                (op, issue["issue_stream"]) in wait_edges,
                "collective missing its own completion readiness event",
            )
            if op:
                ordering = launch_edges if launch_edges else pg_edges
                require((op - 1, op) in ordering, "missing consecutive communication order")
    return (
        rows,
        dict(orders),
        {
            "events": len(rows),
            "event_counts": dict(counts),
            "collectives": len(issues),
            "buffer_peak_logical_bytes": peak_bytes,
            "recorded_dependency_edges_passed": True,
            "ordering_scope": "host_launch"
            if launch_edges
            else ("legacy_gpu_completion" if cuda else "host_program"),
            "nccl_launch_order_contract": launch_contract,
            "completion_sources": dict(completion_sources),
            "physical_event_completion_verified": False,
        },
    )


def audit_steps(rows, layers, steps, parameters):
    for step in steps:
        selected = [row for row in rows if row["step_id"] == step]
        gathers = [row for row in selected if row["event_type"] == "materialize_issue"]
        reductions = [row for row in selected if row["event_type"] == "unit_grads_ready"]
        require(len(gathers) == 2 * layers + 2, f"step {step}: AG materialization count")
        require(
            not any(row["phase"] == "BWD" and row["unit_id"] in (0, layers + 1) for row in gathers),
            "tied embedding/finalnorm unexpectedly regathered",
        )
        require(
            len(reductions) == layers + 2
            and len({row["unit_id"] for row in reductions}) == layers + 2,
            f"step {step}: missing/duplicate unit gradient reduction",
        )
        leaf_names = [
            row["param_id"] for row in selected if row["event_type"] == "leaf_grad_committed"
        ]
        require(
            len(leaf_names) == len(set(leaf_names)) == parameters, "missing/duplicate leaf commit"
        )
        require(
            sum(row["event_type"] == "optimizer_ready" for row in selected) == 1
            and sum(row["event_type"] == "step_end" for row in selected) == 1,
            "missing/duplicate step boundary",
        )


def audit_pg_sequences(orders, f, t):
    """Compare every communication role, including identical-member separate PGs."""
    for pg, groups in (
        ("world", [list(range(f * t))]),
        ("tp", [[r * t + i for i in range(t)] for r in range(f)]),
        ("fsdp", [[i * t + r for i in range(f)] for r in range(t)]),
        ("tp_grad", [[r * t + i for i in range(t)] for r in range(f)]),
        ("fsdp_grad", [[i * t + r for i in range(f)] for r in range(t)]),
    ):
        for members in groups:
            reference = orders[members[0]].get(pg, [])
            require(
                all(orders[rank].get(pg, []) == reference for rank in members),
                f"PG order/input/output shape divergence: {pg} {members}",
            )


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def overlap(left, right):
    left, right = union(left), union(right)
    total = i = j = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def kernel_summary(kernels, unit):
    groups = defaultdict(lambda: {"nccl": [], "compute": [], "streams": set()})
    nccl = 0
    for start, end, name, process, device, stream in kernels:
        group = groups[(process, device)]
        group["streams"].add(stream)
        group["nccl" if "nccl" in name.lower() else "compute"].append((start, end))
        nccl += "nccl" in name.lower()
    return {
        "gpu_kernels": len(kernels),
        "nccl_kernels": nccl,
        "time_unit": unit,
        "groups": [
            {
                "process": key[0],
                "device": key[1],
                "kernel_streams": sorted(group["streams"]),
                "nccl_compute_overlap": overlap(group["nccl"], group["compute"]),
            }
            for key, group in groups.items()
        ],
        "note": "Kernel intersection is observation, not dependency correctness or speedup.",
    }


def torch_trace(path):
    events = read_json(path)["traceEvents"]
    kernels = [
        (
            row["ts"],
            row["ts"] + row.get("dur", 0),
            row["name"],
            row.get("pid", 0),
            row.get("args", {}).get("device", -1),
            row.get("tid", -1),
        )
        for row in events
        if row.get("cat") == "kernel" and row.get("ph") == "X"
    ]
    return kernel_summary(kernels, "microseconds")


def nsys_trace(path):
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        require({"start", "end", "deviceId", "streamId"} <= columns, "missing Nsight kernel schema")
        name = "demangledName" if "demangledName" in columns else "shortName"
        process = "k.globalPid" if "globalPid" in columns else "0"
        query = (
            f"SELECT k.start,k.end,s.value,{process},k.deviceId,k.streamId "
            f"FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON k.{name}=s.id"
        )
        kernels = [
            (start, end, label or "", pid, device, stream)
            for start, end, label, pid, device, stream in db.execute(query)
        ]
    return kernel_summary(kernels, "nanoseconds")


def audit_profile(directory, rank, cuda, args, expected_steps):
    path = directory / f"profiling.rank{rank}.json"
    if not path.is_file() and not args.require_profile:
        return {
            "mode": "unavailable",
            "artifacts": [],
            "trace": None,
            "note": "Historical result has no profiling metadata; no profile claim.",
        }
    require(path.is_file(), "missing profiling metadata")
    data = read_json(path)
    require(not data["errors"], "profiling recorded errors")
    mode = data["mode"]
    if args.require_profile:
        require(mode != "off", "profiling required but disabled")
    artifacts = []
    if cuda and mode != "off":
        for suffix in ("start", "end"):
            snapshot = directory / f"memory.{suffix}.rank{rank}.pickle"
            require(snapshot.is_file() and snapshot.stat().st_size > 0, "missing CUDA snapshot")
            artifacts.append({"path": str(snapshot), "bytes": snapshot.stat().st_size})
    trace = None
    if mode == "torch":
        path = directory / f"profile.rank{rank}.json"
        require(path.is_file() and path.stat().st_size > 0, "missing Torch timeline")
        artifacts.append({"path": str(path), "bytes": path.stat().st_size})
        if args.inspect_traces:
            trace = torch_trace(path)
            if cuda:
                require(trace["gpu_kernels"] > 0, "Torch timeline contains no GPU kernels")
    memory = data["memory"]
    if cuda and mode != "off":
        require(len(memory) == expected_steps, "missing allocator step measurements")
        for row in memory:
            require(
                all(
                    finite(row[key]) and row[key] >= 0
                    for key in ("allocated", "reserved", "max_allocated", "max_reserved")
                ),
                "invalid allocator measurement",
            )
            require(
                row["max_reserved"] >= row["max_allocated"] >= row["allocated"]
                and row["reserved"] >= row["allocated"],
                "allocator peak/end inconsistency",
            )
    return {
        "mode": mode,
        "artifacts": artifacts,
        "trace": trace,
        "allocator_max_allocated": max((row["max_allocated"] for row in memory), default=None),
        "allocator_max_reserved": max((row["max_reserved"] for row in memory), default=None),
        "allocator_end_drift": memory[-1]["allocated"] - memory[0]["allocated"] if memory else None,
        "snapshot_content_validated": False,
        "note": "File presence checked; allocator metadata excludes direct CUDA/NCCL memory.",
    }


def audit_case(directory, kind, args):
    prefix = "qwen_correctness" if kind == "worker" else "benchmark"
    first = read_json(directory / f"{prefix}.rank0.json")
    config_path = directory / "resolved_config.rank0.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else None
    if kind == "worker":
        f, t, layers, steps = first["F"], first["T"], first["layers"], range(first["steps"])
        policy = first["policy"]
        flags = first["flags"]
    else:
        require(config is not None, "benchmark requires resolved config")
        f, t = config["parallel"]["fsdp_degree"], config["parallel"]["tp_degree"]
        layers = first["model_source"]["config"]["num_hidden_layers"]
        policy = first["communication_policy"]
        flags = "".join(
            str(int(first["runtime_switches"][key]))
            for key in ("forward_prefetch", "backward_prefetch", "async_grad_rs")
        )
        steps = range(first["warmup"] + first["steady_steps_requested"])
    ranks, orders, payloads = [], [], []
    for rank in range(f * t):
        data = read_json(directory / f"{prefix}.rank{rank}.json")
        rank_config_path = directory / f"resolved_config.rank{rank}.yaml"
        require(rank_config_path.is_file(), "missing per-rank resolved config")
        rank_config = yaml.safe_load(rank_config_path.read_text())
        resolved_flags = "".join(
            str(int(rank_config["runtime"][key]))
            for key in ("forward_prefetch", "backward_prefetch", "async_grad_rs")
        )
        require(resolved_flags == flags, "flags do not match resolved per-rank config")
        require(
            rank_config["runtime"]["communication_policy"] == policy,
            "policy does not match resolved per-rank config",
        )
        require(
            rank_config["parallel"]["fsdp_degree"] == f
            and rank_config["parallel"]["tp_degree"] == t,
            "mesh does not match resolved per-rank config",
        )
        if kind == "worker":
            require(
                data["flags"] == flags
                and data["policy"] == policy
                and data["SP"] == rank_config["parallel"]["sp_enabled"],
                "worker rank/config policy, flags or SP disagree",
            )
        cuda = data.get("device", "").startswith("cuda")
        backend = data.get("backend", config["parallel"]["backend"] if config else "unknown")
        if args.require_nccl:
            require(
                f * t > 1 and cuda and backend == "nccl", "actual multi-rank CUDA/NCCL required"
            )
            require(
                kind != "worker" or data.get("actual_NCCL") is True,
                "worker did not verify initialized NCCL",
            )
            environment = read_json(directory / f"environment.rank{rank}.json")
            require(
                environment["nccl_available"] and environment["world_size"] == f * t,
                "NCCL/environment world-size mismatch",
            )
            require(
                data["device"] == f"cuda:{environment['local_rank']}",
                "runtime device does not match local rank binding",
            )
        if kind == "worker":
            detail = audit_worker(data, rank, f, t)
            parameters = 12 * layers + 2
        else:
            require(data["status"] == "completed" and data["rank"] == rank, "benchmark status/rank")
            require(
                data["world_size"] == f * t
                and not data["errors"]
                and not data["oom"]
                and not data["nonfinite_detected"],
                "benchmark failure flags",
            )
            own = data["ownership"]
            parameters, numel = expected_shards(data["model_source"]["config"], f, t)
            require(
                own["persistent_shard_count"] == parameters
                and own["persistent_shard_numel_including_padding"] == numel,
                "benchmark shard counts",
            )
            require(
                own["parameter_bytes"] == own["gradient_bytes"] == numel * 4,
                "FP32 shard/grad bytes",
            )
            require(
                own["stable_shard_identity"]
                and own["units_reduced_once_last_microbatch"]
                and own["live_buffers"] == own["live_fullparam_leases"] == 0,
                "benchmark ownership",
            )
            state_count = own["optimizer_state_parameter_count"]
            if data["optimizer"] == "adamw" and state_count:
                require(
                    state_count == parameters
                    and own["optimizer_tensor_bytes"] == numel * 8 + parameters * 4,
                    "AdamW FP32 local state bytes/count",
                )
            if data["optimizer"] == "sgd":
                require(own["optimizer_tensor_bytes"] == 0, "unexpected SGD optimizer state")
            detail = {
                "persistent_shard_numel": numel,
                "persistent_shard_bytes": numel * 4,
                "hf_reference_gate_passed": False,
                "saved_fullparam_check": "not provided by benchmark; run independent worker gate",
            }
        rows, pg, events = audit_events(
            directory / f"runtime_events.rank{rank}.jsonl", rank, f, t, policy, cuda
        )
        # Benchmark may use multiple microbatches per optimizer step.
        if kind == "worker" or data.get("accumulation_steps", 1) == 1:
            require(
                kind == "worker" or data["model_source"]["config"]["tie_word_embeddings"],
                "untied benchmark unit plan requires separate lifecycle audit",
            )
            audit_steps(rows, layers, steps, parameters)
        capture_steps = data["steps"] if kind == "worker" else data["steady_steps_requested"]
        profiles = audit_profile(directory, rank, cuda, args, capture_steps)
        if args.require_nccl and profiles["trace"] is not None:
            require(profiles["trace"]["nccl_kernels"] > 0, "timeline has no NCCL kernels")
        ranks.append(
            {
                "rank": rank,
                "device": data.get("device"),
                "backend": backend,
                **detail,
                "events": events,
                "profiling": profiles,
            }
        )
        orders.append(pg)
        payloads.append(data)
    audit_pg_sequences(orders, f, t)
    if kind == "benchmark":
        require(
            all(len(data["steps"]) == first["steady_steps_requested"] for data in payloads),
            "benchmark incomplete measured steps",
        )
        for index in range(first["steady_steps_requested"]):
            samples = [data["steps"][index] for data in payloads]
            duration = max(row["seconds"] for row in samples)
            tokens = samples[0]["valid_tokens"]
            require(
                duration > 0 and all(finite(row["loss"]) for row in samples),
                "nonfinite/timing sample",
            )
            require(
                all(
                    row["valid_tokens"] == tokens
                    and close(row["max_rank_seconds"], duration)
                    and close(row["tokens_per_second"], tokens / duration)
                    for row in samples
                ),
                "incorrect global token/max-rank timing reduction",
            )
            require(
                len({row["overflow"] for row in samples}) == 1,
                "ranks disagreed on overflow/optimizer update",
            )
        durations = sorted(row["max_rank_seconds"] for row in first["steps"])
        for quantile, key in ((0.5, "step_seconds_p50"), (0.95, "step_seconds_p95")):
            position = (len(durations) - 1) * quantile
            lo, hi = math.floor(position), math.ceil(position)
            expected = durations[lo] + (durations[hi] - durations[lo]) * (position - lo)
            require(
                all(close(data[key], expected) for data in payloads),
                "benchmark percentile mismatch",
            )
        throughput = sum(row["valid_tokens"] for row in first["steps"]) / sum(durations)
        require(
            all(close(data["tokens_per_second"], throughput) for data in payloads),
            "benchmark summary throughput mismatch",
        )
    native = []
    for location in (directory, directory.parent):
        for path in location.glob("*.nsys-rep"):
            native.append({"path": str(path), "bytes": path.stat().st_size})
    if any(row["profiling"]["mode"] == "nsight" for row in ranks):
        require(
            native and all(row["bytes"] > 0 for row in native),
            "Nsight mode has no nonempty native .nsys-rep",
        )
    sqlite = []
    if args.inspect_traces:
        for location in (directory, directory.parent):
            for path in location.glob("*.sqlite"):
                sqlite.append({"path": str(path), "trace": nsys_trace(path)})
        if any(row["profiling"]["mode"] == "nsight" for row in ranks):
            require(sqlite, "export Nsight SQLite before requesting trace inspection")
            if args.require_nccl:
                require(
                    any(row["trace"]["nccl_kernels"] > 0 for row in sqlite),
                    "Nsight capture has no NCCL kernel evidence",
                )
    return {
        "status": "passed",
        "directory": str(directory),
        "kind": kind,
        "F": f,
        "T": t,
        "SP": first.get("SP", config["parallel"]["sp_enabled"] if config else None),
        "layers": layers,
        "policy": policy,
        "flags": flags,
        "ranks": ranks,
        "pg_sequences_passed": True,
        "native_traces": native,
        "nsys_sqlite": sqlite,
        "physical_dependencies_verified": False,
        "overlap_success_verified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-nccl", action="store_true")
    parser.add_argument("--require-profile", action="store_true")
    parser.add_argument(
        "--inspect-traces",
        action="store_true",
        help="Parse Chrome/SQLite kernels; may use substantial RAM",
    )
    parser.add_argument("--kind", choices=("worker", "benchmark", "all"), default="all")
    parser.add_argument(
        "--require-full-flags",
        action="store_true",
        help="Require 4-layer F2T2/SP HF worker gates for 8 flags x 2 policies",
    )
    parser.add_argument(
        "--required-depths",
        default="",
        help="Require F2T2/SP conservative000 and stream_ordered111, e.g. 4,12,28",
    )
    args = parser.parse_args(argv)
    cases = []
    for directory, kind in discover(args.root):
        if args.kind != "all" and args.kind != kind:
            continue
        try:
            cases.append(audit_case(directory, kind, args))
        except (ValueError, KeyError, OSError, sqlite3.Error) as error:
            cases.append(
                {"status": "failed", "directory": str(directory), "kind": kind, "error": str(error)}
            )
    coverage = defaultdict(set)
    for case in cases:
        if (
            case["status"] == "passed"
            and case["kind"] == "worker"
            and (case["F"], case["T"], case["SP"]) == (2, 2, True)
        ):
            coverage[case["layers"]].add((case["policy"], case["flags"]))
    missing = []
    if args.require_full_flags:
        for policy in ("conservative", "stream_ordered"):
            for value in range(8):
                if (policy, f"{value:03b}") not in coverage[4]:
                    missing.append(f"N4 F2T2/SP {policy}/{value:03b}")
    try:
        depths = [int(value) for value in args.required_depths.split(",") if value]
    except ValueError:
        parser.error("required-depths must be comma-separated integers")
    for layers in depths:
        for policy, flags in (("conservative", "000"), ("stream_ordered", "111")):
            if (policy, flags) not in coverage[layers]:
                missing.append(f"N{layers} F2T2/SP {policy}/{flags}")
    passed = bool(cases) and not missing and all(case["status"] == "passed" for case in cases)
    result = {
        "status": "passed" if passed else "failed",
        "cases": cases,
        "case_count": len(cases),
        "tensor_comparisons": sum(
            rank.get("tensor_comparisons", 0)
            for case in cases
            if case["status"] == "passed"
            for rank in case["ranks"]
        ),
        "max_abs_error": max(
            (
                rank["max_abs_error"]
                for case in cases
                if case["status"] == "passed"
                for rank in case["ranks"]
                if "max_abs_error" in rank
            ),
            default=None,
        ),
        "hf_worker_flags_coverage": {
            str(layers): [f"{policy}/{flags}" for policy, flags in sorted(combinations)]
            for layers, combinations in sorted(coverage.items())
        },
        "required_matrix_missing": missing,
        "scope": "Offline numeric/log/lifetime/profile artifact audit. "
        "Recorded edges do not prove physical completion; overlap does not prove speedup.",
        "physical_dependency_correctness_verified": False,
        "overlap_success_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    for case in cases:
        if case["status"] == "failed":
            print(f"FAIL {case['directory']}: {case['error']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
