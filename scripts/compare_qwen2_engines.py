"""Compare two unprofiled four-rank runs; this is not a correctness oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import yaml

from sharded_runtime.training.performance import qwen_performance_summary


class ComparisonError(ValueError):
    """Incomplete results or mismatched paired conditions."""


def require(condition, message):
    if not condition:
        raise ComparisonError(message)


def finite(value, *, positive=False):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (not positive or value > 0)
    )


def same(actual, expected, label):
    require(actual == expected, f"{label} differs: {actual!r} versus {expected!r}")


def needed(record, key, label):
    require(key in record and record[key] is not None, f"Missing {label}.{key}")
    return record[key]


def read_json(path):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ComparisonError(f"Cannot read {path}: {exc}") from exc
    require(isinstance(record, dict), f"Expected JSON object in {path}")
    return record


def resolved_config(directory, rank):
    path = directory / f"resolved_config.rank{rank}.yaml"
    if path.is_file():
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ComparisonError(f"Cannot read {path}: {exc}") from exc
    else:
        path = directory / f"environment.rank{rank}.json"
        config = needed(read_json(path), "config", str(path))
    require(isinstance(config, dict), f"Invalid resolved config in {path}")
    for section in ("model", "parallel", "training"):
        require(isinstance(config.get(section), dict), f"Missing {path}:{section}")
    return config, path


def topology(directory):
    status_path = directory / "gpu_topology_status.json"
    status = read_json(status_path)
    same(status.get("returncode"), 0, f"{directory}: successful topology capture")
    path = directory / "gpu_topology.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ComparisonError(f"Cannot read {path}: {exc}") from exc
    normalized = "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())
    require(all(f"GPU{rank}" in normalized for rank in range(4)), f"Incomplete topology in {path}")
    return normalized, [path, status_path]


def validate_steps(record, config, label):
    warmup = needed(record, "warmup", label)
    requested = needed(config["training"], "steps", label + ".training")
    require(type(warmup) is int and warmup > 0, label + ": positive warmup required")
    require(type(requested) is int and requested > 0, label + ": invalid measured steps")
    steps = needed(record, "steps", label)
    warmup_steps = needed(record, "warmup_steps", label)
    require(isinstance(steps, list) and len(steps) == requested, label + ": incomplete steps")
    require(
        isinstance(warmup_steps, list) and len(warmup_steps) == warmup,
        label + ": incomplete warmup",
    )
    same(record.get("steady_steps_completed"), requested, label + ": completed steps")
    if "steady_steps_requested" in record:
        same(record["steady_steps_requested"], requested, label + ": requested steps")
    same(record.get("optimizer_updates"), requested, label + ": optimizer update count")
    same(record.get("overflow_steps"), 0, label + ": measured overflow")
    same(record.get("warmup_overflow_steps"), 0, label + ": warmup overflow")
    for index, row in enumerate(warmup_steps + steps):
        require(isinstance(row, dict), label + ": invalid step record")
        same(row.get("step"), index, label + ": optimizer step index")
        same(row.get("overflow"), False, label + ": overflow flag")
        require(finite(row.get("loss")), label + ": nonfinite/missing loss")
        require(finite(row.get("loss_scale"), positive=True), label + ": invalid loss scale")
        same(row.get("accumulation_steps"), record["accumulation_steps"], label + ": accumulation")
        require(
            type(row.get("valid_tokens")) is int and row["valid_tokens"] > 0,
            label + ": invalid global valid-token count",
        )
        if index >= warmup:
            same(row.get("measured_step"), index - warmup, label + ": measured step index")
            for field in ("seconds", "max_rank_seconds", "cuda_elapsed_ms"):
                require(finite(row.get(field), positive=True), label + ": invalid " + field)
    return steps


def load_run(directory, engine):
    directory = Path(directory).resolve()
    require(directory.is_dir(), f"Run directory does not exist: {directory}")
    same(
        {path.name for path in directory.glob("benchmark.rank*.json")},
        {f"benchmark.rank{rank}.json" for rank in range(4)},
        f"{directory}: exactly four rank results",
    )
    require(
        not any(
            path.is_file()
            for pattern in ("*.nsys-rep", "*.qdrep", "*.sqlite")
            for path in directory.rglob(pattern)
        ),
        f"{directory}: Nsight capture artifacts found in unprofiled run",
    )
    records, configs, inputs = [], [], []
    for rank in range(4):
        path = directory / f"benchmark.rank{rank}.json"
        record = read_json(path)
        label = f"{engine}.rank{rank}"
        for field, expected in {
            "engine": engine,
            "rank": rank,
            "world_size": 4,
            "coordinate": {"fsdp": rank // 2, "tp": rank % 2},
            "status": "completed",
            "errors": [],
            "oom": False,
            "nonfinite_loss": False,
            "nonfinite_detected": False,
            "profile_mode": "off",
        }.items():
            same(record.get(field), expected, label + ": " + field)
        if engine == "shardtrain":
            same(record.get("runtime_events_enabled"), False, label + ": runtime event logging")
            require(
                not (directory / f"runtime_events.rank{rank}.jsonl").exists(),
                label + ": runtime event log in unprofiled run",
            )
        profile_path = directory / f"profiling.rank{rank}.json"
        if profile_path.is_file():
            profile = read_json(profile_path)
            same(profile.get("mode"), "off", label + ": saved profiling mode")
            same(profile.get("errors"), [], label + ": profiling errors")
            inputs.append(profile_path)
        require(
            not (directory / f"profile.rank{rank}.json").exists()
            and not any(directory.glob(f"memory.*.rank{rank}.pickle")),
            label + ": profiling artifacts found",
        )
        config, config_path = resolved_config(directory, rank)
        parallel, training = config["parallel"], config["training"]
        for field, expected in {
            "fsdp_degree": 2,
            "tp_degree": 2,
            "sp_enabled": True,
            "device": "cuda",
            "backend": "nccl",
        }.items():
            same(parallel.get(field), expected, label + ".parallel." + field)
        same(config["model"].get("model_type"), "qwen2", label + ": Qwen model")
        for field, expected in {
            "dtype": training.get("dtype"),
            "optimizer": training.get("optimizer"),
            "micro_batch": training.get("local_batch_size"),
            "accumulation_steps": training.get("accumulation_steps"),
            "sequence_length": config["model"].get("seq_len"),
        }.items():
            require(expected is not None, label + ": missing resolved " + field)
            same(record.get(field), expected, label + ": resolved " + field)
        for field in ("micro_batch", "accumulation_steps", "sequence_length"):
            require(type(record[field]) is int and record[field] > 0, label + ": invalid " + field)
        same(
            record.get("global_batch"),
            record["micro_batch"] * record["accumulation_steps"] * 2,
            label + ": unique global batch",
        )
        source = needed(record, "model_source", label)
        require(isinstance(source.get("config"), dict), label + ": missing model config")
        same(source.get("seed"), training.get("seed"), label + ": source seed")
        require(type(source.get("seed")) is int, label + ": missing integer source seed")
        require(
            type(source.get("global_unique_parameter_numel")) is int
            and source["global_unique_parameter_numel"] > 0,
            label + ": missing global unique parameter count",
        )
        for field in ("num_hidden_layers", "hidden_size"):
            require(
                type(source["config"].get(field)) is int and source["config"][field] > 0,
                label + ": invalid model " + field,
            )
        needed(source, "initialization", label + ".model_source")
        require(
            source["initialization"] == "random",
            label + ": checkpoint-initialized pairing requires an actual shared weight digest; "
            "the current benchmark schema only identifies random initialization by source seed",
        )
        needed(source, "tied_aliases", label + ".model_source")
        digest = needed(record, "input_batch_digest", label)
        require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), label + ": digest"
        )
        policy = needed(record, "precision_policy", label)
        require(isinstance(policy, dict), label + ": invalid precision policy")
        require(record["dtype"] in {"fp16", "bf16", "fp32"}, label + ": unknown compute dtype")
        for field in ("compute", "linear_backward"):
            same(policy.get(field), record["dtype"], label + ".precision_policy." + field)
        for field in (
            "parameters",
            "parameter_all_gather",
            "gradient_reduce",
            "optimizer_state",
            "norm_accumulation",
            "loss",
        ):
            same(policy.get(field), "fp32", label + ".precision_policy." + field)
        defaults = needed(record, "optimizer_hyperparameters", label)
        require(isinstance(defaults, dict), label + ": invalid optimizer hyperparameters")
        same(defaults.get("lr"), training.get("lr"), label + ": optimizer learning rate")
        versions = needed(record, "versions", label)
        for field in ("torch", "cuda_build", "nccl", "platform"):
            needed(versions, field, label + ".versions")
        for field in ("device", "device_name", "device_capability", "attention_backend"):
            needed(record, field, label)
        same(record["device"], f"cuda:{rank}", label + ": device mapping")
        require(len(record["device_capability"]) == 2, label + ": invalid CUDA capability")
        steps = validate_steps(record, config, label)
        for field in ("max_allocated", "max_reserved", "allocated_end", "reserved_end"):
            require(
                type(record.get(field)) is int and record[field] > 0, label + ": invalid " + field
            )
        performance = needed(record, "performance", label)
        peak = needed(performance, "peak_tflops_per_gpu", label + ".performance")
        require(finite(peak, positive=True), label + ": peak must be explicit and positive")
        expected = qwen_performance_summary(
            parameter_numel=source["global_unique_parameter_numel"],
            layers=source["config"]["num_hidden_layers"],
            hidden_size=source["config"]["hidden_size"],
            sequence_length=record["sequence_length"],
            global_batch=record["global_batch"],
            steps=steps,
            world_size=4,
            peak_tflops_per_gpu=peak,
            profile_mode="off",
        )
        for field, value in expected.items():
            observed = needed(performance, field, label + ".performance")
            if finite(value):
                require(
                    finite(observed)
                    and math.isclose(observed, value, rel_tol=1e-10, abs_tol=1e-12),
                    label + ": stale/inconsistent performance." + field,
                )
            else:
                same(observed, value, label + ".performance." + field)
        records.append(record)
        configs.append(config)
        inputs.extend([path, config_path])
    reference = records[0]
    for rank, record in enumerate(records):
        for field in (
            "dtype",
            "optimizer",
            "optimizer_hyperparameters",
            "precision_policy",
            "global_batch",
            "micro_batch",
            "sequence_length",
            "accumulation_steps",
            "warmup",
            "device_name",
            "device_capability",
            "attention_backend",
        ):
            same(record[field], reference[field], f"{engine}: ranks disagree on {field}")
        for field in (
            "config",
            "seed",
            "initialization",
            "global_unique_parameter_numel",
            "tied_aliases",
        ):
            same(
                record["model_source"][field],
                reference["model_source"][field],
                engine + ": " + field,
            )
        same(configs[rank]["training"], configs[0]["training"], engine + ": training configs")
        same(configs[rank]["parallel"], configs[0]["parallel"], engine + ": parallel configs")
        for field in ("torch", "cuda_build", "nccl", "platform"):
            same(
                record["versions"][field],
                reference["versions"][field],
                engine + ": versions." + field,
            )
        same(record["performance"], reference["performance"], engine + ": rank summaries")
    for index in range(len(reference["steps"])):
        maximum = max(record["steps"][index]["seconds"] for record in records)
        for record in records:
            row = record["steps"][index]
            require(
                math.isclose(row["max_rank_seconds"], maximum, rel_tol=1e-10, abs_tol=1e-12),
                engine + ": max-rank duration differs from four-rank maximum",
            )
            same(
                row["valid_tokens"], reference["steps"][index]["valid_tokens"], engine + ": tokens"
            )
    for replica in (0, 1):
        same(
            records[2 * replica]["input_batch_digest"],
            records[2 * replica + 1]["input_batch_digest"],
            engine + ": TP peers must consume the same complete input batch",
        )
    topo, topo_files = topology(directory)
    inputs.extend(topo_files)
    return {
        "directory": directory,
        "records": records,
        "configs": configs,
        "topology": topo,
        "inputs": inputs,
        "performance": reference["performance"],
    }


def compare_runs(shardtrain_directory, native_directory):
    custom = load_run(shardtrain_directory, "shardtrain")
    native = load_run(native_directory, "pytorch_fsdp2_dtensor")
    require(custom["directory"] != native["directory"], "Paired runs require distinct directories")
    same(custom["topology"], native["topology"], "GPU/CPU/link topology")
    for rank in range(4):
        left, right = custom["records"][rank], native["records"][rank]
        for field in (
            "dtype",
            "optimizer",
            "optimizer_hyperparameters",
            "precision_policy",
            "global_batch",
            "micro_batch",
            "sequence_length",
            "accumulation_steps",
            "warmup",
            "input_batch_digest",
            "device",
            "device_name",
            "device_capability",
            "attention_backend",
            "steady_steps_completed",
        ):
            same(left[field], right[field], f"Paired rank{rank}: {field}")
        for field in (
            "config",
            "seed",
            "initialization",
            "global_unique_parameter_numel",
            "tied_aliases",
        ):
            same(
                left["model_source"][field], right["model_source"][field], "Paired source: " + field
            )
        for field in ("torch", "cuda_build", "nccl", "platform"):
            same(left["versions"][field], right["versions"][field], "Paired versions: " + field)
        same(custom["configs"][rank]["training"], native["configs"][rank]["training"], "Training")
        same(custom["configs"][rank]["parallel"], native["configs"][rank]["parallel"], "Parallel")
        for rows_key in ("warmup_steps", "steps"):
            for lhs, rhs in zip(left[rows_key], right[rows_key], strict=True):
                for field in (
                    "step",
                    "valid_tokens",
                    "loss_scale",
                    "accumulation_steps",
                    "overflow",
                ):
                    same(lhs[field], rhs[field], "Paired " + rows_key + ": " + field)
    left, right = custom["performance"], native["performance"]
    same(left["peak_tflops_per_gpu"], right["peak_tflops_per_gpu"], "Explicit nominal peak")
    for field in (
        "flops_formula",
        "model_unique_parameters",
        "global_input_tokens_per_step",
        "six_n_flops_per_step",
        "attention_flops_per_step",
        "estimated_model_flops_per_step",
    ):
        same(left[field], right[field], "FLOPs accounting: " + field)
    ratio_fields = (
        "mean_max_rank_seconds",
        "global_input_tokens_per_second",
        "global_valid_tokens_per_second",
        "global_sequences_per_second",
        "all_gpu_effective_model_tflops",
        "mfu_percent",
    )
    memory = {}
    for field in ("max_allocated", "max_reserved", "allocated_end", "reserved_end"):
        lhs = [record[field] for record in custom["records"]]
        rhs = [record[field] for record in native["records"]]
        memory[field] = {
            "shardtrain_bytes_by_rank": lhs,
            "native_bytes_by_rank": rhs,
            "shardtrain_over_native_by_rank": [a / b for a, b in zip(lhs, rhs, strict=True)],
            "max_rank_shardtrain_over_native": max(lhs) / max(rhs),
            "sum_rank_shardtrain_over_native": sum(lhs) / sum(rhs),
        }
    return {
        "status": "paired_conditions_match",
        "world_size": 4,
        "engines": ["shardtrain", "pytorch_fsdp2_dtensor"],
        "shardtrain_directory": str(custom["directory"]),
        "native_directory": str(native["directory"]),
        "shardtrain_performance": left,
        "native_performance": right,
        "shardtrain_over_native": {field: left[field] / right[field] for field in ratio_fields},
        "shardtrain_speedup": right["mean_max_rank_seconds"] / left["mean_max_rank_seconds"],
        "memory": memory,
        "checked_conditions": [
            "all four completed ranks, F2T2 SP, same rank/device and GPU/CPU/link topology",
            "same architecture, source seed, unique parameters and tied aliases",
            "same per-rank token batch digest, batch, sequence and accumulation",
            "same dtype, explicit precision policy and optimizer hyperparameters",
            "same positive warmup/steady lengths, zero overflow, loss scales and update counts",
            "same Torch/CUDA/NCCL/platform and attention backend",
            "profiling off, custom events off, identical explicit nominal peak",
            "same FLOPs formula recomputed from four-rank timings and checked against summaries",
        ],
        "correctness": {
            "gradient_and_update_equivalence_proven": False,
            "requires_independent_oracle": True,
            "loss_similarity_used_as_evidence": False,
            "note": "Metadata and update counts do not prove equal gradients, parameters "
            "or optimizer states. Run an independent same-start gradient/update oracle separately.",
        },
        "limitations": [
            "One pair cannot establish variance or eliminate run-order effects; use ABBA repeats.",
            "Only recorded conditions are checked; clocks, competing jobs and unrecorded driver "
            "settings affect timing. GPU topology/name/capability do not prove physical identity.",
            "MFU is architecture FLOPs divided by supplied nominal peak, not GPU utilization.",
            "Memory is the PyTorch allocator, excluding direct CUDA/NCCL. Benchmark measured peaks "
            "reset after warmup exclude dense native startup.",
            "Parameter ownership, tied-weight lifetime and communication scheduling differ.",
        ],
        "input_artifacts": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for run in (custom, native)
            for path in run["inputs"]
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--shardtrain-run", required=True, type=Path)
    parser.add_argument("--native-run", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, help="New output JSON; never overwrite existing files"
    )
    args = parser.parse_args(argv)
    try:
        result = compare_runs(args.shardtrain_run, args.native_run)
        serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(serialized)
        else:
            print(serialized, end="")
    except (ComparisonError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
