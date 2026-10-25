"""Serial full Qwen2.5-1.5B V100 timing and independent native Nsight captures.

No weights are downloaded. A checkpoint must be an explicit local path.
Dry run writes an immutable plan/config/source manifest without querying CUDA.
Warmup overflow is recorded; every measured step must be finite and update weights.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import pickle
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import yaml
from run_matrix import execute

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = 1_543_714_304
OWNED_ELEMENTS = 444_293_888
MASTER_BYTES = 1_777_175_552
ADAM_BYTES = 3_554_352_456
PRIMARY_MODES = [("conservative", "000"), ("stream_ordered", "111")]
FULL_MODEL = {
    "vocab_size": 151936,
    "num_hidden_layers": 28,
    "hidden_size": 1536,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "intermediate_size": 8960,
    "tie_word_embeddings": True,
}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(path):
    path = path.resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def source_manifest(base_path, hf_path):
    paths = {
        Path(__file__).resolve(),
        ROOT / "scripts/benchmark_qwen2.py",
        ROOT / "scripts/run_matrix.py",
        base_path,
        hf_path,
        *(ROOT / "src/sharded_runtime").rglob("*.py"),
    }
    provenance = hf_path.with_suffix(".provenance.json")
    if provenance.is_file():
        paths.add(provenance)
    return [describe(path) for path in sorted(paths)]


def scaler_source():
    path = ROOT / "src/sharded_runtime/training/qwen2_trainer.py"
    source = path.read_text(encoding="utf-8")
    calls = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "GradScaler"
        ):
            values = {
                keyword.arg: ast.get_source_segment(source, keyword.value)
                for keyword in node.keywords
            }
            calls.append(dict(line=node.lineno, keyword_source=values))
    return dict(
        path=str(path),
        sha256=sha256(path),
        calls=calls,
        interpretation=(
            "Source arguments are recorded, not changed. Actual scale and overflow "
            "are recorded by each rank; steady overflow stops the matrix."
        ),
    )


def resolve_project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def local_checkpoint(value, dry_run):
    if not value:
        return dict(path=None, initialization="random", files=[], official_origin_verified=False)
    path = resolve_project_path(value)
    if not path.exists():
        if not dry_run:
            raise ValueError(f"Local checkpoint does not exist: {path}")
        return dict(
            path=str(path),
            initialization="local_safetensors",
            exists=False,
            files=[],
            official_origin_verified=False,
        )
    if path.is_file():
        if not path.name.endswith((".safetensors", ".safetensors.index.json")):
            raise ValueError(f"Checkpoint file must be safetensors or its index: {path}")
        files = [path]
        if path.name.endswith(".safetensors.index.json"):
            index = json.loads(path.read_text(encoding="utf-8"))
            for name in sorted(set(index["weight_map"].values())):
                shard = (path.parent / name).resolve()
                if not shard.is_relative_to(path.parent.resolve()) or not shard.is_file():
                    raise ValueError(f"Index shard is absent/outside checkpoint directory: {name}")
                files.append(shard)
    else:
        files = sorted([*path.rglob("*.safetensors"), *path.glob("*.safetensors.index.json")])
        if (path / "config.json").is_file():
            files.append(path / "config.json")
    if not any(file.name.endswith((".safetensors", ".safetensors.index.json")) for file in files):
        raise ValueError(f"Checkpoint directory has no local safetensors: {path}")
    return dict(
        path=str(path),
        initialization="local_safetensors",
        exists=True,
        files=[describe(file) for file in files],
        official_origin_verified=False,
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--config", default="configs/qwen25_1p5b_v100.yaml")
    result.add_argument("--output-dir", "--output", dest="output", required=True)
    result.add_argument("--phase", choices=("timing", "nsight", "both"), default="both")
    result.add_argument("--seq-lengths", default="128,256,512,1024")
    result.add_argument("--warmup", type=int, help="Override warmup for all selected phases")
    result.add_argument("--steps", type=int, help="Override measured steps for all selected phases")
    result.add_argument("--nsys", help="Explicit Nsight Systems executable path")
    result.add_argument("--checkpoint", help="Explicit local safetensors directory/file/index")
    result.add_argument(
        "--expected-sm70-device-name",
        help="Explicit exact device name for SM70 hardware reported without the V100 label",
    )
    result.add_argument(
        "--modes",
        help=(
            "Exact comma-separated policy:flags cases; e.g. "
            "stream_ordered:111,stream_ordered:001. Defaults to "
            "conservative:000,stream_ordered:111. Cannot combine with --cross-switches."
        ),
    )
    result.add_argument(
        "--cross-switches",
        action="store_true",
        help="Nsight: also conservative111 and stream_ordered000",
    )
    result.add_argument("--timeout", type=int, default=7200)
    result.add_argument("--export-timeout", type=int, default=1800)
    result.add_argument("--dry-run", action="store_true")
    return result


def selected_primary_modes(args):
    supplied = getattr(args, "modes", None)
    if supplied is None:
        return list(PRIMARY_MODES)
    if args.cross_switches:
        raise ValueError("Explicit --modes cannot be combined with --cross-switches")
    modes = []
    for value in supplied.split(","):
        parts = [part.strip() for part in value.split(":")]
        if (
            len(parts) != 2
            or parts[0] not in {"conservative", "stream_ordered"}
            or len(parts[1]) != 3
            or any(bit not in "01" for bit in parts[1])
        ):
            raise ValueError(
                "modes must be comma-separated policy:flags pairs, with "
                "policy conservative/stream_ordered and exactly three binary flags"
            )
        mode = tuple(parts)
        if mode in modes:
            raise ValueError(f"Duplicate mode: {value.strip()}")
        modes.append(mode)
    return modes


def prepare(args):
    if not args.output.strip():
        raise ValueError("Output directory must not be empty")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a fresh directory: {output}")
    try:
        sequences = [int(value.strip()) for value in args.seq_lengths.split(",")]
    except ValueError as error:
        raise ValueError("seq-lengths must be comma-separated integers") from error
    if (
        not sequences
        or len(set(sequences)) != len(sequences)
        or any(value < 2 or value % 2 for value in sequences)
    ):
        raise ValueError("Sequence lengths must be unique positive even integers >=2")
    if args.warmup is not None and args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.steps is not None and args.steps < 1:
        raise ValueError("steps must be positive")
    if min(args.timeout, args.export_timeout) <= 0:
        raise ValueError("timeouts must be positive")
    primary_modes = selected_primary_modes(args)
    if args.phase != "timing" and not args.nsys:
        raise ValueError("Nsight phase requires an explicit --nsys executable path")
    if args.phase != "timing" and not args.dry_run and shutil.which(args.nsys) is None:
        raise ValueError(f"Nsight executable is unavailable: {args.nsys}")
    base_path = resolve_project_path(args.config)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if base["model"].get("model_type") != "qwen2":
        raise ValueError("Base configuration must select Qwen2")
    hf_path = resolve_project_path(base["model"]["hf_config_path"])
    hf = json.loads(hf_path.read_text(encoding="utf-8"))
    if any(hf.get(key) != value for key, value in FULL_MODEL.items()):
        raise ValueError("This runner requires the full 28-layer Qwen2.5-1.5B config")
    if base["training"].get("dtype") != "fp16" or base["training"].get("optimizer") != "adamw":
        raise ValueError("V100 workload requires FP16 AMP with AdamW FP32 master shards")
    checkpoint = local_checkpoint(
        args.checkpoint
        if args.checkpoint is not None
        else base["model"].get("checkpoint_path", ""),
        args.dry_run,
    )
    sources = source_manifest(base_path, hf_path)
    output.mkdir(parents=True, exist_ok=False)
    plan = []
    phases = ["timing", "nsight"] if args.phase == "both" else [args.phase]
    for phase in phases:
        modes = primary_modes + (
            [("conservative", "111"), ("stream_ordered", "000")]
            if phase == "nsight" and args.cross_switches
            else []
        )
        for sequence in sequences if phase == "timing" else [128]:
            for policy, flags in modes:
                identifier = f"{phase}_S{sequence}_{policy}_flags{flags}"
                directory = output / identifier
                artifacts = directory / "artifacts"
                artifacts.mkdir(parents=True, exist_ok=False)
                config = json.loads(json.dumps(base))
                config["model"].update(
                    hf_config_path=str(hf_path),
                    checkpoint_path=checkpoint["path"] or "",
                    seq_len=sequence,
                )
                config["parallel"] = dict(
                    fsdp_degree=2, tp_degree=2, sp_enabled=True, device="cuda", backend="nccl"
                )
                config["runtime"].update(
                    communication_policy=policy,
                    reshard_after_forward=True,
                    param_grouping="block",
                    forward_prefetch=flags[0] == "1",
                    backward_prefetch=flags[1] == "1",
                    async_grad_rs=flags[2] == "1",
                )
                steps = args.steps if args.steps is not None else (10 if phase == "timing" else 2)
                warmup = args.warmup if args.warmup is not None else (3 if phase == "timing" else 1)
                config["training"]["steps"] = steps
                config["debug"]["output_dir"] = str(artifacts)
                config_path = directory / "config.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--nnodes=1",
                    "--nproc_per_node=4",
                    "--rdzv_backend=c10d",
                    "--rdzv_endpoint=127.0.0.1:0",
                    "--max_restarts=0",
                    str(ROOT / "scripts/benchmark_qwen2.py"),
                    "--config",
                    str(config_path),
                    "--warmup",
                    str(warmup),
                    "--profile-mode",
                    "off" if phase == "timing" else "nsight",
                ]
                if phase == "nsight":
                    command = [
                        args.nsys,
                        "profile",
                        "--trace=cuda,nvtx,osrt",
                        "--sample=none",
                        "--cpuctxsw=none",
                        "--cuda-memory-usage=true",
                        "--force-overwrite=false",
                        "--output",
                        str(directory / "timeline"),
                        *command,
                    ]
                item = dict(
                    id=identifier,
                    phase=phase,
                    sequence_length=sequence,
                    policy=policy,
                    flags=flags,
                    warmup=warmup,
                    steps=steps,
                    directory=str(directory),
                    artifacts=str(artifacts),
                    command=command,
                    config_sha256=sha256(config_path),
                    expected_config=config,
                    expected_sm70_device_name=args.expected_sm70_device_name,
                )
                write_json(directory / "command.json", item)
                plan.append(item)
    document = dict(
        dry_run=args.dry_run,
        status="planned",
        cases=plan,
        sources=sources,
        checkpoint=checkpoint,
        scaler_source=scaler_source(),
        expected_parameter_numel=PARAMETERS,
        expected_owned_size_per_rank=dict(
            persistent_shard_numel_including_padding=OWNED_ELEMENTS,
            persistent_shard_valid_numel=OWNED_ELEMENTS,
            parameter_bytes=MASTER_BYTES,
            unique_optimizer_leaves=338,
            adam_tensor_bytes=ADAM_BYTES,
        ),
        world_size=4,
        warmup_overflow_policy=(
            "Record every warmup overflow and scale. Only all-finite, zero-overflow, "
            "full-update steady windows pass; unexplained nonfinite flags fail. "
            "No automatic scale change or rerun."
        ),
        timing_scope=(
            "Profiler off; synthetic repeated tokens, combined policy/switch "
            "configuration comparison; no HF/FSDP2 reference speedup claim."
        ),
        nsight_scope=(
            "Full process includes initialization and warmup; per-rank allocator "
            "snapshots/history cover only measured steps. Allocator does not include "
            "direct NCCL/CUDA allocations."
        ),
    )
    write_json(output / "plan.json", document)
    write_json(output / "checkpoint_source.json", checkpoint)
    return output, document


def gate_rank(path, item, rank):
    data = json.loads(path.read_text(encoding="utf-8"))
    problems = []

    def check(condition, message):
        if not condition:
            problems.append(message)

    steady, warmup = data.get("steps", []), data.get("warmup_steps", [])
    config = item["expected_config"]
    check(data.get("status") == "completed", "benchmark did not complete")
    check(data.get("rank") == rank and data.get("world_size") == 4, "rank/world mismatch")
    check(data.get("oom") is False and data.get("errors") == [], "OOM or runtime/profiling error")
    check(data.get("nonfinite_loss") is False, "nonfinite/missing loss status")
    check(
        len(steady)
        == data.get("steady_steps_completed")
        == data.get("steady_steps_requested")
        == item["steps"],
        "incomplete measured steps",
    )
    check(len(warmup) == data.get("warmup") == item["warmup"], "incomplete warmup")
    check(
        all(
            isinstance(row.get("loss"), (int, float)) and math.isfinite(row["loss"])
            for row in [*warmup, *steady]
        ),
        "nonfinite/missing warmup or steady loss",
    )
    check(
        all(row.get("overflow") is False for row in steady) and data.get("overflow_steps") == 0,
        "steady overflow; stop and reconsider scale/warmup",
    )
    check(data.get("optimizer_updates") == item["steps"], "measured optimizer updates missing")
    warmup_overflows = sum(bool(row.get("overflow")) for row in warmup)
    check(data.get("warmup_overflow_steps") == warmup_overflows, "warmup overflow count mismatch")
    check(
        not data.get("nonfinite_detected") or warmup_overflows > 0, "unexplained nonfinite_detected"
    )
    check(
        data.get("sequence_length") == item["sequence_length"]
        and data.get("communication_policy") == item["policy"],
        "workload/policy mismatch",
    )
    check(
        data.get("profile_mode") == ("off" if item["phase"] == "timing" else "nsight"),
        "profiler mode mismatch",
    )
    check(
        data.get("dtype") == "fp16" and data.get("optimizer") == "adamw", "dtype/optimizer mismatch"
    )
    expected_device = item.get("expected_sm70_device_name")
    actual_device = data.get("device_name", "")
    check(
        data.get("device_capability") == [7, 0]
        and (
            actual_device == expected_device if expected_device else "v100" in actual_device.lower()
        ),
        "actual device does not match the expected SM70 hardware name",
    )
    check(data.get("versions", {}).get("nccl") is not None, "NCCL unavailable")
    source = data.get("model_source", {})
    check(source.get("global_unique_parameter_numel") == PARAMETERS, "not the full 1.5B model")
    check(
        all(source.get("config", {}).get(key) == value for key, value in FULL_MODEL.items()),
        "model config differs from full 1.5B",
    )
    check(
        source.get("initialization")
        == ("local_safetensors" if config["model"]["checkpoint_path"] else "random"),
        "checkpoint source mismatch",
    )
    own = data.get("ownership", {})
    check(
        own.get("live_buffers") == own.get("live_fullparam_leases") == 0,
        "live communication buffer/FullParam lease",
    )
    check(
        own.get("stable_shard_identity") is True
        and own.get("units_reduced_once_last_microbatch") is True,
        "shard identity/reduction ownership failed",
    )
    check(
        own.get("unit_count") == 30 and own.get("persistent_shard_count") == 338,
        "unexpected unit/unique optimizer leaf count",
    )
    check(
        own.get("persistent_shard_numel_including_padding")
        == own.get("persistent_shard_valid_numel")
        == OWNED_ELEMENTS,
        "F2T2 owned element count differs from exact 1.5B theory",
    )
    check(own.get("parameter_bytes") == MASTER_BYTES, "F2T2 FP32 master byte count differs")
    check(own.get("parameter_dtypes") == ["torch.float32"], "master shards are not FP32")
    check(
        own.get("parameter_bytes", 0) > 0
        and own.get("gradient_bytes") == own.get("parameter_bytes"),
        "gradient/master storage byte mismatch",
    )
    check(
        own.get("optimizer_state_parameter_count") == own.get("persistent_shard_count"),
        "AdamW state missing for some leaves",
    )
    check(
        own.get("optimizer_tensor_bytes") == ADAM_BYTES,
        "AdamW moments/step tensor storage mismatch",
    )
    expected_tokens = (
        2
        * config["training"]["local_batch_size"]
        * config["training"].get("accumulation_steps", 1)
        * (item["sequence_length"] - 1)
    )
    check(
        all(row.get("valid_tokens") == expected_tokens for row in steady),
        "global shifted-token count mismatch",
    )
    check(
        all(
            math.isfinite(row.get("max_rank_seconds", float("nan"))) and row["max_rank_seconds"] > 0
            for row in steady
        ),
        "invalid measured durations",
    )
    switches = data.get("runtime_switches", {})
    check(
        all(
            switches.get(key) is expected
            for key, expected in zip(
                ["forward_prefetch", "backward_prefetch", "async_grad_rs"],
                [char == "1" for char in item["flags"]],
                strict=True,
            )
        ),
        "runtime switches mismatch",
    )
    return dict(
        rank=rank,
        problems=problems,
        warmup_overflows=warmup_overflows,
        warmup_steps=warmup,
        steady_steps=steady,
        ownership=own,
        step_seconds_p50=data.get("step_seconds_p50"),
        step_seconds_p95=data.get("step_seconds_p95"),
        tokens_per_second=data.get("tokens_per_second"),
        max_allocated=data.get("max_allocated"),
        max_reserved=data.get("max_reserved"),
        allocated_end=data.get("allocated_end"),
        source=source,
        device_name=actual_device,
        device_capability=data.get("device_capability"),
        expected_sm70_device_name=expected_device,
    )


def benchmark_gate(item):
    directory = Path(item["artifacts"])
    ranks, problems = [], []
    for rank in range(4):
        try:
            result = gate_rank(directory / f"benchmark.rank{rank}.json", item, rank)
            ranks.append(result)
            problems.extend(f"rank{rank}: {message}" for message in result["problems"])
            environment = json.loads(
                (directory / f"environment.rank{rank}.json").read_text(encoding="utf-8")
            )
            if (
                environment.get("local_rank") != rank
                or environment.get("nccl_available") is not True
            ):
                problems.append(f"rank{rank}: local device assignment/NCCL metadata mismatch")
            profiling = json.loads(
                (directory / f"profiling.rank{rank}.json").read_text(encoding="utf-8")
            )
            if profiling.get("errors") != []:
                problems.append(f"rank{rank}: profiling errors/missing error status")
            if profiling.get("mode") != ("off" if item["phase"] == "timing" else "nsight"):
                problems.append(f"rank{rank}: profiling metadata mode mismatch")
            if item["phase"] == "nsight":
                if len(profiling.get("memory", [])) != item["steps"]:
                    problems.append(f"rank{rank}: missing measured allocator rows")
                result["snapshots"] = {}
                for phase in ["start", "end"]:
                    snapshot = directory / f"memory.{phase}.rank{rank}.pickle"
                    if not snapshot.is_file() or snapshot.stat().st_size == 0:
                        problems.append(f"rank{rank}: missing {phase} allocator snapshot")
                        continue
                    # These are fresh local artifacts produced by our own benchmark.
                    with snapshot.open("rb") as reader:
                        payload = pickle.load(reader)
                    if (
                        not isinstance(payload, dict)
                        or not {"segments", "device_traces"} <= payload.keys()
                    ):
                        raise ValueError(f"Invalid allocator snapshot schema: {snapshot}")
                    history = sum(len(events) for events in payload["device_traces"])
                    result["snapshots"][phase] = dict(
                        segments=len(payload["segments"]),
                        reserved_bytes=sum(
                            segment["total_size"] for segment in payload["segments"]
                        ),
                        allocated_bytes=sum(
                            block["size"]
                            for segment in payload["segments"]
                            for block in segment["blocks"]
                            if block["state"] == "active_allocated"
                        ),
                        history_entries=history,
                        history_at_limit=history >= profiling.get("memory_history_max_entries", 0),
                    )
                    state = result["snapshots"][phase]
                    if not state["reserved_bytes"] >= state["allocated_bytes"] >= MASTER_BYTES:
                        problems.append(
                            f"rank{rank}: {phase} snapshot lacks persistent FP32 master storage"
                        )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            pickle.UnpicklingError,
            EOFError,
        ) as error:
            problems.append(f"rank{rank}: artifact validation failed: {error!r}")
    if len(ranks) == 4:
        for stage in ["warmup_steps", "steady_steps"]:
            signatures = [
                [
                    (
                        row.get("step"),
                        row.get("overflow"),
                        row.get("loss_scale"),
                        row.get("valid_tokens"),
                    )
                    for row in result[stage]
                ]
                for result in ranks
            ]
            if any(signature != signatures[0] for signature in signatures):
                problems.append(f"{stage}: rank scale/overflow/token decisions differ")
            if all(len(result[stage]) == len(ranks[0][stage]) for result in ranks):
                for index in range(len(ranks[0][stage])):
                    losses = [result[stage][index].get("loss") for result in ranks]
                    if all(isinstance(loss, (int, float)) for loss in losses):
                        if max(losses) - min(losses) > 1e-6:
                            problems.append(f"{stage} {index}: global loss differs across ranks")
    return dict(status="passed" if not problems else "failed", problems=problems, ranks=ranks)


def sqlite_gate(path, item):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        table = next(
            (
                name
                for name in ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"]
                if name in tables
            ),
            None,
        )
        if table is None or "NVTX_EVENTS" not in tables:
            raise ValueError("Export is missing GPU kernel or NVTX tables")
        counts = dict(
            connection.execute(f"SELECT deviceId,count(*) FROM {table} GROUP BY deviceId")
        )
        kernel_count = sum(counts.values())
        nvtx_count = connection.execute("SELECT count(*) FROM NVTX_EVENTS").fetchone()[0]
        stream_counts = [
            dict(device=row[0], stream=row[1], kernels=row[2])
            for row in connection.execute(
                f"SELECT deviceId,streamId,count(*) FROM {table} GROUP BY deviceId,streamId"
            )
        ]
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        columns = {row[1] for row in connection.execute("PRAGMA table_info(NVTX_EVENTS)")}
        texts = (
            [
                row[0] or strings.get(row[1], "")
                for row in connection.execute("SELECT text,textId FROM NVTX_EVENTS")
            ]
            if {"text", "textId"} <= columns
            else []
        )
        ranges = Counter(texts)
        expected = 4 * (item["warmup"] + item["steps"])
        names = [
            "model_forward",
            "loss",
            "backward",
            "optimizer",
            *[f"layer{index}" for index in range(28)],
        ]
        roles = {name: ranges["shardtrain/qwen/" + name] for name in names}
        problems = []
        if kernel_count <= 0 or nvtx_count <= 0:
            problems.append("Zero GPU kernel or NVTX events")
        if len(counts) != 4 or any(count <= 0 for count in counts.values()):
            problems.append("Native timeline did not capture kernels on all four GPUs")
        if any(count != expected for count in roles.values()):
            problems.append("Missing/duplicate full-process phase or 28-layer NVTX ranges")
        return dict(
            status="passed" if not problems else "failed",
            problems=problems,
            gpu_kernel_count=kernel_count,
            nvtx_event_count=nvtx_count,
            kernel_counts_per_device=counts,
            kernel_counts_per_device_stream=stream_counts,
            phase_layer_counts=roles,
            expected_each_phase_layer_count=expected,
            communication_overlap_claim=False,
        )


def run(args, output, plan):
    summaries = []
    for index, item in enumerate(plan["cases"]):
        directory = Path(item["directory"])
        print(f"[{index + 1}/{len(plan['cases'])}] {item['id']}", flush=True)
        sources = source_manifest(
            resolve_project_path(args.config),
            Path(item["expected_config"]["model"]["hf_config_path"]),
        )
        write_json(
            directory / "source_manifest.json", dict(files=sources, scaler_source=scaler_source())
        )
        result = dict(
            id=item["id"],
            status="failed",
            command=item["command"],
            source_manifest_sha256=sha256(directory / "source_manifest.json"),
            config_sha256=item["config_sha256"],
            checkpoint_manifest_sha256=sha256(output / "checkpoint_source.json"),
            errors=[],
        )
        try:
            result["returncode"] = execute(item["command"], directory, args.timeout)
            result["benchmark_gate"] = benchmark_gate(item)
            if item["phase"] == "nsight":
                trace, sqlite = directory / "timeline.nsys-rep", directory / "timeline.sqlite"
                if not trace.is_file() or trace.stat().st_size == 0:
                    raise ValueError("Native Nsight report is missing/empty")
                export_dir = directory / "export"
                export_dir.mkdir(exist_ok=False)
                command = [
                    args.nsys,
                    "export",
                    "--type",
                    "sqlite",
                    "--output",
                    str(sqlite),
                    str(trace),
                ]
                result["export_command"] = command
                result["export_returncode"] = execute(command, export_dir, args.export_timeout)
                if result["export_returncode"] != 0 or not sqlite.is_file():
                    raise ValueError("Native SQLite export failed; raw trace/logs preserved")
                result["trace_gate"] = sqlite_gate(sqlite, item)
            if (
                result["returncode"] == 0
                and result["benchmark_gate"]["status"] == "passed"
                and result.get("trace_gate", {}).get("status", "passed") == "passed"
            ):
                result["status"] = "passed"
        except (OSError, ValueError, TypeError, sqlite3.Error) as error:
            result["errors"].append(repr(error))
        result["artifact_manifest"] = [
            describe(path) for path in sorted(directory.rglob("*")) if path.is_file()
        ]
        write_json(directory / "validation.json", result)
        summaries.append(result)
        document = dict(
            status="running" if result["status"] == "passed" else "failed",
            cases=summaries,
            planned_cases=len(plan["cases"]),
            completed_cases=len(summaries),
            stopped_after_failure=result["status"] != "passed",
            actual_NCCL_execution_verified=all(
                summary.get("benchmark_gate", {}).get("status") == "passed" for summary in summaries
            ),
            full_checkpoint_logits_reference_verified=False,
            communication_overlap_claim=False,
        )
        write_json(output / "summary.json", document)
        if result["status"] != "passed":
            raise RuntimeError(
                f"Failed gate: {directory}; all artifacts retained, later cases stopped. "
                "Inspect warmup scales and steady overflow before changing scale/warmup."
            )
    document["status"] = "passed"
    write_json(output / "summary.json", document)
    print(f"Passed {len(summaries)} serial cases: {output}", flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        output, plan = prepare(args)
        if args.dry_run:
            print(f"Planned {len(plan['cases'])} cases without CUDA queries or execution: {output}")
            return 0
        run(args, output, plan)
    except (OSError, ValueError, TypeError, RuntimeError, yaml.YAMLError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
