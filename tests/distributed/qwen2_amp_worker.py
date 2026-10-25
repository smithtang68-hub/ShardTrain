"""CUDA AMP same-start HF step oracle and global overflow gate for FSDP/TP/SP.

Example: torchrun --standalone --nproc-per-node=4 tests/distributed/qwen2_amp_worker.py
    --f 2 --t 2 --sp --optimizer adamw --output-dir runs/qwen_amp_adamw

Use --plan-only for a CPU-only harness/config check. No GPU is touched in that mode.
The runtime trains continuously. Before each checked step, the independent dense HF
reference receives the runtime's starting weights and optimizer state. Different
FP16 TP/F reductions need not produce identical independent AdamW rollouts. This
gate checks a same-start step and replays every local optimizer leaf bit-exactly;
it does not claim equality of separate long-running numerical trajectories.
Profiling includes HF, runtime, synchronization and replay; no performance claim.
"""

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from sharded_runtime.config import (
    Config,
    DebugConfig,
    ModelConfig,
    ParallelConfig,
    RuntimeConfig,
    TrainingConfig,
)
from sharded_runtime.distributed.context import WorkerEnvironment
from sharded_runtime.launch.bootstrap import WorkerSession
from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig
from sharded_runtime.observability.profiling import ProfileCapture
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.training.qwen2_trainer import QwenTrainer

ATOL = RTOL = 3e-3
SEED = 313
CLEAN_STEPS = 3


def token_batch(replica, step):
    generator = torch.Generator().manual_seed(2718 + replica * 100 + step)
    tokens = torch.randint(1, 64, (2, 8), generator=generator)
    mask = torch.ones_like(tokens)
    mask[1, -(2 + 2 * replica) :] = 0
    labels = tokens.masked_fill(mask.eq(0), -100)
    labels[0, 4 + replica % 3] = -100
    # Row 1's token-4 label crosses the T=2 SP boundary when it is valid.
    positions = torch.arange(17 + step, 25 + step).expand(2, -1)
    return tokens, labels, mask, positions


def optimizer_snapshot(optimizer):
    return {
        parameter: {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }
        for parameter, state in optimizer.state.items()
    }


def optimizer_unchanged(optimizer, before):
    return len(optimizer.state) == len(before) and all(
        parameter in before
        and state.keys() == before[parameter].keys()
        and all(
            torch.equal(value, before[parameter][key])
            if isinstance(value, torch.Tensor)
            else value == before[parameter][key]
            for key, value in state.items()
        )
        for parameter, state in optimizer.state.items()
    )


def assert_master_state(trainer, identities):
    runtime = trainer.runtime
    leaves = runtime.sharded_parameters()
    assert len(leaves) == len({id(parameter) for parameter in leaves})
    assert trainer.model.lm_head.weight is trainer.model.model.embed_tokens.weight
    assert all(parameter.dtype == torch.float32 for parameter in leaves)
    assert all(
        parameter.grad is not None and parameter.grad.dtype == torch.float32 for parameter in leaves
    )
    for state in trainer.optimizer.state.values():
        assert all(
            not isinstance(value, torch.Tensor) or value.dtype == torch.float32
            for value in state.values()
        )
    assert identities == {
        name: (id(handle.local_shard), handle.local_shard.data_ptr())
        for name, handle in runtime.registry.items()
    }
    assert not runtime.comm.live_buffers and all(unit.lease is None for unit in runtime.units)
    for handle in runtime.registry.values():
        valid = handle.spec.valid_len(runtime.materializer.rank)
        assert torch.count_nonzero(handle.local_shard.grad[valid:]) == 0


def expected_shard(runtime, name, value):
    handle = runtime.registry[name]
    local = runtime.qwen_partitions[name].local_slice(value).contiguous()
    valid = handle.spec.valid_len(runtime.materializer.rank)
    start = runtime.materializer.rank * handle.spec.fsdp_shard_numel
    shard = torch.zeros_like(handle.local_shard)
    shard[:valid].copy_(local.reshape(-1)[start : start + valid])
    return shard


def join_fsdp_shards(spec, shards):
    """Oracle-only F reconstruction: discard each rank's storage padding."""
    if len(shards) * spec.fsdp_shard_numel != spec.numel + spec.padding_numel:
        raise AssertionError("Wrong F shard count")
    pieces = []
    for rank, shard in enumerate(shards):
        if shard.numel() != spec.fsdp_shard_numel:
            raise AssertionError("Wrong F shard storage size")
        pieces.append(shard.reshape(-1)[: spec.valid_len(rank)])
    return torch.cat(pieces).view(spec.tp_local_shape)


def join_tp_parts(partition, parts):
    """Oracle-only inverse of the published TP partition, including replicas."""
    if len(parts) != partition.degree:
        raise AssertionError("Wrong T part count")
    if partition.dim is None:
        if any(not torch.equal(parts[0], part) for part in parts[1:]):
            raise AssertionError("TP replicated parameter/state differs between ranks")
        result = parts[0]
    else:
        result = torch.cat(parts, dim=partition.dim)
    if tuple(result.shape) != partition.global_shape:
        raise AssertionError("Reconstructed logical shape differs from partition")
    return result


def validate_reference_layout(runtime, reference):
    """Derive expected axes from the HF module roles, independently of the adapter."""
    parameters = dict(reference.named_parameters())
    if parameters.keys() != runtime.registry.keys():
        raise AssertionError("Runtime keys differ from independent HF canonical parameters")
    column = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
    row = {"o_proj", "down_proj"}
    for name, parameter in parameters.items():
        module_name, kind = name.rsplit(".", 1)
        module = reference.get_submodule(module_name)
        role = module_name.rsplit(".", 1)[-1]
        if isinstance(module, torch.nn.Linear) and role in column:
            axis = 0
        elif isinstance(module, torch.nn.Linear) and role in row and kind == "weight":
            axis = 1
        elif role in {
            "embed_tokens",
            "lm_head",
            "input_layernorm",
            "post_attention_layernorm",
            "norm",
        }:
            axis = None
        else:
            raise AssertionError(f"Unknown independent HF parameter role: {name}")
        partition = runtime.qwen_partitions[name]
        spec = runtime.registry[name].spec
        local_shape = list(parameter.shape)
        if axis is not None:
            if local_shape[axis] % runtime.tp.degree:
                raise AssertionError("HF partition axis cannot divide TP degree")
            local_shape[axis] //= runtime.tp.degree
        if (
            partition.dim != axis
            or partition.global_shape != tuple(parameter.shape)
            or partition.degree != runtime.tp.degree
            or partition.rank != runtime.tp.rank
            or spec.global_shape != tuple(parameter.shape)
            or spec.tp_local_shape != tuple(local_shape)
            or spec.numel != parameter.numel() // (runtime.tp.degree if axis is not None else 1)
            or spec.fsdp_shard_numel
            != (spec.numel + runtime.materializer.degree - 1) // runtime.materializer.degree
            or spec.padding_numel
            != spec.fsdp_shard_numel * runtime.materializer.degree - spec.numel
            or spec.dtype != parameter.dtype
            or runtime.registry[name].local_shard.dtype != parameter.dtype
            or runtime.registry[name].local_shard.device != runtime.device
        ):
            raise AssertionError(f"Runtime partition differs from independent HF structure: {name}")


def oracle_gather(runtime, value, axis, kind):
    """Keep acceptance collectives in the existing communicator's PG ordering."""
    if axis == "fsdp":
        group, degree, channel = (
            runtime.materializer.group,
            runtime.materializer.degree,
            "fsdp_ag",
        )
    else:
        group, degree, channel = runtime.tp.process_group, runtime.tp.degree, "sp_ag"
    value = value.detach().contiguous().to(runtime.device)
    outputs = [torch.empty_like(value) for _ in range(degree)]
    if degree == 1:
        outputs[0].copy_(value)
    else:
        runtime.comm.unit_id, runtime.comm.phase = None, "AMP_ORACLE_SYNC"
        runtime.comm.issue(
            axis,
            kind,
            group,
            value,
            outputs[0],
            lambda async_: dist.all_gather(outputs, value, group=group, async_op=async_),
            channel=channel,
            asynchronous=False,
            group_role="oracle_metadata",
            extra_outputs=outputs[1:],
        )
    return outputs


@torch.no_grad()
def reconstruct_start(runtime, optimizer):
    """Read the actual trajectory's logical weights and Adam moments on every rank."""
    weights = {
        name: join_tp_parts(
            runtime.qwen_partitions[name], oracle_gather(runtime, value, "tp", "amp_weight_sync")
        )
        for name, value in runtime.reconstruct_tp_local().items()
    }
    states = {}
    for name, handle in runtime.registry.items():
        state = optimizer.state.get(handle.local_shard, {})
        states[name] = {}
        # State schema is fixed by the optimizer; synchronize its presence first so
        # a missing state on one rank fails uniformly before variable collectives.
        schema = torch.tensor(
            [int(key in state) for key in ("step", "exp_avg", "exp_avg_sq")],
            dtype=torch.int32,
            device=runtime.device,
        )
        if set(state) - {"step", "exp_avg", "exp_avg_sq"}:
            raise AssertionError("Unexpected optimizer state schema")
        for axis in ("fsdp", "tp"):
            pieces = oracle_gather(runtime, schema, axis, "amp_optimizer_schema")
            if any(not torch.equal(schema, piece) for piece in pieces):
                raise AssertionError("Optimizer state schema differs between ranks")
        for key in ("step", "exp_avg", "exp_avg_sq"):
            if key not in state:
                continue
            value = state[key]
            if key == "step":
                scalar = value.reshape(1)
                for axis in ("fsdp", "tp"):
                    pieces = oracle_gather(runtime, scalar, axis, "amp_optimizer_step_sync")
                    if any(not torch.equal(pieces[0], piece) for piece in pieces[1:]):
                        raise AssertionError("Optimizer step differs between ranks")
                states[name][key] = pieces[0].reshape(value.shape).to(value.device)
            else:
                local = join_fsdp_shards(
                    handle.spec, oracle_gather(runtime, value, "fsdp", "amp_moment_f_sync")
                )
                states[name][key] = join_tp_parts(
                    runtime.qwen_partitions[name],
                    oracle_gather(runtime, local, "tp", "amp_moment_t_sync"),
                )
    return weights, states


@torch.no_grad()
def copy_reference_start(reference, optimizer, weights, states):
    """Copy into HF leaves without replacing the tied Parameter or its optimizer key."""
    parameters = dict(reference.named_parameters())
    if parameters.keys() != weights.keys() or parameters.keys() != states.keys():
        raise AssertionError("Oracle logical keys differ from runtime canonical keys")
    optimizer.state.clear()
    for name, parameter in parameters.items():
        if (
            parameter.shape != weights[name].shape
            or parameter.dtype != weights[name].dtype
            or parameter.device != weights[name].device
        ):
            raise AssertionError("Oracle starting weight shape/dtype/device differs")
        for key, value in states[name].items():
            if key in {"exp_avg", "exp_avg_sq"} and (
                value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or value.device != parameter.device
            ):
                raise AssertionError("Oracle starting moment shape/dtype/device differs")
            if key == "step" and (value.ndim != 0 or value.dtype != torch.float32):
                raise AssertionError("Oracle starting optimizer step shape/dtype differs")
        parameter.copy_(weights[name])
        optimizer.state[parameter] = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in states[name].items()
        }
    if reference.config.tie_word_embeddings:
        assert reference.lm_head.weight is reference.model.embed_tokens.weight


def start_hash(weights, states):
    digest = hashlib.sha256()
    for name in sorted(weights):
        for key, value in [("weight", weights[name]), *sorted(states[name].items())]:
            digest.update(f"{name}/{key}/{value.dtype}/{tuple(value.shape)}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_initial_weights(runtime, reference):
    """Check the original independent seed oracle before any same-start overwrite."""
    parameters = dict(reference.named_parameters())
    assert parameters.keys() == runtime.registry.keys()
    for name, handle in runtime.registry.items():
        if not torch.equal(handle.local_shard, expected_shard(runtime, name, parameters[name])):
            raise AssertionError(f"Original HF initialization shard differs: {name}")


def assert_optimizer_contract(optimizer, kind):
    """Independent fixture mathematics; numerical implementation flags may differ."""
    intended = {"sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}[kind]
    assert type(optimizer) is intended, "Optimizer type differs from fixture contract"
    assert len(optimizer.param_groups) == 1, "Optimizer group count differs from fixture contract"
    expected = dict(lr=0.001, maximize=False, differentiable=False)
    if kind == "adamw":
        expected.update(
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False, capturable=False
        )
    else:
        expected.update(weight_decay=0, momentum=0, dampening=0, nesterov=False)
    for key, value in expected.items():
        assert optimizer.param_groups[0].get(key) == value, (
            "Optimizer option differs from fixture contract",
            key,
            optimizer.param_groups[0].get(key),
            value,
        )


def assert_update_count(optimizer, updates):
    leaves = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if type(optimizer) is torch.optim.AdamW:
        if updates == 0:
            assert all(not state for state in optimizer.state.values()), (
                "Adam state before first update"
            )
            return
        assert len(optimizer.state) == len(leaves), "Missing Adam leaf state"
        for parameter in leaves:
            assert parameter in optimizer.state, "Missing Adam leaf state"
            state = optimizer.state[parameter]
            assert set(state) == {"step", "exp_avg", "exp_avg_sq"}, (
                "Missing/unexpected Adam state keys"
            )
            step = state["step"]
            assert step.ndim == 0 and step.dtype == torch.float32, "Wrong Adam step shape/dtype"
            assert float(step) == updates, (step, updates)
    elif type(optimizer) is torch.optim.SGD:
        assert all(not state for state in optimizer.state.values()), "Unexpected SGD momentum state"
    else:
        raise AssertionError("Unsupported fixture optimizer")


def finalize_result(directory, rank, payload, capture):
    """Save the gate result even when profiling stop/snapshot/metadata fails."""
    cleanup_error = None
    try:
        if capture is not None:
            capture.close()
    except BaseException as error:
        cleanup_error = error
        payload["status"] = "failed"
        payload["errors"].append(repr(error))
    finally:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"qwen_amp.rank{rank}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    if cleanup_error is not None:
        raise cleanup_error


def _bit_equal(actual, expected):
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        return (
            actual.dtype == expected.dtype
            and actual.device == expected.device
            and actual.shape == expected.shape
            and torch.equal(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def optimizer_replay(optimizer, parameter_before, state_before, names, records):
    """Replay the actual unscaled gradient with the same native optimizer/options."""
    leaves = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    replicas = {
        parameter: torch.nn.Parameter(parameter_before[parameter].detach().clone())
        for parameter in leaves
    }
    groups = [
        {
            **{key: value for key, value in group.items() if key != "params"},
            "params": [replicas[parameter] for parameter in group["params"]],
        }
        for group in optimizer.param_groups
    ]
    replay = type(optimizer)(groups)
    for parameter, replica in replicas.items():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        replica.grad = parameter.grad.detach().clone()
        replay.state[replica] = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in state_before.get(parameter, {}).items()
        }
    replay.step()
    for parameter, replica in replicas.items():
        actual_state = optimizer.state.get(parameter, {})
        expected_state = replay.state.get(replica, {})
        state_equal = actual_state.keys() == expected_state.keys() and all(
            _bit_equal(value, expected_state[key]) for key, value in actual_state.items()
        )
        record = dict(
            name=names[parameter],
            numel=parameter.numel(),
            parameter_bit_exact=_bit_equal(parameter.detach(), replica.detach()),
            optimizer_state_bit_exact=state_equal,
            max_abs_parameter_error=float((parameter.detach() - replica.detach()).abs().max()),
            state_keys=sorted(actual_state),
            state_dtype_device={
                key: {"dtype": str(value.dtype), "device": str(value.device)}
                for key, value in actual_state.items()
                if isinstance(value, torch.Tensor)
            },
        )
        records.append(record)
        assert record["parameter_bit_exact"] and record["optimizer_state_bit_exact"], record


def compare(actual, expected, name, comparisons):
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    error = (actual - expected).abs()
    record = dict(
        name=name,
        shape=list(actual.shape),
        max_abs_error=float(error.max()),
        max_rel_error=float((error / expected.abs().clamp_min(1e-12)).max()),
        nonfinite_count=int((~torch.isfinite(actual)).sum()),
        reference_nonfinite_count=int((~torch.isfinite(expected)).sum()),
        atol=ATOL,
        rtol=RTOL,
        passed=bool(torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)),
    )
    comparisons.append(record)
    assert (
        record["passed"] and record["nonfinite_count"] == record["reference_nonfinite_count"] == 0
    ), record


def rank_status(
    trainer, result, *, unchanged=True, state_unchanged=True, changed=False, hook_calls=0
):
    runtime = trainer.runtime
    local_finite = all(
        bool(torch.isfinite(parameter.grad).all()) for parameter in runtime.sharded_parameters()
    )
    fields = [
        "overflow",
        "scale",
        "tracker",
        "valid_tokens",
        "loss",
        "local_finite",
        "parameters_unchanged",
        "optimizer_state_unchanged",
        "parameters_changed",
        "injection_hook_calls",
    ]
    values = [
        result.overflow,
        result.loss_scale,
        trainer.scaler.state_dict()["_growth_tracker"],
        result.valid_tokens,
        result.loss,
        local_finite,
        unchanged,
        state_unchanged,
        changed,
        hook_calls,
    ]
    token = torch.tensor(values, device=runtime.device, dtype=torch.float64)
    outputs = [
        torch.empty_like(token)
        for _ in range(dist.get_world_size() if dist.is_initialized() else 1)
    ]
    runtime.comm.unit_id, runtime.comm.phase = None, "AMP_ORACLE"
    if len(outputs) > 1:
        runtime.comm.issue(
            "world",
            "amp_oracle_status",
            dist.group.WORLD,
            token,
            outputs[0],
            lambda async_: dist.all_gather(outputs, token, async_op=async_),
            channel="fsdp_rs",
            asynchronous=False,
            group_role="oracle_metadata",
        )
    else:
        outputs[0].copy_(token)
    rows = [
        dict(rank=rank, **dict(zip(fields, output.cpu().tolist(), strict=True)))
        for rank, output in enumerate(outputs)
    ]
    for key in ["overflow", "scale", "tracker", "valid_tokens"]:
        assert len({row[key] for row in rows}) == 1, (key, rows)
    assert max(row["loss"] for row in rows) - min(row["loss"] for row in rows) <= ATOL
    return rows


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--f", type=int, default=2)
    result.add_argument("--t", type=int, default=2)
    result.add_argument("--sp", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--layers", type=int, default=4)
    result.add_argument("--optimizer", choices=("sgd", "adamw"), required=True)
    result.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    result.add_argument("--device", choices=("cuda",), default="cuda")
    result.add_argument("--profile-mode", choices=("off", "torch", "nsight"), default="off")
    result.add_argument(
        "--policy", choices=("conservative", "stream_ordered"), default="conservative"
    )
    result.add_argument(
        "--flags", choices=tuple(f"{value:03b}" for value in range(8)), default="000"
    )
    result.add_argument("--inject-rank", type=int, default=None)
    result.add_argument("--output-dir", "--debug.output_dir", dest="directory", required=True)
    result.add_argument("--plan-only", action="store_true")
    return result


def plan(args):
    if args.f < 1 or args.t < 1 or args.layers < 1 or 2 % args.t:
        raise ValueError("Positive F/T/layers required; T must divide 2 KV heads")
    if args.sp and args.t == 1:
        raise ValueError("SP requires T>1")
    inject_rank = args.f * args.t - 1 if args.inject_rank is None else args.inject_rank
    if not 0 <= inject_rank < args.f * args.t:
        raise ValueError("inject-rank must belong to the mesh")
    counts = [int(token_batch(replica, 0)[1][:, 1:].ne(-100).sum()) for replica in range(args.f)]
    if args.f > 1 and len(set(counts)) < 2:
        raise ValueError("The oracle must exercise unequal valid counts")
    return dict(
        F=args.f,
        T=args.t,
        SP=args.sp,
        world_size=args.f * args.t,
        optimizer=args.optimizer,
        policy=args.policy,
        flags=args.flags,
        layers=args.layers,
        clean_steps=CLEAN_STEPS,
        continuous_training_steps=CLEAN_STEPS + 2,
        successful_updates=CLEAN_STEPS + 1,
        overflow_inject_rank=inject_rank,
        valid_counts_by_replica=counts,
        global_valid_tokens=sum(counts),
        dtype="fp16",
        master_dtype="fp32",
        atol=ATOL,
        rtol=RTOL,
        growth_interval=2,
        seed=SEED,
        oracle_scope="independent HF same-start single-step; not independent rollout equivalence",
        reference_synced_from_runtime="weights and optimizer state before each clean/recovery step",
        runtime_continuous_no_reset=True,
        optimizer_replay="all local leaves, actual unscaled gradients, same options, bit-exact",
        profiling_scope="HF, runtime, oracle sync and optimizer replay; no performance claim",
    )


def run(args, planned):
    if importlib.metadata.version("transformers") != "4.57.3":
        raise RuntimeError("Qwen AMP oracle requires transformers==4.57.3")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    worker = WorkerEnvironment.read()
    directory = Path(args.directory)
    if not args.directory.strip():
        raise ValueError("Output directory must not be empty")
    for prefix in ["qwen_amp", "profiling"]:
        if (directory / f"{prefix}.rank{worker.rank}.json").exists():
            raise ValueError("Use a fresh output directory to preserve previous evidence")
    if not torch.cuda.is_available():
        raise RuntimeError("Real FP16 AMP gate requires CUDA; --plan-only uses CPU exclusively")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    forward_prefetch, backward_prefetch, async_grad_rs = (bit == "1" for bit in args.flags)
    config = Config(
        model=ModelConfig(
            model_type="qwen2",
            vocab_size=64,
            num_key_value_heads=2,
            layers=args.layers,
            hidden_size=32,
            num_heads=4,
            mlp_hidden_size=64,
            seq_len=8,
            tie_word_embeddings=True,
        ),
        parallel=ParallelConfig(
            fsdp_degree=args.f,
            tp_degree=args.t,
            sp_enabled=args.sp,
            backend=args.backend,
            device=args.device,
        ),
        runtime=RuntimeConfig(
            reshard_after_forward=True,
            param_grouping="block",
            forward_prefetch=forward_prefetch,
            backward_prefetch=backward_prefetch,
            async_grad_rs=async_grad_rs,
            communication_policy=args.policy,
        ),
        training=TrainingConfig(
            dtype="fp16", optimizer=args.optimizer, lr=0.001, steps=5, seed=SEED
        ),
        debug=DebugConfig(output_dir=args.directory, distributed_timeout_seconds=180),
    )
    payload = dict(
        status="failed",
        rank=worker.rank,
        plan=planned,
        comparisons=[],
        optimizer_replay=[],
        oracle_starts=[],
        step_metrics=[],
        errors=[],
        continuous_runtime_no_reset=True,
        independent_rollout_equivalence_claimed=False,
        historical_independent_rollout_failures_reclassified=False,
    )
    capture = None
    try:
        with WorkerSession(config, "runtime") as session:
            context = session.context
            device = context.device
            qconfig = Qwen2ModelConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=args.layers,
                num_attention_heads=4,
                num_key_value_heads=2,
                tie_word_embeddings=True,
                max_position_embeddings=128,
                use_cache=False,
            )
            hfconfig = Qwen2Config(**qconfig.to_dict())
            hfconfig._attn_implementation = "sdpa"
            torch.manual_seed(SEED)
            reference = Qwen2ForCausalLM(hfconfig).train()
            model = Qwen2CausalLM(qconfig).train()
            model.load_state_dict(reference.state_dict(), strict=True)
            reference.to(device)
            runtime = ShardedRuntime(
                device=device,
                reshard_after_forward=True,
                forward_prefetch=forward_prefetch,
                backward_prefetch=backward_prefetch,
                async_grad_rs=async_grad_rs,
                communication_policy=args.policy,
                directory=session.directory,
            ).register(model, context, sp=args.sp)
            trainer = QwenTrainer(model, runtime, lr=0.001, dtype="fp16", optimizer=args.optimizer)
            trainer.scaler.set_growth_interval(2)
            optimizer_type = {"sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}[args.optimizer]
            ref_optimizer = optimizer_type(reference.parameters(), lr=0.001, foreach=False)
            ref_scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=2)
            references = dict(reference.named_parameters())
            validate_reference_layout(runtime, reference)
            assert_initial_weights(runtime, reference)
            assert_optimizer_contract(trainer.optimizer, args.optimizer)
            assert_optimizer_contract(ref_optimizer, args.optimizer)
            payload["initialization"] = dict(
                original_seed_HF_shards_bit_exact=True,
                seed=SEED,
                logical_parameter_count=len(references),
            )
            assert len(trainer.optimizer.param_groups) == len(ref_optimizer.param_groups) == 1
            # The independent HF optimizer uses every actual option, including an
            # inferred foreach setting, eps, betas, weight decay and maximize.
            ref_optimizer.param_groups[0].update(
                {
                    key: value
                    for key, value in trainer.optimizer.param_groups[0].items()
                    if key != "params"
                }
            )
            leaf_names = {handle.local_shard: name for name, handle in runtime.registry.items()}
            identities = {
                name: (id(handle.local_shard), handle.local_shard.data_ptr())
                for name, handle in runtime.registry.items()
            }
            captured = {}
            hook = model.register_forward_hook(
                lambda _module, _args, output: captured.update(
                    logits=output.logits.detach().clone()
                )
            )
            payload.update(
                torch=torch.__version__,
                transformers="4.57.3",
                gpu=torch.cuda.get_device_name(device),
                capability=torch.cuda.get_device_capability(device),
                cuda=torch.version.cuda,
                backend=dist.get_backend() if dist.is_initialized() else "none",
                actual_NCCL=dist.is_initialized() and dist.get_backend() == "nccl",
                coordinate=asdict(context.coordinate),
                model_config=qconfig.to_dict(),
                policy=args.policy,
                flags=args.flags,
                actual_optimizer_options={
                    key: value
                    for key, value in trainer.optimizer.param_groups[0].items()
                    if key != "params"
                },
            )
            capture = ProfileCapture(session.directory, worker.rank, device, mode=args.profile_mode)
            capture.start()

            def clean_step(index, stage):
                assert_optimizer_contract(trainer.optimizer, args.optimizer)
                previous_parameters = [
                    parameter.detach().clone() for parameter in runtime.sharded_parameters()
                ]
                previous_state = optimizer_snapshot(trainer.optimizer)
                before_by_leaf = dict(
                    zip(runtime.sharded_parameters(), previous_parameters, strict=True)
                )
                updates_before = index if index < CLEAN_STEPS else CLEAN_STEPS
                assert_update_count(trainer.optimizer, updates_before)
                expected_start = [(128.0, 0), (128.0, 1), (256.0, 0)]
                scale_start, tracker_start = (
                    expected_start[index] if index < CLEAN_STEPS else (128.0, 0)
                )
                assert trainer.scaler.get_scale() == scale_start
                assert trainer.scaler.state_dict()["_growth_tracker"] == tracker_start
                weights, states = reconstruct_start(runtime, trainer.optimizer)
                for name, handle in runtime.registry.items():
                    assert torch.equal(
                        handle.local_shard, expected_shard(runtime, name, weights[name])
                    )
                digest = start_hash(weights, states)
                copy_reference_start(reference, ref_optimizer, weights, states)
                for name, parameter in references.items():
                    assert torch.equal(parameter, weights[name])
                ref_scaler.load_state_dict(trainer.scaler.state_dict())
                payload["oracle_starts"].append(
                    dict(
                        stage=stage,
                        oracle_start_hash=digest,
                        reference_synced_from_runtime=True,
                        logical_parameter_count=len(weights),
                        optimizer_updates_before=updates_before,
                        scale=scale_start,
                        tracker=tracker_start,
                    )
                )
                del weights, states
                batches = [token_batch(replica, index) for replica in range(args.f)]
                full = [
                    torch.cat([batch[column] for batch in batches]).to(device)
                    for column in range(4)
                ]
                ref_optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        expected = reference(
                            input_ids=full[0],
                            labels=full[1],
                            attention_mask=full[2],
                            position_ids=full[3],
                            use_cache=False,
                        )
                ref_scaler.scale(expected.loss).backward()
                ref_scaler.unscale_(ref_optimizer)
                local = [value.to(device) for value in batches[context.coordinate.fsdp]]
                result = trainer.train_step(
                    local[0], local[1], attention_mask=local[2], position_ids=local[3]
                )
                assert not result.overflow
                assert_optimizer_contract(trainer.optimizer, args.optimizer)
                assert_master_state(trainer, identities)
                replay_records = []
                payload["optimizer_replay"].append(dict(stage=stage, leaves=replay_records))
                optimizer_replay(
                    trainer.optimizer, before_by_leaf, previous_state, leaf_names, replay_records
                )
                assert len(replay_records) == len(leaf_names)
                assert_update_count(trainer.optimizer, updates_before + 1)
                replica = slice(context.coordinate.fsdp * 2, context.coordinate.fsdp * 2 + 2)
                compare(
                    captured["logits"],
                    runtime.input_for_rank(expected.logits[replica]),
                    f"{stage}/logits",
                    payload["comparisons"],
                )
                compare(
                    torch.tensor(result.loss),
                    expected.loss,
                    f"{stage}/global_loss",
                    payload["comparisons"],
                )
                counts = [int(batch[1][:, 1:].ne(-100).sum()) for batch in batches]
                assert result.valid_tokens == sum(counts)
                for name, handle in runtime.registry.items():
                    compare(
                        handle.local_shard.grad,
                        expected_shard(runtime, name, references[name].grad),
                        f"{stage}/{name}/gradient_shard",
                        payload["comparisons"],
                    )
                ref_scaler.step(ref_optimizer)
                ref_scaler.update()
                assert_update_count(ref_optimizer, updates_before + 1)
                for name, value in runtime.reconstruct_tp_local().items():
                    compare(
                        value,
                        runtime.qwen_partitions[name].local_slice(references[name]),
                        f"{stage}/{name}/tp_local_update",
                        payload["comparisons"],
                    )
                for name, handle in runtime.registry.items():
                    compare(
                        handle.local_shard,
                        expected_shard(runtime, name, references[name]),
                        f"{stage}/{name}/persistent_update_shard",
                        payload["comparisons"],
                    )
                    for key, value in trainer.optimizer.state.get(handle.local_shard, {}).items():
                        expected_value = ref_optimizer.state[references[name]][key]
                        target = (
                            expected_shard(runtime, name, expected_value)
                            if value.ndim
                            else expected_value
                        )
                        compare(
                            value, target, f"{stage}/{name}/optimizer_{key}", payload["comparisons"]
                        )
                assert result.loss_scale == ref_scaler.get_scale()
                assert (
                    trainer.scaler.state_dict()["_growth_tracker"]
                    == ref_scaler.state_dict()["_growth_tracker"]
                )
                scale_expected, tracker_expected = (
                    [(128.0, 1), (256.0, 0), (256.0, 1)][index]
                    if index < CLEAN_STEPS
                    else (128.0, 1)
                )
                assert result.loss_scale == scale_expected
                assert trainer.scaler.state_dict()["_growth_tracker"] == tracker_expected
                changed = any(
                    not torch.equal(parameter, previous)
                    for parameter, previous in zip(
                        runtime.sharded_parameters(), previous_parameters, strict=True
                    )
                )
                statuses = rank_status(
                    trainer,
                    result,
                    unchanged=not changed,
                    state_unchanged=optimizer_unchanged(trainer.optimizer, previous_state),
                    changed=changed,
                )
                assert all(row["local_finite"] == 1 for row in statuses)
                assert all(row["parameters_changed"] == 1 for row in statuses)
                payload["step_metrics"].append(
                    dict(
                        stage=stage,
                        result=asdict(result),
                        valid_counts_by_replica=counts,
                        rank_status=statuses,
                    )
                )
                capture.step(result.step)
                return result

            try:
                for index in range(CLEAN_STEPS):
                    clean_step(index, f"clean{index}")
                before = [parameter.detach().clone() for parameter in runtime.sharded_parameters()]
                state_before = optimizer_snapshot(trainer.optimizer)
                scale_before = trainer.scaler.get_scale()
                assert_optimizer_contract(trainer.optimizer, args.optimizer)
                assert trainer.scaler.state_dict()["_growth_tracker"] == 1
                selected = next(
                    handle
                    for handle in runtime.registry.values()
                    if handle.spec.valid_len(runtime.materializer.rank) > 0
                )
                hook_calls = 0

                def inject(gradient):
                    nonlocal hook_calls
                    hook_calls += 1
                    gradient = gradient.clone()
                    gradient[0] = float("inf")
                    return gradient

                injection = (
                    selected.local_shard.register_hook(inject)
                    if worker.rank == planned["overflow_inject_rank"]
                    else None
                )
                try:
                    local = [
                        value.to(device)
                        for value in token_batch(context.coordinate.fsdp, CLEAN_STEPS)
                    ]
                    overflow = trainer.train_step(
                        local[0], local[1], attention_mask=local[2], position_ids=local[3]
                    )
                finally:
                    if injection is not None:
                        injection.remove()
                unchanged = all(
                    torch.equal(parameter, previous)
                    for parameter, previous in zip(
                        runtime.sharded_parameters(), before, strict=True
                    )
                )
                state_unchanged = optimizer_unchanged(trainer.optimizer, state_before)
                assert_optimizer_contract(trainer.optimizer, args.optimizer)
                statuses = rank_status(
                    trainer,
                    overflow,
                    unchanged=unchanged,
                    state_unchanged=state_unchanged,
                    hook_calls=hook_calls,
                )
                payload["overflow"] = dict(
                    scale_before=scale_before, result=asdict(overflow), rank_status=statuses
                )
                assert (
                    overflow.overflow
                    and overflow.loss_scale == scale_before * trainer.scaler.get_backoff_factor()
                )
                assert all(
                    row["overflow"]
                    == row["parameters_unchanged"]
                    == row["optimizer_state_unchanged"]
                    == 1
                    and row["tracker"] == 0
                    for row in statuses
                )
                assert [int(row["rank"]) for row in statuses if not row["local_finite"]] == [
                    planned["overflow_inject_rank"]
                ]
                assert [int(row["rank"]) for row in statuses if row["injection_hook_calls"]] == [
                    planned["overflow_inject_rank"]
                ]
                assert sum(row["injection_hook_calls"] for row in statuses) == 1
                assert_master_state(trainer, identities)
                assert_update_count(trainer.optimizer, CLEAN_STEPS)
                capture.step(overflow.step)
                recovery = clean_step(CLEAN_STEPS + 1, "recovery")
                changed = any(
                    not torch.equal(parameter, previous)
                    for parameter, previous in zip(
                        runtime.sharded_parameters(), before, strict=True
                    )
                )
                statuses = rank_status(trainer, recovery, unchanged=not changed, changed=changed)
                payload["recovery"] = dict(result=asdict(recovery), rank_status=statuses)
                assert recovery.loss_scale == overflow.loss_scale
                assert all(
                    row["parameters_changed"] == row["tracker"] == 1 and not row["overflow"]
                    for row in statuses
                )
                assert trainer.step_id == CLEAN_STEPS + 2
                payload.update(
                    status="passed",
                    FP32_master_parameters_gradients_optimizer_state=True,
                    stable_shard_identity=True,
                    actual_tied_parameter_alias=True,
                    live_buffers=0,
                    live_fullparam_leases=0,
                    runtime_training_steps=trainer.step_id,
                    successful_optimizer_updates=CLEAN_STEPS + 1,
                    all_local_leaf_optimizer_replays_bit_exact=True,
                    independent_optimizer_math_contract_verified=True,
                    same_start_HF_oracle=True,
                )
            finally:
                hook.remove()
                finishing = capture
                capture = None
                finishing.close()
    except BaseException as error:
        payload["status"] = "failed"
        payload["errors"].append(repr(error))
        raise
    finally:
        finalize_result(directory, worker.rank, payload, capture)


def main(argv=None):
    args = parser().parse_args(argv)
    planned = plan(args)
    if args.plan_only:
        print(json.dumps(planned, indent=2))
        return
    run(args, planned)


if __name__ == "__main__":
    main()
