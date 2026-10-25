"""Four-GPU diagnostic: retain strict FAIL and continue ten independent updates.

Only finite numerical comparison failures are deferred. Runtime, lifecycle,
optimizer replay, scaler and nonfinite failures stop the diagnostic. No profiler
or performance claim. Production modules and the old oracle are left unchanged.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from experiments.finalization_4gpu.mapping_worker import (  # noqa: E402
    import_fixture,
    input_hashes,
    observed_fixture,
    verify_source_metadata,
)
from experiments.fp16_numerics.common import (  # noqa: E402
    ATOL,
    BIAS,
    OPTIMIZER,
    RTOL,
    cpu,
    distribution,
    global_index,
    immutable_json,
    named_hash,
    pack,
    sha,
    tensor_hash,
)


def slice_state(fixture, runtime, state):
    result = dict(state)
    result["parameter"] = cpu(fixture.expected_shard(runtime, BIAS, state["parameter"]))
    if state["gradient"] is not None:
        result["gradient"] = cpu(fixture.expected_shard(runtime, BIAS, state["gradient"]))
    result["state"] = {
        key: cpu(fixture.expected_shard(runtime, BIAS, value))
        if torch.is_tensor(value) and value.ndim
        else value
        for key, value in state["state"].items()
    }
    return result


def run(args):
    if args.updates != 10:
        raise ValueError("Primary pre-registered diagnostic requires exactly ten effective updates")
    if importlib.metadata.version("transformers") != "4.57.3":
        raise RuntimeError("Versioned independent oracle requires transformers==4.57.3")
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("BLOCKED_RESOURCE: actual four-GPU NCCL required")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    fixture = import_fixture("tests/distributed/qwen2_amp_worker.py", "_p0_frozen_amp_fixture")
    rank = int(os.environ["RANK"])
    output = Path(args.output)
    result_file = output / f"trajectory.rank{rank}.json"
    if result_file.exists():
        raise FileExistsError(result_file)
    payload = {
        "schema_version": 1,
        "capture_status": "FAILED",
        "strict_status": "NOT_RUN",
        "mode": args.mode,
        "reference_synced_each_step": args.mode == "same_start",
        "rank": rank,
        "mapping_id": args.mapping_id,
        "requested_effective_updates": 10,
        "completed_effective_updates": 0,
        "numerically_accepted_updates": 0,
        "errors": [],
        "steps": [],
        "historical_failures_reclassified": False,
        "atol": ATOL,
        "rtol": RTOL,
        "profile_mode": "off",
        "performance_claim_allowed": False,
        "oracle_source_sha256": sha(PROJECT / "tests/distributed/qwen2_amp_worker.py"),
        "placement_manifest_sha256": sha(args.placement_manifest),
        "source_manifest_sha256": os.environ.get("SHARDTRAIN_SOURCE_MANIFEST_SHA256"),
        "actual_distributed_collectives": True,
        "actual_custom_runtime_executed": True,
        "evidence_class": "CURRENT_GPU_REPRODUCTION",
        "input_fingerprints": input_hashes(fixture.token_batch, 10, 2),
        "optimizer_contract": OPTIMIZER,
        "distribution_scope": (
            "per-rank actual leaf storage (valid elements only), plus each "
            "parameter; TP replicated leaves are not global-unique weighted"
        ),
        "diagnostic_interference": (
            "Target reducer tensors and gradients copied to CPU; "
            "synchronization/IO permitted; no timing comparison allowed"
        ),
    }
    began = time.perf_counter()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = fixture.Config(
        model=fixture.ModelConfig(
            model_type="qwen2",
            vocab_size=64,
            num_key_value_heads=2,
            layers=4,
            hidden_size=32,
            num_heads=4,
            mlp_hidden_size=64,
            seq_len=8,
            tie_word_embeddings=True,
        ),
        parallel=fixture.ParallelConfig(
            fsdp_degree=2, tp_degree=2, sp_enabled=True, backend="nccl", device="cuda"
        ),
        runtime=fixture.RuntimeConfig(
            reshard_after_forward=True,
            param_grouping="block",
            forward_prefetch=True,
            backward_prefetch=True,
            async_grad_rs=True,
            communication_policy="stream_ordered",
        ),
        training=fixture.TrainingConfig(
            dtype="fp16", optimizer="adamw", lr=0.001, steps=10, seed=313
        ),
        debug=fixture.DebugConfig(output_dir=args.output, distributed_timeout_seconds=180),
    )
    try:
        verify_source_metadata(payload)
        with observed_fixture(fixture, args, payload):
            with fixture.WorkerSession(cfg, "runtime") as session:
                context, device = session.context, session.context.device
                qconfig = fixture.Qwen2ModelConfig(
                    vocab_size=64,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=4,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    tie_word_embeddings=True,
                    max_position_embeddings=128,
                    use_cache=False,
                )
                hfconfig = Qwen2Config(**qconfig.to_dict())
                hfconfig._attn_implementation = "sdpa"
                torch.manual_seed(313)
                reference = Qwen2ForCausalLM(hfconfig).train()
                model = fixture.Qwen2CausalLM(qconfig).train()
                model.load_state_dict(reference.state_dict(), strict=True)
                reference.to(device)
                runtime = fixture.ShardedRuntime(
                    device=device,
                    reshard_after_forward=True,
                    forward_prefetch=True,
                    backward_prefetch=True,
                    async_grad_rs=True,
                    communication_policy="stream_ordered",
                    directory=session.directory,
                ).register(model, context, sp=True)
                trainer = fixture.QwenTrainer(
                    model, runtime, lr=0.001, dtype="fp16", optimizer="adamw"
                )
                trainer.scaler.set_growth_interval(2)
                ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, foreach=False)
                ref_scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=2)
                refs = dict(reference.named_parameters())
                fixture.validate_reference_layout(runtime, reference)
                fixture.assert_initial_weights(runtime, reference)
                identities = dict(runtime.identities)
                names = {handle.local_shard: name for name, handle in runtime.registry.items()}
                target = runtime.registry[BIAS].local_shard
                spec = runtime.registry[BIAS].spec
                partition = runtime.qwen_partitions[BIAS]
                indices = [
                    global_index(i, spec, partition, context.coordinate.fsdp)
                    for i in range(target.numel())
                ]
                payload["initialization"] = {
                    "HF_seed": 313,
                    "global_unique_parameters": sum(value.numel() for value in refs.values()),
                    "reference_initial_parameter_sha256": named_hash(refs),
                    "initial_shards_bit_exact_HF": True,
                    "model_config": qconfig.to_dict(),
                    "global_k_bias_indices": indices,
                }
                payload["parameter_layout"] = {
                    name: {
                        "global_shape": list(handle.spec.global_shape),
                        "tp_local_shape": list(handle.spec.tp_local_shape),
                        "tp_dim": runtime.qwen_partitions[name].dim,
                        "tp_rank": runtime.qwen_partitions[name].rank,
                        "fsdp_rank": context.coordinate.fsdp,
                        "fsdp_shard_numel": handle.spec.fsdp_shard_numel,
                        "valid_numel": handle.spec.valid_len(runtime.materializer.rank),
                    }
                    for name, handle in runtime.registry.items()
                }
                payload["environment"] = {
                    "torch": torch.__version__,
                    "transformers": importlib.metadata.version("transformers"),
                    "cuda": torch.version.cuda,
                    "nccl": torch.cuda.nccl.version(),
                    "gpu": torch.cuda.get_device_name(device),
                    "actual_uuid": payload["placement"]["local"]["actual_uuid"],
                    "capability": list(torch.cuda.get_device_capability(device)),
                    "global_rank": rank,
                    "local_rank": int(os.environ["LOCAL_RANK"]),
                    "world_size": dist.get_world_size(),
                }
                captured = {}

                def retain(key):
                    def hook(_module, _args, value):
                        value.retain_grad()
                        captured[key] = value

                    return hook

                hooks = [
                    reference.model.embed_tokens.register_forward_hook(retain("ref_embedding")),
                    reference.model.layers[3].self_attn.k_proj.register_forward_hook(
                        retain("ref_k_output")
                    ),
                    model.model.embed_tokens.register_forward_hook(retain("runtime_embedding")),
                    model.register_forward_hook(
                        lambda _m, _a, value: captured.update(logits=cpu(value.logits))
                    ),
                ]
                original_reduce, original_unscale, original_step = (
                    runtime.materializer.reduce,
                    trainer.scaler.unscale_,
                    trainer.optimizer.step,
                )
                from sharded_runtime.runtime.qwen2_operators import QwenProjection

                original_apply = QwenProjection.apply
                original_apply_descriptor = QwenProjection.__dict__.get("apply")

                def observed_apply(*values):
                    outputs = original_apply(*values)
                    offset = 0
                    for position, provider in enumerate(values[2]):
                        if provider.bias_slot and provider.bias_slot.handle.spec.param_id == BIAS:
                            actual_bias = values[4 + offset + 1]
                            captured["k_bias_forward_materialized_master_tp_local"] = cpu(
                                actual_bias
                            )
                            captured["k_bias_forward_effective_compute_tp_local"] = cpu(
                                actual_bias.to(outputs[position].dtype)
                            )

                            def observe_k_gradient(gradient, compute_dtype=outputs[position].dtype):
                                captured["k_output_scaled_gradient"] = cpu(gradient)
                                summed = gradient.float().sum(tuple(range(gradient.ndim - 1)))
                                captured["k_bias_sum_fp32_before_compute_cast"] = cpu(summed)
                                captured["k_bias_sum_after_compute_cast"] = cpu(
                                    summed.to(compute_dtype).float()
                                )
                                return None

                            outputs[position].register_hook(observe_k_gradient)
                        offset += 2 if provider.bias_slot else 1
                    return outputs

                QwenProjection.apply = observed_apply

                def observed_reduce(unit, gradients):
                    locations = [
                        i for i, handle in enumerate(unit.handles) if handle.spec.param_id == BIAS
                    ]
                    if locations:
                        captured["k_bias_pre_fsdp_scaled_tp_local"] = cpu(gradients[locations[0]])
                    result = original_reduce(unit, gradients)
                    if locations:
                        captured["k_bias_post_fsdp_average_scaled_shard"] = cpu(
                            result[locations[0]]
                        )
                    return result

                def observed_unscale(optimizer):
                    captured["k_bias_reduced_scaled_shard"] = cpu(target.grad)
                    result = original_unscale(optimizer)
                    captured["k_bias_reduced_unscaled_shard"] = cpu(target.grad)
                    captured["found_inf_per_device"] = {
                        str(key): float(value)
                        for key, value in trainer.scaler._per_optimizer_states[id(optimizer)][
                            "found_inf_per_device"
                        ].items()
                    }
                    return result

                def observed_step(*values, **kwargs):
                    captured["runtime_pre_optimizer"] = pack(target, trainer.optimizer)
                    result = original_step(*values, **kwargs)
                    captured["runtime_post_optimizer"] = pack(target, trainer.optimizer)
                    return result

                runtime.materializer.reduce, trainer.scaler.unscale_, trainer.optimizer.step = (
                    observed_reduce,
                    observed_unscale,
                    observed_step,
                )
                try:
                    for step in range(10):
                        fixture.assert_update_count(trainer.optimizer, step)
                        fixture.assert_update_count(ref_optimizer, step)
                        if args.mode == "same_start":
                            weights, states = fixture.reconstruct_start(runtime, trainer.optimizer)
                            fixture.copy_reference_start(reference, ref_optimizer, weights, states)
                            ref_scaler.load_state_dict(trainer.scaler.state_dict())
                            del weights, states
                        before = {p: p.detach().clone() for p in runtime.sharded_parameters()}
                        states_before = fixture.optimizer_snapshot(trainer.optimizer)
                        ref_before = {p: p.detach().clone() for p in reference.parameters()}
                        ref_states_before = fixture.optimizer_snapshot(ref_optimizer)
                        reference_pre_forward = cpu(refs[BIAS])
                        runtime_pre_forward = cpu(target)
                        pre_hash = {
                            "runtime_shards": named_hash(
                                {
                                    name: handle.local_shard
                                    for name, handle in runtime.registry.items()
                                }
                            ),
                            "dense_reference": named_hash(refs),
                            "cpu_rng": tensor_hash(torch.get_rng_state()),
                            "cuda_rng": tensor_hash(torch.cuda.get_rng_state(device)),
                        }
                        scale_before = trainer.scaler.get_scale()
                        if trainer.scaler.state_dict() != ref_scaler.state_dict():
                            raise AssertionError("Scaler states differ before clean step")
                        batches = [fixture.token_batch(replica, step) for replica in range(2)]
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
                        dense_raw_scaled = cpu(refs[BIAS].grad)
                        ref_scaler.unscale_(ref_optimizer)
                        dense_found_inf = {
                            str(key): float(value)
                            for key, value in ref_scaler._per_optimizer_states[id(ref_optimizer)][
                                "found_inf_per_device"
                            ].items()
                        }
                        dense_pre = pack(refs[BIAS], ref_optimizer)
                        local = [value.to(device) for value in batches[context.coordinate.fsdp]]
                        result = trainer.train_step(
                            local[0], local[1], attention_mask=local[2], position_ids=local[3]
                        )
                        if result.overflow or not all(
                            torch.isfinite(p.grad).all() for p in reference.parameters()
                        ):
                            raise FloatingPointError(
                                "BLOCKED_NONFINITE: effective clean update could not be completed"
                            )
                        fixture.assert_master_state(trainer, identities)
                        fixture.assert_update_count(trainer.optimizer, step + 1)
                        replay = []
                        fixture.optimizer_replay(
                            trainer.optimizer, before, states_before, names, replay
                        )
                        replica = slice(
                            context.coordinate.fsdp * 2, context.coordinate.fsdp * 2 + 2
                        )
                        row = {
                            "step": step,
                            "effective_update": step + 1,
                            "reference_synced_from_runtime": args.mode == "same_start",
                            "pre_forward_hashes": pre_hash,
                            "loss_runtime": result.loss,
                            "loss_reference": float(expected.loss),
                            "scale_before": scale_before,
                            "scale_after": trainer.scaler.get_scale(),
                            "overflow": result.overflow,
                            "valid_tokens": result.valid_tokens,
                            "runtime_result": asdict(result),
                            "runtime_optimizer_replay": replay,
                            "logits": distribution(
                                captured["logits"], runtime.input_for_rank(expected.logits[replica])
                            ),
                            "loss": distribution(torch.tensor(result.loss), expected.loss),
                            "decoder_input_gradient": distribution(
                                captured["runtime_embedding"].grad / scale_before,
                                runtime.input_for_rank(
                                    captured["ref_embedding"].grad[replica] * 2 / scale_before
                                ),
                            ),
                            "diagnostic_k_output_gradient": distribution(
                                captured["k_output_scaled_gradient"] / scale_before,
                                captured["ref_k_output"].grad[
                                    replica, :, partition.rank * 8 : (partition.rank + 1) * 8
                                ]
                                * 2
                                / scale_before,
                            ),
                            "parameter_gradients": {},
                            "parameters_after": {},
                            "optimizer_states": {},
                        }
                        if result.valid_tokens != sum(
                            int(batch[1][:, 1:].ne(-100).sum()) for batch in batches
                        ):
                            raise AssertionError("Invalid loss normalization denominator")
                        ga, gb, pa, pb = [], [], [], []
                        for name, handle in runtime.registry.items():
                            expected_gradient = fixture.expected_shard(
                                runtime, name, refs[name].grad
                            )
                            valid = handle.spec.valid_len(runtime.materializer.rank)
                            stats = distribution(handle.local_shard.grad, expected_gradient)
                            for item in stats["topk"]:
                                if item["flat_index"] < valid:
                                    item["global_flat_index"] = global_index(
                                        item["flat_index"],
                                        handle.spec,
                                        runtime.qwen_partitions[name],
                                        context.coordinate.fsdp,
                                    )
                            row["parameter_gradients"][name] = stats
                            ga.append(cpu(handle.local_shard.grad[:valid]))
                            gb.append(cpu(expected_gradient[:valid]))
                        ref_scaler.step(ref_optimizer)
                        ref_scaler.update()
                        dense_post = pack(refs[BIAS], ref_optimizer)
                        ref_replay = []
                        fixture.optimizer_replay(
                            ref_optimizer,
                            ref_before,
                            ref_states_before,
                            {p: name for name, p in refs.items()},
                            ref_replay,
                        )
                        row["reference_optimizer_replay"] = ref_replay
                        fixture.assert_update_count(ref_optimizer, step + 1)
                        for name, handle in runtime.registry.items():
                            target_value = fixture.expected_shard(runtime, name, refs[name])
                            valid = handle.spec.valid_len(runtime.materializer.rank)
                            stats = distribution(handle.local_shard, target_value)
                            for item in stats["topk"]:
                                if item["flat_index"] < valid:
                                    item["global_flat_index"] = global_index(
                                        item["flat_index"],
                                        handle.spec,
                                        runtime.qwen_partitions[name],
                                        context.coordinate.fsdp,
                                    )
                            row["parameters_after"][name] = stats
                            pa.append(cpu(handle.local_shard[:valid]))
                            pb.append(cpu(target_value[:valid]))
                            row["optimizer_states"][name] = {}
                            for key, value in trainer.optimizer.state[handle.local_shard].items():
                                wanted = ref_optimizer.state[refs[name]][key]
                                if value.ndim:
                                    wanted = fixture.expected_shard(runtime, name, wanted)
                                row["optimizer_states"][name][key] = distribution(value, wanted)
                        if trainer.scaler.state_dict() != ref_scaler.state_dict():
                            raise AssertionError("Scaler schedule diverged after finite update")
                        row["all_local_gradient_distribution"] = distribution(
                            torch.cat(ga), torch.cat(gb)
                        )
                        row["all_local_parameter_distribution"] = distribution(
                            torch.cat(pa), torch.cat(pb)
                        )
                        comparisons = [
                            ("post_forward/logits", row["logits"]),
                            ("post_forward/loss", row["loss"]),
                            ("post_backward/decoder_input_gradient", row["decoder_input_gradient"]),
                        ]
                        comparisons += [
                            ("post_unscale/" + name, stats)
                            for name, stats in row["parameter_gradients"].items()
                        ]
                        comparisons += [
                            ("post_optimizer/" + name, stats)
                            for name, stats in row["parameters_after"].items()
                        ]
                        comparisons += [
                            ("post_optimizer/" + name + "/" + key, stats)
                            for name, states in row["optimizer_states"].items()
                            for key, stats in states.items()
                        ]
                        row["strict_failures"] = [
                            {"transition": name, **stats}
                            for name, stats in comparisons
                            if not stats["passed"]
                        ]
                        row["first_nonbitwise_transition"] = next(
                            (name for name, stats in comparisons if not stats["bitwise_equal"]),
                            None,
                        )
                        hard_nonfinite = any(
                            stats["actual_nonfinite"] or stats["reference_nonfinite"]
                            for _, stats in comparisons
                        )
                        flags = [None] * 4
                        dist.all_gather_object(
                            flags, [bool(row["strict_failures"]), hard_nonfinite]
                        )
                        row["failed_logical_ranks"] = [r for r, flag in enumerate(flags) if flag[0]]
                        row["hard_nonfinite_logical_ranks"] = [
                            r for r, flag in enumerate(flags) if flag[1]
                        ]
                        row["strict_status"] = "FAIL" if row["failed_logical_ranks"] else "PASS"
                        capture = {
                            "schema_version": 1,
                            "step": step,
                            "effective_update": step + 1,
                            "mode": args.mode,
                            "reference_synced_from_runtime": args.mode == "same_start",
                            "rank": rank,
                            "logical_parameter": BIAS,
                            "global_indices": indices,
                            "source_manifest_sha256": payload["source_manifest_sha256"],
                            "runtime": {
                                "pre": captured["runtime_pre_optimizer"],
                                "post": captured["runtime_post_optimizer"],
                            },
                            "reference": {
                                "pre": slice_state(fixture, runtime, dense_pre),
                                "post": slice_state(fixture, runtime, dense_post),
                            },
                            "reference_full_bias_pre": dense_pre,
                            "reference_full_bias_post": dense_post,
                            "pre_forward": {
                                "runtime_master": runtime_pre_forward,
                                "runtime_compute_fp16": runtime_pre_forward.half(),
                                "reference_master": reference_pre_forward,
                                "reference_compute_fp16": reference_pre_forward.half(),
                            },
                            "gradient_transitions": {
                                key: value
                                for key, value in captured.items()
                                if key.startswith("k_bias_")
                            },
                            "reference_scaled_gradient_full": dense_raw_scaled,
                            "distribution_shards": {
                                "parameters_actual": {
                                    name: cpu(handle.local_shard)
                                    for name, handle in runtime.registry.items()
                                },
                                "parameters_expected": {
                                    name: cpu(fixture.expected_shard(runtime, name, refs[name]))
                                    for name in runtime.registry
                                },
                                "gradients_actual": {
                                    name: cpu(handle.local_shard.grad)
                                    for name, handle in runtime.registry.items()
                                },
                                "gradients_expected": {
                                    name: cpu(
                                        fixture.expected_shard(runtime, name, refs[name].grad)
                                    )
                                    for name in runtime.registry
                                },
                            },
                            "k_output_scaled_gradient_tp_local": captured[
                                "k_output_scaled_gradient"
                            ],
                            "reference_k_output_scaled_gradient_full": cpu(
                                captured["ref_k_output"].grad
                            ),
                            "amp": {
                                "scale_before": scale_before,
                                "runtime_state_after": trainer.scaler.state_dict(),
                                "reference_state_after": ref_scaler.state_dict(),
                                "runtime_found_inf_per_device": captured["found_inf_per_device"],
                                "reference_found_inf_per_device": dense_found_inf,
                                "found_inf_contract": (
                                    "actual pinned-version GradScaler state captured "
                                    "immediately after unscale; global finite branch also "
                                    "required"
                                ),
                                "skipped": False,
                            },
                        }
                        tensor_path = output / f"states.step{step:02d}.rank{rank}.pt"
                        output.mkdir(parents=True, exist_ok=True)
                        if tensor_path.exists():
                            raise FileExistsError(tensor_path)
                        torch.save(capture, tensor_path)
                        row.update(tensor_path=tensor_path.name, tensor_sha256=sha(tensor_path))
                        immutable_json(output / f"step{step:02d}.rank{rank}.json", row)
                        payload["steps"].append(row)
                        payload["completed_effective_updates"] += 1
                        payload["numerically_accepted_updates"] += not bool(
                            row["failed_logical_ranks"]
                        )
                        payload["strict_status"] = (
                            "FAIL"
                            if any(item["strict_status"] == "FAIL" for item in payload["steps"])
                            else "PASS"
                        )
                        if row["strict_failures"] and "first_local_strict_failure" not in payload:
                            payload["first_local_strict_failure"] = {
                                "step": step,
                                **row["strict_failures"][0],
                            }
                        captured.clear()
                        del expected, before, states_before, ref_before, ref_states_before
                        if row["hard_nonfinite_logical_ranks"]:
                            raise FloatingPointError(
                                "BLOCKED_NONFINITE: state/parameter/activation contains "
                                "nonfinite values; evidence retained"
                            )
                    payload["strict_status"] = (
                        "FAIL"
                        if any(row["strict_status"] == "FAIL" for row in payload["steps"])
                        else "PASS"
                    )
                    payload["capture_status"] = "COMPLETED"
                    payload["stable_leaf_tied_lifecycle"] = True
                    payload["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    payload["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                finally:
                    runtime.materializer.reduce, trainer.scaler.unscale_, trainer.optimizer.step = (
                        original_reduce,
                        original_unscale,
                        original_step,
                    )
                    if original_apply_descriptor is None:
                        del QwenProjection.apply
                    else:
                        QwenProjection.apply = original_apply_descriptor
                    for hook in hooks:
                        hook.remove()
    except BaseException as error:
        payload["errors"].append(repr(error))
        # No new collectives during failure handling; retain whatever real state
        # exists locally, without fabricating missing transition observations.
        if "trainer" in locals() and "reference" in locals():
            failure_path = output / f"failure_context.rank{rank}.pt"
            try:
                output.mkdir(parents=True, exist_ok=True)
                if failure_path.exists():
                    raise FileExistsError(failure_path)
                torch.save(
                    {
                        "schema_version": 1,
                        "source_manifest_sha256": payload["source_manifest_sha256"],
                        "error": repr(error),
                        "mode": args.mode,
                        "rank": rank,
                        "completed_effective_updates": payload["completed_effective_updates"],
                        "runtime_current_actual_state": pack(target, trainer.optimizer),
                        "reference_current_actual_state": pack(refs[BIAS], ref_optimizer),
                        "captured_transitions_available": {
                            key: cpu(value)
                            for key, value in captured.items()
                            if torch.is_tensor(value)
                        },
                    },
                    failure_path,
                )
                payload.update(
                    failure_context_path=failure_path.name, failure_context_sha256=sha(failure_path)
                )
            except BaseException as save_error:
                payload["errors"].append("failure-context save: " + repr(save_error))
        raise
    finally:
        payload["diagnostic_wall_seconds"] = time.perf_counter() - began
        immutable_json(result_file, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement-manifest", required=True)
    parser.add_argument("--mapping-id", choices=["M0"], default="M0")
    parser.add_argument("--mode", choices=["independent", "same_start"], required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-strict-pass", action="store_true")
    args = parser.parse_args()
    result = run(args)
    # Diagnostic completion is separate from mathematical PASS, explicitly on stdout.
    print(
        json.dumps(
            {
                "capture_status": result["capture_status"],
                "strict_status": result["strict_status"],
                "mode": args.mode,
            }
        )
    )
    if args.require_strict_pass and result["strict_status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
