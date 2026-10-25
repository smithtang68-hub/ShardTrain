"""Ten-update FP16 gates using the existing independent HF fixture and tolerances.

same_start and independent_rollout are separate fresh-process cases. Only the
former copies Runtime starting weights/moments into HF. Historical rollout
failures are never reclassified by a successful same-start/replay gate.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapping_worker import (  # noqa: E402
    PROJECT,
    import_fixture,
    input_hashes,
    observed_fixture,
    verify_source_metadata,
    write_json,
)


class DeferredNumericalChecks:
    """Keep original failed records, then fail all ranks at one oracle boundary."""

    def __init__(self, compare):
        self.original_compare = compare
        self.errors = []

    def compare(self, *args):
        try:
            return self.original_compare(*args)
        except AssertionError as error:
            self.errors.append(repr(error))
            return None


def synchronize_comparisons(runtime, checks, payload, stage):
    import torch
    import torch.distributed as dist

    value = torch.tensor([int(bool(checks.errors))], device=runtime.device, dtype=torch.int32)
    outputs = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    runtime.comm.unit_id, runtime.comm.phase = None, "FINALIZATION_ORACLE"
    ticket = runtime.comm.issue(
        "world",
        "finalization_comparison_status",
        dist.group.WORLD,
        value,
        outputs[0],
        lambda async_: dist.all_gather(outputs, value, async_op=async_),
        channel="fsdp_rs",
        asynchronous=False,
        group_role="oracle_metadata",
        extra_outputs=outputs[1:],
    )
    runtime.comm.consume(ticket, consumer="all_rank_numerical_check")
    failed_ranks = [rank for rank, flag in enumerate(outputs) if bool(flag.cpu().item())]
    if failed_ranks:
        payload["numerical_failure"] = {
            "stage": stage,
            "failed_ranks": failed_ranks,
            "local_failed": bool(checks.errors),
            "local_errors": checks.errors,
            "peer_failure_propagated": not checks.errors,
        }
        raise AssertionError(
            f"Independent numerical comparison failed at {stage}, ranks={failed_ranks}"
        )


def fixture_plan(args):
    if args.effective_updates < 10:
        raise ValueError("The two long gates require at least ten effective updates")
    return {
        "mode": args.mode,
        "mapping_id": args.mapping_id,
        "F": 2,
        "T": 2,
        "SP": True,
        "layers": 4,
        "hidden_size": 32,
        "seq_len": 8,
        "local_batch_size": 2,
        "optimizer": "AdamW",
        "lr": 0.001,
        "atol": 3e-3,
        "rtol": 3e-3,
        "requested_effective_updates": 3 if args.mode == "overflow" else args.effective_updates,
        "continuous_runtime_no_reset": True,
        "reference_synced_each_step": args.mode in {"same_start", "overflow"},
        "independent_rollout_equivalence_checked": args.mode == "independent_rollout",
        "independent_rollout_equivalence_claimed": False,
        "historical_failures_reclassified": False,
        "overflow_scope": "fresh case: two clean updates, rank3 inf injection, one recovery update",
        "seed": 313,
        "initial_scale": 128.0,
        "growth_interval": 2,
        "profile_mode": "off",
        "performance_claim_allowed": False,
    }


def run(args):
    import torch
    import torch.distributed as dist

    fixture = import_fixture("tests/distributed/qwen2_amp_worker.py", "_finalization_amp_fixture")
    planned = fixture_plan(args)
    rank = int(os.environ.get("RANK", "0"))
    directory = Path(args.output)
    result_path = directory / f"amp_{args.mode}.rank{rank}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    payload = {
        "status": "FAIL",
        "rank": rank,
        "mapping_id": args.mapping_id,
        "plan": planned,
        "errors": [],
        "comparisons": [],
        "optimizer_replay": [],
        "oracle_starts": [],
        "step_metrics": [],
        "successful_optimizer_updates": 0,
        "actual_optimizer_updates": 0,
        "oracle_source": "tests/distributed/qwen2_amp_worker.py",
        "oracle_sha256": hashlib.sha256(
            (PROJECT / "tests/distributed/qwen2_amp_worker.py").read_bytes()
        ).hexdigest(),
        "placement_manifest_sha256": hashlib.sha256(
            Path(args.placement_manifest).read_bytes()
        ).hexdigest(),
        "source_manifest_sha256": os.environ.get("SHARDTRAIN_SOURCE_MANIFEST_SHA256"),
    }
    trainer = reference = None
    try:
        verify_source_metadata(payload)
        if fixture.importlib.metadata.version("transformers") != "4.57.3":
            raise RuntimeError("Independent HF fixture requires transformers==4.57.3")
        from transformers import Qwen2Config, Qwen2ForCausalLM

        if not torch.cuda.is_available() or not dist.is_nccl_available():
            raise RuntimeError("Real four-GPU CUDA/NCCL required; no CPU math fallback")
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
                dtype="fp16",
                optimizer="adamw",
                lr=0.001,
                steps=args.effective_updates,
                seed=fixture.SEED,
            ),
            debug=fixture.DebugConfig(output_dir=args.output, distributed_timeout_seconds=180),
        )
        total_steps = 4 if args.mode == "overflow" else args.effective_updates
        payload["input_fingerprints"] = input_hashes(fixture.token_batch, total_steps, 2)
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
                torch.manual_seed(fixture.SEED)
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
                fixture.assert_optimizer_contract(trainer.optimizer, "adamw")
                fixture.assert_optimizer_contract(ref_optimizer, "adamw")
                payload["initialization"] = {
                    "seed_HF_shards_bit_exact": True,
                    "reference_initial_fingerprints": {
                        name: __import__("mapping_worker").tensor_hash(value)
                        for name, value in refs.items()
                    },
                }
                identities = dict(runtime.identities)
                names = {handle.local_shard: name for name, handle in runtime.registry.items()}
                captured = {}

                def retain(key):
                    def hook(_module, _args, output):
                        output.retain_grad()
                        captured[key] = output

                    return hook

                hooks = [
                    reference.model.embed_tokens.register_forward_hook(retain("ref_embedding")),
                    model.model.embed_tokens.register_forward_hook(retain("runtime_embedding")),
                    model.register_forward_hook(
                        lambda _m, _a, output: captured.update(
                            logits=output.logits.detach().clone()
                        )
                    ),
                ]

                def clean_step(index, stage):
                    checks = DeferredNumericalChecks(fixture.compare)
                    updates_before = payload["successful_optimizer_updates"]
                    fixture.assert_update_count(trainer.optimizer, updates_before)
                    before = {
                        parameter: parameter.detach().clone()
                        for parameter in runtime.sharded_parameters()
                    }
                    states_before = fixture.optimizer_snapshot(trainer.optimizer)
                    if args.mode != "independent_rollout":
                        weights, states = fixture.reconstruct_start(runtime, trainer.optimizer)
                        # copy_reference_start clones tensors, never aliases the input reference.
                        immutable_digest = fixture.start_hash(weights, states)
                        fixture.copy_reference_start(reference, ref_optimizer, weights, states)
                        if fixture.start_hash(weights, states) != immutable_digest:
                            raise AssertionError(
                                "HF setter mutated the independent starting snapshot"
                            )
                        ref_scaler.load_state_dict(trainer.scaler.state_dict())
                        payload["oracle_starts"].append(
                            {
                                "stage": stage,
                                "sha256": immutable_digest,
                                "reference_synced_from_runtime": True,
                            }
                        )
                        del weights, states
                    else:
                        payload["oracle_starts"].append(
                            {"stage": stage, "reference_synced_from_runtime": False}
                        )
                    scale_start = trainer.scaler.get_scale()
                    ref_scale_start = ref_scaler.get_scale()
                    if trainer.scaler.state_dict() != ref_scaler.state_dict():
                        raise AssertionError("Independent scaler states differ before clean step")
                    batches = [fixture.token_batch(replica, index) for replica in range(2)]
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
                    if result.overflow:
                        raise AssertionError(
                            "Unexpected clean-step overflow; effective update not completed"
                        )
                    payload["actual_optimizer_updates"] += 1
                    fixture.assert_master_state(trainer, identities)
                    fixture.assert_update_count(trainer.optimizer, updates_before + 1)
                    replay = []
                    payload["optimizer_replay"].append({"stage": stage, "leaves": replay})
                    fixture.optimizer_replay(
                        trainer.optimizer, before, states_before, names, replay
                    )
                    replica = slice(context.coordinate.fsdp * 2, context.coordinate.fsdp * 2 + 2)
                    checks.compare(
                        captured["logits"],
                        runtime.input_for_rank(expected.logits[replica]),
                        stage + "/logits",
                        payload["comparisons"],
                    )
                    checks.compare(
                        torch.tensor(result.loss),
                        expected.loss,
                        stage + "/global_loss",
                        payload["comparisons"],
                    )
                    checks.compare(
                        captured["runtime_embedding"].grad / scale_start,
                        runtime.input_for_rank(
                            captured["ref_embedding"].grad[replica] * 2 / ref_scale_start
                        ),
                        stage + "/decoder_input_gradient",
                        payload["comparisons"],
                    )
                    counts = [int(batch[1][:, 1:].ne(-100).sum()) for batch in batches]
                    if result.valid_tokens != sum(counts):
                        raise AssertionError("Global valid-token denominator mismatch")
                    for name, handle in runtime.registry.items():
                        checks.compare(
                            handle.local_shard.grad,
                            fixture.expected_shard(runtime, name, refs[name].grad),
                            f"{stage}/{name}/gradient_shard",
                            payload["comparisons"],
                        )
                    ref_scaler.step(ref_optimizer)
                    ref_scaler.update()
                    fixture.assert_update_count(ref_optimizer, updates_before + 1)
                    for name, handle in runtime.registry.items():
                        checks.compare(
                            handle.local_shard,
                            fixture.expected_shard(runtime, name, refs[name]),
                            f"{stage}/{name}/persistent_update_shard",
                            payload["comparisons"],
                        )
                        for key, value in trainer.optimizer.state[handle.local_shard].items():
                            target = ref_optimizer.state[refs[name]][key]
                            if value.ndim:
                                target = fixture.expected_shard(runtime, name, target)
                            checks.compare(
                                value,
                                target,
                                f"{stage}/{name}/optimizer_{key}",
                                payload["comparisons"],
                            )
                    if trainer.scaler.state_dict() != ref_scaler.state_dict():
                        raise AssertionError("Scaler states differ after clean update")
                    synchronize_comparisons(runtime, checks, payload, stage)
                    changed = any(
                        not torch.equal(parameter, before[parameter]) for parameter in before
                    )
                    statuses = fixture.rank_status(
                        trainer, result, unchanged=not changed, changed=changed
                    )
                    if not all(
                        row["parameters_changed"] and row["local_finite"] for row in statuses
                    ):
                        raise AssertionError("An effective update was missing or nonfinite")
                    payload["successful_optimizer_updates"] += 1
                    payload["step_metrics"].append(
                        {"stage": stage, "result": asdict(result), "rank_status": statuses}
                    )
                    captured.pop("ref_embedding", None)
                    captured.pop("runtime_embedding", None)
                    return result

                try:
                    if args.mode == "overflow":
                        clean_step(0, "clean0")
                        clean_step(1, "clean1")
                        before = [
                            parameter.detach().clone() for parameter in runtime.sharded_parameters()
                        ]
                        states_before = fixture.optimizer_snapshot(trainer.optimizer)
                        scale_before = trainer.scaler.get_scale()
                        selected = next(
                            handle
                            for handle in runtime.registry.values()
                            if handle.spec.valid_len(runtime.materializer.rank)
                        )
                        hook_calls = 0

                        def inject(gradient):
                            nonlocal hook_calls
                            hook_calls += 1
                            gradient = gradient.clone()
                            gradient[0] = float("inf")
                            return gradient

                        injection = (
                            selected.local_shard.register_hook(inject) if rank == 3 else None
                        )
                        try:
                            local = [
                                value.to(device)
                                for value in fixture.token_batch(context.coordinate.fsdp, 2)
                            ]
                            overflow = trainer.train_step(
                                local[0], local[1], attention_mask=local[2], position_ids=local[3]
                            )
                        finally:
                            if injection is not None:
                                injection.remove()
                        unchanged = all(
                            torch.equal(parameter, value)
                            for parameter, value in zip(
                                runtime.sharded_parameters(), before, strict=True
                            )
                        )
                        state_unchanged = fixture.optimizer_unchanged(
                            trainer.optimizer, states_before
                        )
                        statuses = fixture.rank_status(
                            trainer,
                            overflow,
                            unchanged=unchanged,
                            state_unchanged=state_unchanged,
                            hook_calls=hook_calls,
                        )
                        payload["overflow"] = {
                            "result": asdict(overflow),
                            "rank_status": statuses,
                            "scale_before": scale_before,
                        }
                        if (
                            not overflow.overflow
                            or overflow.loss_scale
                            != scale_before * trainer.scaler.get_backoff_factor()
                        ):
                            raise AssertionError("Global overflow/backoff contract failed")
                        if not all(
                            row["overflow"]
                            and row["parameters_unchanged"]
                            and row["optimizer_state_unchanged"]
                            and row["tracker"] == 0
                            for row in statuses
                        ):
                            raise AssertionError(
                                "Overflow update/state/scale was not skipped uniformly"
                            )
                        if [int(row["rank"]) for row in statuses if not row["local_finite"]] != [
                            3
                        ] or sum(row["injection_hook_calls"] for row in statuses) != 1:
                            raise AssertionError(
                                "Overflow injection did not affect exactly the declared rank once"
                            )
                        fixture.assert_update_count(trainer.optimizer, 2)
                        fixture.assert_master_state(trainer, identities)
                        recovery = clean_step(3, "recovery")
                        if (
                            recovery.overflow
                            or payload["successful_optimizer_updates"] != 3
                            or trainer.step_id != 4
                        ):
                            raise AssertionError(
                                "Overflow recovery did not restore one effective update"
                            )
                    else:
                        for index in range(args.effective_updates):
                            clean_step(index, f"clean{index}")
                        if payload["successful_optimizer_updates"] != args.effective_updates:
                            raise AssertionError("Requested effective update count not reached")
                    payload.update(
                        status="PASS",
                        runtime_training_steps=trainer.step_id,
                        stable_shard_identity=True,
                        live_buffers=0,
                        live_fullparam_leases=0,
                        all_local_leaf_optimizer_replays_bit_exact=True,
                    )
                    if args.mode == "independent_rollout":
                        payload["plan"]["independent_rollout_equivalence_claimed"] = True
                finally:
                    for hook in hooks:
                        hook.remove()
    except BaseException as exc:
        payload["errors"].append(repr(exc))
        # Save local tensors without new collectives: peers may already be failing.
        if trainer is not None and reference is not None:
            directory.mkdir(parents=True, exist_ok=True)
            diagnostic = directory / f"failure_state.rank{rank}.pt"
            try:
                torch.save(
                    {
                        "reference_parameters": {
                            name: value.detach().cpu().clone()
                            for name, value in reference.named_parameters()
                        },
                        "reference_gradients": {
                            name: None if value.grad is None else value.grad.detach().cpu().clone()
                            for name, value in reference.named_parameters()
                        },
                        "runtime_shards": {
                            name: handle.local_shard.detach().cpu().clone()
                            for name, handle in trainer.runtime.registry.items()
                        },
                        "runtime_gradients": {
                            name: None
                            if handle.local_shard.grad is None
                            else handle.local_shard.grad.detach().cpu().clone()
                            for name, handle in trainer.runtime.registry.items()
                        },
                        "runtime_optimizer": trainer.optimizer.state_dict(),
                        "runtime_scaler": trainer.scaler.state_dict(),
                        "reference_optimizer": ref_optimizer.state_dict(),
                        "reference_scaler": ref_scaler.state_dict(),
                    },
                    diagnostic,
                )
                payload["failure_tensor_path"] = str(diagnostic)
            except BaseException as diagnostic_error:
                payload["errors"].append("failure artifact: " + repr(diagnostic_error))
        raise
    finally:
        write_json(result_path, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--placement-manifest", required=True)
    parser.add_argument("--mapping-id", choices=["M0", "M1"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", choices=["same_start", "independent_rollout", "overflow"], required=True
    )
    parser.add_argument("--effective-updates", type=int, default=10)
    args = parser.parse_args()
    fixture_plan(args)
    run(args)


if __name__ == "__main__":
    main()
