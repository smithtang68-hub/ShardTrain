"""Independent HF oracle for real multi-process Qwen2 FSDP/TP/SP training."""

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM

from sharded_runtime.config import (
    Config,
    DebugConfig,
    ModelConfig,
    ParallelConfig,
    RuntimeConfig,
    TrainingConfig,
)
from sharded_runtime.launch.bootstrap import WorkerSession
from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig
from sharded_runtime.observability.profiling import ProfileCapture
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.training.reference import compare_tensor


def token_batch(replica, step, vocabulary=64):
    """Unequal counts, boundary label and prefix context; identical within a TP replica."""
    generator = torch.Generator().manual_seed(2718 + replica * 100 + step)
    tokens = torch.randint(1, vocabulary, (2, 8), generator=generator)
    mask = torch.ones_like(tokens)
    mask[1, -(2 + 2 * replica) :] = 0
    labels = tokens.clone()
    labels[mask == 0] = -100
    labels[0, 4 + replica] = -100
    # Position 3's target is at 4, across the T=2 SP partition boundary.
    labels[1, 4] = tokens[1, 4] if mask[1, 4] else -100
    positions = torch.arange(17 + step, 25 + step).expand(2, -1)
    return tokens, labels, mask, positions


def main():
    if transformers.__version__ != "4.57.3":
        raise RuntimeError("Qwen numerical oracle requires transformers==4.57.3")
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=int, required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--profile-mode", choices=("off", "torch", "nsight"), default="off")
    parser.add_argument("--expect-init-mismatch", action="store_true")
    parser.add_argument(
        "--policy", choices=("conservative", "stream_ordered"), default="conservative"
    )
    parser.add_argument(
        "--flags", choices=tuple(f"{value:03b}" for value in range(8)), default="000"
    )
    parser.add_argument("--debug.output_dir", dest="directory", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    forward_prefetch, backward_prefetch, async_grad_rs = (bit == "1" for bit in args.flags)
    config = Config(
        model=ModelConfig(
            model_type="qwen2",
            vocab_size=64,
            num_key_value_heads=2,
            layers=args.layers,
            hidden_size=16,
            num_heads=4,
            mlp_hidden_size=32,
            seq_len=8,
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
        training=TrainingConfig(lr=0.01, steps=args.steps, seed=777),
        debug=DebugConfig(output_dir=args.directory, distributed_timeout_seconds=90),
    )
    with WorkerSession(config, "runtime") as session:
        context = session.context
        qconfig = Qwen2ModelConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=args.layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
            max_position_embeddings=128,
        )
        hfconfig = Qwen2Config(**qconfig.to_dict())
        # Match the runtime's documented boolean SDPA mask semantics. HF eager
        # gives uniform attention on an entirely blocked row with its finite-min
        # additive mask, so it is a different oracle for all-padding samples.
        hfconfig._attn_implementation = "sdpa"
        torch.manual_seed(777)
        reference = Qwen2ForCausalLM(hfconfig).train()
        model = Qwen2CausalLM(qconfig).train()
        model.load_state_dict(reference.state_dict(), strict=True)
        assert model.model.embed_tokens.weight is model.lm_head.weight
        runtime = ShardedRuntime(
            device=context.device,
            reshard_after_forward=True,
            forward_prefetch=forward_prefetch,
            backward_prefetch=backward_prefetch,
            async_grad_rs=async_grad_rs,
            communication_policy=args.policy,
            directory=session.directory,
        )
        if args.expect_init_mismatch:
            if session.worker.rank == args.f * args.t - 1:
                with torch.no_grad():
                    model.model.embed_tokens.weight[0, 0].add_(0.125)
            source_parameters = dict(model.named_parameters())
            source_identities = {
                name: (id(parameter), parameter.data_ptr(), tuple(parameter.shape))
                for name, parameter in source_parameters.items()
            }
            source_values = {
                name: parameter.detach().clone() for name, parameter in source_parameters.items()
            }
            try:
                runtime.register(model, context, sp=args.sp)
            except ValueError as error:
                assert "initialization source/config differs across ranks" in str(error)
                assert not runtime.units and not runtime.registry
                assert not hasattr(model, "_sharded_runtime")
                assert type(model.model.embed_tokens) is torch.nn.Embedding
                assert model.model.embed_tokens.weight is model.lm_head.weight
                assert source_identities == {
                    name: (id(parameter), parameter.data_ptr(), tuple(parameter.shape))
                    for name, parameter in model.named_parameters()
                }
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, source_values[name], atol=0, rtol=0)
                rejection = {
                    "status": "passed",
                    "rank": session.worker.rank,
                    "F": args.f,
                    "T": args.t,
                    "error": str(error),
                    "model_unchanged": True,
                    "partial_shards": len(runtime.registry),
                }
                (
                    session.directory / f"qwen_rejected_init.rank{session.worker.rank}.json"
                ).write_text(json.dumps(rejection, indent=2), encoding="utf-8")
                return
            raise AssertionError("Different rank source weights were silently combined")
        runtime.register(model, context, sp=args.sp)
        assert model.model.embed_tokens.weight is model.lm_head.weight
        optimizer = torch.optim.SGD(runtime.sharded_parameters(), lr=0.01)
        runtime.bind_optimizer(optimizer)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        references = dict(reference.named_parameters())
        embedding_outputs = {}

        def retain_embedding_output(key):
            def retain(module, args, output):
                output.retain_grad()
                embedding_outputs[key] = output

            return retain

        reference.model.embed_tokens.register_forward_hook(retain_embedding_output("reference"))
        model.model.embed_tokens.register_forward_hook(retain_embedding_output("runtime"))

        def forbid_saved_full_parameter(tensor):
            live_weights = {
                lease.tensor.untyped_storage().data_ptr()
                for unit in runtime.units
                if unit.lease is not None
                for lease in unit.lease.full_param_storages.values()
                if lease.tensor is not None
            }
            assert tensor.untyped_storage().data_ptr() not in live_weights, (
                "Autograd saved FullParam"
            )
            return tensor

        identities = {
            name: (id(handle.local_shard), handle.local_shard.data_ptr())
            for name, handle in runtime.registry.items()
        }
        comparisons, step_rows = [], []
        capture = ProfileCapture(
            session.directory, session.worker.rank, context.device, mode=args.profile_mode
        )
        capture.start()
        try:
            with runtime.compute_context():
                for step in range(args.steps):
                    all_batches = [token_batch(replica, step) for replica in range(args.f)]
                    full_tokens, full_labels, full_mask, full_positions = [
                        torch.cat([batch[index] for batch in all_batches]) for index in range(4)
                    ]
                    expected_optimizer.zero_grad(set_to_none=True)
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        expected = reference(
                            input_ids=full_tokens,
                            labels=full_labels,
                            attention_mask=full_mask,
                            position_ids=full_positions,
                            use_cache=False,
                        )
                    expected.loss.backward()
                    tokens, labels, mask, positions = [
                        tensor.to(context.device) for tensor in all_batches[context.coordinate.fsdp]
                    ]
                    runtime.begin_step(step)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autograd.graph.saved_tensors_hooks(
                        forbid_saved_full_parameter, lambda x: x
                    ):
                        actual = model(
                            input_ids=tokens,
                            labels=labels,
                            attention_mask=mask,
                            position_ids=positions,
                            use_cache=False,
                        )
                    replica_slice = slice(
                        context.coordinate.fsdp * 2, context.coordinate.fsdp * 2 + 2
                    )
                    expected_logits = runtime.input_for_rank(
                        expected.logits.detach()[replica_slice]
                    )
                    comparisons.append(
                        compare_tensor(
                            actual.logits,
                            expected_logits,
                            f"step{step}/logits",
                            atol=3e-6,
                            rtol=3e-5,
                        )
                    )
                    assert torch.isfinite(actual.loss)
                    actual.loss.backward()
                    expected_dx = runtime.input_for_rank(
                        embedding_outputs["reference"].grad[replica_slice] * args.f
                    )
                    comparisons.append(
                        compare_tensor(
                            embedding_outputs["runtime"].grad,
                            expected_dx,
                            f"step{step}/decoder_input_gradient",
                            atol=4e-6,
                            rtol=5e-5,
                        )
                    )
                    runtime.finalize_backward()
                    reported_loss = runtime.qwen_reported_loss()
                    comparisons.append(
                        compare_tensor(
                            reported_loss,
                            expected.loss.detach(),
                            f"step{step}/global_loss",
                            atol=3e-6,
                            rtol=3e-5,
                        )
                    )
                    reconstructed = runtime.reconstruct_tp_local(gradients=True)
                    for name, handle in runtime.registry.items():
                        expected_gradient = runtime.qwen_partitions[name].local_slice(
                            references[name].grad
                        )
                        comparisons.append(
                            compare_tensor(
                                reconstructed[name],
                                expected_gradient,
                                f"step{step}/{name}/gradient",
                                atol=4e-6,
                                rtol=5e-5,
                            )
                        )
                        valid = handle.spec.valid_len(context.coordinate.fsdp)
                        shard_length = handle.spec.fsdp_shard_numel
                        expected_shard = torch.zeros_like(handle.local_shard)
                        expected_shard[:valid].copy_(
                            expected_gradient.reshape(-1)[context.coordinate.fsdp * shard_length :][
                                :valid
                            ]
                        )
                        comparisons.append(
                            compare_tensor(
                                handle.local_shard.grad,
                                expected_shard,
                                f"step{step}/{name}/gradient_shard",
                                atol=4e-6,
                                rtol=5e-5,
                            )
                        )
                        assert torch.count_nonzero(handle.local_shard.grad[valid:]) == 0
                    optimizer.step()
                    expected_optimizer.step()
                    runtime.end_step()
                    assert all(unit.lease is None for unit in runtime.units)
                    for name, value in runtime.reconstruct_tp_local().items():
                        expected_weight = runtime.qwen_partitions[name].local_slice(
                            references[name]
                        )
                        comparisons.append(
                            compare_tensor(
                                value,
                                expected_weight,
                                f"step{step}/{name}/update",
                                atol=3e-6,
                                rtol=3e-5,
                            )
                        )
                    assert not runtime.comm.live_buffers
                    assert identities == {
                        name: (id(handle.local_shard), handle.local_shard.data_ptr())
                        for name, handle in runtime.registry.items()
                    }
                    step_rows.append(
                        {
                            "step": step,
                            "expected_global_loss": expected.loss.item(),
                            "reported_global_loss": reported_loss.item(),
                            "local_backward_loss": actual.loss.item(),
                            "valid_counts_by_replica": [
                                int(batch[1][:, 1:].ne(-100).sum()) for batch in all_batches
                            ],
                            "live_buffers": len(runtime.comm.live_buffers),
                        }
                    )
                    capture.step(step)
        finally:
            capture.close()
        result = {
            "status": "passed",
            "rank": session.worker.rank,
            "F": args.f,
            "T": args.t,
            "SP": args.sp,
            "layers": args.layers,
            "steps": args.steps,
            "policy": args.policy,
            "flags": args.flags,
            "backend": args.backend,
            "device": str(context.device),
            "actual_NCCL": (
                args.f * args.t > 1
                and torch.distributed.is_initialized()
                and str(torch.distributed.get_backend()) == "nccl"
            ),
            "transformers": __import__("transformers").__version__,
            "reference_attention_backend": "HF SDPA math (boolean causal/padding mask)",
            "torch": torch.__version__,
            "comparisons": comparisons,
            "step_metrics": step_rows,
            "leaf_identities_stable": True,
            "live_buffers": 0,
            "shared_parameter_owners": sum(
                name == "model.embed_tokens.weight" for name in runtime.registry
            ),
            "autograd_saved_full_parameters": False,
            "runtime_tied_parameter_alias": model.model.embed_tokens.weight is model.lm_head.weight,
            "logical_parameter_numel": sum(
                parameter.numel() for parameter in reference.parameters()
            ),
            "tp_local_parameter_numel": sum(
                handle.spec.numel for handle in runtime.registry.values()
            ),
            "persistent_shard_numel": sum(
                handle.local_shard.numel() for handle in runtime.registry.values()
            ),
            "persistent_shard_bytes": sum(
                handle.local_shard.numel() * handle.local_shard.element_size()
                for handle in runtime.registry.values()
            ),
            "parameter_grad_dtype": str(
                next(iter(runtime.registry.values())).local_shard.grad.dtype
            ),
        }
        Path(session.directory / f"qwen_correctness.rank{session.worker.rank}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
