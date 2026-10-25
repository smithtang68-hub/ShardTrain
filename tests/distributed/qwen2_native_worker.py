"""Real native FSDP2/TP/SP versus independent HF dense loss/grad/update oracle.

Use torchrun --standalone --nproc-per-node=4 tests/distributed/qwen2_native_worker.py
  --f 2 --t 2 --sp --dtype fp16 --optimizer adamw --output-dir runs/native_amp
This gate is instrumented for verification, not performance measurement.
"""

import argparse
import json
import os
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sharded_runtime.config import (
    Config,
    DebugConfig,
    ModelConfig,
    ParallelConfig,
    RuntimeConfig,
    TrainingConfig,
)
from sharded_runtime.training.qwen2_native_baseline import (
    NativeQwenTrainer,
    build_native_qwen,
    local_tensor,
)


def token_batch(replica, accumulation):
    generator = torch.Generator().manual_seed(2718 + replica)
    ids = torch.randint(1, 64, (2 * accumulation, 8), generator=generator)
    mask = torch.ones_like(ids)
    mask[1::2, -(2 + replica) :] = 0
    labels = ids.masked_fill(mask.eq(0), -100)
    labels[0::2, 2 + replica] = -100
    positions = torch.arange(17, 25).expand(ids.shape[0], -1)
    return ids, labels, mask, positions


def reconstruct(model, *, gradient=False):
    values = {}
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradient else parameter
        assert isinstance(value, DTensor), f"Not a native DTensor: {name}"
        values[name] = value.full_tensor().detach().clone()
    return values


def assert_close(actual, expected, stage, *, dtype):
    atol, rtol = (
        (3e-3, 3e-3) if dtype == "fp16" else (4e-2, 3e-2) if dtype == "bf16" else (2e-5, 2e-4)
    )
    if actual.keys() != expected.keys():
        raise AssertionError(f"{stage} canonical parameter names differ")
    maximum = 0.0
    for name, value in actual.items():
        torch.testing.assert_close(
            value, expected[name], atol=atol, rtol=rtol, msg=f"{stage}/{name}"
        )
        maximum = max(maximum, float((value - expected[name]).abs().max()))
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f", type=int, default=2)
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--untied", action="store_true")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if int(os.environ["WORLD_SIZE"]) != args.f * args.t:
        parser.error("WORLD_SIZE must equal f*t")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    mesh = init_device_mesh("cuda", (args.f, args.t), mesh_dim_names=("fsdp", "tp"))
    coordinate = mesh.get_coordinate()
    config = Config(
        model=ModelConfig(
            model_type="qwen2",
            vocab_size=64,
            hidden_size=32,
            num_heads=4,
            num_key_value_heads=2,
            mlp_hidden_size=64,
            layers=2,
            seq_len=8,
            tie_word_embeddings=not args.untied,
        ),
        parallel=ParallelConfig(args.f, args.t, args.sp, "cuda", "nccl"),
        runtime=RuntimeConfig(
            reshard_after_forward=True,
            param_grouping="block",
            forward_prefetch=True,
            backward_prefetch=True,
            async_grad_rs=True,
            communication_policy="stream_ordered",
        ),
        training=TrainingConfig(
            dtype=args.dtype,
            optimizer=args.optimizer,
            lr=0.001,
            accumulation_steps=args.accumulation,
            seed=313,
        ),
        debug=DebugConfig(output_dir=args.output_dir, distributed_timeout_seconds=180),
    )
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"status": "running", "rank": rank, "args": vars(args), "errors": []}
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        native, policy = build_native_qwen(config, mesh, device)
        trainer = NativeQwenTrainer(native, mesh, config)
        assert (native.lm_head.weight is native.model.embed_tokens.weight) == (not args.untied)
        assert all(p.dtype == torch.float32 for p in native.parameters())
        starts = reconstruct(native)
        hfconfig = Qwen2Config(**native.config.to_dict())
        hfconfig._attn_implementation = "sdpa"
        hfconfig.use_cache = False
        reference = Qwen2ForCausalLM(hfconfig).to(device).train()
        missing, unexpected = reference.load_state_dict(starts, strict=False)
        if not args.untied:
            missing = [name for name in missing if name != "lm_head.weight"]
        assert not missing and not unexpected, (missing, unexpected)
        if not args.untied:
            assert reference.lm_head.weight is reference.model.embed_tokens.weight
        optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.SGD
        reference_optimizer = optimizer_class(reference.parameters(), lr=0.001, foreach=False)
        # Native DTensor shard reconstruction must exactly reproduce the source.
        assert_close(starts, dict(reference.named_parameters()), "initial", dtype="fp32")
        full_cpu = [
            torch.cat([token_batch(f, args.accumulation)[i] for f in range(args.f)])
            for i in range(4)
        ]
        full = [value.to(device) for value in full_cpu]
        local = [value.to(device) for value in token_batch(coordinate[0], args.accumulation)]
        with (
            torch.autocast(
                "cuda",
                dtype=trainer.compute_dtype,
                enabled=args.dtype != "fp32",
                cache_enabled=False,
            ),
            sdpa_kernel(SDPBackend.MATH),
        ):
            reference_logits = reference(
                input_ids=full[0], attention_mask=full[2], position_ids=full[3], use_cache=False
            ).logits
            ref_num = F.cross_entropy(
                reference_logits[:, :-1].float().reshape(-1, 64),
                full[1][:, 1:].reshape(-1),
                reduction="sum",
                ignore_index=-100,
            )
            ref_count = full[1][:, 1:].ne(-100).sum()
            reference_loss = ref_num / ref_count.clamp_min(1)
        (reference_loss * trainer.loss_scale).backward()
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.grad.mul_(1.0 / trainer.loss_scale)
        actual_loss, count, finite = trainer.backward_batch(
            local[0], local[1], attention_mask=local[2], position_ids=local[3]
        )
        assert finite and count == ref_count
        torch.testing.assert_close(actual_loss, reference_loss, atol=3e-3, rtol=3e-3)
        gradients = reconstruct(native, gradient=True)
        gradient_error = assert_close(
            gradients,
            {name: p.grad for name, p in reference.named_parameters()},
            "global_gradient",
            dtype=args.dtype,
        )
        trainer.optimizer_step(finite)
        reference_optimizer.step()
        updated = reconstruct(native)
        update_error = assert_close(
            updated, dict(reference.named_parameters()), "optimizer_update", dtype=args.dtype
        )
        identities = {name: id(p) for name, p in native.named_parameters()}
        # A single-rank overflow must propagate globally and leave every owner
        # and optimizer moment untouched. The finite decision is tested in the
        # normal backward path using an output hook, not by overriding the gate.
        if args.dtype == "fp16":
            before = {p: local_tensor(p).detach().clone() for p in native.parameters()}
            state_before = {
                p: {
                    key: local_tensor(value).detach().clone()
                    for key, value in state.items()
                    if isinstance(value, torch.Tensor)
                }
                for p, state in trainer.optimizer.state.items()
            }

            def inject(_module, _inputs, output):
                if rank == 0:
                    output.logits.register_hook(lambda grad: grad * float("inf"))

            handle = native.register_forward_hook(inject)
            scale = trainer.loss_scale
            overflow = trainer.train_step(
                local[0], local[1], attention_mask=local[2], position_ids=local[3]
            )
            handle.remove()
            assert overflow.overflow and trainer.loss_scale == scale * 0.5
            for parameter in native.parameters():
                assert torch.equal(local_tensor(parameter), before[parameter])
            for parameter, states in trainer.optimizer.state.items():
                for key, value in states.items():
                    if isinstance(value, torch.Tensor):
                        assert torch.equal(local_tensor(value), state_before[parameter][key])
            recovery = trainer.train_step(
                local[0], local[1], attention_mask=local[2], position_ids=local[3]
            )
            assert not recovery.overflow
            payload["overflow"] = asdict(overflow)
            payload["recovery"] = asdict(recovery)
        assert identities == {name: id(p) for name, p in native.named_parameters()}
        payload.update(
            status="passed",
            policy=policy,
            actual_loss=float(actual_loss),
            reference_loss=float(reference_loss),
            valid_tokens=int(count),
            gradient_max_abs_error=gradient_error,
            update_max_abs_error=update_error,
            unique_parameters=sum(value.numel() for value in starts.values()),
            tied_identity_preserved=not args.untied,
            global_gradient_oracle="independent HuggingFace Qwen2 dense full global batch",
            torch=torch.__version__,
            cuda_build=torch.version.cuda,
            gpu=torch.cuda.get_device_name(device),
        )
    except BaseException as exc:
        payload["status"] = "failed"
        payload["errors"].append(repr(exc))
        raise
    finally:
        (directory / f"native_oracle.rank{rank}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
