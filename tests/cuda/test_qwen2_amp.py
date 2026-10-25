"""Single-GPU AMP evidence; no V100 or multi-rank NCCL claim."""

import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig
from sharded_runtime.observability.profiling import ProfileCapture
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.training.qwen2_trainer import QwenTrainer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
CAPTURE_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_pid{os.getpid()}"


def tiny_config():
    return Qwen2ModelConfig(
        vocab_size=47,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        pad_token_id=0,
        use_cache=False,
    )


def artifact_dir(tmp_path, name):
    configured = os.environ.get("SHARDTRAIN_QWEN_CUDA_ARTIFACT_DIR")
    target = Path(configured) / CAPTURE_ID / name if configured else tmp_path / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def max_error(actual, expected):
    return float((actual.detach().float() - expected.detach().float()).abs().max().item())


def assert_master_state(trainer):
    parameters = list(trainer.runtime.sharded_parameters())
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert trainer.model.lm_head.weight is trainer.model.model.embed_tokens.weight
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    assert all(
        parameter.grad is not None and parameter.grad.dtype == torch.float32
        for parameter in parameters
    )
    for state in trainer.optimizer.state.values():
        assert all(
            not isinstance(value, torch.Tensor) or value.dtype == torch.float32
            for value in state.values()
        )
    assert not trainer.runtime.comm.live_buffers
    assert all(unit.lease is None for unit in trainer.runtime.units)


@pytest.fixture
def hf_model_class():
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Install pinned qwen optional dependencies")
    if version != "4.57.3":
        pytest.skip(f"HF numerical reference pinned to 4.57.3, installed {version}")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    return Qwen2Config, Qwen2ForCausalLM


@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
def test_single_gpu_fp16_amp_hf_three_updates_and_profile(hf_model_class, optimizer_name, tmp_path):
    torch.manual_seed(313)
    config = tiny_config()
    hf_config_class, hf_class = hf_model_class
    hf_config = hf_config_class(**config.to_dict())
    hf_config._attn_implementation = "sdpa"
    reference = hf_class(hf_config).cuda()
    actual = Qwen2CausalLM(config)
    actual.load_state_dict(reference.cpu().state_dict(), strict=True)
    reference.cuda()
    runtime = ShardedRuntime(
        device="cuda",
        reshard_after_forward=True,
        forward_prefetch=True,
        backward_prefetch=True,
        async_grad_rs=True,
        communication_policy="stream_ordered",
    ).register(actual)
    trainer = QwenTrainer(actual, runtime, lr=0.001, dtype="fp16", optimizer=optimizer_name)
    optimizer_class = {"sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}[optimizer_name]
    reference_optimizer = optimizer_class(reference.parameters(), lr=0.001, foreach=False)
    reference_scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    tokens = torch.tensor(
        [[1, 3, 5, 7, 11, 13, 0, 0], [0, 0, 17, 19, 23, 29, 31, 37]], device="cuda"
    )
    mask = tokens.ne(0).long()
    labels = tokens.masked_fill(mask.eq(0), -100)
    labels[1, 5] = -100
    positions = torch.stack((torch.arange(8) + 7, torch.arange(8) + 23)).cuda()
    captured = {}
    hook = actual.register_forward_hook(
        lambda _module, _args, output: captured.update(logits=output.logits.detach().clone())
    )
    directory = artifact_dir(tmp_path, optimizer_name)
    capture_mode = os.environ.get("SHARDTRAIN_QWEN_PROFILE_MODE", "torch")
    profiling = ProfileCapture(
        directory, 0, "cuda", mode=capture_mode if optimizer_name == "sgd" else "off"
    )
    torch.cuda.reset_peak_memory_stats()
    profiling.start()
    metrics = []
    try:
        for step in range(3):
            reference_optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    expected = reference(
                        tokens,
                        attention_mask=mask,
                        position_ids=positions,
                        labels=labels,
                        use_cache=False,
                    )
            reference_scaler.scale(expected.loss).backward()
            reference_scaler.unscale_(reference_optimizer)
            result = trainer.train_step(tokens, labels, attention_mask=mask, position_ids=positions)
            torch.cuda.synchronize()
            assert not result.overflow
            assert_master_state(trainer)
            torch.testing.assert_close(captured["logits"], expected.logits, atol=3e-3, rtol=3e-3)
            assert abs(result.loss - expected.loss.item()) < 3e-3
            ref_parameters = dict(reference.named_parameters())
            gradient_max = 0.0
            for name, handle in runtime.registry.items():
                actual_gradient = handle.local_shard.grad.view(handle.spec.tp_local_shape)
                gradient_max = max(
                    gradient_max, max_error(actual_gradient, ref_parameters[name].grad)
                )
                torch.testing.assert_close(
                    actual_gradient, ref_parameters[name].grad, atol=3e-3, rtol=3e-3
                )
            reference_scaler.step(reference_optimizer)
            reference_scaler.update()
            parameter_max = 0.0
            for name, value in runtime.reconstruct_tp_local().items():
                parameter_max = max(parameter_max, max_error(value, ref_parameters[name]))
                torch.testing.assert_close(
                    value,
                    ref_parameters[name],
                    atol=3e-3,
                    rtol=3e-3,
                    msg=lambda error, name=name: f"{name}: {error}",
                )
            metrics.append(
                dict(
                    step=step,
                    logits_max_abs=max_error(captured["logits"], expected.logits),
                    loss_abs=abs(result.loss - expected.loss.item()),
                    gradient_max_abs=gradient_max,
                    parameter_max_abs=parameter_max,
                    loss_scale=result.loss_scale,
                    valid_tokens=result.valid_tokens,
                    overflow=result.overflow,
                    live_buffers=len(runtime.comm.live_buffers),
                )
            )
            profiling.step(step)
            del expected
    finally:
        hook.remove()
        profiling.close()
    evidence = dict(
        torch=torch.__version__,
        transformers="4.57.3",
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        ranks=1,
        actual_NCCL=False,
        V100_validated=False,
        optimizer=optimizer_name,
        model_config=config.to_dict(),
        seed=313,
        batch_size=2,
        seq_length=8,
        profiling_scope="three-step HF comparison; both reference and runtime live on GPU",
        profiling_mode=capture_mode if optimizer_name == "sgd" else "off",
        dtype="FP16 AMP with FP32 master shards/gradients/state",
        attention_backend="SDPA math, explicit KV repeat",
        atol=3e-3,
        rtol=3e-3,
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        steps=metrics,
    )
    (directory / "metrics.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    (directory / "runtime_events.rank0.json").write_text(
        json.dumps(runtime.comm.events, indent=2) + "\n", encoding="utf-8"
    )
    print("QWEN_CUDA_METRICS=" + json.dumps(evidence, sort_keys=True))


def test_amp_overflow_skips_all_parameters_and_recovers(tmp_path):
    torch.manual_seed(409)
    model = Qwen2CausalLM(tiny_config())
    runtime = ShardedRuntime(device="cuda", reshard_after_forward=True).register(model)
    trainer = QwenTrainer(model, runtime, dtype="fp16", optimizer="adamw")
    trainer.scaler.set_growth_interval(2)
    tokens = torch.tensor([[1, 3, 5, 7, 11, 13]], device="cuda")
    first_clean = trainer.train_step(tokens, tokens)
    first_clean_tracker = trainer.scaler.state_dict()["_growth_tracker"]
    assert not first_clean.overflow and first_clean_tracker == 1
    assert_master_state(trainer)
    before = [parameter.detach().clone() for parameter in runtime.sharded_parameters()]
    before_optimizer_state = {
        parameter: {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }
        for parameter, state in trainer.optimizer.state.items()
    }
    initial_scale = trainer.scaler.get_scale()
    first = runtime.sharded_parameters()[0]

    def overflow(gradient):
        result = gradient.clone()
        result[0] = float("inf")  # A valid element: F=1 has no padded tail here.
        return result

    hook = first.register_hook(overflow)
    try:
        failed = trainer.train_step(tokens, tokens)
    finally:
        hook.remove()
    torch.cuda.synchronize()
    overflow_tracker = trainer.scaler.state_dict()["_growth_tracker"]
    parameters_unchanged = all(
        torch.equal(parameter, previous)
        for parameter, previous in zip(runtime.sharded_parameters(), before, strict=True)
    )
    optimizer_state_unchanged = len(trainer.optimizer.state) == len(before_optimizer_state) and all(
        state.keys() == before_optimizer_state[parameter].keys()
        and all(
            torch.equal(value, before_optimizer_state[parameter][key])
            if isinstance(value, torch.Tensor)
            else value == before_optimizer_state[parameter][key]
            for key, value in state.items()
        )
        for parameter, state in trainer.optimizer.state.items()
    )
    directory = artifact_dir(tmp_path, "overflow")
    observation = dict(
        gpu=torch.cuda.get_device_name(),
        ranks=1,
        growth_interval=trainer.scaler.get_growth_interval(),
        first_clean_loss_scale=first_clean.loss_scale,
        first_clean_growth_tracker=first_clean_tracker,
        overflow_loss_scale=failed.loss_scale,
        overflow_growth_tracker=overflow_tracker,
        expected_overflow_growth_tracker=0,
        all_parameters_unchanged_after_overflow=parameters_unchanged,
        all_optimizer_states_unchanged_after_overflow=optimizer_state_unchanged,
    )
    (directory / "overflow_observation.json").write_text(
        json.dumps(observation, indent=2) + "\n", encoding="utf-8"
    )
    assert failed.overflow and failed.loss_scale == initial_scale / 2
    assert overflow_tracker == 0
    assert parameters_unchanged
    assert optimizer_state_unchanged
    assert not runtime.comm.live_buffers
    recovered = trainer.train_step(tokens, tokens)
    torch.cuda.synchronize()
    assert not recovered.overflow
    recovery_tracker = trainer.scaler.state_dict()["_growth_tracker"]
    assert recovered.loss_scale == failed.loss_scale and recovery_tracker == 1
    assert any(
        not torch.equal(parameter, previous)
        for parameter, previous in zip(runtime.sharded_parameters(), before, strict=True)
    )
    assert_master_state(trainer)
    evidence = dict(
        gpu=torch.cuda.get_device_name(),
        ranks=1,
        actual_NCCL=False,
        V100_validated=False,
        injected_valid_gradient_inf=True,
        all_parameters_unchanged_after_overflow=True,
        all_optimizer_states_unchanged_after_overflow=True,
        loss_scale_before=initial_scale,
        loss_scale_after=failed.loss_scale,
        growth_interval=trainer.scaler.get_growth_interval(),
        first_clean_growth_tracker=first_clean_tracker,
        overflow_growth_tracker=overflow_tracker,
        recovered_loss_scale=recovered.loss_scale,
        recovered_growth_tracker=recovery_tracker,
        clean_step_after_overflow_does_not_grow_scale_early=True,
        subsequent_clean_step_updates=True,
        live_buffers=0,
    )
    (directory / "metrics.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
