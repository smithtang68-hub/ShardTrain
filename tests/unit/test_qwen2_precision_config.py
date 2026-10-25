"""Compute precision and common MFU accounting must not silently change policy."""

from contextlib import nullcontext

import pytest
import torch

from sharded_runtime.config import load_config
from sharded_runtime.runtime.comm import CommunicationCoordinator
from sharded_runtime.training.performance import qwen_performance_summary
from sharded_runtime.training.qwen2_precision import resolve_qwen_compute_dtype


def test_qwen_bf16_configuration_and_native_cuda_guard(monkeypatch):
    config = load_config("configs/qwen25_tiny.yaml", ["--training.dtype", "bf16"])
    config.validate(1, mode="runtime")
    assert resolve_qwen_compute_dtype("bf16", torch.device("cpu")) == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    checks = []

    def supported(*, including_emulation):
        checks.append(including_emulation)
        return False

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)
    with pytest.raises(ValueError, match="V100.*native BF16"):
        resolve_qwen_compute_dtype("bf16", torch.device("cuda:0"))
    assert checks == [False]
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kw: True)
    assert resolve_qwen_compute_dtype("bf16", torch.device("cuda:0")) == torch.bfloat16


def test_perf_model_flops_use_input_tokens_and_max_rank_step():
    result = qwen_performance_summary(
        parameter_numel=1_543_714_304,
        layers=28,
        hidden_size=1536,
        sequence_length=128,
        global_batch=2,
        steps=[
            {"max_rank_seconds": 3.530579652000597, "valid_tokens": 254},
            {"max_rank_seconds": 3.5151275750013156, "valid_tokens": 254},
        ],
        world_size=4,
        peak_tflops_per_gpu=125,
    )
    assert result["global_input_tokens_per_step"] == 256
    assert result["estimated_model_flops_per_step"] == 2_388_056_604_672
    assert result["global_valid_tokens_per_second"] == pytest.approx(72.10063995465846)
    assert result["mfu_percent"] == pytest.approx(0.13557512554708664)
    assert result["unprofiled_timing"]


def test_no_nominal_peak_does_not_invent_mfu():
    result = qwen_performance_summary(
        parameter_numel=100,
        layers=1,
        hidden_size=8,
        sequence_length=4,
        global_batch=1,
        steps=[],
        world_size=1,
        profile_mode="torch",
    )
    assert result["mfu_percent"] is None and result["mean_max_rank_seconds"] is None
    assert not result["unprofiled_timing"]


def test_perf_diagnostics_disable_observation_without_disabling_dependency(tmp_path):
    comm = CommunicationCoordinator("cpu", directory=tmp_path, diagnostics_enabled=False)
    comm.event("a", purpose="performance")
    comm.dependency("DATA_DEP", producer_ticket=0)
    assert comm.events == []
    assert not (tmp_path / "runtime_events.rank0.jsonl").exists()

    class Stream:
        waited = None

        def wait_event(self, event):
            self.waited = event

    stream, ready = Stream(), object()
    comm.wait_event(ready, dependency_type="PG_ORDER", stream=stream)
    assert stream.waited is ready
    assert comm.gpu_waits == []


def test_cpu_bf16_trainer_keeps_fp32_masters(tmp_path):
    from sharded_runtime.training.qwen2_benchmark import main

    assert (
        main(
            [
                "--config",
                "configs/qwen25_tiny.yaml",
                "--training.dtype",
                "bf16",
                "--training.steps",
                "2",
                "--warmup",
                "0",
                "--model.layers",
                "1",
                "--debug.output_dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    import json

    data = json.loads((tmp_path / "benchmark.rank0.json").read_text(encoding="utf-8"))
    assert data["precision_policy"]["compute"] == "bf16"
    assert data["precision_policy"]["linear_backward"] == "bf16"
    assert data["ownership"]["parameter_dtypes"] == ["torch.float32"]
    assert data["optimizer_updates"] == 2 and data["overflow_steps"] == 0
    assert not data["runtime_events_enabled"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
def test_cuda_observation_toggle_preserves_gradients_and_updates(dtype):
    from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig
    from sharded_runtime.runtime import ShardedRuntime
    from sharded_runtime.training.qwen2_trainer import QwenTrainer

    if dtype == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("GPU has no native BF16")
    torch.manual_seed(773)
    config = Qwen2ModelConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    initial = Qwen2CausalLM(config).state_dict()
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], device="cuda")
    results = []
    for enabled in (True, False):
        model = Qwen2CausalLM(config)
        model.load_state_dict(initial)
        runtime = ShardedRuntime(
            device="cuda",
            forward_prefetch=True,
            backward_prefetch=True,
            async_grad_rs=True,
            communication_policy="stream_ordered",
            diagnostics_enabled=enabled,
        ).register(model)
        trainer = QwenTrainer(model, runtime, lr=1e-4, dtype=dtype, optimizer="adamw")
        rows = []
        for _ in range(2):
            result = trainer.train_step(ids)
            assert not result.overflow and not runtime.comm.live_buffers
            rows.append(
                (
                    result.loss,
                    [p.detach().cpu().clone() for p in runtime.sharded_parameters()],
                    [p.grad.detach().cpu().clone() for p in runtime.sharded_parameters()],
                )
            )
        if not enabled:
            assert not runtime.comm.events and not runtime.comm.pending_last_reads
        results.append(rows)
    for observed, quiet in zip(*results, strict=True):
        assert observed[0] == quiet[0]
        for left, right in zip(observed[1] + observed[2], quiet[1] + quiet[2], strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
