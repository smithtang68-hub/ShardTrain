"""Forward-effective GEMM precision and FP32 master-gradient regression checks.

CUDA cases run only on available, natively supported devices. CPU BF16 checks
do not claim Tensor Core, NCCL or V100 performance validation.
"""

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.runtime.comm import CommunicationCoordinator, RuntimeTPGroup
from sharded_runtime.runtime.qwen2_operators import QwenProjection, QwenRowLinear


class BackwardPrecision(TorchDispatchMode):
    """Observe actual dispatched GEMM operands, rather than returned grad dtype."""

    def __init__(self):
        super().__init__()
        self.gemms, self.sums = [], []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
            self.gemms.append((args[0].dtype, args[1].dtype))
        if func == torch.ops.aten.sum.dim_IntList:
            self.sums.append(args[0].dtype)
        return func(*args, **(kwargs or {}))


CASES = [
    pytest.param("cpu", torch.float32, False, id="cpu-fp32"),
    pytest.param("cpu", torch.bfloat16, True, id="cpu-bf16-autocast"),
    pytest.param("cuda", torch.float32, False, id="cuda-fp32"),
    pytest.param("cuda", torch.float16, True, id="cuda-fp16-autocast"),
    pytest.param("cuda", torch.bfloat16, True, id="cuda-bf16-autocast"),
]


def require_device(device, dtype):
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
            pytest.skip("Native CUDA BF16 unavailable; do not enable V100 BF16 emulation")


@contextmanager
def operator_compute(group):
    """Order fixture inputs and gradient checks across the caller and compute streams."""
    comm = group.comm
    if not comm.streams:
        with comm.stream_context("compute"):
            yield
        return
    caller = torch.cuda.current_stream(comm.device)
    compute = comm.streams["compute"]
    compute.wait_stream(caller)
    try:
        with comm.stream_context("compute"):
            yield
    finally:
        caller.wait_stream(compute)


def fixture_operators(device, widths):
    group = RuntimeTPGroup(None, CommunicationCoordinator(device))
    unit = SimpleNamespace(annotate=lambda phase: None, prefetch_after_input=lambda *args: None)
    weights = [torch.randn(width, 16, device=device).mul_(0.1).requires_grad_() for width in widths]
    providers, biases = [], []
    for weight in weights:
        bias = torch.randn(weight.shape[0], device=device).mul_(0.05).requires_grad_()
        handle = SimpleNamespace(unit=unit, weight=lambda weight=weight: weight)
        providers.append(
            SimpleNamespace(weight_slot=SimpleNamespace(handle=handle), bias_slot=True)
        )
        biases.append(bias)
    return group, weights, biases, providers


@pytest.mark.parametrize("device,compute_dtype,amp", CASES)
@pytest.mark.parametrize("operator", ["qkv", "row", "replicated_head"])
def test_backward_gemms_use_forward_dtype_and_match_dense_autograd(
    device, compute_dtype, amp, operator
):
    require_device(device, compute_dtype)
    torch.manual_seed(703)
    widths = (16, 8, 8) if operator == "qkv" else (24,)
    group, weights, biases, providers = fixture_operators(device, widths)
    x = torch.randn(2, 5, 16, device=device).mul_(0.2).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    reference_weights = [w.detach().clone().requires_grad_() for w in weights]
    reference_biases = [b.detach().clone().requires_grad_() for b in biases]
    with operator_compute(group):
        with torch.autocast(device, dtype=compute_dtype, enabled=amp, cache_enabled=False):
            if operator == "qkv":
                flat = tuple(value for pair in zip(weights, biases, strict=True) for value in pair)
                actual = QwenProjection.apply(x, group, providers, True, *flat)
                expected = tuple(
                    F.linear(reference_x, w, b)
                    for w, b in zip(reference_weights, reference_biases, strict=True)
                )
            else:
                actual = (
                    QwenRowLinear.apply(
                        x,
                        weights[0],
                        providers[0].weight_slot,
                        group,
                        True,
                        operator == "replicated_head",
                    ),
                )
                expected = (F.linear(reference_x, reference_weights[0]),)
        assert all(y.dtype == compute_dtype for y in actual)
        assert all(y.grad_fn.compute_dtype == compute_dtype for y in actual)
        upstream = [torch.randn_like(y).mul_(0.1) for y in actual]
        loss = sum((y.float() * g.float()).sum() for y, g in zip(actual, upstream, strict=True))
        reference_loss = sum(
            (y.float() * g.float()).sum() for y, g in zip(expected, upstream, strict=True)
        )
        # An unrelated ambient backward autocast must not choose the GEMM dtype.
        alternate = torch.bfloat16 if compute_dtype != torch.bfloat16 else torch.float16
        with torch.autocast(device, dtype=alternate, enabled=True), BackwardPrecision() as observed:
            loss.backward()
        reference_loss.backward()
        assert observed.gemms == [(compute_dtype, compute_dtype)] * (2 * len(weights))
        if operator == "qkv":
            assert observed.sums == [torch.float32] * len(weights)
        atol, rtol = (2e-5, 2e-5) if compute_dtype == torch.float32 else (3e-3, 3e-3)
        for value, target in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, target, atol=atol, rtol=rtol)
        torch.testing.assert_close(x.grad, reference_x.grad, atol=atol, rtol=rtol)
        assert x.grad.dtype == x.dtype == torch.float32
        for weight, reference in zip(weights, reference_weights, strict=True):
            assert weight.dtype == weight.grad.dtype == torch.float32
            torch.testing.assert_close(weight.grad, reference.grad, atol=atol, rtol=rtol)
        if operator == "qkv":
            for bias, reference in zip(biases, reference_biases, strict=True):
                assert bias.dtype == bias.grad.dtype == torch.float32
                torch.testing.assert_close(bias.grad, reference.grad, atol=atol, rtol=rtol)
    assert not group.comm.live_buffers


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_explicit_low_precision_without_autocast(dtype):
    require_device("cuda", dtype)
    torch.manual_seed(705)
    group, weights, _, providers = fixture_operators("cuda", (24,))
    x = torch.randn(2, 5, 16, device="cuda", dtype=dtype, requires_grad=True)
    with operator_compute(group):
        output = QwenRowLinear.apply(x, weights[0], providers[0].weight_slot, group, False, True)
        loss = output.float().square().mean()
        with BackwardPrecision() as observed:
            loss.backward()
    assert observed.gemms == [(dtype, dtype)] * 2
    assert x.grad.dtype == dtype and weights[0].grad.dtype == torch.float32
    assert torch.isfinite(x.grad).all() and torch.isfinite(weights[0].grad).all()


def test_cpu_bf16_full_runtime_gradients_and_master_updates():
    torch.manual_seed(709)
    cfg = Qwen2ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    model = Qwen2CausalLM(cfg)
    reference = copy.deepcopy(model)
    runtime = ShardedRuntime(
        reshard_after_forward=True,
        forward_prefetch=True,
        backward_prefetch=True,
        async_grad_rs=True,
        communication_policy="stream_ordered",
    ).register(model)
    optimizer = runtime.bind_optimizer(torch.optim.SGD(runtime.sharded_parameters(), lr=0.003))
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.003)
    leaves = runtime.sharded_parameters()
    identities = [id(p) for p in leaves]
    tokens = torch.tensor([[1, 3, 5, 7, 11, 13], [2, 4, 6, 8, 10, 12]])
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        runtime.begin_step(step)
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
            actual, expected = model(tokens, labels=tokens), reference(tokens, labels=tokens)
        torch.testing.assert_close(actual.logits, expected.logits, atol=3e-3, rtol=3e-3)
        torch.testing.assert_close(actual.loss, expected.loss, atol=3e-3, rtol=3e-3)
        actual.loss.backward()
        expected.loss.backward()
        runtime.finalize_backward()
        for name, handle in runtime.registry.items():
            assert handle.local_shard.dtype == handle.local_shard.grad.dtype == torch.float32
            assert torch.isfinite(handle.local_shard.grad).all()
            gradient = handle.local_shard.grad.view(handle.spec.tp_local_shape)
            torch.testing.assert_close(
                gradient, dict(reference.named_parameters())[name].grad, atol=3e-3, rtol=3e-3
            )
        optimizer.step()
        reference_optimizer.step()
        runtime.end_step()
        assert identities == [id(p) for p in runtime.sharded_parameters()]
        assert not runtime.comm.live_buffers and all(unit.lease is None for unit in runtime.units)
        for name, weight in runtime.reconstruct_tp_local().items():
            torch.testing.assert_close(
                weight, dict(reference.named_parameters())[name], atol=3e-3, rtol=3e-3
            )
