"""F=1 exercises bucket-copy readiness without any NCCL completion ticket."""

from types import SimpleNamespace

import pytest
import torch

from sharded_runtime.runtime.comm import CommunicationCoordinator, RuntimeTPGroup
from sharded_runtime.runtime.layout import ParameterSpec
from sharded_runtime.runtime.materializer import Materializer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("buckets", [("norm",), ("main",), ("norm", "main")])
def test_degree_one_delayed_producer_bucket_copy_oracle(buckets):
    comm = CommunicationCoordinator("cuda", communication_policy="stream_ordered")
    gradients = [
        torch.arange(1.0, 8193.0, device="cuda")
        if bucket == "norm"
        else torch.arange(1.0, 65537.0, device="cuda").reshape(256, 256)
        for bucket in buckets
    ]
    specs = [
        ParameterSpec.create(
            bucket, 0, gradient, gradient.shape, "replicated" if bucket == "norm" else "column", 1
        )
        for bucket, gradient in zip(buckets, gradients, strict=True)
    ]
    runtime = SimpleNamespace(
        fsdp_grad_group=None,
        communication_policy="stream_ordered",
        async_grad_rs=True,
        sp=False,
        tp=RuntimeTPGroup(None, comm),
    )
    unit = SimpleNamespace(
        runtime=runtime,
        unit_id=0,
        handles=[SimpleNamespace(spec=spec) for spec in specs],
    )
    materializer = Materializer(None, comm)
    # Warm the allocation/copy/scaling kernels before imposing a producer delay.
    for gradient in gradients:
        gradient.clone().div_(1)
    torch.cuda._sleep(1)
    torch.cuda.synchronize()
    rs_stream = comm.streams["fsdp_rs"]
    rs_stream.wait_stream(torch.cuda.current_stream())
    with comm.stream_context("fsdp_rs"):
        torch.cuda._sleep(100_000_000)
        actual = materializer.reduce(unit, gradients)
    torch.cuda.current_stream().wait_stream(rs_stream)
    for output, expected in zip(actual, gradients, strict=True):
        torch.testing.assert_close(output, expected.flatten(), atol=0, rtol=0)
    if "norm" in buckets:
        edges = [
            event
            for event in comm.events
            if event["event_type"] == "dependency_edge"
            and event.get("consumer") == "norm_gradient_bucket"
        ]
        assert edges
        assert all(edge["dependency_type"] == "DATA_DEP" for edge in edges)
        assert all(edge["wait_stream"] == "compute" for edge in edges)
    assert not comm.live_buffers
    assert not comm.tickets  # F=1, T=1; safety must come from the producer event.
