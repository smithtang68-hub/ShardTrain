"""Independent mathematical and delayed-Work checks for the Norm/main RS branches."""

from types import SimpleNamespace

import pytest
import torch

from sharded_runtime.runtime.comm import CommunicationCoordinator
from sharded_runtime.runtime.layout import ParameterSpec
from sharded_runtime.runtime.materializer import Materializer


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("sp", [False, True])
def test_small_norm_sum_does_not_consume_pending_main_rs(monkeypatch, degree, rank, sp):
    rank %= degree
    grads = [
        torch.arange(1.0, 6.0),
        torch.arange(1.0, 13.0).reshape(4, 3),
        torch.arange(1.0, 4.0),
        torch.arange(1.0, 19.0).reshape(6, 3),
    ]
    layouts = ["replicated", "column", "column", "replicated"]
    specs = [
        ParameterSpec.create(str(i), 0, grad, grad.shape, layout, degree)
        for i, (grad, layout) in enumerate(zip(grads, layouts, strict=True))
    ]
    comm = CommunicationCoordinator("cpu", communication_policy="stream_ordered")
    works, sums = [], []

    class Work:
        def __init__(self, output, packed):
            self.output, self.packed, self.waited = output, packed, False

        def wait(self):
            if not self.waited:
                # Independent F-coordinate contributions are g and 3*g.
                start = rank * self.output.numel()
                self.output.copy_(self.packed[start : start + self.output.numel()] * 4)
                self.waited = True

    def reduce_scatter(output, packed, *, group, async_op):
        assert async_op
        work = Work(output, packed)
        works.append(work)
        return work

    def tp_sum(gradient, *, kind, group_role=None, channel="sp_rs"):
        assert kind == "tp_norm_allreduce"
        sums.append((group_role, channel))
        if group_role == "norm_gradient_bucket" and degree > 1:
            assert works[0].waited  # The small Norm RS is safe to read.
            assert not works[1].waited  # The main output is still deliberately unavailable.
        # Independent T-coordinate contributions are h and 5*h; this is SUM, not AVG.
        return gradient.clone() * 6

    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", reduce_scatter)
    runtime = SimpleNamespace(
        fsdp_grad_group=None,
        communication_policy="stream_ordered",
        async_grad_rs=True,
        sp=sp,
        tp=SimpleNamespace(degree=2, sum=tp_sum),
    )
    unit = SimpleNamespace(
        runtime=runtime,
        unit_id=0,
        handles=[SimpleNamespace(spec=spec) for spec in specs],
    )
    materializer = Materializer(None, comm)
    materializer.degree, materializer.rank = degree, rank
    materializer.backend = "nccl"  # Simulated delayed RS; no CUDA/NCCL installation needed.
    actual = materializer.reduce(unit, grads)

    for value, grad, spec in zip(actual, grads, specs, strict=True):
        # Oracle performs the F sum and mean and then the T sum explicitly.
        expected_full = grad.clone() if degree == 1 else torch.stack((grad, grad * 3)).sum(0) / 2
        if spec.tp_grad_reduce == "sum":
            if not sp:
                expected_full = expected_full / 2
            expected_full = torch.stack((expected_full, expected_full * 5)).sum(0)
        expected = torch.zeros(spec.fsdp_shard_numel)
        valid = spec.valid_len(rank)
        expected[:valid] = expected_full.flatten()[
            rank * spec.fsdp_shard_numel : rank * spec.fsdp_shard_numel + valid
        ]
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
    assert sums == [("norm_gradient_bucket", "compute"), (None, "sp_rs")]
    assert not comm.live_buffers
    assert all(work.waited for work in works)
    issued = [row["bucket"] for row in comm.events if row["event_type"] == "gradient_bucket_issued"]
    assert issued == ["norm", "main"]
    assert [ticket.group_role for ticket in comm.tickets] == (
        ["norm_gradient_bucket", "main_gradient_bucket"] if degree > 1 else []
    )


@pytest.mark.parametrize("layout", ["replicated", "column"])
def test_single_bucket_and_missing_gradient_validation(layout):
    gradient = torch.arange(1.0, 6.0)
    spec = ParameterSpec.create("p", 0, gradient, gradient.shape, layout, 1)
    comm = CommunicationCoordinator("cpu", communication_policy="stream_ordered")
    runtime = SimpleNamespace(
        communication_policy="stream_ordered",
        async_grad_rs=True,
        sp=True,
        tp=SimpleNamespace(degree=1, sum=lambda x, **kwargs: x.clone()),
    )
    unit = SimpleNamespace(runtime=runtime, unit_id=0, handles=[SimpleNamespace(spec=spec)])
    materializer = Materializer(None, comm)
    torch.testing.assert_close(materializer.reduce(unit, [gradient])[0], gradient)
    assert not comm.live_buffers
    with pytest.raises(RuntimeError, match="Missing full parameter gradient"):
        materializer.reduce(unit, [None])
