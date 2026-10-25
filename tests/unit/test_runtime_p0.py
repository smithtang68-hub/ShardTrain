import copy
import weakref

import pytest
import torch

from sharded_runtime.config import ModelConfig
from sharded_runtime.models.reference import build_reference_model
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.runtime.layout import ParameterSpec, copy_out, offsets, pack_gradients


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("sizes", [(1,), (1, 3, 17, 24)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_packed_rank_major_copy_out_padding_and_gradient_layout(degree, sizes, dtype):
    weights = [torch.arange(n, dtype=dtype) + j * 100 for j, n in enumerate(sizes)]
    specs = [
        ParameterSpec.create(str(j), 0, w, tuple(w.shape), "column", degree)
        for j, w in enumerate(weights)
    ]
    starts, length = offsets(specs)
    storage = torch.full((degree * length,), -999.0, dtype=dtype)
    for rank in range(degree):
        for spec, w, start in zip(specs, weights, starts, strict=True):
            valid = spec.valid_len(rank)
            storage[rank * length + start :][:valid].copy_(
                w[rank * spec.fsdp_shard_numel :][:valid]
            )
    weak = weakref.ref(storage)
    result = copy_out(storage, specs, degree)
    pointers = [x.untyped_storage().data_ptr() for x in result.values()]
    assert len(set(pointers)) == len(pointers)
    assert storage.data_ptr() not in pointers
    del storage
    assert weak() is None
    for spec, expected in zip(specs, weights, strict=True):
        torch.testing.assert_close(result[spec.param_id], expected, atol=0, rtol=0)
    packed = pack_gradients(weights, specs, degree)
    for rank in range(degree):
        for spec, expected, start in zip(specs, weights, starts, strict=True):
            valid = spec.valid_len(rank)
            slot = packed[rank * length + start :][: spec.fsdp_shard_numel]
            torch.testing.assert_close(
                slot[:valid], expected[rank * spec.fsdp_shard_numel :][:valid]
            )
            assert torch.count_nonzero(slot[valid:]) == 0


@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("requires_input_grad", [False, True])
@pytest.mark.parametrize("reshard", [False, True])
def test_p0_multi_output_bridge_leaf_identity_saved_weights_and_three_steps(
    layers, requires_input_grad, reshard
):
    config = ModelConfig(layers=layers, hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    dense = build_reference_model(config)
    model = copy.deepcopy(dense)
    runtime = ShardedRuntime(reshard_after_forward=reshard).register(model)
    optimizer = torch.optim.SGD(runtime.sharded_parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(dense.parameters(), lr=0.01)
    full_storage_refs, saved_pointers = {}, []

    def snapshot(unit):
        def hook(module, args):
            full_storage_refs[unit.unit_id] = [
                weakref.ref(lease.tensor.untyped_storage())
                for lease in unit.lease.full_param_storages.values()
            ]

        return hook

    hooks = [unit.block.register_forward_pre_hook(snapshot(unit)) for unit in runtime.units]
    for step in range(3):
        runtime.begin_step(step)
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        a = torch.randn(2, 4, 8, requires_grad=requires_input_grad)
        b = a.detach().clone().requires_grad_(requires_input_grad)
        target = torch.randn_like(a)

        def save(tensor):
            saved_pointers.append(tensor.untyped_storage().data_ptr())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(save, lambda x: x):
            actual = model(a)
            if reshard and layers == 2:
                assert all(ref() is None for ref in full_storage_refs[0])
            assert runtime.units[-1].state == "FWD_RETAINED"
            expected = dense(b)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            runtime.local_loss(actual, target).backward()
            (expected - target).square().mean().backward()
        runtime.finalize_backward()
        if requires_input_grad:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5)
        for name, handle in runtime.registry.items():
            torch.testing.assert_close(
                handle.local_shard.grad.view(handle.spec.tp_local_shape),
                dict(dense.named_parameters())[name].grad,
                atol=3e-6,
                rtol=3e-5,
            )
            assert handle.local_shard.is_leaf
        optimizer.step()
        reference_optimizer.step()
        runtime.end_step()
        for name, value in runtime.reconstruct_tp_local().items():
            torch.testing.assert_close(
                value, dict(dense.named_parameters())[name], atol=3e-6, rtol=3e-5
            )
        assert all(unit.reductions == 1 for unit in runtime.units)
        assert not runtime.comm.live_buffers
    for hook in hooks:
        hook.remove()


def test_boundaries_fail_fast_missing_gradient_repeat_forward_and_stale_lease():
    config = ModelConfig(hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    model = build_reference_model(config)
    runtime = ShardedRuntime().register(model)
    with pytest.raises(RuntimeError, match="Premature"):
        runtime.end_step()
    with pytest.raises(RuntimeError, match="outside"):
        model(torch.randn(2, 4, 8))
    runtime.begin_step(0)
    output = model(torch.randn(2, 4, 8))
    with pytest.raises(RuntimeError, match="Missing gradient readiness"):
        runtime.finalize_backward()
    with pytest.raises(RuntimeError, match="Stale/early"):
        runtime.units[0].handles[0].weight()
    with pytest.raises(RuntimeError, match="repeated"):
        model(torch.randn(2, 4, 8))
    output.square().mean().backward()
    runtime.finalize_backward()
    runtime.end_step()
