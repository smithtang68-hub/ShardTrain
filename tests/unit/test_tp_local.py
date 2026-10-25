import copy

import pytest
import torch
from torch import nn

from sharded_runtime.config import ConfigError, ModelConfig, config_from_dict
from sharded_runtime.models.reference import build_reference_model, synthetic_batch
from sharded_runtime.parallel.mappings import TPGroup, gather_sequence, reduce_scatter_sequence
from sharded_runtime.parallel.model import parallelize_tp_sp
from sharded_runtime.parallel.tp_linear import ColumnParallelLinear, RowParallelLinear
from sharded_runtime.training.trainer import Trainer


def test_tp_one_preserves_names_outputs_gradients_and_parameter_storage():
    config = ModelConfig(hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    reference = build_reference_model(config)
    model = copy.deepcopy(reference)
    plan = parallelize_tp_sp(model)
    assert set(plan.partitions) == set(dict(reference.named_parameters()))
    a = synthetic_batch(config, 2, 4321).requires_grad_()
    b = a.detach().clone().requires_grad_()
    expected, actual = reference(a), model(b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    plan.backward_loss(actual).backward()
    plan.finalize_gradients()
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    for name, gradient in plan.global_parameters(gradients=True).items():
        torch.testing.assert_close(
            gradient, dict(reference.named_parameters())[name].grad, rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="own TP/SP plan"):
        Trainer(model, 0.01)
    Trainer(model, 0.01, tp_plan=plan).train_step(b.detach())


def test_singleton_linear_and_mapping_backward():
    group = TPGroup()
    column = ColumnParallelLinear(nn.Linear(4, 6, bias=False), group)
    row = RowParallelLinear(nn.Linear(6, 4, bias=False), group)
    x = torch.randn(2, 4, 4, requires_grad=True)
    row(column(x)).sum().backward()
    assert x.grad.shape == x.shape
    x = torch.randn(2, 4, 4, requires_grad=True)
    y = reduce_scatter_sequence(gather_sequence(x, group), group)
    y.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x)


def test_unsupported_modules_shared_parameters_and_double_conversion():
    with pytest.raises(ValueError, match="only supports"):
        parallelize_tp_sp(nn.Linear(4, 4))
    config = ModelConfig(hidden_size=8, num_heads=2, mlp_hidden_size=12)
    model = build_reference_model(config)
    model[0].attention.k_proj.weight = model[0].attention.q_proj.weight
    with pytest.raises(ValueError, match="Shared parameters"):
        parallelize_tp_sp(model)
    model = build_reference_model(config)
    parallelize_tp_sp(model)
    with pytest.raises(ValueError, match="already"):
        parallelize_tp_sp(model)
    with pytest.raises(ValueError, match="SP requires"):
        parallelize_tp_sp(build_reference_model(config), sp=True)
    model = build_reference_model(config)
    model[0].mlp_norm = nn.Identity()
    with pytest.raises(ValueError, match="nine parameters|original Block"):
        parallelize_tp_sp(model)


def test_unsupported_linear_and_invalid_sequence():
    group = TPGroup()
    with pytest.raises(ValueError, match="bias-free"):
        ColumnParallelLinear(nn.Linear(4, 4), group)
    with pytest.raises(ValueError, match="input feature"):
        ColumnParallelLinear(nn.Linear(4, 4, bias=False), group)(torch.ones(2, 3))
    with pytest.raises(ValueError, match="divisible"):
        group.sequence_shard(torch.ones(2, 0, 4))
    with pytest.raises(ValueError, match="explicit ProcessGroup"):
        TPGroup(degree=2)


def test_m1_training_config_gates():
    for sp in (False, True):
        cfg = config_from_dict({"parallel": {"tp_degree": 2, "sp_enabled": sp}})
        cfg.validate(2, mode="train")
    with pytest.raises(ConfigError, match="M2"):
        config_from_dict({"parallel": {"fsdp_degree": 2}}).validate(2, mode="train")
    with pytest.raises(ConfigError, match="only tp_degree"):
        config_from_dict({"parallel": {"tp_degree": 4}}).validate(4, mode="train")
