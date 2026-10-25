"""CPU structural/negative checks for the same-start distributed AMP harness.

These tests exercise the acceptance machinery; they do not validate CUDA AMP.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig
from sharded_runtime.parallel.tp_linear import ParameterPartition
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.runtime.layout import ParameterSpec
from sharded_runtime.training.qwen2_trainer import QwenTrainer

WORKER = Path(__file__).resolve().parents[1] / "distributed/qwen2_amp_worker.py"
spec = importlib.util.spec_from_file_location("amp_oracle_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.mark.parametrize("axis,shape", [(0, (6, 3)), (1, (3, 6)), (None, (3, 3))])
def test_f_t_reconstruction_and_padding(axis, shape):
    full = torch.arange(torch.tensor(shape).prod()).view(shape).float()
    parts = []
    for rank in range(2):
        partition = ParameterPartition(shape, axis, rank, 2)
        local = partition.local_slice(full).contiguous()
        layout = ParameterSpec.create("weight", 0, local, shape, "test", 2)
        # Odd numel deliberately exercises storage padding; padding is discarded.
        chunks = []
        for f in range(2):
            chunk = torch.full((layout.fsdp_shard_numel,), 999.0)
            count = layout.valid_len(f)
            start = f * layout.fsdp_shard_numel
            chunk[:count].copy_(local.flatten()[start : start + count])
            chunks.append(chunk)
        rebuilt = worker.join_fsdp_shards(layout, chunks)
        assert torch.equal(rebuilt, local)
        with pytest.raises(AssertionError, match="Wrong F shard count"):
            worker.join_fsdp_shards(layout, chunks[:1])
        parts.append(rebuilt)
    assert torch.equal(worker.join_tp_parts(partition, parts), full)
    with pytest.raises(AssertionError, match="Wrong T part count"):
        worker.join_tp_parts(partition, parts[:1])
    if axis is None:
        parts[1] = parts[1].clone()
        parts[1].flatten()[0] += 1
        with pytest.raises(AssertionError, match="replicated parameter/state differs"):
            worker.join_tp_parts(partition, parts)
    else:
        wrong_axis = ParameterPartition(shape, 1 - axis, 0, 2)
        with pytest.raises(AssertionError, match="logical shape"):
            worker.join_tp_parts(wrong_axis, parts)
        assert not torch.equal(worker.join_tp_parts(partition, parts[::-1]), full)


def tiny_config():
    return Qwen2ModelConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )


@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
def test_real_runtime_start_mapping_and_native_replay(optimizer):
    torch.manual_seed(913)
    model = Qwen2CausalLM(tiny_config())
    reference = copy.deepcopy(model)
    runtime = ShardedRuntime().register(model)
    trainer = QwenTrainer(model, runtime, optimizer=optimizer)
    ref_type = {"sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}[optimizer]
    reference_optimizer = ref_type(reference.parameters(), lr=0.001, foreach=False)
    worker.validate_reference_layout(runtime, reference)
    leaves = runtime.sharded_parameters()
    names = {handle.local_shard: name for name, handle in runtime.registry.items()}
    identities = {name: id(parameter) for name, parameter in reference.named_parameters()}
    ids = torch.tensor([[1, 2, 3, 4]])
    for step in range(2):
        weights, states = worker.reconstruct_start(runtime, trainer.optimizer)
        digest = worker.start_hash(weights, states)
        # Deliberately desynchronize both weight and state before syncing again.
        with torch.no_grad():
            reference.model.embed_tokens.weight.add_(10)
        worker.copy_reference_start(reference, reference_optimizer, weights, states)
        assert reference.lm_head.weight is reference.model.embed_tokens.weight
        assert identities == {
            name: id(parameter) for name, parameter in reference.named_parameters()
        }
        assert digest == worker.start_hash(
            dict(reference.named_parameters()),
            {
                name: reference_optimizer.state[parameter]
                for name, parameter in reference.named_parameters()
            },
        )
        worker.assert_update_count(reference_optimizer, step)
        before = {parameter: parameter.detach().clone() for parameter in leaves}
        state_before = worker.optimizer_snapshot(trainer.optimizer)
        trainer.train_step(ids)
        records = []
        worker.optimizer_replay(trainer.optimizer, before, state_before, names, records)
        assert len(records) == len(leaves)
        assert all(
            record["parameter_bit_exact"] and record["optimizer_state_bit_exact"]
            for record in records
        )
        worker.assert_update_count(trainer.optimizer, step + 1)
        assert trainer.step_id == step + 1  # Sync/replay never reset the runtime trajectory.


def hf_model():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.3":
        pytest.skip("Independent HF structure check pins transformers 4.57.3")
    return transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**tiny_config().to_dict()))


def test_independent_hf_layout_rejects_axis_and_shape_faults():
    reference = hf_model()
    model = Qwen2CausalLM(tiny_config())
    runtime = ShardedRuntime().register(model)
    worker.validate_reference_layout(runtime, reference)
    name = "model.layers.0.self_attn.k_proj.weight"
    original = runtime.qwen_partitions[name]
    runtime.qwen_partitions[name] = ParameterPartition(original.global_shape, 1, 0, 1)
    with pytest.raises(AssertionError, match="independent HF structure"):
        worker.validate_reference_layout(runtime, reference)
    runtime.qwen_partitions[name] = ParameterPartition((1, 1), 0, 0, 1)
    with pytest.raises(AssertionError, match="independent HF structure"):
        worker.validate_reference_layout(runtime, reference)


def test_initial_weights_checked_before_reference_sync_rejects_bad_init():
    torch.manual_seed(731)
    model = Qwen2CausalLM(tiny_config())
    original_reference = copy.deepcopy(model)
    runtime = ShardedRuntime().register(model)
    worker.assert_initial_weights(runtime, original_reference)
    with torch.no_grad():
        next(iter(runtime.registry.values())).local_shard[0] += 0.25
    with pytest.raises(AssertionError, match="Original HF initialization shard differs"):
        worker.assert_initial_weights(runtime, original_reference)


@pytest.mark.parametrize(
    "kind,key,bad_value",
    [
        ("sgd", "lr", 0.002),
        ("sgd", "weight_decay", 0.01),
        ("sgd", "momentum", 0.1),
        ("sgd", "maximize", True),
        ("adamw", "lr", 0.002),
        ("adamw", "weight_decay", 0.0),
        ("adamw", "betas", (0.8, 0.999)),
        ("adamw", "eps", 1e-6),
        ("adamw", "amsgrad", True),
        ("adamw", "maximize", True),
        ("adamw", "capturable", True),
    ],
)
def test_optimizer_contract_rejects_bad_options(kind, key, bad_value):
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer_type = {"sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}[kind]
    optimizer = optimizer_type([parameter], lr=0.001)
    worker.assert_optimizer_contract(optimizer, kind)
    optimizer.param_groups[0][key] = bad_value
    with pytest.raises(AssertionError, match="Optimizer option differs"):
        worker.assert_optimizer_contract(optimizer, kind)


def test_optimizer_contract_rejects_wrong_type():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(2))], lr=0.001)
    with pytest.raises(AssertionError, match="Optimizer type differs"):
        worker.assert_optimizer_contract(optimizer, "adamw")


@pytest.mark.parametrize("fault", ["missing_leaf", "missing_step", "empty_state"])
def test_adam_update_count_rejects_missing_states(fault):
    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    optimizer = torch.optim.AdamW(parameters, lr=0.001)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    worker.assert_update_count(optimizer, 0)
    optimizer.step()
    worker.assert_update_count(optimizer, 1)
    if fault == "missing_leaf":
        del optimizer.state[parameters[0]]
    elif fault == "missing_step":
        del optimizer.state[parameters[0]]["step"]
    else:
        optimizer.state.clear()
    with pytest.raises(AssertionError, match="Missing"):
        worker.assert_update_count(optimizer, 1)


def test_failed_profiling_still_saves_failed_result_once(tmp_path):
    class BrokenCapture:
        calls = 0

        def close(self):
            self.calls += 1
            raise RuntimeError("profile stop/metadata failure")

    capture = BrokenCapture()
    payload = dict(
        status="passed", errors=[], optimizer_replay=[{"leaves": [{"failed_leaf": True}]}]
    )
    with pytest.raises(RuntimeError, match="profile stop/metadata failure"):
        worker.finalize_result(tmp_path, 1, payload, capture)
    assert capture.calls == 1
    data = json.loads((tmp_path / "qwen_amp.rank1.json").read_text())
    assert data["status"] == "failed" and "profile stop/metadata failure" in data["errors"][0]
    assert data["optimizer_replay"][0]["leaves"][0]["failed_leaf"]


def test_same_start_independent_hf_forward_gradients_and_optimizer():
    torch.manual_seed(731)
    reference = hf_model()
    reference.config._attn_implementation = "sdpa"
    model = Qwen2CausalLM(tiny_config())
    model.load_state_dict(reference.state_dict())
    runtime = ShardedRuntime().register(model)
    trainer = QwenTrainer(model, runtime, optimizer="adamw")
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, foreach=False)
    ids = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 0]])
    labels = ids.clone()
    labels[1, 3] = -100
    mask = ids.ne(0)
    references = dict(reference.named_parameters())
    captured = {}
    hook = model.register_forward_hook(
        lambda _module, _args, output: captured.update(logits=output.logits.detach().clone())
    )
    comparisons = []
    try:
        for index in range(2):
            weights, states = worker.reconstruct_start(runtime, trainer.optimizer)
            worker.copy_reference_start(reference, optimizer, weights, states)
            optimizer.zero_grad(set_to_none=True)
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                expected = reference(input_ids=ids, labels=labels, attention_mask=mask)
            expected.loss.backward()
            result = trainer.train_step(ids, labels, attention_mask=mask)
            worker.compare(captured["logits"], expected.logits, "logits", comparisons)
            worker.compare(torch.tensor(result.loss), expected.loss, "loss", comparisons)
            for name, handle in runtime.registry.items():
                worker.compare(
                    handle.local_shard.grad,
                    worker.expected_shard(runtime, name, references[name].grad),
                    name + "/gradient",
                    comparisons,
                )
            optimizer.step()
            worker.assert_update_count(optimizer, index + 1)
            for name, handle in runtime.registry.items():
                worker.compare(
                    handle.local_shard,
                    worker.expected_shard(runtime, name, references[name]),
                    name + "/update",
                    comparisons,
                )
                for key, value in trainer.optimizer.state[handle.local_shard].items():
                    target = optimizer.state[references[name]][key]
                    if value.ndim:
                        target = worker.expected_shard(runtime, name, target)
                    worker.compare(value, target, name + "/" + key, comparisons)
    finally:
        hook.remove()
    assert trainer.step_id == 2 and all(record["passed"] for record in comparisons)


@pytest.mark.parametrize("fault", ["missing_key", "weight_shape", "moment_shape", "step_dtype"])
def test_reference_sync_rejects_bad_mapping(fault):
    reference = Qwen2CausalLM(tiny_config())
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, foreach=False)
    weights = {name: parameter.detach().clone() for name, parameter in reference.named_parameters()}
    states = {name: {} for name in weights}
    name = "model.embed_tokens.weight"
    expected_error = {
        "missing_key": "logical keys",
        "weight_shape": "weight shape/dtype/device",
        "moment_shape": "moment shape/dtype",
        "step_dtype": "step shape/dtype",
    }
    if fault == "missing_key":
        del weights[name]
    elif fault == "weight_shape":
        weights[name] = torch.zeros(1)
    elif fault == "moment_shape":
        states[name] = {"exp_avg": torch.zeros(1)}
    else:
        states[name] = {"step": torch.tensor(1, dtype=torch.int64)}
    with pytest.raises(AssertionError, match=expected_error[fault]):
        worker.copy_reference_start(reference, optimizer, weights, states)


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.AdamW])
@pytest.mark.parametrize(
    "fault", [None, "update", "gradient", "starting_state", "final_state", "options"]
)
def test_native_replay_rejects_parameter_gradient_state_and_options_faults(optimizer_type, fault):
    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.5, 1.5]))
    options = dict(lr=0.017, weight_decay=0.031, foreach=False)
    if optimizer_type is torch.optim.AdamW:
        options.update(betas=(0.8, 0.95), eps=0.0001)
        state_key = "exp_avg"
    else:
        options.update(momentum=0.7, dampening=0.1)
        state_key = "momentum_buffer"
    optimizer = optimizer_type([parameter], **options)
    parameter.grad = torch.tensor([0.3, -0.1, 0.2])
    optimizer.step()
    before = {parameter: parameter.detach().clone()}
    state_before = worker.optimizer_snapshot(optimizer)
    parameter.grad = torch.tensor([-0.2, 0.4, -0.3])
    optimizer.step()
    if fault == "update":
        with torch.no_grad():
            parameter[0] += 0.001
    elif fault == "gradient":
        parameter.grad[0] += 0.1
    elif fault == "starting_state":
        state_before[parameter][state_key][0] += 0.1
    elif fault == "final_state":
        optimizer.state[parameter][state_key][0] += 0.1
    elif fault == "options":
        optimizer.param_groups[0]["weight_decay"] += 0.1
    records = []
    if fault is None:
        worker.optimizer_replay(optimizer, before, state_before, {parameter: "leaf"}, records)
        assert records[0]["parameter_bit_exact"] and records[0]["optimizer_state_bit_exact"]
    else:
        with pytest.raises(AssertionError, match="bit_exact"):
            worker.optimizer_replay(optimizer, before, state_before, {parameter: "leaf"}, records)


def test_plan_is_cpu_only_and_names_scope(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Plan must not query CUDA"))
    worker.main(["--optimizer", "adamw", "--output-dir", "unused", "--plan-only"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["clean_steps"] == 3 and plan["continuous_training_steps"] == 5
    assert plan["successful_updates"] == 4 and plan["runtime_continuous_no_reset"]
    assert plan["atol"] == plan["rtol"] == 3e-3
    assert plan["valid_counts_by_replica"] == [11, 9]
    assert "not independent rollout" in plan["oracle_scope"]


def test_scalar_update_count_rejects_drift():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    optimizer.state[parameter] = {
        "step": torch.tensor(2.0),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    }
    with pytest.raises(AssertionError):
        worker.assert_update_count(optimizer, 1)
