"""Native HF TP2 linear-rounding diagnosis; never a formal AMP acceptance gate.

Run on exactly one GPU through the root agent, or use --device cpu for preflight.
No ShardTrain model, autograd Function, communicator or distributed runtime is
imported or executed. Fixture/config definitions are read from the exact worker
and model source using AST, without importing those modules.
"""

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/sharded_runtime").is_dir())
WORKER = ROOT / "tests/distributed/qwen2_amp_worker.py"
CONFIG_SOURCE = ROOT / "src/sharded_runtime/models/qwen2.py"
MODEL_NAMES = ("hf_dense_b4", "hf_tp2_b4", "hf_tp2_f2_b2")
PAIRS = (
    (MODEL_NAMES[1], MODEL_NAMES[0]),
    (MODEL_NAMES[2], MODEL_NAMES[0]),
    (MODEL_NAMES[2], MODEL_NAMES[1]),
)
BIAS_NAME = "model.layers.3.self_attn.k_proj.bias"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_contract():
    """Execute only the original pure token_batch function and literal constants."""
    tree = ast.parse(WORKER.read_text(encoding="utf8"))
    wanted = {"SEED", "CLEAN_STEPS", "ATOL", "RTOL"}
    selected = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)
    ]
    selected.extend(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "token_batch"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(WORKER), "exec"), namespace)
    assert {x: namespace[x] for x in wanted} == {
        "SEED": 313,
        "CLEAN_STEPS": 3,
        "ATOL": 3e-3,
        "RTOL": 3e-3,
    }
    model_tree = ast.parse(CONFIG_SOURCE.read_text(encoding="utf8"))
    cls = next(
        n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen2ModelConfig"
    )
    config = {
        n.target.id: ast.literal_eval(n.value)
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None
    }
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "Qwen2ModelConfig"
    ]
    assert len(calls) == 1
    for keyword in calls[0].keywords:
        if keyword.arg == "num_hidden_layers":
            assert ast.unparse(keyword.value) == "args.layers"
            config[keyword.arg] = 4
        else:
            config[keyword.arg] = ast.literal_eval(keyword.value)
    assert config["rope_theta"] == 1_000_000.0
    return config, namespace


def column_forward(self, x):
    """Separate native cast+linear branches, then concatenate output partitions."""
    assert self.out_features % 2 == 0
    width = self.out_features // 2
    outputs = []
    for rank in range(2):
        start, end = rank * width, (rank + 1) * width
        # Each .to() is a separate autograd cast when x is FP32. Sharing one
        # half cast would sum both branches' dx in half before promotion.
        source = x.to(torch.get_autocast_dtype(x.device.type))
        outputs.append(
            F.linear(
                source, self.weight[start:end], None if self.bias is None else self.bias[start:end]
            )
        )
    assert all(y.dtype == torch.float16 for y in outputs)
    self._tp_native_calls += 1
    self._tp_input_dtypes.add(str(x.dtype))
    return torch.cat(outputs, dim=-1)


def row_forward(self, x):
    """Native input-column GEMMs and an FP16 pairwise sum of the partials."""
    assert self.in_features % 2 == 0 and self.bias is None
    width = self.in_features // 2
    partials = [
        F.linear(
            x[..., rank * width : (rank + 1) * width],
            self.weight[:, rank * width : (rank + 1) * width],
        )
        for rank in range(2)
    ]
    assert all(y.dtype == torch.float16 for y in partials)
    result = partials[0] + partials[1]
    assert result.dtype == torch.float16
    self._tp_native_calls += 1
    self._tp_input_dtypes.add(str(x.dtype))
    return result


def patch_native_linears(model):
    patched = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.startswith("model.layers."):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}:
            module.forward = MethodType(column_forward, module)
            kind = "column"
        elif leaf in {"o_proj", "down_proj"}:
            module.forward = MethodType(row_forward, module)
            kind = "row"
        else:
            raise AssertionError(name)
        module._tp_native_kind = kind
        module._tp_native_calls = 0
        module._tp_input_dtypes = set()
        patched.append(name)
    assert len(patched) == 28
    return patched


def comparison(actual, expected, atol, rtol):
    a, b = actual.detach().cpu().float(), expected.detach().cpu().float()
    error = (a - b).abs()
    location = int(error.reshape(-1).argmax())
    return dict(
        shape=list(a.shape),
        max_abs_error=float(error.max()),
        max_rel_error=float((error / b.abs().clamp_min(1e-12)).max()),
        original_tolerance_met=bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        actual_nonfinite=int((~torch.isfinite(a)).sum()),
        expected_nonfinite=int((~torch.isfinite(b)).sum()),
        max_error_flat_index=location,
        actual_at_max=float(a.reshape(-1)[location]),
        expected_at_max=float(b.reshape(-1)[location]),
        near_adam_eps_count=int((b.abs() <= 1e-8).sum()),
    )


def cpu_values(mapping):
    return {name: value.detach().cpu().clone() for name, value in mapping.items()}


def array_values(mapping):
    return {
        name: dict(shape=list(value.shape), values=value.tolist())
        for name, value in mapping.items()
    }


def state_values(optimizer, parameters):
    return {
        name: {key: value.detach().cpu().clone() for key, value in optimizer.state[param].items()}
        for name, param in parameters.items()
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def model_digest(parameters):
    digest = hashlib.sha256()
    for name, value in sorted(parameters.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(args):
    if importlib.metadata.version("transformers") != "4.57.3":
        raise RuntimeError(
            "Requires transformers==4.57.3; this is an independent versioned HF oracle"
        )
    from transformers import Qwen2Config, Qwen2ForCausalLM

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a fresh output directory to preserve evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    config, fixture = fixture_contract()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    torch.manual_seed(fixture["SEED"])
    hfconfig = Qwen2Config(**config)
    hfconfig._attn_implementation = "sdpa"
    base = Qwen2ForCausalLM(hfconfig).train()
    models = {name: copy.deepcopy(base).to(device) for name in MODEL_NAMES}
    del base
    patched = {name: patch_native_linears(models[name]) for name in MODEL_NAMES[1:]}
    parameters = {name: dict(model.named_parameters()) for name, model in models.items()}
    digests = {name: model_digest(values) for name, values in parameters.items()}
    assert len(set(digests.values())) == 1
    optimizers = {
        name: torch.optim.AdamW(values.values(), lr=0.001, foreach=False)
        for name, values in parameters.items()
    }
    scalers = {
        name: torch.amp.GradScaler(device.type, init_scale=128.0, growth_interval=2)
        for name in MODEL_NAMES
    }
    contract = dict(
        scope="Native HF TP2 linear-rounding emulator; numerical diagnosis, not formal acceptance",
        formal_acceptance=False,
        actual_custom_runtime_executed=False,
        actual_distributed_collectives=False,
        actual_SP_execution=False,
        cuda_executed=device.type == "cuda",
        device=str(device),
        torch=torch.__version__,
        transformers="4.57.3",
        config=config,
        hf_class=type(models[MODEL_NAMES[0]]).__name__,
        hf_attention_backend="sdpa",
        torch_sdpa_backend="math",
        compute_dtype="fp16",
        master_dtype="fp32",
        seed=fixture["SEED"],
        steps=fixture["CLEAN_STEPS"],
        F=2,
        T=2,
        learning_rate=0.001,
        optimizer="AdamW default betas/eps/weight_decay; foreach=False",
        init_scale=128.0,
        growth_interval=2,
        atol=fixture["ATOL"],
        rtol=fixture["RTOL"],
        source_sha256={
            "diagnose.py": sha(Path(__file__)),
            "tests/distributed/qwen2_amp_worker.py": sha(WORKER),
            "src/sharded_runtime/models/qwen2.py": sha(CONFIG_SOURCE),
        },
        initial_parameter_sha256=digests,
        native_tp_patched_modules=patched,
        limitations=[
            "CPU preflight is not V100/CUDA numerical evidence",
            "HF attention remains full-head native HF, not per-TP-rank batched SDPA shapes",
            "HF loss remains native full-sequence mean; no SP-local CE slicing",
            "F2 branch scales native mean by local_count*F/global_count; FP32 gradient average",
            "Row output pair addition emulates half rounding, not NCCL transport or scheduling",
            "Models roll independently; after clean0 parameters/optimizer states can differ",
        ],
        steps_data=[],
        errors=[],
    )
    if device.type == "cuda":
        contract.update(
            gpu=torch.cuda.get_device_name(device),
            capability=list(torch.cuda.get_device_capability(device)),
            cuda=torch.version.cuda,
        )
    write_json(
        output_dir / "contract.json", {k: v for k, v in contract.items() if k != "steps_data"}
    )
    try:
        for step in range(fixture["CLEAN_STEPS"]):
            batches = [
                [x.to(device) for x in fixture["token_batch"](replica, step)]
                for replica in range(2)
            ]
            full = [torch.cat([batch[column] for batch in batches]) for column in range(4)]
            counts = [int(batch[1][:, 1:].ne(-100).sum()) for batch in batches]
            count = sum(counts)
            snapshots, all_logits = {}, {}
            for name in MODEL_NAMES:
                model, optimizer, scaler = models[name], optimizers[name], scalers[name]
                params = parameters[name]
                before = cpu_values(params)
                optimizer.zero_grad(set_to_none=True)
                is_f2 = name == MODEL_NAMES[2]
                selected = batches if is_f2 else [full]
                accumulated = {key: torch.zeros_like(value) for key, value in params.items()}
                bias_contributions, logits_parts, loss_total = [], [], 0.0
                scale_before = scaler.get_scale()
                for batch in selected:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, dtype=torch.float16, cache_enabled=False):
                        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                            result = model(
                                input_ids=batch[0],
                                labels=batch[1],
                                attention_mask=batch[2],
                                position_ids=batch[3],
                                use_cache=False,
                            )
                    local_count = int(batch[1][:, 1:].ne(-100).sum())
                    objective = result.loss * local_count * 2 / count if is_f2 else result.loss
                    loss_total += float(result.loss.detach()) * local_count / count
                    logits_parts.append(result.logits.detach().cpu())
                    scaler.scale(objective).backward()
                    bias_contributions.append(
                        {
                            key: value.grad.detach().cpu().tolist()
                            for key, value in params.items()
                            if key.endswith("k_proj.bias")
                        }
                    )
                    for key, value in params.items():
                        accumulated[key].add_(value.grad)
                for key, value in params.items():
                    value.grad = accumulated[key] / (2 if is_f2 else 1)
                scaler.unscale_(optimizer)
                gradients = cpu_values({key: value.grad for key, value in params.items()})
                finite = all(bool(torch.isfinite(g).all()) for g in gradients.values())
                if not finite:
                    raise RuntimeError(
                        f"Unexpected nonfinite diagnostic gradient: {name}/step{step}"
                    )
                scaler.step(optimizer)
                scaler.update()
                after = cpu_values(params)
                snapshots[name] = dict(
                    parameters_before=before,
                    gradients=gradients,
                    parameters_after=after,
                    parameter_delta={key: after[key] - before[key] for key in params},
                    optimizer_state=state_values(optimizer, params),
                    loss=loss_total,
                    scale_before=scale_before,
                    scale_after=scaler.get_scale(),
                    growth_tracker=scaler.state_dict()["_growth_tracker"],
                    replica_scaled_k_bias_gradients=bias_contributions,
                )
                all_logits[name] = torch.cat(logits_parts)
            pair_rows = []
            for actual_name, expected_name in PAIRS:
                actual, expected = snapshots[actual_name], snapshots[expected_name]
                records = {}
                for key in parameters[actual_name]:
                    record = {
                        field: comparison(
                            actual[field][key],
                            expected[field][key],
                            fixture["ATOL"],
                            fixture["RTOL"],
                        )
                        for field in [
                            "parameters_before",
                            "gradients",
                            "parameters_after",
                            "parameter_delta",
                        ]
                    }
                    record["optimizer_state"] = {
                        state_key: comparison(
                            value,
                            expected["optimizer_state"][key][state_key],
                            fixture["ATOL"],
                            fixture["RTOL"],
                        )
                        for state_key, value in actual["optimizer_state"][key].items()
                    }
                    records[key] = record
                pair_rows.append(
                    dict(
                        pair=[actual_name, expected_name],
                        logits=comparison(
                            all_logits[actual_name],
                            all_logits[expected_name],
                            fixture["ATOL"],
                            fixture["RTOL"],
                        ),
                        loss_abs_error=abs(actual["loss"] - expected["loss"]),
                        max_gradient_error=max(
                            x["gradients"]["max_abs_error"] for x in records.values()
                        ),
                        max_parameter_error=max(
                            x["parameters_after"]["max_abs_error"] for x in records.values()
                        ),
                        parameter_failures=[
                            key
                            for key, row in records.items()
                            if not row["parameters_after"]["original_tolerance_met"]
                        ],
                        key_bias=records[BIAS_NAME],
                        all_parameter_comparisons=records,
                    )
                )
            tensor_file = output_dir / f"tensors.step{step}.json"
            tensor_data = {}
            for name, data in snapshots.items():
                tensor_data[name] = {
                    field: array_values(data[field])
                    for field in [
                        "parameters_before",
                        "gradients",
                        "parameters_after",
                        "parameter_delta",
                    ]
                }
                tensor_data[name]["optimizer_state"] = {
                    key: array_values(value) for key, value in data["optimizer_state"].items()
                }
                tensor_data[name]["replica_scaled_k_bias_gradients"] = data[
                    "replica_scaled_k_bias_gradients"
                ]
            write_json(tensor_file, tensor_data)
            row = dict(
                step=step,
                valid_tokens=count,
                local_valid_counts=counts,
                per_model={
                    name: {
                        key: data[key]
                        for key in ["loss", "scale_before", "scale_after", "growth_tracker"]
                    }
                    for name, data in snapshots.items()
                },
                tensor_file=str(tensor_file),
                tensor_sha256=sha(tensor_file),
                pairs=pair_rows,
            )
            contract["steps_data"].append(row)
            write_json(output_dir / "diagnosis.json", contract)
            print(
                json.dumps(
                    {
                        "step": step,
                        "device": str(device),
                        "valid_tokens": count,
                        "pairs": [
                            {
                                key: x[key]
                                for key in [
                                    "pair",
                                    "max_gradient_error",
                                    "max_parameter_error",
                                    "parameter_failures",
                                ]
                            }
                            for x in pair_rows
                        ],
                    }
                ),
                flush=True,
            )
        contract["native_tp_call_metadata"] = {
            name: {
                key: dict(
                    kind=module._tp_native_kind,
                    calls=module._tp_native_calls,
                    input_dtypes=sorted(module._tp_input_dtypes),
                )
                for key, module in model.named_modules()
                if hasattr(module, "_tp_native_kind")
            }
            for name, model in models.items()
            if name in patched
        }
        contract["diagnostic_execution_completed"] = True
        write_json(output_dir / "diagnosis.json", contract)
    except BaseException as error:
        contract["errors"].append(repr(error))
        contract["diagnostic_execution_completed"] = False
        write_json(output_dir / "diagnosis.json", contract)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
