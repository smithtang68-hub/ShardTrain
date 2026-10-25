"""Reproducible audit of the historical M1 path versus v0.7 shard-aware consumers."""

import argparse
import json
from pathlib import Path

import torch

from sharded_runtime.config import ModelConfig
from sharded_runtime.models.reference import build_reference_model
from sharded_runtime.parallel.model import parallelize_tp_sp
from sharded_runtime.runtime import ShardedRuntime


def audit(mode):
    config = ModelConfig(layers=2, hidden_size=8, num_heads=2, mlp_hidden_size=12, seq_len=4)
    model = build_reference_model(config)
    runtime = None
    if mode == "historical_m1":
        parallelize_tp_sp(model)
        pointers = {p.untyped_storage().data_ptr() for p in model.parameters()}
    else:
        runtime = ShardedRuntime().register(model)
        runtime.begin_step(0)
        pointers = set()
        for unit in runtime.units:

            def snapshot(module, args, unit=unit):
                pointers.update(
                    lease.tensor.untyped_storage().data_ptr()
                    for lease in unit.lease.full_param_storages.values()
                )

            unit.block.register_forward_pre_hook(snapshot)
    rows = []

    def save(tensor):
        # A retired allocation's address can be reused by attention activations.
        # Compare against CURRENT live FullParams, never historical pointer integers.
        live_pointers = (
            pointers
            if runtime is None
            else {
                lease.tensor.untyped_storage().data_ptr()
                for unit in runtime.units
                if unit.lease is not None
                for lease in unit.lease.full_param_storages.values()
            }
        )
        rows.append(
            dict(
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
                tensor_bytes=tensor.numel() * tensor.element_size(),
                storage_bytes=tensor.untyped_storage().nbytes(),
                aliases_full_weight=tensor.untyped_storage().data_ptr() in live_pointers,
            )
        )
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda x: x):
        output = model(torch.randn(2, 4, 8))
        output.square().mean().backward()
    if runtime:
        runtime.finalize_backward()
        runtime.end_step()
    return dict(
        mode=mode,
        saved_tensors=rows,
        full_weight_saved_count=sum(r["aliases_full_weight"] for r in rows),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/v07-audit/saved_tensors.json")
    args = parser.parse_args()
    results = [audit(mode) for mode in ("historical_m1", "v07_runtime")]
    assert results[0]["full_weight_saved_count"] > 0
    assert results[1]["full_weight_saved_count"] == 0
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print([(r["mode"], r["full_weight_saved_count"]) for r in results])


if __name__ == "__main__":
    main()
