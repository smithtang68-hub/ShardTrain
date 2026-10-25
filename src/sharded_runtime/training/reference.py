"""Independent functional dense SDPA oracle, including every Block output."""

import torch
from torch.nn import functional as F


def dense_oracle(x, parameters, config, *, sublayers=None):
    outputs = []
    for layer in range(config.layers):
        p = {
            name.removeprefix(f"{layer}."): value
            for name, value in parameters.items()
            if name.startswith(f"{layer}.")
        }

        def norm(value, gamma):
            return (
                value
                / torch.sqrt(value.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)
                * gamma
            )

        z = norm(x, p["attention_norm.weight"])
        batch, seq, hidden = z.shape
        q, k, v = [
            F.linear(z, p[f"attention.{name}_proj.weight"])
            .reshape(batch, seq, config.num_heads, hidden // config.num_heads)
            .transpose(1, 2)
            for name in ("q", "k", "v")
        ]
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attention = F.linear(
            attended.transpose(1, 2).reshape(batch, seq, hidden), p["attention.out_proj.weight"]
        )
        if sublayers is not None:
            sublayers[layer, "attention"] = attention
        x = x + attention
        z = norm(x, p["mlp_norm.weight"])
        gate = F.linear(z, p["mlp.gate_proj.weight"])
        mlp = F.linear(
            gate * torch.sigmoid(gate) * F.linear(z, p["mlp.up_proj.weight"]),
            p["mlp.down_proj.weight"],
        )
        if sublayers is not None:
            sublayers[layer, "mlp"] = mlp
        x = x + mlp
        outputs.append(x)
    return x, outputs


def compare_tensor(actual, expected, name, *, atol=1e-5, rtol=1e-4):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    error = (actual - expected).abs()
    record = dict(
        name=name,
        shape=list(actual.shape),
        dtype=str(actual.dtype),
        max_abs_error=error.max().item(),
        max_rel_error=(error / expected.abs().clamp_min(1e-12)).max().item(),
        nan_count=torch.isnan(actual).sum().item(),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)
    return record
