"""Check the DP/accumulation oracle mathematics on CPU, not FSDP correctness."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from q09_param_representation import (
    data_for_rank,
    frozen_gradient_row,
    make_model,
    optimizer_for,
    write_json,
)


def verify_dp_reference(frozen: bool) -> dict:
    device = torch.device("cpu")
    base = make_model(1234, device, frozen=frozen)
    replicas = [copy.deepcopy(base) for _ in range(4)]
    packed = copy.deepcopy(base)
    batches = [data_for_rank(1234, rank, device) for rank in range(4)]
    for model, (ids, targets) in zip(replicas, batches, strict=True):
        F.mse_loss(model(ids), targets).backward()
    packed_ids = torch.cat([batch[0] for batch in batches])
    packed_targets = torch.cat([batch[1] for batch in batches])
    F.mse_loss(packed(packed_ids), packed_targets).backward()
    checks = []
    parameters = [dict(model.named_parameters()) for model in replicas]
    for name, parameter in packed.named_parameters():
        if not parameter.requires_grad:
            checks.append({"name": name, "pass": all(p[name].grad is None for p in parameters)})
            continue
        average = torch.stack([p[name].grad for p in parameters]).mean(0)
        checks.append(
            {
                "name": name,
                "pass": bool(torch.allclose(average, parameter.grad, atol=2e-7, rtol=1e-6)),
                "max_abs": float((average - parameter.grad).abs().max()),
            }
        )
        parameters[0][name].grad = average
    distributed_optimizer = optimizer_for(replicas[0], True)
    packed_optimizer = optimizer_for(packed, True)
    distributed_optimizer.step()
    packed_optimizer.step()
    for name, parameter in packed.named_parameters():
        checks.append(
            {
                "name": f"update/{name}",
                "pass": bool(torch.allclose(parameters[0][name], parameter, atol=2e-7, rtol=1e-6)),
                "max_abs": float((parameters[0][name] - parameter).detach().abs().max()),
            }
        )
    return {
        "test": "equal-sized DP mean gradient == one packed dense batch",
        "frozen_embedding": frozen,
        "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = [verify_dp_reference(False), verify_dp_reference(True)]
    fixture_cases = [
        (None, True),
        (torch.zeros(31, 16), True),
        (torch.ones(31, 16), False),
        (torch.full((31, 16), float("nan")), False),
        (torch.zeros(16, 31), False),
    ]
    checks = [
        {
            "name": f"frozen_view_fixture_{index}",
            "pass": frozen_gradient_row("test", observed, (31, 16))["pass"] == expected,
        }
        for index, (observed, expected) in enumerate(fixture_cases)
    ]
    results.append(
        {
            "test": "Frozen full observer zero view accepted; nonzero/NaN/wrong shape rejected",
            "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
            "checks": checks,
        }
    )
    write_json(
        Path(args.output),
        {
            "scope": "CPU oracle mathematics only; NOT GPU FSDP R1/R2/R3 PASS",
            "torch_version": torch.__version__,
            "results": results,
        },
    )
    return 0 if all(row["status"] == "PASS" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
