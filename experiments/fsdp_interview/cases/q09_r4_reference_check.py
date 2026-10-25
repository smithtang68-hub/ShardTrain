"""CPU fixture guard: public API input mutation must not alter the R4 oracle."""

from __future__ import annotations

import json

import torch
from q09_param_representation import clone_cpu_model_state, error_row, make_model


def main() -> int:
    checkpoint = make_model(1234, torch.device("cpu")).state_dict()
    reference = clone_cpu_model_state(checkpoint)
    api_input = clone_cpu_model_state(reference)
    for name in reference:
        assert reference[name].device.type == "cpu"
        assert reference[name].data_ptr() != api_input[name].data_ptr()
        assert reference[name].data_ptr() != checkpoint[name].data_ptr()
    for name in api_input:
        # Simulate both mutations allowed by a state-dict setter: in-place
        # updates of a value and replacing the mapping's Tensor object.
        api_input[name].add_(1)
        api_input[name] = torch.full_like(api_input[name], 2)
        assert torch.equal(reference[name], checkpoint[name])
        assert error_row(name, reference[name], checkpoint[name], 0.0, 0.0)["pass"]
    name = next(iter(reference))
    modified = reference[name].clone()
    modified.reshape(-1)[0] += 1e-4
    assert not error_row(name, modified, reference[name], 0.0, 0.0)["pass"]
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "CPU fixture alias/isolation and exact comparison only",
                "checks": [
                    "API mapping can mutate without changing CPU oracle",
                    "nonzero update rejected at zero tolerance",
                ],
                "distributed_gpu_roundtrip": "NOT_RUN_BY_THIS_CHECK",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
