"""Read-only R4 audit negative fixtures; no GPU and no checkpoint writes."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import audit_q09_roundtrip as audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    original = json.loads(args.result.read_text(encoding="utf-8"))[0]
    assert audit.verify_row(original, "load", 2, 4, 0)["status"] == "PASS"
    nonzero = copy.deepcopy(original)
    nonzero["checks"][0]["max_abs"] = 1e-9
    assert audit.verify_row(nonzero, "load", 2, 4, 0)["status"] == "FAIL"
    alias = copy.deepcopy(original)
    alias["input_state_api_observation"]["reference_is_separate_from_api_input"] = False
    assert audit.verify_row(alias, "load", 2, 4, 0)["status"] == "FAIL"
    wrong_device = copy.deepcopy(original)
    wrong_device["input_state_api_observation"]["before"]["embedding.weight"]["device"] = "cuda:0"
    assert audit.verify_row(wrong_device, "load", 2, 4, 0)["status"] == "FAIL"
    wrong_world = copy.deepcopy(original)
    wrong_world["origin_world_size"] = 2
    assert audit.verify_row(wrong_world, "load", 2, 4, 0)["status"] == "FAIL"
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "CPU audit fixtures only; real load result remains unchanged",
                "checks": [
                    "nonzero parameter error rejected despite PASS flag",
                    "aliased CPU reference rejected",
                    "non-CPU reference input rejected",
                    "wrong source world size rejected",
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
