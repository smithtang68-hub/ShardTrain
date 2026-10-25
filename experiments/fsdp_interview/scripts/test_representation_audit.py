"""Read-only CPU checks for Q9/Q10 result-auditor safeguards, not GPU tests."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from unittest.mock import patch

import analyze_representation as audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q9-dir", required=True)
    parser.add_argument("--q10-correctness-dirs", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert audit.audit_q9([args.q9_dir], 4, {})["status"] == "PASS"
    assert audit.audit_q10_correctness(args.q10_correctness_dirs, 4, {})["status"] == "PASS"
    original = audit.load_json

    def nonzero_frozen_view(path, provenance):
        value = original(path, provenance)
        if isinstance(value, list):
            value = copy.deepcopy(value)
            for item in value:
                if item.get("experiment_id") == "Q9_R2_fsdp1_orig":
                    for row in item.get("checks", []):
                        if row["name"] == "gradient/embedding.weight":
                            row["nonzero_elements"] = 1
        return value

    with patch.object(audit, "load_json", nonzero_frozen_view):
        assert audit.audit_q9([args.q9_dir], 4, {})["status"] == "FAIL"

    def different_model_config(path, provenance):
        value = original(path, provenance)
        if isinstance(value, dict) and value.get("config", {}).get("implementation") == "fsdp2":
            value = copy.deepcopy(value)
            value["config"]["ffn"] += 1
        return value

    with patch.object(audit, "load_json", different_model_config):
        assert audit.audit_q10_correctness(args.q10_correctness_dirs, 4, {})["status"] == "FAIL"
    assert (
        audit.audit_q9([str(Path(args.q9_dir) / "missing_test_input")], 4, {})["status"]
        == "NOT_RUN"
    )
    timing = audit.audit_timing([args.q10_correctness_dirs[0]], 4, {})
    assert timing["status"] == "NOT_RUN" and timing["runs"][0]["status"] == "EXCLUDED"
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    audit.dump(
        output,
        {
            "status": "PASS",
            "scope": "CPU result-auditor checks only; original GPU files never modified",
            "checks": [
                "Actual Q9/Q10 correctness inputs pass read-only audit",
                "Nonzero frozen gradient view rejected even if its reported pass flag is True",
                "Mismatched model controls reject cross-engine comparison",
                "Missing result input marked NOT_RUN",
                "Correctness-only step time excluded from performance comparison",
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
