"""Check assembled local evidence links and completion labels before delivery."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.report.read_text(encoding="utf-8")
    questions = ["Q1", "Q3", "Q5", "Q6", "Q9", "Q10", "Q12"]
    for question in questions:
        match = re.search(rf"^### {question}：.*?(?=^### Q[0-9]+：|\Z)", content, re.M | re.S)
        assert match and re.search(r"运行状态[^\n]*PASS", match.group()), question
    table_start = content.index("| 组别 | P50 / P95 ms")
    table_lines = content[table_start:].splitlines()[:8]
    assert len(table_lines) == 8 and all(line.startswith("|") for line in table_lines)
    assert table_lines[1].startswith("|---")
    checked, missing = [], []
    for target in re.findall(r"\]\((<[^>]+>|[^)]+)\)", content):
        value = target[1:-1] if target.startswith("<") and target.endswith(">") else target
        if value.startswith("#") or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
            continue
        value = unquote(value.partition("#")[0])
        value = re.sub(r":\d+$", "", value)
        path = Path(value)
        if not path.is_absolute():
            path = args.report.parent / path
        row = {"target": target, "resolved": str(path.resolve()), "exists": path.exists()}
        checked.append(row)
        if not row["exists"]:
            missing.append(row)
    payload = {
        "status": "PASS" if not missing else "FAIL",
        "report": str(args.report.resolve()),
        "sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "question_sections": questions,
        "local_links": checked,
        "missing": missing,
        "scope": "Local evidence existence and seven completion labels; "
        "web-source semantics audited in individual chapters.",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"status": payload["status"], "local_links": len(checked), "missing": missing})
    )
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
