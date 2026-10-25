"""One owned, watchdog-bounded stage; exact argv and telemetry are always saved."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from owned_stage import run_stage  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--env-json", default="{}")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("An explicit command is required")
    r = run_stage(args.stage, command, timeout=args.timeout, extra_env=json.loads(args.env_json))
    return 0 if r["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
