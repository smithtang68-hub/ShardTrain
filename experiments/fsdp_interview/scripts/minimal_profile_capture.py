"""Independent Nsight allocator capture without all-ATen NVTX or Python stacks.

PyTorch 2.9.1 accepts context=None to disable allocation tracebacks; stacks
remains the supported string 'python'. Alloc/free history and snapshots remain.
Only observability changes. Step/collective NVTX is supplied by the runner/hooks.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch


class MinimalProfileCapture:
    def __init__(self, directory, rank, device, *, max_entries=80000):
        if not 1 <= max_entries <= 200000:
            raise ValueError("Bounded memory history is required")
        self.directory, self.rank, self.device = Path(directory), int(rank), device
        self.max_entries = int(max_entries)
        self.started = False
        self.closed = False
        self.steps = []

    def start(self):
        parameters = inspect.signature(torch.cuda.memory._record_memory_history).parameters
        if not {"enabled", "context", "stacks", "max_entries"} <= parameters.keys():
            raise RuntimeError("Installed allocator history API is not qualified")
        for stem in ["memory.start", "memory.end", "profiling"]:
            suffix = "json" if stem == "profiling" else "pickle"
            if (self.directory / f"{stem}.rank{self.rank}.{suffix}").exists():
                raise FileExistsError("Preserve earlier profiler evidence")
        torch.cuda.synchronize(self.device)
        torch.cuda.memory._record_memory_history(
            enabled="all", context=None, stacks="python", max_entries=self.max_entries
        )
        self.started = True
        torch.cuda.memory._dump_snapshot(
            str(self.directory / f"memory.start.rank{self.rank}.pickle")
        )

    def step(self, step):
        self.steps.append(int(step))

    def close(self):
        if not self.started or self.closed:
            return
        try:
            torch.cuda.synchronize(self.device)
            torch.cuda.memory._dump_snapshot(
                str(self.directory / f"memory.end.rank{self.rank}.pickle")
            )
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
            self.closed = True
            (self.directory / f"profiling.rank{self.rank}.json").write_text(
                json.dumps(
                    {
                        "rank": self.rank,
                        "mode": "nsight",
                        "detail": "minimal",
                        "torch": torch.__version__,
                        "torch_git_version": torch.version.git_version,
                        "memory_history": {
                            "enabled": "all",
                            "context": None,
                            "stacks": "python",
                            "max_entries": self.max_entries,
                        },
                        "all_aten_emit_nvtx": False,
                        "step_and_collective_nvtx": True,
                        "steps": self.steps,
                        "closed": True,
                        "scope": "profile mechanism only; performance measured separately",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
