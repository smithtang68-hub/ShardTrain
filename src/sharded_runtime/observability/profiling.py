"""Per-rank CUDA traces and bounded allocator history, including failed runs."""

import os
from pathlib import Path

import torch

from sharded_runtime.observability.metadata import write_json


class ProfileCapture:
    def __init__(
        self, directory, rank, device, *, mode="torch", max_entries=200_000, detail="low_overhead"
    ):
        if mode not in {"torch", "nsight", "off"}:
            raise ValueError("Unknown profile mode")
        if detail not in {"low_overhead", "full"}:
            raise ValueError("Unknown profile detail")
        self.directory = Path(directory)
        self.rank = rank
        self.device = torch.device(device)
        self.mode = mode
        self.max_entries = max_entries
        self.detail = detail
        self.memory_context = "all" if detail == "full" else "alloc"
        self.memory_stacks = "all" if detail == "full" else "python"
        self.profiler = None
        self.nvtx = None
        self.memory = []
        self.previous_nvtx = None
        self.active = False

    def start(self):
        if self.mode == "off":
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda":
            torch.cuda.memory._record_memory_history(
                context=self.memory_context,
                stacks=self.memory_stacks,
                max_entries=self.max_entries,
            )
            self.active = True
            torch.cuda.memory._dump_snapshot(
                str(self.directory / f"memory.start.rank{self.rank}.pickle")
            )
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.previous_nvtx = os.environ.get("SHARDTRAIN_NVTX")
            os.environ["SHARDTRAIN_NVTX"] = "1"
        if self.mode == "torch":
            self.profiler = torch.profiler.profile(
                # Shape tracing can retain tensors and perturb reshard/storage-lifetime checks.
                activities=activities,
                record_shapes=False,
                profile_memory=True,
                with_stack=self.detail == "full",
            )
            self.profiler.start()
        elif self.device.type == "cuda":
            # Nsight owns CUPTI in this mode; do not also start the Kineto CUDA profiler.
            self.nvtx = torch.autograd.profiler.emit_nvtx(record_shapes=self.detail == "full")
            self.nvtx.__enter__()

    def step(self, step):
        if self.device.type == "cuda" and self.mode != "off":
            self.memory.append(
                dict(
                    step=step,
                    allocated=torch.cuda.memory_allocated(self.device),
                    reserved=torch.cuda.memory_reserved(self.device),
                    max_allocated=torch.cuda.max_memory_allocated(self.device),
                    max_reserved=torch.cuda.max_memory_reserved(self.device),
                )
            )
            torch.cuda.nvtx.mark(f"shardtrain/step_complete/{step}/rank{self.rank}")
        if self.profiler is not None:
            self.profiler.step()

    def close(self):
        errors = []
        try:
            if self.profiler is not None:
                self.profiler.stop()
                self.profiler.export_chrome_trace(
                    str(self.directory / f"profile.rank{self.rank}.json")
                )
            if self.nvtx is not None:
                self.nvtx.__exit__(None, None, None)
        except Exception as exc:
            errors.append(repr(exc))
        finally:
            if self.active:
                try:
                    torch.cuda.memory._dump_snapshot(
                        str(self.directory / f"memory.end.rank{self.rank}.pickle")
                    )
                except Exception as exc:
                    errors.append(repr(exc))
                finally:
                    torch.cuda.memory._record_memory_history(enabled=None)
                    if self.previous_nvtx is None:
                        os.environ.pop("SHARDTRAIN_NVTX", None)
                    else:
                        os.environ["SHARDTRAIN_NVTX"] = self.previous_nvtx
                    self.active = False
            write_json(
                self.directory / f"profiling.rank{self.rank}.json",
                dict(
                    mode=self.mode,
                    rank=self.rank,
                    device=str(self.device),
                    memory_history_max_entries=self.max_entries,
                    torch_record_shapes=False,
                    profile_detail=self.detail,
                    nsight_record_shapes=self.mode == "nsight" and self.detail == "full",
                    torch_with_stack=self.mode == "torch" and self.detail == "full",
                    memory_history_context=self.memory_context,
                    memory_history_stacks=self.memory_stacks,
                    memory=self.memory,
                    errors=errors,
                    scope="training window; start/end snapshots with bounded allocation history",
                    note="Allocator snapshots exclude memory allocated directly by NCCL/CUDA",
                ),
            )
        if errors:
            raise RuntimeError(f"Failed to save profiling artifacts: {errors}")
