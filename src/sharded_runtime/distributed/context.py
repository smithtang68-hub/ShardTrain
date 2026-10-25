"""Single owner of device binding, default group and mesh subgroup lifetimes."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from sharded_runtime.config import ConfigError, ParallelConfig
from sharded_runtime.distributed.topology import Topology


@dataclass(frozen=True)
class WorkerEnvironment:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    launched: bool = False

    @classmethod
    def read(cls, environ: Mapping[str, str] | None = None) -> WorkerEnvironment:
        env = os.environ if environ is None else environ
        keys = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        present = [key in env for key in keys]
        if not any(present):
            return cls()
        if not all(present):
            raise ConfigError("torchrun requires RANK, LOCAL_RANK and WORLD_SIZE together")
        try:
            rank, local_rank, world_size = (int(env[key]) for key in keys)
        except ValueError as exc:
            raise ConfigError("torchrun rank environment values must be integers") from exc
        if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
            raise ConfigError("Invalid RANK/LOCAL_RANK/WORLD_SIZE in worker environment")
        return cls(rank, local_rank, world_size, True)


class DistributedContext:
    def __init__(
        self,
        parallel: ParallelConfig,
        worker: WorkerEnvironment,
        timeout_seconds: int,
        logger: logging.Logger,
    ) -> None:
        self.logger = logger
        self._close_callbacks: list[Callable[[], None]] = []
        self.cleanup_errors: list[str] = []
        self.parallel = parallel
        self.worker = worker
        self.topology = Topology(parallel.fsdp_degree, parallel.tp_degree)
        self.coordinate = self.topology.coordinate(worker.rank)
        if self.topology.world_size != worker.world_size:
            raise ConfigError("Worker world size does not match topology")
        self.device = torch.device("cpu")
        self.tp_group: Any = None
        self.fsdp_group: Any = None
        self.tp_grad_group: Any = None
        self.fsdp_grad_group: Any = None
        self._owned_groups: list[Any] = []
        self._owns_default = False
        self._closed = False
        timeout = timedelta(seconds=timeout_seconds)
        try:
            self._bind_device()
            if dist.is_initialized():
                raise RuntimeError(
                    "DistributedContext requires ownership of an uninitialized group"
                )
            if worker.launched:
                if not os.environ.get("MASTER_ADDR") or not os.environ.get("MASTER_PORT"):
                    raise ConfigError("torchrun launch requires MASTER_ADDR and MASTER_PORT")
                if not dist.is_available():
                    raise ConfigError("This PyTorch build does not support torch.distributed")
                if parallel.backend == "nccl" and not dist.is_nccl_available():
                    raise ConfigError(
                        "NCCL is unavailable in this PyTorch build (Windows uses Gloo)"
                    )
                if parallel.backend == "gloo" and not dist.is_gloo_available():
                    raise ConfigError("Gloo is unavailable in this PyTorch build")
                if parallel.backend == "gloo" and sys.platform == "win32":
                    try:
                        dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")
                    except RuntimeError as exc:
                        raise ConfigError(
                            "This Windows PyTorch wheel has no usable Gloo transport; "
                            "use Linux/WSL or a libuv-enabled build"
                        ) from exc
                # Windows wheels may omit libuv. Use the documented legacy TCPStore route.
                if sys.platform == "win32":
                    os.environ.setdefault("USE_LIBUV", "0")
                dist.init_process_group(
                    backend=parallel.backend,
                    rank=worker.rank,
                    world_size=worker.world_size,
                    timeout=timeout,
                )
                self._owns_default = True
                for axis, ranks in self.topology.creation_order:
                    self.logger.debug("Creating %s group ranks=%s", axis, ranks)
                    group = dist.new_group(
                        ranks=list(ranks), backend=parallel.backend, timeout=timeout
                    )
                    if worker.rank in ranks:
                        self._owned_groups.append(group)
                        setattr(self, f"{axis}_group", group)
            self.logger.info(
                "Context ready: rank=%d local_rank=%d coordinate=%s device=%s groups=%s",
                worker.rank,
                worker.local_rank,
                self.coordinate,
                self.device,
                self.topology.groups_for_rank(worker.rank),
            )
        except BaseException:
            self.close()
            raise

    def _bind_device(self) -> None:
        if self.parallel.device == "cuda":
            if not torch.cuda.is_available():
                raise ConfigError("CUDA was requested but is unavailable")
            if self.worker.local_rank >= torch.cuda.device_count():
                raise ConfigError("LOCAL_RANK exceeds the number of visible CUDA devices")
            torch.cuda.set_device(self.worker.local_rank)
            self.device = torch.device("cuda", self.worker.local_rank)

    @property
    def sp_group(self) -> Any:
        return self.tp_group

    def smoke_collectives(self) -> dict[str, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot use a closed DistributedContext")
        results = {}
        for axis, ranks in self.topology.groups_for_rank(self.worker.rank).items():
            value = torch.tensor([float(self.worker.rank + 1)], device=self.device)
            gathered = [torch.empty_like(value) for _ in ranks]
            if self._owns_default:
                group = getattr(self, f"{axis}_group")
                dist.all_gather(gathered, value, group=group)
                dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
            else:
                gathered[0].copy_(value)
            expected = sum(rank + 1 for rank in ranks)
            actual = value.item()  # GPU read waits for the collective consumer dependency.
            actual_gather = [item.item() for item in gathered]
            if actual != expected or actual_gather != [float(rank + 1) for rank in ranks]:
                raise AssertionError(f"{axis} collective mismatch: {actual}, {actual_gather}")
            results[axis] = {"ranks": list(ranks), "sum": actual, "gather": actual_gather}
        self.logger.info("Collective smoke passed: %s", results)
        return results

    def add_close_callback(self, callback: Callable[[], None]) -> None:
        """Register local cleanup, such as flushing buffered diagnostic records.

        Callbacks run once before process-group destruction and must not introduce
        collective operations or device synchronization into cleanup.
        """
        if self._closed:
            raise RuntimeError("Cannot register cleanup on a closed DistributedContext")
        if not callable(callback):
            raise TypeError("DistributedContext close callback must be callable")
        self._close_callbacks.append(callback)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        callbacks, self._close_callbacks = self._close_callbacks, []
        for index, callback in enumerate(callbacks):
            try:
                callback()
            except Exception as exc:
                self.cleanup_errors.append(f"close callback {index}: {exc!r}")
                self.logger.exception("Failed to run local close callback %d", index)
        # No barrier on failure: a missing peer must not cause an unbounded cleanup wait.
        for group in reversed(self._owned_groups):
            try:
                dist.destroy_process_group(group)
            except Exception as exc:
                self.cleanup_errors.append(f"destroy subgroup: {exc!r}")
                self.logger.exception("Failed to destroy subgroup")
        self._owned_groups.clear()
        if self._owns_default:
            try:
                dist.destroy_process_group()
            except Exception as exc:
                self.cleanup_errors.append(f"destroy default group: {exc!r}")
                self.logger.exception("Failed to destroy default process group")
            finally:
                self._owns_default = False
        self.tp_group = self.fsdp_group = self.tp_grad_group = self.fsdp_grad_group = None
        self.logger.info("Context closed")

    def __enter__(self) -> DistributedContext:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
        if exc_type is None and self.cleanup_errors:
            raise RuntimeError(
                "DistributedContext cleanup failed: " + "; ".join(self.cleanup_errors)
            )
