"""Diagnostic-only actual dispatcher calls; never enabled in formal timing."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode


class CollectiveProbe(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.training_step = None
        self.rows = []

    def metadata(self, value):
        if isinstance(value, torch.Tensor):
            return dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                bytes=value.numel() * value.element_size(),
                device=str(value.device),
            )
        if isinstance(value, dist.ProcessGroup):
            return dict(
                process_group_name=value.group_name,
                members=dist.get_process_group_ranks(value),
                group_size=value.size(),
            )
        if isinstance(value, (list, tuple)):
            return [self.metadata(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return dict(type=type(value).__name__)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        label = str(func)
        if (
            self.training_step is not None
            and "c10d" in label
            and any(
                word in label.lower()
                for word in [
                    "allgather",
                    "all_gather",
                    "reduce_scatter",
                    "allreduce",
                    "all_reduce",
                    "broadcast",
                ]
            )
        ):
            names = [a.name for a in func._schema.arguments]
            self.rows.append(
                dict(
                    ordinal=len(self.rows),
                    optimizer_step=self.training_step,
                    dispatcher_op=label,
                    arguments={
                        name: self.metadata(value) for name, value in zip(names, args, strict=False)
                    },
                    keyword_arguments=self.metadata(list((kwargs or {}).items())),
                    output=self.metadata(output),
                    scope="Actual training dispatcher invocation, not wire byte counter",
                )
            )
        return output
