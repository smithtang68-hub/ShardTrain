"""Explicit synchronous TP/SP operators; no official parallel wrapper dependency."""

from sharded_runtime.parallel.model import TensorParallelPlan, parallelize_tp_sp
from sharded_runtime.parallel.tp_linear import ColumnParallelLinear, RowParallelLinear

__all__ = ["ColumnParallelLinear", "RowParallelLinear", "TensorParallelPlan", "parallelize_tp_sp"]
