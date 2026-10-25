"""Profile-only local ATen GEMM metadata and exact NVTX invocation labels.

No numerical/DTensor rule is replaced. An optional, temporary wrapper re-enables
TorchDispatchMode *inside* the installed DTensor dispatcher so that its final
plain-local ATen calls can be observed. Global DTensor shapes are never reported
as local GEMM dimensions. CPU timestamps are not NSYS/GPU timestamps.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import TorchDispatchMode

TRACKED = {"aten::mm", "aten::addmm", "aten::bmm", "aten::linear"}
VERSION_AUDITED = "5811a8d7da873dd699ff6687092c225caffcf1bb"


def _operation(func) -> str:
    return func._schema.name if hasattr(func, "_schema") else str(func)


def _tensor_values(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for element in value:
            yield from _tensor_values(element)
    elif isinstance(value, dict):
        for element in value.values():
            yield from _tensor_values(element)


def _plain_tensor(value) -> bool:
    # Standard Parameter adds no __torch_dispatch__ implementation. FakeTensor,
    # DTensor and other subclasses are deliberately excluded from trusted rows.
    return type(value) in {torch.Tensor, torch.nn.Parameter}


def _metadata(value) -> dict[str, Any]:
    if not _plain_tensor(value):
        return {"kind": type(value).__name__, "local_shape": None, "reason": "not plain local"}
    return {
        "kind": type(value).__name__,
        "local_shape": [int(size) for size in value.shape],
        "local_stride": [int(size) for size in value.stride()],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "layout": str(value.layout),
        "requires_grad": bool(value.requires_grad),
    }


def _named_arguments(func, args, kwargs) -> dict[str, Any]:
    names = [argument.name for argument in func._schema.arguments]
    result = {name: value for name, value in zip(names, args, strict=False)}
    result.update(kwargs)
    return result


def _matrix_metadata(func, args, kwargs) -> dict[str, Any]:
    operation = _operation(func)
    named = _named_arguments(func, args, kwargs)
    if operation in {"aten::mm", "aten::bmm"}:
        left, right = named.get("self"), named.get("mat2")
    elif operation == "aten::addmm":
        left, right = named.get("mat1"), named.get("mat2")
    elif operation == "aten::linear":
        left, right = named.get("input"), named.get("weight")
    else:
        raise ValueError("not a tracked matrix operation")
    tensors = list(_tensor_values((args, kwargs)))
    result = {
        "operator": str(func),
        "aten_schema": str(func._schema),
        "classification": "UNKNOWN",
        "M": None,
        "N": None,
        "K": None,
        "batch_count": None,
        "left": _metadata(left),
        "right": _metadata(right),
        "bias": _metadata(named["self"])
        if operation == "aten::addmm"
        else _metadata(named["bias"])
        if operation == "aten::linear" and isinstance(named.get("bias"), torch.Tensor)
        else None,
        "reason": None,
    }
    if any(not _plain_tensor(value) for value in tensors) or not tensors:
        result["reason"] = "logical DTensor/other subclass is not an observed local GEMM"
        return result
    if not _plain_tensor(left) or not _plain_tensor(right):
        result["reason"] = "matrix operands not found as plain local tensors"
        return result
    if left.layout != torch.strided or right.layout != torch.strided:
        result["reason"] = "non-strided operands; dense GEMM shape not established"
        return result
    if left.is_meta or right.is_meta:
        result["reason"] = "meta tensor planning is not an executed local GEMM"
        return result
    if operation == "aten::linear":
        # Linear is composite. Its local input/weight are trusted, but flattening
        # or batched decomposition must be observed as an actual mm/addmm/bmm.
        result["reason"] = "local linear is composite; await underlying mm/addmm/bmm metadata"
        result["classification"] = "LOCAL_LINEAR_COMPOSITE"
        return result
    expected_rank = 3 if operation == "aten::bmm" else 2
    if left.ndim != expected_rank or right.ndim != expected_rank:
        result["reason"] = "actual operands have unsupported matrix rank"
        return result
    if left.shape[-1] != right.shape[-2]:
        result["reason"] = "actual operands have inconsistent contraction dimensions"
        return result
    if operation == "aten::bmm" and left.shape[0] != right.shape[0]:
        result["reason"] = "actual batch counts differ"
        return result
    result.update(
        classification="LOCAL",
        M=int(left.shape[-2]),
        N=int(right.shape[-1]),
        K=int(left.shape[-1]),
        batch_count=int(left.shape[0]) if operation == "aten::bmm" else 1,
    )
    # CPU scalars alpha/beta are saved as metadata only. Never read device data.
    for name in ("alpha", "beta"):
        value = named.get(name)
        if isinstance(value, (int, float)):
            result[name] = value
    return result


class _LocalGemmMode(TorchDispatchMode):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if _operation(func) not in TRACKED or not self.recorder.active:
            return func(*args, **kwargs)
        tensors = tuple(_tensor_values((args, kwargs)))
        if any(isinstance(value, DTensor) for value in tensors):
            # This scope only records UNKNOWN metadata, never an NVTX GEMM range.
            # The installed DTensor dispatcher is responsible for real local ops.
            return self.recorder.logical_call(func, args, kwargs)
        return self.recorder.local_call(func, args, kwargs)


class GemmShapeCapture:
    """Use only in a separate shape-diagnostic profile, never timing baselines.

    with capture_gemm_shapes(directory, rank=rank, implementation="native") as c:
        c.set_step(optimizer_step=10, measured_step=0)
        trainer.train_step(...)

    Metadata contains no Tensor references. Tensor operands remain ordinary
    inputs passed unchanged to the original operation. No .item(), to_local(),
    CUDA synchronization, Event, or stream wait is introduced by this module.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        run_id="fsdp_interview",
        experiment_id="gemm_shapes",
        implementation="unknown",
        rank=None,
        capture_nvtx=True,
        patch_dtensor=True,
        max_records=80000,
    ):
        if not 1 <= max_records <= 200000:
            raise ValueError("max_records must be bounded to 1..200000")
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.run_id = str(run_id)
        self.experiment_id = str(experiment_id)
        self.implementation = str(implementation)
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else int(rank)
        self.capture_nvtx, self.patch_dtensor = bool(capture_nvtx), bool(patch_dtensor)
        self.max_records = max_records
        self.rows: list[dict[str, Any]] = []
        self.active = False
        self._lock = threading.RLock()
        self._tls = threading.local()
        self._ordinal = 0
        self.dropped_records = 0
        self._step = {"optimizer_step": None, "measured_step": None, "phase": "unknown"}
        self._dispatch_type = type(DTensor._op_dispatcher)
        self._original_dispatch = None
        self._dispatch_wrapper = None
        self._base_mode = None
        self.restored = False
        self.exception = None

    def set_step(self, optimizer_step=None, measured_step=None, phase="unknown", **extra) -> None:
        with self._lock:
            self._step = {
                "optimizer_step": optimizer_step,
                "measured_step": measured_step,
                "phase": str(phase),
                **extra,
            }

    @contextmanager
    def phase(self, name: str):
        previous = getattr(self._tls, "phase", None)
        self._tls.phase = str(name)
        try:
            yield
        finally:
            self._tls.phase = previous

    def _parents(self):
        if not hasattr(self._tls, "parents"):
            self._tls.parents = []
        return self._tls.parents

    def _row(self, func, args, kwargs, *, source):
        with self._lock:
            ordinal = self._ordinal
            self._ordinal += 1
            step = dict(self._step)
        try:
            metadata = _matrix_metadata(func, args, kwargs)
        except Exception as error:
            metadata = {
                "operator": str(func),
                "classification": "UNKNOWN",
                "M": None,
                "N": None,
                "K": None,
                "batch_count": None,
                "reason": f"metadata unavailable: {type(error).__name__}: {error}",
            }
        parents = self._parents()
        row = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "implementation": self.implementation,
            "rank": self.rank,
            "cpu_ordinal": ordinal,
            "cpu_native_thread_id": threading.get_native_id(),
            "cpu_python_thread_id": threading.get_ident(),
            "shape_source": source,
            "dtensor_dispatch_depth": getattr(self._tls, "dispatch_depth", 0),
            "dtensor_parent_ordinal": parents[-1] if parents else None,
            **step,
            "phase": getattr(self._tls, "phase", None) or step.get("phase", "unknown"),
            **metadata,
            "nvtx_label": None,
            "nvtx_emitted": False,
            "user_stream_pointer": None,
            "gpu_trace_status": "UNMATCHED",
            "cpu_start_monotonic_ns": None,
            "cpu_end_monotonic_ns": None,
            "cpu_invocation_ms": None,
            "operation_status": "STARTED",
        }
        return row

    def _append(self, row):
        with self._lock:
            if len(self.rows) < self.max_records:
                self.rows.append(row)
            else:
                self.dropped_records += 1

    def logical_call(self, func, args, kwargs):
        row = self._row(func, args, kwargs, source="LOGICAL_DTENSOR_NOT_LOCAL")
        row.update(classification="UNKNOWN", M=None, N=None, K=None, batch_count=None)
        row["reason"] = "global DTensor operands excluded; inspect linked local dispatcher rows"
        self._parents().append(row["cpu_ordinal"])
        row["cpu_start_monotonic_ns"] = time.perf_counter_ns()
        try:
            result = func(*args, **kwargs)
            row["operation_status"] = "RETURNED"
            return result
        except BaseException as error:
            row["operation_status"] = "FAILED"
            row["exception"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            row["cpu_end_monotonic_ns"] = time.perf_counter_ns()
            row["cpu_invocation_ms"] = (
                row["cpu_end_monotonic_ns"] - row["cpu_start_monotonic_ns"]
            ) / 1e6
            self._parents().pop()
            self._append(row)

    def local_call(self, func, args, kwargs):
        source = (
            "DTENSOR_DISPATCH_PLAIN_LOCAL_ATEN"
            if getattr(self._tls, "dispatch_depth", 0) > 0
            else "PLAIN_LOCAL_ATEN"
        )
        row = self._row(func, args, kwargs, source=source)
        tensors = tuple(_tensor_values((args, kwargs)))
        cuda_devices = {value.device for value in tensors if _plain_tensor(value) and value.is_cuda}
        label = (
            f"interview/gemm/rank={self.rank}/ordinal={row['cpu_ordinal']}"
            f"/op={str(func)}/M={row.get('M')}/N={row.get('N')}/K={row.get('K')}"
            f"/B={row.get('batch_count')}/dtype={row.get('left', {}).get('dtype')}"
        )
        if row["classification"] == "LOCAL" and len(cuda_devices) == 1:
            row["user_stream_pointer"] = int(
                torch.cuda.current_stream(next(iter(cuda_devices))).cuda_stream
            )
            row["nvtx_label"] = label
            if self.capture_nvtx:
                torch.cuda.nvtx.range_push(label)
                row["nvtx_emitted"] = True
        row["cpu_start_monotonic_ns"] = time.perf_counter_ns()
        try:
            result = func(*args, **kwargs)
            row["output"] = _metadata(result) if isinstance(result, torch.Tensor) else None
            row["operation_status"] = "RETURNED"
            return result
        except BaseException as error:
            row["operation_status"] = "FAILED"
            row["exception"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            row["cpu_end_monotonic_ns"] = time.perf_counter_ns()
            row["cpu_invocation_ms"] = (
                row["cpu_end_monotonic_ns"] - row["cpu_start_monotonic_ns"]
            ) / 1e6
            if row["nvtx_emitted"]:
                torch.cuda.nvtx.range_pop()
            self._append(row)

    def __enter__(self):
        if self.active:
            raise RuntimeError("nested/reused active GEMM capture is unsupported")
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if (self.output_dir / f"gemm_shapes.rank{self.rank}.jsonl").exists():
                raise FileExistsError(
                    "preserve previous GEMM shapes; choose a fresh output directory"
                )
        original = self._dispatch_type.dispatch
        if getattr(original, "_interview_gemm_capture", False):
            raise RuntimeError("another GEMM capture already wraps the DTensor dispatcher")
        if self.patch_dtensor and list(inspect.signature(original).parameters) != [
            "self",
            "op_call",
            "args",
            "kwargs",
        ]:
            raise RuntimeError("installed DTensor dispatcher signature is not the audited form")
        self._original_dispatch = original
        self.active = True
        recorder = self

        def dispatch(dispatcher, op_call, args, kwargs):
            if not recorder.active:
                return original(dispatcher, op_call, args, kwargs)
            depth = getattr(recorder._tls, "dispatch_depth", 0)
            recorder._tls.dispatch_depth = depth + 1
            try:
                # This is exactly the installed official dispatcher. Re-enable
                # metadata observation around its final local ATen invocation.
                with _LocalGemmMode(recorder):
                    return original(dispatcher, op_call, args, kwargs)
            finally:
                recorder._tls.dispatch_depth = depth

        dispatch._interview_gemm_capture = True
        self._dispatch_wrapper = dispatch
        try:
            if self.patch_dtensor:
                self._dispatch_type.dispatch = dispatch
            self._base_mode = _LocalGemmMode(self)
            self._base_mode.__enter__()
        except BaseException:
            self.active = False
            if self.patch_dtensor:
                self._dispatch_type.dispatch = original
            self.restored = self._dispatch_type.dispatch is original
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exception = f"{exc_type.__name__}: {exc}" if exc_type else None
        try:
            if self._base_mode is not None:
                self._base_mode.__exit__(exc_type, exc, tb)
        finally:
            self.active = False
            if self.patch_dtensor:
                self._dispatch_type.dispatch = self._original_dispatch
            self.restored = self._dispatch_type.dispatch is self._original_dispatch
        try:
            self.flush()
        except BaseException:
            # Original mathematical errors must remain the primary exception.
            if exc_type is None:
                raise
        return False

    def report(self) -> dict[str, Any]:
        local_rows = [row for row in self.rows if row["classification"] == "LOCAL"]
        children = {row["dtensor_parent_ordinal"] for row in local_rows}
        logical = [row for row in self.rows if row["shape_source"] == "LOGICAL_DTENSOR_NOT_LOCAL"]
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "implementation": self.implementation,
            "rank": self.rank,
            "torch": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "audited_dispatch_commit": VERSION_AUDITED,
            "official_dispatcher_type": self._dispatch_type.__module__
            + "."
            + self._dispatch_type.__name__,
            "patch_dtensor": self.patch_dtensor,
            "restored": self.restored,
            "active": self.active,
            "exception": self.exception,
            "recorded_rows": len(self.rows),
            "total_invocation_ordinals": self._ordinal,
            "dropped_records": self.dropped_records,
            "observed_plain_local_matrix_ops": len(local_rows),
            "observed_dtensor_internal_local_matrix_ops": sum(
                row["shape_source"] == "DTENSOR_DISPATCH_PLAIN_LOCAL_ATEN" for row in local_rows
            ),
            "logical_dtensor_rows_excluded_from_local_counts": len(logical),
            "logical_dtensor_rows_with_local_children": sum(
                row["cpu_ordinal"] in children for row in logical
            ),
            "unknown_nonlogical_rows": sum(
                row["classification"] not in {"LOCAL", "LOCAL_LINEAR_COMPOSITE"}
                and row["shape_source"] != "LOGICAL_DTENSOR_NOT_LOCAL"
                for row in self.rows
            ),
            "status": "PARTIAL" if self.dropped_records or self.exception else "RECORDED",
            "limits": [
                "Profile-only CPU overhead changes; never use instrumented step times as baseline.",
                "M/N/K/B are local ATen mathematical dimensions, not CUDA grid/tile dimensions.",
                "Linear is composite; actual GEMM shape needs an observed mm/addmm/bmm call.",
                "GPU duration needs exact NVTX/API/correlation/kernel matching.",
                "CPU monotonic time is separate from GPU/NSYS time.",
                "No device item/to_local reads or retained tensor references.",
            ],
        }

    def flush(self) -> None:
        if self.output_dir is None:
            return
        rows = sorted(self.rows, key=lambda row: row["cpu_ordinal"])
        with (self.output_dir / f"gemm_shapes.rank{self.rank}.jsonl").open(
            "w", encoding="utf-8"
        ) as target:
            for row in rows:
                target.write(json.dumps(row, sort_keys=True) + "\n")
        with (self.output_dir / f"gemm_shapes.rank{self.rank}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as target:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(target, fieldnames=fields or ["status"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value) if isinstance(value, (dict, list)) else value
                        for key, value in row.items()
                    }
                )
        (self.output_dir / f"gemm_shapes_manifest.rank{self.rank}.json").write_text(
            json.dumps(self.report(), indent=2) + "\n", encoding="utf-8"
        )


def capture_gemm_shapes(output_dir=None, **kwargs) -> GemmShapeCapture:
    """Return an independent profile-only context manager."""
    return GemmShapeCapture(output_dir, **kwargs)


def cpu_selfcheck(*, allow_fake_cpu_dtensor=False) -> dict[str, Any]:
    """Validate plain FWD/BWD, local CPU DTensor, bounded rows and restoration."""
    generator = torch.Generator().manual_seed(1234)
    left = torch.randn((3, 5), generator=generator, requires_grad=True)
    right = torch.randn((5, 7), generator=generator, requires_grad=True)
    bias = torch.randn(7, generator=generator)
    batch_left, batch_right = torch.randn((2, 3, 5)), torch.randn((2, 5, 7))
    reference = left.detach() @ right.detach()
    dispatcher_type = type(DTensor._op_dispatcher)
    original_dispatch = dispatcher_type.dispatch
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    original_mode = _get_current_dispatch_mode()
    with capture_gemm_shapes(capture_nvtx=False, implementation="cpu_plain") as plain:
        plain.set_step(optimizer_step=0, measured_step=0)
        with plain.phase("FWD"):
            output = left @ right
            torch.addmm(bias, left, right)
            torch.bmm(batch_left, batch_right)
            torch.nn.functional.linear(left, right.t(), bias)
        with plain.phase("BWD"):
            output.sum().backward()
    assert torch.equal(output.detach(), reference)
    assert dispatcher_type.dispatch is original_dispatch and plain.restored
    assert _get_current_dispatch_mode() is original_mode
    assert any(row.get("M") == 3 and row.get("N") == 7 and row.get("K") == 5 for row in plain.rows)
    assert any(row.get("batch_count") == 2 for row in plain.rows)
    assert any(row.get("phase") == "BWD" for row in plain.rows)
    assert not any(
        isinstance(value, torch.Tensor) for row in plain.rows for value in _tensor_values(row)
    )
    try:
        with capture_gemm_shapes(capture_nvtx=False) as failed:
            torch.mm(torch.randn(2, 3), torch.randn(4, 2))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected invalid GEMM to fail")
    assert dispatcher_type.dispatch is original_dispatch and failed.restored
    assert _get_current_dispatch_mode() is original_mode
    with capture_gemm_shapes(capture_nvtx=False, max_records=1) as bounded:
        torch.mm(left.detach(), right.detach())
        torch.mm(left.detach(), right.detach())
    assert len(bounded.rows) == 1 and bounded.dropped_records == 1
    native = {"status": "NOT_RUN", "reason": "CPU Gloo unavailable or another process group active"}
    import torch.distributed as dist

    if dist.is_gloo_available() and not dist.is_initialized():
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate

        store_directory = tempfile.TemporaryDirectory()
        store = dist.FileStore(str(Path(store_directory.name) / "gemm_cpu_gloo.store"), 1)
        backend = "gloo"
        real_gloo_error = None
        try:
            dist.init_process_group("gloo", store=store, rank=0, world_size=1)
        except RuntimeError as error:
            real_gloo_error = f"{type(error).__name__}: {error}"
            native = {"status": "BLOCKED", "reason": real_gloo_error}
            if allow_fake_cpu_dtensor:
                # Test harness only: no collective correctness claim. Replicated
                # world1 mm still runs the installed real local ATen dispatcher.
                from torch._C._distributed_c10d import FakeProcessGroup

                backend = "interview_fake_cpu"
                if backend not in dist.Backend.backend_list:
                    dist.Backend.register_backend(
                        backend,
                        lambda store, rank, world_size, timeout: FakeProcessGroup(rank, world_size),
                        devices=["cpu"],
                    )
                dist.init_process_group(backend, store=store, rank=0, world_size=1)
            else:
                del store
                store_directory.cleanup()
        initialized = dist.is_initialized()
        try:
            if initialized:
                mesh = init_device_mesh("cpu", (1,))
                cpu_left = left.detach().clone().requires_grad_(True)
                cpu_right = right.detach().clone().requires_grad_(True)
                dleft = DTensor.from_local(cpu_left, mesh, [Replicate()])
                dright = DTensor.from_local(cpu_right, mesh, [Replicate()])
                with capture_gemm_shapes(
                    capture_nvtx=False, implementation="cpu_dtensor"
                ) as native_capture:
                    with native_capture.phase("FWD"):
                        native_output = torch.mm(dleft, dright)
                    with native_capture.phase("BWD"):
                        native_output.sum().backward()
                assert torch.equal(native_output.to_local(), reference)
                assert torch.allclose(cpu_left.grad, left.grad)
                assert torch.allclose(cpu_right.grad, right.grad)
                assert dispatcher_type.dispatch is original_dispatch and native_capture.restored
                assert _get_current_dispatch_mode() is original_mode
                local_rows = [
                    row
                    for row in native_capture.rows
                    if row["shape_source"] == "DTENSOR_DISPATCH_PLAIN_LOCAL_ATEN"
                    and row["classification"] == "LOCAL"
                ]
                assert local_rows, "No final plain local DTensor operation captured"
                assert any((row["M"], row["N"], row["K"]) == (3, 7, 5) for row in local_rows)
                assert any(row["phase"] == "BWD" for row in local_rows)
                assert all(
                    row["M"] is None
                    for row in native_capture.rows
                    if row["shape_source"] == "LOGICAL_DTENSOR_NOT_LOCAL"
                )
                native = {
                    "status": "PASS" if backend == "gloo" else "PASS_LOCAL_DISPATCH_ONLY",
                    "backend": backend,
                    "real_gloo_error": real_gloo_error,
                    "report": native_capture.report(),
                    "rows": native_capture.rows,
                    "scope": "CPU world1 local operation; fake backend proves no communication",
                }
                try:
                    with capture_gemm_shapes(capture_nvtx=False) as failed_dtensor:
                        torch.mm(dleft, dright.t())
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("expected invalid DTensor GEMM to fail")
                assert dispatcher_type.dispatch is original_dispatch and failed_dtensor.restored
                assert _get_current_dispatch_mode() is original_mode
                native["exception_restore"] = failed_dtensor.report()
        finally:
            if initialized:
                dist.destroy_process_group()
                del store
                store_directory.cleanup()
    return {
        "status": "PASS",
        "scope": "CPU only; no CUDA/SSH/GPU run",
        "plain_tensor": plain.report(),
        "plain_rows": plain.rows,
        "exception_restore": failed.report(),
        "bounded_capture": bounded.report(),
        "dtensor_world1": native,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-selfcheck", action="store_true")
    parser.add_argument("--allow-fake-cpu-dtensor", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.cpu_selfcheck:
        parser.error(
            "CLI is only a CPU selfcheck; import capture_gemm_shapes for diagnostic traces"
        )
    result = cpu_selfcheck(allow_fake_cpu_dtensor=args.allow_fake_cpu_dtensor)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / "gemm_shapes_cpu_selfcheck.json"
        if destination.exists():
            raise FileExistsError("preserve earlier selfcheck records")
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "dtensor": result["dtensor_world1"]["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
