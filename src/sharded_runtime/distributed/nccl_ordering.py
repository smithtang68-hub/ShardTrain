"""Backend contract for host-ordered, independently completing NCCL collectives."""

import ctypes
import os
from functools import lru_cache

import torch
import torch.distributed as dist


@lru_cache(maxsize=1)
def _launch_order_capabilities():
    if not dist.is_nccl_available():
        raise RuntimeError("stream_ordered NCCL execution requires an NCCL-enabled PyTorch build")
    nccl_version = tuple(torch.cuda.nccl.version())
    if nccl_version < (2, 26, 0):
        raise RuntimeError("stream_ordered requires NCCL >= 2.26 for implicit launch ordering")
    cuda_build = torch.version.cuda
    if cuda_build is None or tuple(map(int, cuda_build.split(".")[:2])) < (12, 3):
        raise RuntimeError("stream_ordered requires a CUDA >= 12.3 PyTorch build")

    # Query the CUDA runtime resolved by its SONAME, as used by standard dynamic
    # PyTorch CUDA wheels, not nvidia-smi's maximum supported CUDA version.
    # These version queries do not create a device context or launch GPU work.
    major = cuda_build.split(".")[0]
    names = (
        (f"cudart64_{major}.dll",) if os.name == "nt" else (f"libcudart.so.{major}", "libcudart.so")
    )
    library = None
    for name in names:
        try:
            library = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if library is None:
        raise RuntimeError("Cannot inspect the CUDA runtime required by stream_ordered")
    versions = {}
    for field, symbol in (
        ("cuda_runtime_version", "cudaRuntimeGetVersion"),
        ("cuda_driver_version", "cudaDriverGetVersion"),
    ):
        query = getattr(library, symbol)
        query.argtypes = [ctypes.POINTER(ctypes.c_int)]
        query.restype = ctypes.c_int
        version = ctypes.c_int()
        result = query(ctypes.byref(version))
        if result != 0:
            raise RuntimeError(f"{symbol} failed with CUDA error {result}")
        versions[field] = version.value
        if version.value < 12030:
            raise RuntimeError("stream_ordered requires CUDA runtime and driver >= 12.3")
    return {"nccl_version": list(nccl_version), "cuda_build": cuda_build, **versions}


def require_nccl_launch_order():
    """Check only: changing this env after communicator initialization is too late."""
    if os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT") != "1":
        raise RuntimeError(
            "stream_ordered requires NCCL_LAUNCH_ORDER_IMPLICIT=1 before creating NCCL "
            "communicators; use WorkerSession or set it before distributed initialization"
        )
    return {
        **_launch_order_capabilities(),
        "NCCL_LAUNCH_ORDER_IMPLICIT": "1",
        "ordering_scope": "host_launch",
        "gpu_completion_chain": False,
    }


def prepare_nccl_launch_order():
    """Prepare the contract before DistributedContext initializes any communicator."""
    if dist.is_initialized():
        raise RuntimeError("Prepare NCCL launch ordering before distributed initialization")
    if os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT", "1") != "1":
        raise RuntimeError("stream_ordered cannot use NCCL_LAUNCH_ORDER_IMPLICIT other than 1")
    _launch_order_capabilities()
    os.environ.setdefault("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    return require_nccl_launch_order()
