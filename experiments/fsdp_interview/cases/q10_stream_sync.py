"""Versioned FSDP stream-source audit plus exact NVTX/API correlation validation.

This audits installed PyTorch rather than assuming all FSDP versions use the same
stream layout. Source mechanisms and measured trace findings are separate fields.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections import defaultdict
from pathlib import Path

SOURCE_CASES = {
    "fsdp1_runtime": {
        "module": "torch.distributed.fsdp._runtime_utils",
        "repository_path": "torch/distributed/fsdp/_runtime_utils.py",
        "symbols": {
            "root_stream_initialization": "def _init_streams(",
            "compute_stream": "state._default_stream = state._device_handle.current_stream()",
            "unshard_stream": "state._unshard_stream = state._device_handle.Stream",
            "pre_unshard_stream": "state._pre_unshard_stream = state._device_handle.Stream",
            "post_backward_stream": "state._post_backward_stream = state._device_handle.Stream",
            "all_reduce_stream": "state._all_reduce_stream = (",
            "unshard_function": "def _unshard(",
            "cpu_rate_limiter": "event.synchronize()",
            "free_event_queue": "state._free_event_queue.dequeue_if_needed()",
            "gradient_producer_wait": "state._post_backward_stream.wait_stream(",
            "allocator_record_stream": "_no_dispatch_record_stream(",
        },
        "mechanism": (
            "Root shares compute/unshard/pre-unshard/post-backward streams. "
            "HSDP has a separate all-reduce stream; ordinary sharding aliases it to compute. "
            "limit_all_gathers may synchronize a CPU event from the free-event queue. "
            "Gradient producer waits and record_stream protect post-backward reads."
        ),
    },
    "fsdp2_parameter_group": {
        "module": "torch.distributed.fsdp._fully_shard._fsdp_param_group",
        "repository_path": "torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py",
        "symbols": {
            "comm_context": "class FSDPCommContext:",
            "copy_in_stream": "self.all_gather_copy_in_stream =",
            "all_gather_stream": "self.all_gather_stream =",
            "reduce_scatter_stream": "self.reduce_scatter_stream =",
            "all_reduce_stream": "self.all_reduce_stream =",
            "select_ag_streams": "def get_all_gather_streams(",
            "wait_for_unshard": "def wait_for_unshard(",
            "all_gather_buffer_state": "self.comm_ctx.all_gather_state = AllGatherState(",
            "reduce_scatter_buffer_state": (
                "self.comm_ctx.reduce_scatter_state = ReduceScatterState("
            ),
            "implicit_backward_prefetch": "def _backward_prefetch(",
            "explicit_prefetch_entry": "def _prefetch_unshard(",
            "backward_finalization": "def finalize_backward(",
        },
        "mechanism": (
            "Shared CommContext has copy-in, AG, RS and HSDP all-reduce streams. "
            "Forward/pre-backward non-async unshard uses separate AG streams. "
            "AllGatherState and ReduceScatterState retain tensor references and events. "
            "Reverse post-forward order drives implicit backward prefetch; "
            "explicit target lists are also supported."
        ),
    },
    "fsdp2_collectives": {
        "module": "torch.distributed.fsdp._fully_shard._fsdp_collectives",
        "repository_path": "torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py",
        "symbols": {
            "all_gather_packing": "def foreach_all_gather(",
            "copy_in_op": "def all_gather_copy_in_cuda(",
            "copy_in_to_ag_wait": "all_gather_stream.wait_stream(all_gather_copy_in_stream)",
            "all_gather_api": "return dist.all_gather_into_tensor(",
            "all_gather_copy_out": "def foreach_all_gather_copy_out(",
            "gradient_reduction": "def foreach_reduce(",
            "reduce_scatter_api": "return dist.reduce_scatter_tensor(",
        },
        "mechanism": (
            "Parameter groups pack their communication tensor; per-parameter DTensor "
            "representation does not imply one NCCL kernel per parameter. "
            "Copy-in→AG and AG→copy-out synchronization are explicit. "
            "Tensor reference ownership and events protect shared buffers across streams."
        ),
    },
    "fsdp2_state": {
        "module": "torch.distributed.fsdp._fully_shard._fsdp_state",
        "repository_path": "torch/distributed/fsdp/_fully_shard/_fsdp_state.py",
        "symbols": {
            "root_forward": "def _root_pre_forward(",
            "forward": "def _pre_forward(",
            "last_all_gather_buffer_cleanup": "self._comm_ctx.all_gather_state = None",
            "pre_backward": "def _pre_backward(",
            "backward_callback": "def _root_post_backward_final_callback(",
        },
        "mechanism": (
            "Root coordinates state hooks. The final forward state releases the deferred "
            "AG result with stream-event protection. Pre-backward chooses default or explicit "
            "prefetch, and the root backward callback finalizes reductions."
        ),
    },
}


def source_audit():
    import torch

    version = str(torch.__version__)
    commit = torch.version.git_version
    files = {}
    for label, case in SOURCE_CASES.items():
        module = importlib.import_module(case["module"])
        path = Path(module.__file__)
        data = path.read_bytes()
        lines = data.decode("utf-8").splitlines()
        markers = {}
        for name, needle in case["symbols"].items():
            matches = [index + 1 for index, line in enumerate(lines) if needle in line]
            markers[name] = {
                "needle": needle,
                "line_numbers": matches,
                "status": "FOUND" if matches else "NOT_FOUND",
                "source_links": [
                    f"https://github.com/pytorch/pytorch/blob/{commit}/{case['repository_path']}#L{line}"
                    for line in matches
                ],
            }
        files[label] = {
            "installed_path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "repository_path": case["repository_path"],
            "markers": markers,
            "mechanism": case["mechanism"],
            "reference_tag": f"https://github.com/pytorch/pytorch/blob/v2.9.1/{case['repository_path']}",
        }
    return {
        "torch_version": version,
        "torch_git_commit": commit,
        "version_scope": "PyTorch 2.9.1"
        if version.split("+", 1)[0] == "2.9.1"
        else "UNQUALIFIED_VERSION",
        "files": files,
        "nccl_backend_source": {
            "version": "v2.9.1",
            "verification": "reviewed official source; check actual trace correlation separately",
            "source": (
                "https://github.com/pytorch/pytorch/blob/v2.9.1/"
                "torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3371-L3377"
            ),
            "sync_collective_return": (
                "https://github.com/pytorch/pytorch/blob/v2.9.1/"
                "torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3505"
            ),
            "mechanism": (
                "Ordinary collective asyncOp=False submits on current CUDA stream and returns "
                "nullptr. asyncOp=True uses the backend's ncclStreams_ and syncStream input "
                "dependency; Work.wait joins its completion to the caller stream. "
                "P2P and coalescing have separate paths and are outside this qualification."
            ),
        },
        "evidence_scope": (
            "Installed source markers; no measured stream IDs or performance assertions"
        ),
        "physical_stream_rule": (
            "Use exact NVTX range→CUDA API correlation→GPU kernel stream. "
            "Never infer kernel stream from Python stream count."
        ),
        "comparison_scope": (
            "FSDP1 vs FSDP2 strict timing comparison must be FSDP-only with matching "
            "model/precision/input; ShardTrain FSDP+TP/SP is mechanism-only against these runs"
        ),
    }


def physical_stream_audit(events, matched_kernels):
    """joined kernels MUST carry exact launch API's enclosing nvtx_label.

    A kernel-name substring or nearest timestamp is deliberately insufficient.
    Each row should have nvtx_label, correlation_id, stream_id, start_ns, end_ns.
    Multiple kernels per API/range are legal; all stream IDs are preserved.
    """
    by_label = defaultdict(list)
    for kernel in matched_kernels:
        if kernel.get("nvtx_label") and kernel.get("correlation_id") is not None:
            by_label[kernel["nvtx_label"]].append(kernel)
    result = []
    for event in events:
        label = event.get("nvtx_label")
        kernels = by_label.get(label, [])
        result.append(
            {
                "rank": event["rank"],
                "runtime_op_id": event["runtime_op_id"],
                "kind": event["collective_type"],
                "unit_id": event["unit_id"],
                "phase": event["phase"],
                "channel": event["channel"],
                "logical_stream_ptr": event.get("logical_stream_ptr"),
                "actual_submission_stream_ptr": event.get("actual_submission_stream_ptr"),
                "execution_mode": event.get("execution_mode"),
                "nvtx_label": label,
                "status": "MATCHED" if kernels else "UNMATCHED",
                "nsys_stream_ids": sorted({row["stream_id"] for row in kernels}),
                "correlation_ids": sorted({row["correlation_id"] for row in kernels}),
                "gpu_kernel_count": len(kernels),
                "note": (
                    "Pointer and NSYS stream integer are different namespaces; "
                    "API correlation establishes association"
                ),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = source_audit()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "stream_source_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "torch_version": result["torch_version"],
                "version_scope": result["version_scope"],
                "missing_markers": [
                    f"{label}/{name}"
                    for label, case in result["files"].items()
                    for name, item in case["markers"].items()
                    if item["status"] != "FOUND"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
