"""Bounded, independent pinned-memory experiments (P1-P7).

No ShardTrain operator is imported. CUDA event times and CPU wall times have
separate columns. Validation copies are deliberately outside timed regions.
Run on one idle CUDA device; the root experiment runner owns process timeouts.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

MIB = 1024 * 1024
CASES = ("P1", "P2", "P3", "P4_serial", "P4", "P5", "P6", "P7")


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile, including small independent repeats."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        try:
            fields = Path("/proc/self/statm").read_text().split()
            return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            return None


def host_stats() -> dict[str, Any]:
    function = getattr(torch.cuda.memory, "host_memory_stats", None)
    return dict(function()) if function else {"status": "UNAVAILABLE"}


def json_write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@contextlib.contextmanager
def nvtx(label: str):
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def timed_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(MIB), b""):
            digest.update(part)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={Path.cwd().as_posix()}", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def allocator_configuration() -> dict[str, Any]:
    # PYTORCH_ALLOC_CONF is preferred by this PyTorch version; the CUDA alias
    # is accepted. Actual CUDA API calls must additionally be checked in NSYS.
    raw = os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    settings: dict[str, Any] = {
        "pinned_use_cuda_host_register": False,
        "pinned_num_register_threads": 1,
        "pinned_use_background_threads": False,
    }
    for part in raw.split(","):
        if ":" not in part:
            continue
        key, value = (item.strip() for item in part.split(":", 1))
        if key in settings:
            settings[key] = value == "True" if value in {"True", "False"} else value
    return {
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "configured_or_source_default": settings,
        "actual_cuda_allocation_api": "UNMATCHED until inspected in trace",
    }


def payload(size: int, seed: int, tokens_path: Path | None) -> tuple[torch.Tensor, dict[str, Any]]:
    if tokens_path:
        array = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
        if array.size == 0 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("token npy must be a non-empty integer array")
        # The sweep uses exact byte sizes: repeated bytes of actual tokens,
        # rather than pretending a 128 MiB copy is one model microbatch.
        raw = np.array(array.reshape(-1)[: min(array.size, MIB // array.itemsize)], copy=True)
        byte_data = np.resize(raw.view(np.uint8), size)
        tensor = torch.from_numpy(byte_data)
        return tensor, {
            "kind": "real_token_bytes_repeated_for_transfer_sweep",
            "path": str(tokens_path),
            "npy_dtype": str(array.dtype),
            "npy_shape": list(array.shape),
        }
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.randint(0, 256, (size,), dtype=torch.uint8, generator=generator)
    return tensor, {"kind": "synthetic_byte_buffer", "seed": seed}


class GemmWork:
    """Independent FP16 GEMMs: transferred bytes are not GEMM inputs."""

    def __init__(self, device: torch.device, dimension: int, repeats: int):
        self.left = torch.randn((dimension, dimension), device=device, dtype=torch.float16)
        self.right = torch.randn_like(self.left)
        self.output = torch.empty_like(self.left)
        self.repeats = repeats

    def run(self) -> None:
        for _ in range(self.repeats):
            torch.mm(self.left, self.right, out=self.output)

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        for tensor in (self.left, self.right, self.output):
            tensor.record_stream(stream)


def measure_one(
    case: str,
    pageable: torch.Tensor,
    pinned: torch.Tensor,
    destination: torch.Tensor,
    pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    transfer: torch.cuda.Stream,
    compute: torch.cuda.Stream,
    gemm: GemmWork,
    iteration: int,
    repetition: int,
    warmup: bool,
) -> dict[str, Any]:
    size = pageable.numel()
    current = torch.cuda.current_stream()
    copy_stream = transfer if case in {"P4", "P5", "P7"} else current
    use_gemm = case in {"P4", "P4_serial"}
    gpu_compute_stream = compute if case == "P4" else current
    origin, copy_start, copy_end, compute_start, compute_end, done = [
        timed_event() for _ in range(6)
    ]
    label = (
        f"q01q03/{case}/bytes={size}/repeat={repetition}/iteration={iteration}/warmup={int(warmup)}"
    )
    pin_ms = 0.0
    pin_allocated_new_storage = False
    host_pointer = pinned.data_ptr()
    with nvtx(label):
        wall_start = time.perf_counter_ns()
        if case == "P6":
            pin_start = time.perf_counter_ns()
            with nvtx("q01q03/pin_cpu_copy"):
                source = pageable.pin_memory()
            pin_ms = (time.perf_counter_ns() - pin_start) / 1e6
            pin_allocated_new_storage = source.data_ptr() != pageable.data_ptr()
            host_pointer = source.data_ptr()
        else:
            source = pageable if case == "P1" else pinned
        host_output = None
        pool_slot = -1
        if case == "P5":
            # Fresh D2H destination allocation is not part of the measured copy.
            # P5 destination is the existing pinned buffer, initialized before
            # the iteration. CPU does not read it until its copy_end event.
            host_output = pinned
        if case == "P7":
            pool_slot = iteration % len(pool)
            source, host_output, destination = pool[pool_slot]
            host_pointer = source.data_ptr()
        origin.record(current)
        if copy_stream != current:
            copy_stream.wait_event(origin)
        if gpu_compute_stream != current:
            gpu_compute_stream.wait_event(origin)
        non_blocking = case not in {"P1", "P2"}
        with torch.cuda.stream(copy_stream):
            copy_start.record(copy_stream)
            submit_start = time.perf_counter_ns()
            with nvtx("q01q03/d2h_submit" if case == "P5" else "q01q03/h2d_submit"):
                if case == "P5":
                    assert host_output is not None
                    host_output.copy_(destination, non_blocking=True)
                else:
                    destination.copy_(source, non_blocking=non_blocking)
            submit_ms = (time.perf_counter_ns() - submit_start) / 1e6
            copy_end.record(copy_stream)
            destination.record_stream(copy_stream)
        if use_gemm:
            with torch.cuda.stream(gpu_compute_stream), nvtx("q01q03/independent_gemm_submit"):
                compute_start.record(gpu_compute_stream)
                gemm.run()
                compute_end.record(gpu_compute_stream)
                gemm.record_stream(gpu_compute_stream)
        d2h_start = d2h_end = None
        if case == "P7":
            # Same slot H2D then D2H, ordered on one stream. Slot remains owned
            # until done.synchronize() and exact CPU validation below.
            assert host_output is not None
            d2h_start, d2h_end = timed_event(), timed_event()
            with torch.cuda.stream(copy_stream):
                d2h_start.record(copy_stream)
                with nvtx("q01q03/pool_d2h_submit"):
                    host_output.copy_(destination, non_blocking=True)
                d2h_end.record(copy_stream)
        current.wait_event(d2h_end if d2h_end else copy_end)
        if use_gemm:
            current.wait_event(compute_end)
        done.record(current)
        with nvtx("q01q03/completion_event_wait"):
            done.synchronize()
        end_to_end_ms = (time.perf_counter_ns() - wall_start) / 1e6
    # CPU read and comparison are outside the timed interval and occur only
    # after done, which depends on the D2H completion when applicable.
    with nvtx("q01q03/validate_outside_timing"):
        observed = host_output if host_output is not None else destination.cpu()
        valid = bool(torch.equal(observed, pageable))
        if use_gemm:
            valid = valid and bool(torch.isfinite(gemm.output).all().item())
    if not valid:
        raise AssertionError(f"data mismatch: {label}")
    memcpy_ms = copy_start.elapsed_time(copy_end)
    copy_begin_ms = origin.elapsed_time(copy_start)
    copy_finish_ms = origin.elapsed_time(copy_end)
    compute_begin_ms = origin.elapsed_time(compute_start) if use_gemm else None
    compute_finish_ms = origin.elapsed_time(compute_end) if use_gemm else None
    temporal_overlap_ms = (
        max(0.0, min(copy_finish_ms, compute_finish_ms) - max(copy_begin_ms, compute_begin_ms))
        if use_gemm
        else 0.0
    )
    h2d_bytes = 0 if case == "P5" else size
    d2h_bytes = size if case in {"P5", "P7"} else 0
    d2h_ms = (
        d2h_start.elapsed_time(d2h_end)
        if d2h_start is not None and d2h_end is not None
        else memcpy_ms
        if case == "P5"
        else 0.0
    )
    return {
        "experiment_id": case,
        "repetition": repetition,
        "iteration": iteration,
        "warmup": warmup,
        "buffer_bytes": size,
        "buffer_mib": size / MIB,
        "cpu_pin_ms": pin_ms,
        "cpu_copy_submit_ms": submit_ms,
        "cpu_end_to_end_ms": end_to_end_ms,
        "cuda_memcpy_event_ms": memcpy_ms,
        "cuda_h2d_event_ms": memcpy_ms if h2d_bytes else 0.0,
        "cuda_d2h_event_ms": d2h_ms,
        "cuda_compute_event_ms": compute_start.elapsed_time(compute_end) if use_gemm else 0.0,
        "cuda_total_event_ms": origin.elapsed_time(done),
        "cuda_copy_start_relative_ms": copy_begin_ms,
        "cuda_copy_end_relative_ms": copy_finish_ms,
        "cuda_compute_start_relative_ms": compute_begin_ms,
        "cuda_compute_end_relative_ms": compute_finish_ms,
        "temporal_overlap_ms": temporal_overlap_ms,
        "h2d_bytes": h2d_bytes,
        "d2h_bytes": d2h_bytes,
        "h2d_event_gbps": h2d_bytes / (memcpy_ms * 1e6) if h2d_bytes and memcpy_ms > 0 else None,
        "d2h_event_gbps": d2h_bytes / (d2h_ms * 1e6) if d2h_bytes and d2h_ms > 0 else None,
        "end_to_end_gbps": (h2d_bytes + d2h_bytes) / (end_to_end_ms * 1e6),
        "non_blocking": non_blocking,
        "copy_user_stream_pointer": int(copy_stream.cuda_stream),
        "compute_user_stream_pointer": int(gpu_compute_stream.cuda_stream) if use_gemm else None,
        "host_pointer": host_pointer,
        "pin_allocated_new_storage": pin_allocated_new_storage,
        "pool_slot": pool_slot,
        "rss_bytes": rss_bytes(),
        "data_equal": valid,
        "status": "PASS",
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for key in sorted({(row["experiment_id"], row["buffer_bytes"]) for row in rows}):
        selected = [
            row
            for row in rows
            if (row["experiment_id"], row["buffer_bytes"]) == key and not row["warmup"]
        ]
        if not selected:
            continue
        record: dict[str, Any] = {
            "experiment_id": key[0],
            "buffer_bytes": key[1],
            "samples": len(selected),
        }
        for field in (
            "cpu_pin_ms",
            "cpu_copy_submit_ms",
            "cpu_end_to_end_ms",
            "cuda_memcpy_event_ms",
            "cuda_compute_event_ms",
            "cuda_total_event_ms",
            "temporal_overlap_ms",
            "h2d_event_gbps",
            "d2h_event_gbps",
            "end_to_end_gbps",
        ):
            values = [float(row[field]) for row in selected if row[field] is not None]
            if values:
                record.update(
                    {
                        f"{field}_p50": statistics.median(values),
                        f"{field}_p95": percentile(values, 0.95),
                        f"{field}_min": min(values),
                        f"{field}_max": max(values),
                    }
                )
        record["data_equal"] = all(row["data_equal"] for row in selected)
        result.append(record)
    return result


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    output: Path, summary: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "NOT_RUN", "reason": "matplotlib unavailable"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for case in CASES:
        selected = sorted(
            [row for row in summary if row["experiment_id"] == case],
            key=lambda row: row["buffer_bytes"],
        )
        if not selected:
            continue
        x = [row["buffer_bytes"] / MIB for row in selected]
        axes[0].plot(x, [row["cpu_end_to_end_ms_p50"] for row in selected], marker="o", label=case)
        direction = "d2h" if case == "P5" else "h2d"
        axes[1].plot(
            x, [row[f"{direction}_event_gbps_p50"] for row in selected], marker="o", label=case
        )
        axes[2].plot(x, [row["cpu_copy_submit_ms_p50"] for row in selected], marker="o", label=case)
    for axis in axes:
        axis.set_xlabel("Buffer size (MiB)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Completed end-to-end latency P50 (ms)")
    axes[1].set_ylabel("CUDA event interval bandwidth P50 (GB/s)")
    axes[2].set_ylabel("CPU copy call P50 (ms)")
    axes[0].legend(fontsize=8)
    figure.savefig(output / "transfer_vs_size.png", dpi=160)
    plt.close(figure)
    samples = [
        row for row in rows if not row["warmup"] and row["experiment_id"] in {"P4", "P4_serial"}
    ]
    if samples:
        maximum_size = max(row["buffer_bytes"] for row in samples)
        representatives = []
        for case in ("P4_serial", "P4"):
            selected = [
                row
                for row in samples
                if row["experiment_id"] == case and row["buffer_bytes"] == maximum_size
            ]
            if selected:
                middle = statistics.median(row["cpu_end_to_end_ms"] for row in selected)
                representatives.append(
                    min(selected, key=lambda row: abs(row["cpu_end_to_end_ms"] - middle))
                )
        figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
        for index, row in enumerate(representatives):
            y = index * 3
            for offset, start_key, end_key, color in (
                (0, "cuda_copy_start_relative_ms", "cuda_copy_end_relative_ms", "tab:blue"),
                (1, "cuda_compute_start_relative_ms", "cuda_compute_end_relative_ms", "tab:orange"),
            ):
                axis.broken_barh(
                    [(row[start_key], row[end_key] - row[start_key])],
                    (y + offset, 0.8),
                    facecolors=color,
                )
        labels = [
            f"{row['experiment_id']} H2D" if offset == 0 else f"{row['experiment_id']} GEMM"
            for row in representatives
            for offset in (0, 1)
        ]
        axis.set_yticks(
            [
                index * 3 + offset + 0.4
                for index in range(len(representatives))
                for offset in (0, 1)
            ],
            labels,
        )
        axis.set_xlabel("CUDA event time relative to per-iteration origin (ms)")
        axis.set_title(
            "Measured event-bracket intervals; exact memcpy/kernel intervals require NSYS"
        )
        axis.grid(axis="x", alpha=0.25)
        figure.savefig(output / "transfer_compute_timeline.png", dpi=160)
        plt.close(figure)
        json_write(output / "timeline_representatives.json", representatives)
    return {
        "status": "PASS",
        "transfer_curve": "transfer_vs_size.png",
        "timeline": "transfer_compute_timeline.png",
    }


def token_probe(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    if args.tokens_npy is None:
        return []
    array = np.load(args.tokens_npy, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[0] < args.token_batch_size:
        raise ValueError("token probe requires [blocks, sequence] npy with sufficient blocks")
    cpu_tokens = torch.from_numpy(
        np.array(array[: args.token_batch_size], dtype=np.int64, copy=True)
    )
    dataset = torch.utils.data.TensorDataset(cpu_tokens)
    rows = []
    for pin in (False, True):
        # num_workers=0 intentionally isolates fetch+collate+pin without worker
        # startup, interprocess transport, or background queue prefetch.
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.token_batch_size, pin_memory=pin, num_workers=0
        )
        for index in range(args.iterations):
            torch.cuda.synchronize(device)
            fetch_start = time.perf_counter_ns()
            with nvtx("q01q03/token_dataloader_fetch_and_pin"):
                batch = next(iter(loader))[0]
            fetch_ms = (time.perf_counter_ns() - fetch_start) / 1e6
            destination = torch.empty_like(batch, device=device)
            start, end = timed_event(), timed_event()
            transfer_start = time.perf_counter_ns()
            start.record()
            submit_start = time.perf_counter_ns()
            with nvtx("q01q03/token_h2d"):
                destination.copy_(batch, non_blocking=pin)
            submit_ms = (time.perf_counter_ns() - submit_start) / 1e6
            end.record()
            end.synchronize()
            completed_ms = (time.perf_counter_ns() - transfer_start) / 1e6
            valid = torch.equal(destination.cpu(), cpu_tokens)
            if not valid:
                raise AssertionError("real token transfer mismatch")
            rows.append(
                {
                    "iteration": index,
                    "pin_memory": pin,
                    "is_pinned": batch.is_pinned(),
                    "shape": list(batch.shape),
                    "dtype": str(batch.dtype),
                    "bytes": batch.numel() * batch.element_size(),
                    "cpu_fetch_collate_pin_ms": fetch_ms,
                    "cpu_copy_submit_ms": submit_ms,
                    "cuda_memcpy_event_ms": start.elapsed_time(end),
                    "cpu_completed_transfer_ms": completed_ms,
                    "data_equal": bool(valid),
                    "status": "PASS",
                }
            )
    return rows


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=1234)
    result.add_argument("--iterations", type=int, default=20)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--sizes-mib", type=int, nargs="+", default=[4, 16, 64, 128])
    result.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--pool-slots", type=int, default=2)
    result.add_argument("--gemm-dimension", type=int, default=2048)
    result.add_argument("--gemm-repeats", type=int, default=8)
    result.add_argument("--tokens-npy", type=Path)
    result.add_argument("--token-batch-size", type=int, default=2)
    result.add_argument(
        "--profile",
        action="store_true",
        help="CUDA profiler API capture range; NVTX is always emitted",
    )
    result.add_argument("--record-memory", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="validate bounded configuration without initializing CUDA",
    )
    return result


def validate_args(args: argparse.Namespace) -> None:
    if (
        not 1 <= args.iterations <= 500
        or not 0 <= args.warmup <= 100
        or not 1 <= args.repetitions <= 10
    ):
        raise ValueError("iterations 1..500, warmup 0..100, repetitions 1..10 required")
    if any(not 1 <= size <= 256 for size in args.sizes_mib):
        raise ValueError("each transfer buffer must be 1..256 MiB")
    if not 1 <= args.pool_slots <= 4:
        raise ValueError("bounded pinned pool requires 1..4 slots")
    if not 128 <= args.gemm_dimension <= 4096 or not 1 <= args.gemm_repeats <= 128:
        raise ValueError("bounded GEMM dimension 128..4096 and repeats 1..128 required")
    if args.token_batch_size < 1:
        raise ValueError("token_batch_size must be positive")
    if (
        args.profile
        and len(args.cases)
        * len(args.sizes_mib)
        * args.repetitions
        * (args.warmup + args.iterations)
        > 128
    ):
        raise ValueError(
            "representative profile must contain at most 128 transfer iterations; "
            "reduce cases/sizes/repetitions/iterations"
        )
    if args.tokens_npy and not args.tokens_npy.is_file():
        raise FileNotFoundError(args.tokens_npy)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    validate_args(args)
    if args.dry_run:
        print(
            json.dumps({"status": "CONFIG_VALID", "cases": args.cases, "sizes_mib": args.sizes_mib})
        )
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("choose a fresh output directory to preserve prior evidence")
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    json_write(args.output_dir / "config.json", configuration)
    json_write(args.output_dir / "command.json", {"argv": sys.argv, "cwd": str(Path.cwd())})
    if not torch.cuda.is_available():
        json_write(
            args.output_dir / "summary.json", {"status": "NOT_RUN", "reason": "CUDA unavailable"}
        )
        return 2
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.cuda.init()
    transfer, compute = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    gemm = GemmWork(device, args.gemm_dimension, args.gemm_repeats)
    gemm.run()
    torch.cuda.synchronize(device)
    json_write(
        args.output_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "cuda_build": torch.version.cuda,
            "git_commit": git_commit(),
            "device": str(device),
            "device_properties": str(torch.cuda.get_device_properties(device)),
            "allocator": allocator_configuration(),
            "rss_bytes": rss_bytes(),
            "source_file_sha256": file_sha256(Path(__file__)),
            "tokens_sha256": file_sha256(args.tokens_npy) if args.tokens_npy else None,
            "user_streams": {
                "compute": int(compute.cuda_stream),
                "transfer": int(transfer.cuda_stream),
                "default": int(torch.cuda.current_stream().cuda_stream),
            },
        },
    )
    if args.record_memory:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=80000
        )
        torch.cuda.memory._dump_snapshot(str(args.output_dir / "memory_start.pickle"))
    rows: list[dict[str, Any]] = []
    allocations = []
    profiled = False
    status, failure = "PASS", None
    try:
        if args.profile:
            torch.cuda.cudart().cudaProfilerStart()
            profiled = True
        for size_mib in args.sizes_mib:
            size = size_mib * MIB
            pageable, data_origin = payload(size, args.seed + size_mib, args.tokens_npy)
            alloc_start = time.perf_counter_ns()
            with nvtx(f"q01q03/pinned_preallocate/bytes={size}"):
                pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            alloc_ms = (time.perf_counter_ns() - alloc_start) / 1e6
            fill_start = time.perf_counter_ns()
            pinned.copy_(pageable)
            fill_ms = (time.perf_counter_ns() - fill_start) / 1e6
            # Explicitly demonstrate the already-pinned identity path outside
            # all performance timings.
            pinned_again = pinned.pin_memory()
            identity = pinned_again.data_ptr() == pinned.data_ptr()
            del pinned_again
            destination = torch.empty_like(pageable, device=device)
            destination.copy_(pageable)
            pool = []
            pool_start = time.perf_counter_ns()
            if "P7" in args.cases:
                for _ in range(args.pool_slots):
                    source = torch.empty_like(pageable, pin_memory=True)
                    source.copy_(pageable)
                    output = torch.empty_like(source, pin_memory=True)
                    gpu = torch.empty_like(pageable, device=device)
                    pool.append((source, output, gpu))
                del source, output, gpu
            torch.cuda.synchronize(device)
            allocations.append(
                {
                    "buffer_bytes": size,
                    "pinned_preallocation_cpu_ms": alloc_ms,
                    "pinned_initial_fill_cpu_ms": fill_ms,
                    "already_pinned_same_pointer": identity,
                    "pool_setup_cpu_ms": (time.perf_counter_ns() - pool_start) / 1e6,
                    "explicit_pool_pinned_bytes": len(pool) * size * 2,
                    "payload_origin": data_origin,
                    "host_stats_after_setup": host_stats(),
                    "rss_bytes_after_setup": rss_bytes(),
                }
            )
            for repetition in range(args.repetitions):
                # Rotate group order to avoid systematically favoring late
                # cache/clock state. Keep P4 and its serial control identical.
                order = (
                    args.cases[repetition % len(args.cases) :]
                    + args.cases[: repetition % len(args.cases)]
                )
                for case in order:
                    for index in range(args.warmup + args.iterations):
                        if case == "P5":
                            pinned.zero_()  # CPU mutation only after prior done.
                        rows.append(
                            measure_one(
                                case,
                                pageable,
                                pinned,
                                destination,
                                pool,
                                transfer,
                                compute,
                                gemm,
                                index,
                                repetition,
                                index < args.warmup,
                            )
                        )
                    # P5 changed pinned destination to the original payload.
                    allocations[-1].setdefault("case_host_stats", []).append(
                        {
                            "case": case,
                            "repetition": repetition,
                            "host_stats": host_stats(),
                            "rss_bytes": rss_bytes(),
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "case": case,
                                "size_mib": size_mib,
                                "repeat": repetition,
                                "status": "PASS",
                            }
                        ),
                        flush=True,
                    )
            torch.cuda.synchronize(device)
            del pool, destination, pinned, pageable
            gc.collect()
            # Private, version-checked host cache purge is between size groups,
            # never a timed transfer; avoids cumulative pinned cache growth.
            purge = getattr(torch._C, "_host_emptyCache", None)
            if purge:
                purge()
            allocations[-1]["host_stats_after_release"] = host_stats()
        real_tokens = token_probe(args, device)
        json_write(args.output_dir / "token_probe.json", real_tokens)
        csv_write(args.output_dir / "token_probe.csv", real_tokens)
    except Exception:
        status, failure = "FAIL", traceback.format_exc()
        (args.output_dir / "failure.log").write_text(failure, encoding="utf-8")
    finally:
        torch.cuda.synchronize(device)
        if args.record_memory:
            torch.cuda.memory._dump_snapshot(str(args.output_dir / "memory_end.pickle"))
            torch.cuda.memory._record_memory_history(enabled=None)
        if profiled:
            torch.cuda.cudart().cudaProfilerStop()
        csv_write(args.output_dir / "samples.csv", rows)
        json_write(args.output_dir / "samples.json", rows)
        json_write(args.output_dir / "allocations.json", allocations)
        groups = summarize(rows)
        csv_write(args.output_dir / "groups.csv", groups)
        plots = make_plots(args.output_dir, groups, rows)
        json_write(
            args.output_dir / "summary.json",
            {
                "status": status,
                "failure_reason": failure,
                "groups": groups,
                "plots": plots,
                "allocator_stats_final": host_stats(),
                "rss_bytes_final": rss_bytes(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "notes": [
                    "CUDA event intervals bracket copies; exact copy-engine intervals need NSYS.",
                    "P4 event-bracket overlap needs NSYS confirmation.",
                    "P4_serial/P4 use identical GEMMs independent of the copied input.",
                    "CPU validation and D2H validation copy are outside benchmark timing.",
                    "P6 pin calls include CPU copy and may reuse the pinned allocator cache.",
                    "P7 fixed pool reuses only after H2D+D2H event completion and CPU validation.",
                    "First pinned allocation per size/process need not be OS-cold.",
                    "num_workers=0 token probe times fetch+collate+pin without worker startup.",
                ],
            },
        )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
