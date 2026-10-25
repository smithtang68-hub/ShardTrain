"""Isolated Linux stage ownership and telemetry, reused from validated v5 runner.

Provenance: runs/qwen25_native_capacity_20261013/remote_driver_v5.py;
PID/starttime + pidfd cleanup never targets unrelated processes.
"""

import ctypes
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

BASE = Path(
    os.environ.get(
        "SHARDTRAIN_INTERVIEW_RUN_ROOT", "/root/shardtrain_fsdp_interview/20261014_run01"
    )
)
PROJECT = BASE / "project"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def base_env():
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(PROJECT / "src"),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        NCCL_LAUNCH_ORDER_IMPLICIT="1",
        NCCL_DEBUG="WARN",
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
        TOKENIZERS_PARALLELISM="false",
    )
    records = list(BASE.glob("source_manifest_v*.json"))
    if records:
        record = max(records, key=lambda p: int(p.stem.rsplit("v", 1)[1]))
        frozen = json.loads(record.read_text())
        env["SHARDTRAIN_SOURCE_GIT_COMMIT"] = frozen["git_commit"]
        env["SHARDTRAIN_SOURCE_MANIFEST"] = str(record)
        env["SHARDTRAIN_SOURCE_MANIFEST_SHA256"] = hashlib.sha256(record.read_bytes()).hexdigest()
    else:
        env["SHARDTRAIN_SOURCE_GIT_COMMIT"] = os.environ.get(
            "SHARDTRAIN_SOURCE_GIT_COMMIT", "recorded_in_source_manifest"
        )
    return env


def state(status, **kwargs):
    write_json(BASE / "progress.json", {"status": status, "unix": time.time(), **kwargs})


common = SimpleNamespace(
    base_env=base_env, digest=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
)
support = SimpleNamespace(state=state)


class Memory(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]


class Nvml:
    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.check(self.lib.nvmlInit_v2())
        self.handles, self.uuids = [], []
        for index in range(4):
            handle = ctypes.c_void_p()
            self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)))
            value = ctypes.create_string_buffer(96)
            self.check(self.lib.nvmlDeviceGetUUID(handle, value, len(value)))
            self.handles.append(handle)
            self.uuids.append(value.value.decode())

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML error code {code}")

    def sample(self):
        rows = []
        for index, (handle, uuid) in enumerate(zip(self.handles, self.uuids, strict=True)):
            memory = Memory()
            self.check(self.lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
            telemetry = {}
            for name, api, selector in [
                ("sm_clock_mhz", "nvmlDeviceGetClockInfo", 1),
                ("memory_clock_mhz", "nvmlDeviceGetClockInfo", 2),
                ("temperature_c", "nvmlDeviceGetTemperature", 0),
                ("power_mw", "nvmlDeviceGetPowerUsage", None),
                ("power_limit_mw", "nvmlDeviceGetPowerManagementLimit", None),
            ]:
                value = ctypes.c_uint()
                function = getattr(self.lib, api, None)
                code = (
                    (
                        function(handle, ctypes.byref(value))
                        if selector is None
                        else function(handle, selector, ctypes.byref(value))
                    )
                    if function
                    else -1
                )
                telemetry[name] = value.value if code == 0 else None
            rows.append(
                {
                    "index": index,
                    "uuid": uuid,
                    **telemetry,
                    **{
                        name + "_bytes": getattr(memory, name) for name in ("total", "free", "used")
                    },
                }
            )
        return {"unix": time.time(), "monotonic": time.monotonic(), "devices": rows}


def process_identity(pid):
    """Read Linux stat without scanning unrelated processes (starttime is field 22)."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {
        "pid": pid,
        "state": fields[0],
        "ppid": int(fields[1]),
        "process_group": int(fields[2]),
        "session": int(fields[3]),
        "starttime_ticks": int(fields[19]),
    }


def process_children(pid):
    """Children belong to individual Linux threads; inspect every thread of this PID."""
    children = set()
    try:
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
    except (FileNotFoundError, ProcessLookupError):
        return children
    for task in tasks:
        try:
            children.update(map(int, (task / "children").read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return children


class PidfdApi:
    """PID-stable Linux signalling independent of the Python build's os exports."""

    ABI_HEADER = Path("/usr/include/x86_64-linux-gnu/asm/unistd_64.h")
    EXPECTED_NR = {"pidfd_open": 434, "pidfd_send_signal": 424}

    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.open_wrapper = getattr(self.libc, "pidfd_open", None)
        self.send_wrapper = getattr(self.libc, "pidfd_send_signal", None)
        self.abi = {
            "system": platform.system(),
            "machine": platform.machine(),
            "pointer_bits": 8 * ctypes.sizeof(ctypes.c_void_p),
            "long_bits": 8 * ctypes.sizeof(ctypes.c_long),
            "pidfd_open_method": "libc" if self.open_wrapper is not None else "syscall",
            "pidfd_send_signal_method": "libc" if self.send_wrapper is not None else "syscall",
        }
        if self.open_wrapper is not None:
            self.open_wrapper.argtypes = [ctypes.c_int, ctypes.c_uint]
            self.open_wrapper.restype = ctypes.c_int
        if self.send_wrapper is not None:
            self.send_wrapper.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint,
            ]
            self.send_wrapper.restype = ctypes.c_int
        if self.open_wrapper is None or self.send_wrapper is None:
            if (
                platform.system() != "Linux"
                or platform.machine().lower() not in ("x86_64", "amd64")
                or ctypes.sizeof(ctypes.c_void_p) != 8
                or ctypes.sizeof(ctypes.c_long) != 8
            ):
                raise RuntimeError("pidfd syscall fallback is verified only for Linux x86_64")
            header = self.ABI_HEADER.read_text(encoding="utf-8")
            observed = {}
            for name, expected in self.EXPECTED_NR.items():
                matches = re.findall(
                    r"^\s*#\s*define\s+__NR_" + name + r"\s+(\d+)\s*$", header, re.MULTILINE
                )
                if len(matches) != 1 or int(matches[0]) != expected:
                    raise RuntimeError(f"Remote ABI header does not verify __NR_{name}={expected}")
                observed[name] = int(matches[0])
            self.abi.update(
                {
                    "verified_header": str(self.ABI_HEADER),
                    "header_sha256": common.digest(self.ABI_HEADER),
                    "verified_syscall_numbers": observed,
                }
            )
            self.syscall = self.libc.syscall
            self.syscall.argtypes = [ctypes.c_long]  # C variadic arguments are typed below.
            self.syscall.restype = ctypes.c_long

    @staticmethod
    def checked(result, operation):
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, operation + ": " + os.strerror(code))
        return int(result)

    def open(self, pid):
        ctypes.set_errno(0)
        result = (
            self.open_wrapper(pid, 0)
            if self.open_wrapper is not None
            else self.syscall(
                ctypes.c_long(self.EXPECTED_NR["pidfd_open"]), ctypes.c_long(pid), ctypes.c_ulong(0)
            )
        )
        return self.checked(result, "pidfd_open")

    def send_signal(self, fd, sig):
        ctypes.set_errno(0)
        result = (
            self.send_wrapper(fd, int(sig), None, 0)
            if self.send_wrapper is not None
            else (
                self.syscall(
                    ctypes.c_long(self.EXPECTED_NR["pidfd_send_signal"]),
                    ctypes.c_long(fd),
                    ctypes.c_long(int(sig)),
                    ctypes.c_void_p(),
                    ctypes.c_ulong(0),
                )
            )
        )
        self.checked(result, "pidfd_send_signal")


class OwnedProcessTree:
    """Track only stage descendants, including workers that create their own session.

    A temporary child subreaper retains orphan descendants when their launcher exits.
    The driver must have no pre-existing children, and launches only this stage while
    tracking. Thus newly adopted children also have a proven stage ancestry.
    All signals use pidfds, bound after two matching PID/starttime observations.
    No process-group-wide signal can reach an unrelated or PID-reused process.
    """

    def __init__(self):
        self.driver_pid = os.getpid()
        self.driver_key = self.key(process_identity(self.driver_pid))
        self.lock = threading.RLock()
        self.known, self.pidfds = {}, {}
        self.signals, self.reaped, self.errors = [], [], []
        self.root_pid, self.root_key, self.started = None, None, False
        self.closed = False
        if process_children(self.driver_pid):
            raise RuntimeError(
                "Stage driver already has children; exclusive ownership is unavailable"
            )
        self.pidfd = PidfdApi()
        probe = self.pidfd.open(self.driver_pid)
        os.close(probe)
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.prctl.restype = ctypes.c_int
        previous = ctypes.c_int()
        # Constants from Linux include/uapi/linux/prctl.h: SET=36, GET=37.
        if self.libc.prctl(37, ctypes.byref(previous), 0, 0, 0):
            raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
        self.previous_subreaper = previous.value
        if self.libc.prctl(36, ctypes.c_ulong(1), 0, 0, 0):
            raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")

    @staticmethod
    def key(row):
        return row["pid"], row["starttime_ticks"]

    def register(self, pid, parent_key, origin):
        parent_pid = parent_key[0]
        parent = process_identity(parent_pid)
        if parent is None or self.key(parent) != parent_key:
            return None
        row = process_identity(pid)
        if row is None or row["ppid"] != parent_pid:
            return None
        key = self.key(row)
        if key not in self.known:
            try:
                fd = self.pidfd.open(pid)
            except ProcessLookupError:
                return None
            observed = process_identity(pid)
            parent_after = process_identity(parent_pid)
            if (
                observed is None
                or self.key(observed) != key
                or observed["ppid"] != parent_pid
                or parent_after is None
                or self.key(parent_after) != parent_key
            ):
                os.close(fd)
                return None
            self.pidfds[key] = fd
            self.known[key] = {
                **observed,
                "discovered_from_pid": parent_pid,
                "discovered_from_starttime_ticks": parent_key[1],
                "discovery_origin": origin,
                "first_seen_monotonic": time.monotonic(),
            }
        return key

    def start(self, pid):
        with self.lock:
            self.root_pid, self.started = pid, True
            self.root_key = self.register(pid, self.driver_key, "Popen launcher")
            self.refresh()

    def refresh(self):
        with self.lock:
            if not self.started:
                return
            pending = []
            # This catches orphans reparented between two 100ms samples.
            for pid in process_children(self.driver_pid):
                key = self.register(pid, self.driver_key, "exclusive driver/subreaper child")
                if key is not None:
                    pending.append(key)
            pending.extend(self.known)
            scanned = set()
            while pending:
                key = pending.pop()
                if key in scanned:
                    continue
                scanned.add(key)
                row = process_identity(key[0])
                if row is None or self.key(row) != key:
                    continue
                for pid in process_children(row["pid"]):
                    child = self.register(pid, key, "tracked descendant thread children")
                    if child is not None:
                        pending.append(child)

    def active(self):
        with self.lock:
            self.refresh()
            rows = []
            for key in self.known:
                row = process_identity(key[0])
                if row is not None and self.key(row) == key and row["state"] not in ("Z", "X"):
                    rows.append(row)
            return rows

    def reap_adopted(self):
        with self.lock:
            for key in self.known:
                if key == self.root_key:
                    continue  # Popen owns its launcher's wait status.
                row = process_identity(key[0])
                if row is None or self.key(row) != key or row["ppid"] != self.driver_pid:
                    continue
                if row["state"] != "Z":
                    continue
                try:
                    pid, status = os.waitpid(row["pid"], os.WNOHANG)
                except ChildProcessError:
                    continue
                if pid:
                    self.reaped.append(
                        {"pid": pid, "starttime_ticks": key[1], "wait_status": status}
                    )

    def signal_owned(self, row, sig):
        with self.lock:
            key = self.key(row)
            current = process_identity(row["pid"])
            if current is None or self.key(current) != key or current["state"] in ("Z", "X"):
                return
            try:
                self.pidfd.send_signal(self.pidfds[key], sig)
                self.signals.append(
                    {
                        "pid": row["pid"],
                        "starttime_ticks": key[1],
                        "signal": int(sig),
                        "monotonic": time.monotonic(),
                    }
                )
            except ProcessLookupError:
                pass
            except OSError as error:
                self.errors.append(
                    {
                        "pid": row["pid"],
                        "starttime_ticks": key[1],
                        "signal": int(sig),
                        "error": repr(error),
                    }
                )

    def cleanup(self):
        for sig, seconds in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
            sent = set()
            deadline = time.monotonic() + seconds
            while True:
                rows = self.active()
                self.reap_adopted()
                if not rows:
                    return []
                for row in rows:
                    key = self.key(row)
                    if key not in sent:
                        self.signal_owned(row, sig)
                        sent.add(key)
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
        return self.active()

    def wait_for_owned_exit(self, deadline):
        """Track and reap owned descendants until exit/deadline, without signalling."""
        while True:
            rows = self.active()
            self.reap_adopted()
            remaining = deadline - time.monotonic()
            if not rows or remaining <= 0:
                return rows
            time.sleep(min(0.1, remaining))

    def report(self):
        with self.lock:
            return {
                "driver_pid": self.driver_pid,
                "driver_starttime_ticks": self.driver_key[1],
                "launcher_pid": self.root_pid,
                "previous_subreaper": self.previous_subreaper,
                "stage_subreaper": True,
                "discovery": "all-thread /proc/PID/task/TID/children of tracked descendants; "
                "new exclusive driver children include adopted orphans",
                "signal_method": "pidfd; PID/starttime identity verified "
                "before registration/signals",
                "pidfd_abi": self.pidfd.abi,
                "known_processes": list(self.known.values()),
                "signals": list(self.signals),
                "adopted_exit_statuses": list(self.reaped),
                "errors": list(self.errors),
            }

    def close(self):
        with self.lock:
            if self.closed:
                return
            # Refresh first: late adopted zombies must be registered before waitpid.
            active = self.active()
            self.reap_adopted()
            if process_children(self.driver_pid):
                active = self.active()
                self.reap_adopted()
            if active or process_children(self.driver_pid):
                raise RuntimeError("Owned stage children remain; keep subreaper until driver exits")
            if self.libc.prctl(36, ctypes.c_ulong(self.previous_subreaper), 0, 0, 0):
                raise OSError(ctypes.get_errno(), "Restoring child subreaper failed")
            for fd in self.pidfds.values():
                os.close(fd)
            self.closed = True


def run_stage(name, command, *, timeout=1800, extra_env=None):
    directory = BASE / "logs" / name
    directory.mkdir(parents=True, exist_ok=False)
    environment = common.base_env()
    environment.update(extra_env or {})
    recorded = {
        key: value
        for key, value in environment.items()
        if key.startswith(("CUDA_", "NCCL_", "TORCH_", "OMP_", "PYTHON", "SHARDTRAIN_"))
    }
    write_json(
        directory / "command.json",
        {"argv": list(map(str, command)), "cwd": str(PROJECT), "environment_overrides": recorded},
    )
    support.state("running", stage=name)
    started, samples, errors, stop = time.time(), [], [], threading.Event()
    nvml, tracker, thread, process = None, None, None, None
    timed_out, cleanup_needed, normal_exit_residual = False, False, False
    pending_error, remaining_owned, before_cleanup, ownership_errors = None, [], [], []
    active_at_launcher_exit = []
    grace_deadline = None
    normal_exit_grace = {
        "maximum_seconds": 30.0,
        "started_monotonic": None,
        "deadline_monotonic": None,
        "attempts": [],
        "signals_during_grace": False,
    }

    def settle_normal_exit(rows):
        nonlocal grace_deadline
        # Timeout, nonzero exit, driver/tracking/telemetry exceptions retain the
        # immediate owned-only cleanup contract. Only healthy normal exit gets grace.
        if not (
            process is not None
            and process.returncode == 0
            and not timed_out
            and pending_error is None
            and not ownership_errors
            and not errors
        ):
            return rows
        started_wait = time.monotonic()
        if grace_deadline is None:
            grace_deadline = started_wait + normal_exit_grace["maximum_seconds"]
            normal_exit_grace["started_monotonic"] = started_wait
            normal_exit_grace["deadline_monotonic"] = grace_deadline
        if not rows:
            return rows
        # A late finalization check shares the same deadline; grace cannot restart.
        after = tracker.wait_for_owned_exit(grace_deadline)
        normal_exit_grace["attempts"].append(
            {
                "started_monotonic": started_wait,
                "ended_monotonic": time.monotonic(),
                "active_before": rows,
                "active_after": after,
                "deadline_exhausted": bool(after) and time.monotonic() >= grace_deadline,
            }
        )
        return after

    def sampler():
        try:
            with (directory / "nvml_memory.jsonl").open("w") as out:
                while not stop.is_set():
                    try:
                        tracker.refresh()
                    except BaseException as error:
                        failure = {"process_tracking_failure": repr(error)}
                        if failure not in ownership_errors:
                            ownership_errors.append(failure)
                    try:
                        row = nvml.sample()
                        samples.append(row)
                    except Exception as error:
                        row = {"unix": time.time(), "error": str(error)}
                        errors.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    stop.wait(0.1)
        except BaseException as error:
            errors.append({"unix": time.time(), "sampler_failure": repr(error)})

    try:
        tracker = OwnedProcessTree()
        nvml = Nvml()
        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()
        with (
            (directory / "stdout.log").open("wb") as out,
            (directory / "stderr.log").open("wb") as err,
        ):
            process = subprocess.Popen(
                list(map(str, command)),
                cwd=PROJECT,
                env=environment,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            tracker.start(process.pid)
            write_json(
                directory / "process.json",
                {
                    "pid": process.pid,
                    "process_group": process.pid,
                    "owner": "this experiment stage",
                    "identity": tracker.known.get(tracker.root_key),
                    "descendant_tracking": "PID/starttime + pidfd; "
                    "independent worker sessions included",
                },
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
    except BaseException as error:
        pending_error = error
        (directory / "driver_error.txt").write_text(traceback.format_exc())
    finally:
        if tracker is not None:
            try:
                active_at_launcher_exit = tracker.active()
                tracker.reap_adopted()
                before_cleanup = settle_normal_exit(active_at_launcher_exit)
                cleanup_needed = bool(before_cleanup)
                normal_exit_residual = (
                    bool(before_cleanup)
                    and process is not None
                    and (process.returncode is not None and not timed_out and pending_error is None)
                )
                # Healthy profiler helpers may still be flushing/reparenting after
                # their launcher returns zero. Signal only residuals after grace.
                remaining_owned = tracker.cleanup() if before_cleanup else []
            except BaseException as error:
                ownership_errors.append({"cleanup_failure": repr(error)})
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=20)
            except BaseException as error:
                ownership_errors.append({"launcher_wait_failure": repr(error)})
        stop.set()
        if thread is not None:
            thread.join(timeout=20)
        sampler_alive = thread is not None and thread.is_alive()
        if sampler_alive:
            errors.append({"sampler_failure": "Sampler did not finish after stop/join"})
        elif nvml is not None:
            try:
                nvml.check(nvml.lib.nvmlShutdown())
            except Exception as error:
                errors.append({"nvml_shutdown_failure": repr(error)})
        if tracker is not None and not sampler_alive:
            try:
                remaining_owned = settle_normal_exit(tracker.active())
                if remaining_owned:
                    cleanup_needed = True
                    normal_exit_residual = normal_exit_residual or (
                        process is not None
                        and process.returncode is not None
                        and not timed_out
                        and pending_error is None
                    )
                    before_keys = {tracker.key(row) for row in before_cleanup}
                    before_cleanup.extend(
                        row for row in remaining_owned if tracker.key(row) not in before_keys
                    )
                    remaining_owned = tracker.cleanup()
                tracker.reap_adopted()
                tracker.close()
            except BaseException as error:
                ownership_errors.append({"ownership_finalization_failure": repr(error)})
                # A descendant may become adopted during the last close refresh.
                # Clean it before stopping; do not turn this anomaly into a passed stage.
                try:
                    remaining_owned = tracker.cleanup()
                    tracker.reap_adopted()
                except BaseException as cleanup_error:
                    ownership_errors.append({"late_owned_cleanup_failure": repr(cleanup_error)})
        ownership = (
            tracker.report()
            if tracker is not None
            else {"initialization_failed": True, "driver_pid": os.getpid()}
        )
        if tracker is not None and tracker.errors:
            ownership_errors.extend(tracker.errors)
        if grace_deadline is not None:
            normal_exit_grace["signals_during_grace"] = any(
                normal_exit_grace["started_monotonic"] <= row["monotonic"] < grace_deadline
                for row in ownership.get("signals", [])
            )
        normal_exit_grace["aborted_due_to_errors"] = bool(
            ownership_errors or errors or pending_error is not None
        )
        ownership.update(
            {
                "active_before_cleanup": before_cleanup,
                "active_at_launcher_exit": active_at_launcher_exit,
                "remaining_active_owned_processes": remaining_owned,
                "normal_exit_left_owned_descendants": normal_exit_residual,
                "normal_exit_grace": normal_exit_grace,
                "tracking_and_cleanup_errors": ownership_errors,
            }
        )
        write_json(directory / "process_ownership.json", ownership)
    remaining_pids = [row["pid"] for row in remaining_owned]
    remaining_group_pids = [
        row["pid"]
        for row in remaining_owned
        if process is not None and row["process_group"] == process.pid
    ]
    min_free = (
        [min(row["devices"][rank]["free_bytes"] for row in samples) for rank in range(4)]
        if samples
        else []
    )
    record = {
        "stage": name,
        "driver_version": 5,
        "started_unix": started,
        "ended_unix": time.time(),
        "exit_code": process.returncode if process is not None else None,
        "timed_out": timed_out,
        "driver_exception": repr(pending_error) if pending_error else None,
        "remaining_active_group_pids": remaining_group_pids,
        "remaining_active_owned_pids": remaining_pids,
        "process_ownership_artifact": "process_ownership.json",
        "ownership_errors": ownership_errors,
        "normal_exit_left_owned_descendants": normal_exit_residual,
        "normal_exit_grace": normal_exit_grace,
        "residual_group_cleanup": cleanup_needed,
        "status": "passed"
        if process is not None
        and process.returncode == 0
        and not timed_out
        and not cleanup_needed
        and not remaining_pids
        and not pending_error
        and not ownership_errors
        and not errors
        else "failed",
        "gpu_uuids": nvml.uuids if nvml is not None else [],
        "nvml": {
            "samples": len(samples),
            "errors": errors,
            "interval_target_seconds": 0.1,
            "scope": "whole stage including initialization, warmup and measured window",
            "minimum_free_bytes_by_device": min_free,
            "limitations": "Sampling can miss transient device peaks; not allocator accounting.",
        },
    }
    write_json(directory / "status.json", record)
    write_json(directory / "execution.json", record)
    print(json.dumps(record), flush=True)
    if pending_error is not None:
        raise pending_error
    if remaining_pids or sampler_alive or ownership_errors or normal_exit_residual:
        raise RuntimeError("Owned stage descendants/ownership errors after exit; stop suite")
    return record
