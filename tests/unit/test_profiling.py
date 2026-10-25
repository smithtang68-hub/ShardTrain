import json
import pickle
import sys

import pytest
import torch

from sharded_runtime.observability.profiling import ProfileCapture


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocator history required")
@pytest.mark.parametrize("mode", ["torch", "nsight"])
def test_failed_training_window_preserves_trace_and_allocator_history(tmp_path, mode):
    capture = ProfileCapture(tmp_path, 0, "cuda", mode=mode, max_entries=1000)
    capture.start()
    try:
        with pytest.raises(RuntimeError, match="simulated training failure"):
            value = torch.ones(64, 64, device="cuda", requires_grad=True)
            value.square().sum().backward()
            torch.cuda.synchronize()
            capture.step(0)
            raise RuntimeError("simulated training failure")
    finally:
        capture.close()
    with (tmp_path / "memory.end.rank0.pickle").open("rb") as file:
        snapshot = pickle.load(file)
    assert snapshot["segments"]
    assert any(event["action"] == "alloc" for trace in snapshot["device_traces"] for event in trace)
    metadata = json.loads((tmp_path / "profiling.rank0.json").read_text())
    assert metadata["errors"] == []
    assert metadata["memory"][0]["allocated"] > 0
    assert (tmp_path / "memory.start.rank0.pickle").exists()
    if mode == "torch":
        trace = json.loads((tmp_path / "profile.rank0.json").read_text())
        assert any(event.get("name") == "aten::square" for event in trace["traceEvents"])
        # Windows can use the legacy CUDA profiler fallback without kernel trace export.
        if sys.platform != "win32":
            assert any(event.get("cat") == "kernel" for event in trace["traceEvents"])
