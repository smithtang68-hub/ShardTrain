"""Benchmark the public PyTorch FSDP2 plus DTensor Qwen baseline."""

from sharded_runtime.training.qwen2_native_baseline import main

if __name__ == "__main__":
    raise SystemExit(main())
