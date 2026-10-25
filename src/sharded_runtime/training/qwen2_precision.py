"""Explicit compute precision, independent of FP32 optimizer/master storage."""

import torch


def resolve_qwen_compute_dtype(name: str, device: torch.device) -> torch.dtype:
    device = torch.device(device)
    types = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if name not in types:
        raise ValueError("Qwen training dtype must be fp32, fp16 or bf16")
    if name == "fp16" and device.type != "cuda":
        raise ValueError("Qwen fp16 AMP requires CUDA")
    if name == "bf16" and device.type == "cuda":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError(
                    "Qwen bf16 requires native CUDA BF16 support; "
                    "V100 does not support native BF16. Use fp16 on V100."
                )
    return types[name]
