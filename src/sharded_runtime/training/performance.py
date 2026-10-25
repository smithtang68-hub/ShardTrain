"""Identical architecture FLOPs and MFU accounting for both training engines."""

import math


def qwen_performance_summary(
    *,
    parameter_numel,
    layers,
    hidden_size,
    sequence_length,
    global_batch,
    steps,
    world_size,
    peak_tflops_per_gpu=None,
    profile_mode="off",
):
    """Estimate model FLOPs; never reinterpret activity/communication as FLOPs."""
    if peak_tflops_per_gpu is not None and (
        not math.isfinite(peak_tflops_per_gpu) or peak_tflops_per_gpu <= 0
    ):
        raise ValueError("peak_tflops_per_gpu must be a finite positive nominal peak")
    tokens = global_batch * sequence_length
    six_n = 6 * parameter_numel * tokens
    attention = 12 * layers * hidden_size * sequence_length * tokens
    model_flops = six_n + attention
    times = [step["max_rank_seconds"] for step in steps]
    if any(not math.isfinite(value) or value <= 0 for value in times):
        raise ValueError("Measured max-rank step durations must be finite and positive")
    total = sum(times)
    seconds = total / len(times) if times else None
    achieved = model_flops / seconds / 1e12 if seconds else None
    return {
        "flops_formula": "6 * unique_parameters * input_tokens + "
        "12 * layers * hidden_size * sequence * input_tokens",
        "model_unique_parameters": parameter_numel,
        "global_input_tokens_per_step": tokens,
        "six_n_flops_per_step": six_n,
        "attention_flops_per_step": attention,
        "estimated_model_flops_per_step": model_flops,
        "mean_max_rank_seconds": seconds,
        "global_input_tokens_per_second": tokens * len(times) / total if total else None,
        "global_valid_tokens_per_second": sum(step["valid_tokens"] for step in steps) / total
        if total
        else None,
        "global_sequences_per_second": global_batch * len(times) / total if total else None,
        "all_gpu_effective_model_tflops": achieved,
        "peak_tflops_per_gpu": peak_tflops_per_gpu,
        "mfu_percent": 100 * achieved / (world_size * peak_tflops_per_gpu)
        if achieved is not None and peak_tflops_per_gpu is not None
        else None,
        "profile_mode": profile_mode,
        "unprofiled_timing": profile_mode == "off",
        "limitations": "Architecture FLOPs estimate, not hardware counters; includes input "
        "positions rather than only valid next-token labels. Excludes optimizer, "
        "elementwise and communication. Nominal peak must be explicitly supplied.",
    }
