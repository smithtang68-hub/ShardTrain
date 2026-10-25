"""Qwen configuration and meta construction; no implicit network/weight download."""

import torch

from sharded_runtime.models.qwen2 import Qwen2CausalLM, Qwen2ModelConfig


def build_qwen_model(config, seed=1234, *, checkpoint=None):
    if config.hf_config_path:
        qwen = Qwen2ModelConfig.from_file(config.hf_config_path)
    else:
        qwen = Qwen2ModelConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.mlp_hidden_size,
            num_hidden_layers=config.layers,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            tie_word_embeddings=config.tie_word_embeddings,
        )
    qwen.use_cache = False
    model = Qwen2CausalLM(qwen, device="meta", dtype=torch.float32)
    model._qwen_seed = seed
    path = checkpoint or config.checkpoint_path
    if path:
        from sharded_runtime.models.qwen2_checkpoint import QwenCheckpoint

        model._qwen_weight_source = QwenCheckpoint(path)
    return model
