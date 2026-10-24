"""Tiny random-weight bridges for the LRP integration tests, built fully offline.

``build_bridge_from_module`` wraps a freshly initialised HF model with no tokenizer, so no
Hub access is needed; tokens are a fixed tensor of ids below the tiny vocabulary size.
"""

import copy

import pytest
import torch

VOCAB_SIZE = 97
TOKENS = torch.tensor([[1, 5, 7, 42, 9, 61, 23, 88, 3, 14]])
_COMMON = dict(
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=128,
    tie_word_embeddings=False,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)


def _tiny_qwen3():
    """Rotary attention with q/k norms and a gated MLP (RMSNorm model)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        **_COMMON,
    )
    return cfg, Qwen3ForCausalLM, "Qwen3ForCausalLM"


def _tiny_gemma2():
    """Sandwich norms (ln1_post, ln2_post), no q/k norms, Gemma-offset RMSNorm weights."""
    from transformers import Gemma2Config, Gemma2ForCausalLM

    cfg = Gemma2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        query_pre_attn_scalar=16,
        sliding_window=32,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        **_COMMON,
    )
    return cfg, Gemma2ForCausalLM, "Gemma2ForCausalLM"


def _tiny_gpt2():
    """Joint-QKV attention and an ungated MLP (LayerNorm model on the python-norm path)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=VOCAB_SIZE, n_embd=64, n_layer=2, n_head=4, n_positions=128)
    return cfg, GPT2LMHeadModel, "GPT2LMHeadModel"


def _tiny_qwen3_moe():
    """Rotary attention with a mixture-of-experts MLP in every block."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        **_COMMON,
    )
    cfg.architectures = ["Qwen3MoeForCausalLM"]
    return cfg, Qwen3MoeForCausalLM, "Qwen3MoeForCausalLM"


BUILDERS = {
    "qwen3": _tiny_qwen3,
    "gemma2": _tiny_gemma2,
    "gpt2": _tiny_gpt2,
    "qwen3_moe": _tiny_qwen3_moe,
}


def build_bridge(kind: str = "qwen3"):
    """A fresh tiny bridge of the given kind; same kind and seed give identical weights."""
    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    cfg, model_class, architecture = BUILDERS[kind]()
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = model_class(cfg).eval()
    return build_bridge_from_module(
        model, architecture, hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    )


@pytest.fixture
def tokens():
    return TOKENS.clone()


@pytest.fixture
def make_bridge():
    return build_bridge


@pytest.fixture
def tiny_bridge():
    return build_bridge("qwen3")


@pytest.fixture
def tiny_gemma_bridge():
    return build_bridge("gemma2")
