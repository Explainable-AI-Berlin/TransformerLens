"""LRP rules on tiny randomly initialised in-memory bridges.

Each rule is checked against an independent forward-time formulation (a detached norm
scale, a detached softmax, a half-detached product, an activation rewritten as
``x * stop_grad(f(x) / x)``), and all rules together against conservation: on a bias-free
model the input relevance sums to the explained logit.
"""

from __future__ import annotations

import contextlib
import types
import warnings
from typing import Any, Callable, Optional

import jaxtyping
import pytest
import torch
import torch.nn.functional as F
import transformers
from torch import nn

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import NormalizationBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)
from transformer_lens.tools.analysis import (
    EdgeAttributionConfig,
    attribution_patch,
    lrp_rules,
)
from transformer_lens.tools.analysis.lrp import DEFAULT_RULES, LRPRule

_VOCAB = 50
_DIMS: dict[str, Any] = dict(
    hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4
)
_COMMON: dict[str, Any] = dict(vocab_size=_VOCAB, max_position_embeddings=32)
_GQA: dict[str, Any] = dict(num_key_value_heads=2, head_dim=8)
_T5: dict[str, Any] = dict(d_model=32, d_ff=64, num_layers=2, num_heads=4, d_kv=8)
_T5.update(vocab_size=_VOCAB, decoder_start_token_id=0, feed_forward_proj="gated-gelu")
_ENCODER: dict[str, Any] = dict(
    hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64
)
# arch -> (HF model class, config factory)
_MODELS: dict[str, tuple[str, Callable[[], Any]]] = {
    "gpt2": (
        "GPT2LMHeadModel",
        lambda: transformers.GPT2Config(n_layer=2, n_head=4, n_embd=32, vocab_size=_VOCAB),
    ),
    "llama": ("LlamaForCausalLM", lambda: transformers.LlamaConfig(**_DIMS, **_COMMON, **_GQA)),
    "llama_mha": ("LlamaForCausalLM", lambda: transformers.LlamaConfig(**_DIMS, **_COMMON)),
    "gemma2": (
        "Gemma2ForCausalLM",
        lambda: transformers.Gemma2Config(**_DIMS, **_COMMON, **_GQA, final_logit_softcapping=None),
    ),
    "qwen3": ("Qwen3ForCausalLM", lambda: transformers.Qwen3Config(**_DIMS, **_COMMON, **_GQA)),
    "olmo": ("OlmoForCausalLM", lambda: transformers.OlmoConfig(**_DIMS, **_COMMON)),
    "phi3": (
        "Phi3ForCausalLM",
        lambda: transformers.Phi3Config(**_DIMS, **_COMMON, pad_token_id=0),
    ),
    "gptj": (
        "GPTJForCausalLM",
        lambda: transformers.GPTJConfig(
            n_embd=32, n_layer=2, n_head=4, rotary_dim=4, vocab_size=_VOCAB
        ),
    ),
    "opt": (
        "OPTForCausalLM",
        lambda: transformers.OPTConfig(
            hidden_size=32, ffn_dim=64, num_hidden_layers=2, num_attention_heads=4, **_COMMON
        ),
    ),
    "neox": ("GPTNeoXForCausalLM", lambda: transformers.GPTNeoXConfig(**_DIMS, **_COMMON)),
    "mixtral": (
        "MixtralForCausalLM",
        lambda: transformers.MixtralConfig(
            **_DIMS, **_COMMON, **_GQA, num_local_experts=4, num_experts_per_tok=2
        ),
    ),
    "lfm2": (
        "Lfm2ForCausalLM",
        lambda: transformers.Lfm2Config(
            **_DIMS, **_COMMON, num_key_value_heads=2, layer_types=["conv", "full_attention"]
        ),
    ),
    "t5": (
        "T5ForConditionalGeneration",
        lambda: transformers.T5Config(**_T5),
    ),
    "bart": (
        "BartForConditionalGeneration",
        lambda: transformers.BartConfig(
            d_model=32,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            **_COMMON,
        ),
    ),
    "bert": (
        "BertForMaskedLM",
        lambda: transformers.BertConfig(**_ENCODER, **_COMMON),
    ),
    "roberta": (
        "RobertaForMaskedLM",
        lambda: transformers.RobertaConfig(
            **_ENCODER, vocab_size=_VOCAB, max_position_embeddings=40, pad_token_id=1
        ),
    ),
    "hrm_text": (
        "HrmTextForCausalLM",
        lambda: getattr(transformers, "HrmTextConfig")(
            **_ENCODER,
            num_key_value_heads=4,
            vocab_size=_VOCAB,
            num_layers_per_stack=1,
            H_cycles=1,
            L_cycles=1,
        ),
    ),
    "qwen2_moe": (
        "Qwen2MoeForCausalLM",
        lambda: transformers.Qwen2MoeConfig(
            **_DIMS,
            **_COMMON,
            **_GQA,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
        ),
    ),
    "gpt_oss": (
        "GptOssForCausalLM",
        lambda: transformers.GptOssConfig(
            **_DIMS, **_COMMON, **_GQA, num_local_experts=4, num_experts_per_tok=2
        ),
    ),
    "glm4_moe": (
        "Glm4MoeForCausalLM",
        lambda: transformers.Glm4MoeConfig(
            **_DIMS,
            **_COMMON,
            **_GQA,
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            first_k_dense_replace=1,
            n_group=1,
            topk_group=1,
        ),
    ),
    "deepseek_v2": (
        "DeepseekV2ForCausalLM",
        lambda: transformers.DeepseekV2Config(
            **_DIMS,
            **_COMMON,
            moe_intermediate_size=16,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_rope_head_dim=4,
            qk_nope_head_dim=4,
            v_head_dim=8,
            n_routed_experts=4,
            num_experts_per_tok=2,
        ),
    ),
}


_ATTN_LRP: list[LRPRule] = ["ln", "attn_half", "half", "identity"]


def _bridge(
    arch: str,
    dtype: torch.dtype = torch.float64,
    compat: Optional[str] = None,
    bias_free: bool = False,
    **overrides: Any,
) -> TransformerBridge:
    torch.manual_seed(0)
    architecture, make_config = _MODELS[arch]
    hf_config = make_config()
    for key, value in overrides.items():
        setattr(hf_config, key, value)
    hf_model = getattr(transformers, architecture)(hf_config).to(dtype).eval()
    model = build_bridge_from_module(
        hf_model, architecture, hf_config=hf_config, tokenizer=None, device="cpu"
    )
    if compat == "processed":
        model.enable_compatibility_mode()
    elif compat == "no_processing":
        model.enable_compatibility_mode(no_processing=True)
    if bias_free:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith("bias"):
                    param.zero_()
    return model.eval()


def _tokens(seed: int = 0) -> torch.Tensor:
    return torch.randint(0, _VOCAB, (1, 6), generator=torch.Generator().manual_seed(seed))


def _input_grad(
    model: TransformerBridge, rules: Any, fwd_hooks: tuple = (), target: int = 7, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(residual input, its gradient, target logit).

    ``rules`` is ``"plain"`` (no rules), ``"all"`` (``lrp_rules`` defaults) or a rule list.
    """
    store: dict[str, torch.Tensor] = {}
    hooks = [("blocks.0.hook_in", lambda t, hook: store.update(x=t)), *fwd_hooks]
    if rules == "plain":
        ctx: Any = contextlib.nullcontext()
    else:
        ctx = lrp_rules(model, None if rules == "all" else rules)
    with ctx, model.hooks(fwd_hooks=hooks):
        logit = model(kwargs.pop("tokens", _tokens()), **kwargs)[0, -1, target]
        (grad,) = torch.autograd.grad(logit, store["x"])
    return store["x"].detach(), grad, logit.detach()


_CONSERVATION_ARCHS = ["gpt2", "llama", "gemma2", "qwen3", "olmo", "phi3", "gptj", "opt", "neox"]
_CONSERVATION_ARCHS += ["bert", "roberta"]


@pytest.mark.parametrize("arch", _CONSERVATION_ARCHS)
def test_all_rules_conserve_the_logit(arch: str) -> None:
    model = _bridge(arch, bias_free=True)
    x, grad, logit = _input_grad(model, "all")
    # HF RMSNorm computes in float32 even for a float64 model.
    torch.testing.assert_close((x * grad).sum(), logit, rtol=1e-6, atol=1e-7)
    x, grad, logit = _input_grad(model, "plain")
    assert abs((x * grad).sum() - logit) > 1e-4


@pytest.mark.parametrize("arch", ["t5", "bart"])
def test_all_rules_conserve_the_logit_of_encoder_decoders(arch: str) -> None:
    model = _bridge(arch, bias_free=True)
    names = ["encoder_blocks.0.hook_in", "decoder_blocks.0.hook_in"]
    store: dict[str, torch.Tensor] = {}
    keep = [(name, lambda t, hook: store.update({hook.name: t})) for name in names]
    with lrp_rules(model), model.hooks(fwd_hooks=keep):
        logit = model(_tokens(0), decoder_input=_tokens(1))[0, -1, 7]
        grads = torch.autograd.grad(logit, [store[name] for name in names])
    relevance = sum((store[name] * grad).sum() for name, grad in zip(names, grads))
    torch.testing.assert_close(relevance, logit, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("arch", ["llama", "gemma2", "gpt2"])
def test_dropping_any_rule_breaks_conservation(arch: str) -> None:
    model = _bridge(arch, bias_free=True)
    with lrp_rules(model) as sites:
        used = [rule for rule in DEFAULT_RULES if sites[rule]]
    for dropped in used:
        x, grad, logit = _input_grad(model, [rule for rule in used if rule != dropped])
        assert abs((x * grad).sum() - logit) > 1e-5, dropped


_BIT_IDENTICAL = [
    (arch, compat, dtype)
    for arch in ("gpt2", "llama")
    for compat in (None, "processed", "no_processing")
    for dtype in (torch.float32, torch.bfloat16)
]
_BIT_IDENTICAL += [("gptj", None, torch.bfloat16), ("gptj", "no_processing", torch.float32)]
_BIT_IDENTICAL += [("mixtral", None, torch.float32), ("qwen2_moe", None, torch.float32)]


@pytest.mark.parametrize("arch, compat, dtype", _BIT_IDENTICAL)
def test_forward_is_bit_identical(arch: str, compat: Optional[str], dtype: torch.dtype) -> None:
    model = _bridge(arch, dtype=dtype, compat=compat)
    # Processed gated MLPs call the activation as a function (see the bypass test).
    rules: Optional[tuple[LRPRule, ...]] = None
    if compat == "processed" and arch == "llama":
        rules = ("ln", "ah", "half")
    tokens = _tokens()
    plain = model(tokens)
    plain_grad = torch.autograd.grad(plain[0, -1, 3], model.embed.weight)[0]
    with lrp_rules(model, rules):
        logits = model(tokens)
        grad = torch.autograd.grad(logits[0, -1, 3], model.embed.weight)[0]
    assert torch.equal(logits, plain)
    assert not torch.equal(grad, plain_grad)


@pytest.mark.parametrize("arch", ["llama", "gpt2"])
def test_bfloat16_gradients_are_close_to_float64(arch: str) -> None:
    def relative_error(rules: str) -> float:
        _, reference, _ = _input_grad(_bridge(arch), rules)
        _, grad, _ = _input_grad(_bridge(arch, dtype=torch.bfloat16), rules)
        return float((grad.double() - reference).norm() / reference.norm())

    # bfloat16 keeps 8 bits (0.4%); plain gradients come out about 0.5% off here and LRP ones
    # about 1%, so 2% bounds both while catching a wrong rule (tens of percent).
    assert relative_error("all") < 2e-2 and relative_error("plain") < 2e-2


def _hook_counts(model: nn.Module) -> list[tuple[int, int, int]]:
    return [
        (len(m._forward_hooks), len(m._forward_pre_hooks), len(m._backward_hooks))
        for m in model.modules()
    ]


def _fail(t: torch.Tensor, hook: Any) -> None:
    raise KeyError("inside the forward")


@pytest.mark.parametrize(
    "arch, rules, overrides",
    [
        ("gpt2", "all", {}),
        ("llama", "all", {}),
        ("gemma2", _ATTN_LRP, {"final_logit_softcapping": 0.1}),
        ("qwen2_moe", _ATTN_LRP, {"_experts_implementation": "eager"}),
    ],
)
def test_exit_restores_plain_gradients_and_hooks(arch: str, rules: Any, overrides: dict) -> None:
    model = _bridge(arch, **overrides)
    _, plain_grad, _ = _input_grad(model, "plain")
    counts = _hook_counts(model)  # After a forward: blocks wire their own hooks lazily.
    _, rule_grad, _ = _input_grad(model, rules)

    # Inside the block: a forward that fails, then TL's hook reset, then an exception.
    with pytest.raises(KeyError, match="inside the block"):
        with lrp_rules(model, None if rules == "all" else rules):
            with pytest.raises(KeyError, match="inside the forward"):
                _input_grad(model, "plain", fwd_hooks=(("blocks.0.hook_out", _fail),))
            model.reset_hooks(including_permanent=True)
            _, grad, _ = _input_grad(model, "plain")
            assert torch.equal(grad, rule_grad)  # The rules are not TL hooks.
            raise KeyError("inside the block")

    _, grad, _ = _input_grad(model, "plain")
    assert torch.equal(grad, plain_grad)
    assert _hook_counts(model) == counts
    assert model.list_hooks() == {}


def _norm_scale_detached(model: TransformerBridge) -> tuple:
    """Forward hooks holding every norm scale constant, on the python-norm path."""
    names = []
    for module in model.modules():
        if isinstance(module, NormalizationBridge):
            module.use_native_layernorm_autograd = False
            names.append(module.hook_scale.name)
    return tuple((name, lambda t, hook: t.detach()) for name in names)


@pytest.mark.parametrize(
    "arch, compat",
    [("gpt2", None), ("opt", None), ("neox", None), ("gemma2", None), ("llama", "processed")],
)
def test_ln_rule_matches_detached_scale(arch: str, compat: Optional[str]) -> None:
    # Random affine parameters, so the oracle tests more than the unit-weight case.
    model = _bridge(arch, compat=compat)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, NormalizationBridge):
                for param in module.parameters():
                    param.add_(0.3 * torch.randn_like(param))
    _, grad, _ = _input_grad(model, ["ln"])
    oracle_hooks = _norm_scale_detached(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, oracle, _ = _input_grad(model, "plain", fwd_hooks=oracle_hooks)
    # Gemma2's HF RMSNorm computes in float32.
    torch.testing.assert_close(grad, oracle, rtol=1e-5, atol=1e-7 if arch == "gemma2" else 1e-12)


# llama and gpt2 rebuild attention around hook_pattern; gptj and opt run it inside HF.
@pytest.mark.parametrize("arch", ["llama", "gpt2", "gptj", "opt"])
def test_ah_rule_matches_detached_softmax(arch: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Eager HF attention, so the oracle's softmax patch reaches it (SDPA fuses the softmax).
    model = _bridge(arch, _attn_implementation="eager")
    _, grad, _ = _input_grad(model, ["ah"])
    softmax = torch.nn.functional.softmax
    monkeypatch.setattr(torch.nn.functional, "softmax", lambda *a, **k: softmax(*a, **k).detach())
    _, oracle, _ = _input_grad(model, "plain")
    torch.testing.assert_close(grad, oracle, rtol=0, atol=1e-12)


@pytest.mark.parametrize("arch", ["llama", "phi3"])
def test_half_rule_matches_half_detached_product(arch: str) -> None:
    model = _bridge(arch)
    _, grad, _ = _input_grad(model, ["half"])
    half = lambda t, hook: 0.5 * t + 0.5 * t.detach()
    hooks = tuple((f"blocks.{i}.mlp.hook_post", half) for i in range(2))
    _, oracle, _ = _input_grad(model, "plain", fwd_hooks=hooks)
    torch.testing.assert_close(grad, oracle, rtol=0, atol=1e-12)


def _divide_gradient(t: torch.Tensor, factor: float) -> torch.Tensor:
    return t.detach() + (t - t.detach()) / factor


def _hf_twin(arch: str, dtype: torch.dtype = torch.float64, **overrides: Any) -> nn.Module:
    """The HF model ``_bridge`` wraps, built again from the same seed and left unbridged."""
    torch.manual_seed(0)
    architecture, make_config = _MODELS[arch]
    hf_config = make_config()
    for key, value in overrides.items():
        setattr(hf_config, key, value)
    return getattr(transformers, architecture)(hf_config).to(dtype).eval()


def _hf_input_grad(hf_model: Any, target: int = 7) -> torch.Tensor:
    embeds = hf_model.get_input_embeddings()(_tokens()).detach().requires_grad_()
    logit = hf_model(inputs_embeds=embeds).logits[0, -1, target]
    return torch.autograd.grad(logit, embeds)[0]


# AttnLRP's uniform rule on QK^T and pattern @ V, as LXT applies it to HF's eager attention:
# q and k gradients / 4, v gradient / 2. opt runs attention in HF; the others are rebuilt
# (gemma2 with an attention-logit softcap and a sliding window, qwen3 with q/k norms).
_GEMMA2_ATTN = {"attn_logit_softcapping": 1.0, "sliding_window": 3}


@pytest.mark.parametrize(
    "arch, module, overrides",
    [
        ("llama", "llama", {}),
        ("gpt2", "gpt2", {}),
        ("opt", "opt", {}),
        ("gemma2", "gemma2", _GEMMA2_ATTN),
        ("qwen3", "qwen3", {}),
    ],
)
def test_attn_half_rule_matches_uniform_rule_on_hf_attention(
    arch: str, module: str, overrides: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = {**overrides, "_attn_implementation": "eager"}
    _, grad, _ = _input_grad(_bridge(arch, **overrides), ["attn_half"])
    modeling: Any = getattr(getattr(transformers.models, module), f"modeling_{module}")
    attention: Callable[..., Any] = modeling.eager_attention_forward

    def uniform_rule(mod: Any, q: Any, k: Any, v: Any, *args: Any, **kwargs: Any) -> Any:
        q, k, v = _divide_gradient(q, 4), _divide_gradient(k, 4), _divide_gradient(v, 2)
        return attention(mod, q, k, v, *args, **kwargs)

    monkeypatch.setattr(modeling, "eager_attention_forward", uniform_rule)
    oracle = _hf_input_grad(_hf_twin(arch, **overrides))
    # HF RMSNorm computes in float32 even for a float64 model.
    torch.testing.assert_close(grad, oracle, rtol=1e-6, atol=1e-8)


def _lxt_experts_forward(self: Any, hidden_states: Any, top_k_index: Any, weights: Any) -> Any:
    """HF's eager experts loop with the Identity-rule and both Half-rules, as in LXT.

    A copy of HF's loop (``MixtralExperts.forward``), as LXT's own oracle is; it reads the
    experts' ``gate_up_proj``/``down_proj`` layout and needs updating if HF changes it.
    """
    out = torch.zeros_like(hidden_states)
    for expert in range(self.num_experts):
        token, slot = torch.where(top_k_index == expert)
        gate, up = F.linear(hidden_states[token], self.gate_up_proj[expert]).chunk(2, dim=-1)
        act = self.act_fn(gate)
        act = gate * (act / gate).detach()  # Identity-rule.
        product = _divide_gradient(act * up, 2)
        expert_out = F.linear(product, self.down_proj[expert]) * weights[token, slot, None]
        out.index_add_(0, token, _divide_gradient(expert_out, 2))
    return out


def _lxt_gated_mlp_forward(self: Any, x: Any) -> Any:
    gate = self.gate_proj(x)
    act = gate * (self.act_fn(gate) / gate).detach()
    return self.down_proj(_divide_gradient(act * self.up_proj(x), 2))


def _qwen2_moe_block_forward(self: Any, hidden_states: Any) -> Any:
    batch, pos, d_model = hidden_states.shape
    flat = hidden_states.view(-1, d_model)
    _, weights, index = self.gate(flat)
    shared = torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
    return (self.experts(flat, index, weights) + _divide_gradient(shared, 2)).view_as(hidden_states)


def _lxt_moe_twin(arch: str, **kwargs: Any) -> nn.Module:
    twin = _hf_twin(arch, **kwargs)
    for module in twin.modules():
        kind = type(module).__name__
        if kind.endswith("Experts"):
            module.forward = types.MethodType(_lxt_experts_forward, module)
        elif kind.endswith("MLP"):
            module.forward = types.MethodType(_lxt_gated_mlp_forward, module)
        elif kind == "Qwen2MoeSparseMoeBlock":
            module.forward = types.MethodType(_qwen2_moe_block_forward, module)
    return twin


@pytest.mark.parametrize("arch", ["mixtral", "qwen2_moe", "deepseek_v2"])
def test_moe_rules_match_lxt_style_experts(arch: str) -> None:
    # The bridge runs HF's grouped-matmul experts; the oracle the eager loop.
    model = _bridge(arch, dtype=torch.float32)
    _, grad, _ = _input_grad(model, ["half", "identity"])
    _, plain, _ = _input_grad(model, "plain")
    twin = _lxt_moe_twin(arch, dtype=torch.float32)
    oracle = _hf_input_grad(twin)
    torch.testing.assert_close(grad, oracle, rtol=1e-4, atol=1e-6)
    assert (grad - plain).abs().max() > 1e-3


def test_moe_hook_point_gradients_match_lxt_style_products() -> None:
    # Cached gradients at the routed experts' output and at both factors of the shared gate
    # product, against the same tensors in the LXT-style twin.
    model = _bridge("qwen2_moe", _experts_implementation="eager")
    points = ["experts", "shared_expert", "shared_expert_gate"]
    cached: dict[str, torch.Tensor] = {}
    keep = lambda g, hook: cached.update({hook.name.split(".")[-2]: g})
    with model.hooks(bwd_hooks=[(f"blocks.0.mlp.{p}.hook_out", keep) for p in points]):
        _input_grad(model, ["half", "identity"])
    twin: Any = _lxt_moe_twin("qwen2_moe")
    block, outputs = twin.model.layers[0].mlp, {}
    for point in points:
        getattr(block, point).register_forward_hook(
            lambda m, a, out, point=point: outputs.update({point: out})
        )
    embeds = twin.get_input_embeddings()(_tokens()).detach().requires_grad_()
    logit = twin(inputs_embeds=embeds).logits[0, -1, 7]
    for point, grad in zip(points, torch.autograd.grad(logit, [outputs[p] for p in points])):
        torch.testing.assert_close(cached[point], grad.reshape(cached[point].shape), msg=point)


def test_unbridged_gated_mlp_inside_moe_raises() -> None:
    # GLM-4-MoE's adapter does not bridge the block's shared experts (a gated MLP).
    model = _bridge("glm4_moe", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="shared_experts"):
        _input_grad(model, ["ln", "ah", "half"])


def test_softcap_bypasses_raise() -> None:
    model = _bridge("gemma2", final_logit_softcapping=0.1)
    tokens = _tokens()
    with lrp_rules(model, ["identity"]):
        with pytest.raises(RuntimeError, match="softcap"):
            model.original_model(tokens, labels=tokens)  # HF's loss uses the raw softcap.
        with pytest.raises(RuntimeError, match="softcap"):
            model.original_model(tokens, return_dict=False)


def test_final_softcap_gets_the_identity_rule() -> None:
    # A small cap so that tanh is far from linear on these logits.
    model = _bridge("gemma2", bias_free=True, final_logit_softcapping=0.1)
    x, grad, logit = _input_grad(model, "all")
    torch.testing.assert_close((x * grad).sum(), logit, rtol=1e-6, atol=1e-7)
    x, grad, logit = _input_grad(model, ["ln", "ah", "half"])
    assert abs((x * grad).sum() - logit) > 1e-4


@pytest.mark.parametrize("arch", ["llama", "gpt2", "phi3"])
def test_identity_rule_matches_stop_gradient_ratio(arch: str) -> None:
    model = _bridge(arch)
    _, grad, _ = _input_grad(model, ["identity"])
    acts = [
        module
        for module in model.modules()
        if type(module).__module__ == "transformers.activations"
    ]
    handles = [
        act.register_forward_hook(lambda m, args, y: args[0] * (y / args[0]).detach())
        for act in acts
    ]
    try:
        _, oracle, _ = _input_grad(model, "plain")
    finally:
        for handle in handles:
            handle.remove()
    assert len(acts) == 2
    torch.testing.assert_close(grad, oracle, rtol=0, atol=1e-12)


def _reference_grads(
    model: TransformerBridge, names: list[str], loss: bool, rules: Any
) -> dict[str, torch.Tensor]:
    """``torch.autograd.grad`` w.r.t. each hook point's own output tensor."""
    raw: dict[str, torch.Tensor] = {}
    handles = [
        model.hook_dict[name].register_forward_hook(
            lambda m, a, out, name=name: raw.update({name: out})
        )
        for name in names
    ]
    # No-op backward hooks give each hook point a distinct output tensor, as when caching.
    noop = [(name, lambda g, hook: None) for name in names]
    try:
        with lrp_rules(model, rules), model.hooks(bwd_hooks=noop):
            out = model(_tokens(), return_type="loss") if loss else model(_tokens())[0, -1, 3]
            keys = [k for k, v in raw.items() if v.requires_grad]
            grads = torch.autograd.grad(out, [raw[k] for k in keys], allow_unused=True)
    finally:
        for handle in handles:
            handle.remove()
    reference = {}
    for key, grad in zip(keys, grads):
        conversion = model.hook_dict[key].hook_conversion
        if grad is not None:
            reference[key] = conversion.convert(grad) if conversion is not None else grad
    return reference


# Hook points that need a set_use_* flag. (Embedding hook points never fire backward hooks.)
_GATED = ("hook_attn_in", "_input", "hook_result", "hook_mlp_in")


@pytest.mark.parametrize("rules", [None, _ATTN_LRP])
@pytest.mark.parametrize("arch", ["gpt2", "llama", "gptj", "qwen2_moe"])
@pytest.mark.parametrize("norm_internals", [True, False])
def test_cached_gradients_are_the_lrp_gradients(
    arch: str, norm_internals: bool, rules: Any
) -> None:
    # The eager experts loop runs in float64 (grouped matmuls do not).
    model = _bridge(arch, _experts_implementation="eager")
    names = [
        name
        for name, hook_point in model.hook_dict.items()
        if hook_point.name == name
        and not name.split(".")[0].endswith("embed")
        and not name.endswith(_GATED)
    ]
    # Backward hooks on norm internals move the native norm path to the python path.
    if not norm_internals:
        names = [n for n in names if not n.endswith(("hook_scale", "hook_normalized"))]
    cached: dict[str, torch.Tensor] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with lrp_rules(model, rules):
            _, cache = model.run_with_cache(
                _tokens(), return_type="loss", incl_bwd=True, names_filter=names
            )
        bwd = [(n, lambda g, hook: cached.update({hook.name: g})) for n in names]
        with lrp_rules(model, rules), model.hooks(bwd_hooks=bwd):
            torch.autograd.grad(model(_tokens())[0, -1, 3], model.embed.weight)
        for loss, got in (
            (True, {n: cache[f"{n}_grad"] for n in names if f"{n}_grad" in cache}),
            (False, cached),
        ):
            reference = _reference_grads(model, names, loss, rules)
            assert set(got) == set(reference)
            for name, grad in got.items():
                torch.testing.assert_close(grad, reference[name], rtol=0, atol=1e-12, msg=name)
    must_cover = ["blocks.0.ln1.hook_in", "blocks.0.attn.q.hook_out", "blocks.0.attn.v.hook_out"]
    must_cover += ["blocks.0.attn.o.hook_in", "blocks.0.mlp.hook_in"]
    if arch == "qwen2_moe":
        must_cover += ["blocks.0.mlp.experts.hook_in", "blocks.0.mlp.shared_expert.hook_out"]
        must_cover += ["blocks.0.mlp.shared_expert_gate.hook_out"]
    must_cover += ["blocks.0.ln1.hook_scale"] if norm_internals else []
    assert set(must_cover) <= set(cached)


# gpt2: python-path LayerNorm; neox: native-path LayerNorm; llama: native RMSNorm (float32 in HF).
@pytest.mark.parametrize("rules", [None, _ATTN_LRP])
@pytest.mark.parametrize("arch, atol", [("gpt2", 1e-12), ("neox", 1e-12), ("llama_mha", 1e-7)])
def test_relp_edge_scores_add_up_to_node_scores(arch: str, atol: float, rules: Any) -> None:
    # Edge sweeps normalise per-head residual copies through the raw HF ln1 module.
    model = _bridge(arch)
    clean, corrupt = _tokens(0), _tokens(1)
    metric = lambda logits: logits[0, -1, 3] - logits[0, -1, 5]
    with lrp_rules(model, rules):
        nodes = attribution_patch(model, clean, corrupt, metric)
        edges = attribution_patch(
            model, clean, corrupt, metric, EdgeAttributionConfig(granularity="edge")
        )
    for node, score in nodes.node_scores.items():
        assert edges.node_scores[node] == pytest.approx(score, abs=atol)
    qk = [s for (_, r), s in edges.edge_scores.items() if r.kind in ("q_input", "k_input")]
    assert qk and all(score == 0 for score in qk) == (rules is None)


@pytest.mark.parametrize(
    "rules, match",
    [
        ((), "nonempty"),
        (("ln", "softmax"), "nonempty"),
        (("ah", "attn_half"), "without both"),
        (("half", "gate"), "without both"),
    ],
)
def test_invalid_rules_raise(rules: tuple, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        with lrp_rules(_bridge("gpt2"), rules):
            pass


def test_duplicate_rules_apply_once() -> None:
    model = _bridge("llama")
    _, once, _ = _input_grad(model, ["half", "attn_half"])
    _, twice, _ = _input_grad(model, ["half", "attn_half", "half", "attn_half"])
    assert torch.equal(once, twice)


@pytest.mark.parametrize(
    "arch, overrides, rules, match",
    [
        ("gpt_oss", {}, None, "standard gated HF experts.*without 'half', 'gate' and 'identity'"),
        ("lfm2", {}, None, "Lfm2ShortConvBridge"),
        ("gpt2", {"activation_function": "sigmoid"}, None, "Sigmoid.*without 'identity'"),
    ],
)
def test_unsupported_components_raise_on_entry(
    arch: str, overrides: dict, rules: Any, match: str
) -> None:
    model = _bridge(arch, dtype=torch.float32, **overrides)
    with pytest.raises(NotImplementedError, match=match):
        with lrp_rules(model, rules):
            pass


def test_only_components_a_requested_rule_needs_are_checked() -> None:
    # LFM2's gated short convolution is outside the LN-rule's scope.
    model = _bridge("lfm2", dtype=torch.float32)
    _input_grad(model, ["ln"])
    with pytest.raises(NotImplementedError, match="in-place"):
        gpt2 = _bridge("gpt2")
        gpt2.blocks[0].mlp.original_component.act = nn.ReLU(inplace=True)
        with lrp_rules(gpt2):
            pass


def test_nested_contexts_raise() -> None:
    model = _bridge("llama")
    with lrp_rules(model):
        with pytest.raises(RuntimeError, match="already active"):
            with lrp_rules(model):
                pass
    _, grad, _ = _input_grad(model, "plain")  # The failed entry left nothing behind.
    _, plain, _ = _input_grad(_bridge("llama"), "plain")
    assert torch.equal(grad, plain)


def _zero_first(t: torch.Tensor, hook: Any) -> None:
    t[:, 0] = 0


@pytest.mark.parametrize(
    "name", ["attn.hook_attn_scores", "attn.hook_pattern", "mlp.hook_post", "ln1.hook_out"]
)
def test_in_place_user_edits_at_rule_sites(name: str) -> None:
    model = _bridge("llama")
    hook_name = f"blocks.0.{name}"
    in_place = ((hook_name, _zero_first),)
    out_of_place = ((hook_name, lambda t, hook: torch.cat([0 * t[:, :1], t[:, 1:]], dim=1)),)
    _, grad, _ = _input_grad(model, "all", fwd_hooks=in_place)
    _, expected, _ = _input_grad(model, "all", fwd_hooks=out_of_place)
    torch.testing.assert_close(grad, expected, rtol=0, atol=1e-12)


def test_shared_activation_module() -> None:
    model = _bridge("llama")
    _, expected, _ = _input_grad(model, "all")
    mlps = [block.mlp.original_component for block in model.blocks]
    mlps[1].act_fn = mlps[0].act_fn
    _, grad, _ = _input_grad(model, "all")
    torch.testing.assert_close(grad, expected, rtol=0, atol=1e-12)


def test_ln_rule_survives_a_stale_native_path_check(monkeypatch: pytest.MonkeyPatch) -> None:
    # If a future bridge sends a python-path norm through HF's native call, the rule
    # must still hold (or raise), not cut or keep the gradient through the scale.
    model = _bridge("gpt2")
    _, expected, _ = _input_grad(model, ["ln"])

    def native_forward(self: NormalizationBridge, x: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.hook_out(self._hf_autograd_forward_with_hooks(self.hook_in(x)))

    monkeypatch.setattr(NormalizationBridge, "forward", native_forward)
    _, grad, _ = _input_grad(model, ["ln"])
    torch.testing.assert_close(grad, expected, rtol=0, atol=1e-12)


def test_plain_hf_norms_inside_components_get_the_ln_rule() -> None:
    # DeepSeek-V2 wraps q_a_layernorm and kv_a_layernorm as plain components.
    model = _bridge("deepseek_v2", dtype=torch.float32)
    with lrp_rules(model, ["ln", "ah"]) as sites:
        assert any(site.endswith("kv_a_layernorm") for site in sites["ln"])
    _input_grad(model, ["ln", "ah"])


def test_bypassed_sites_raise_during_forward() -> None:
    compat = _bridge("llama", compat="processed")
    with pytest.raises(RuntimeError, match="no_processing=True"):
        _input_grad(compat, "all")
    # A forward-hook edit of a native-path norm internal replaces HF's norm call.
    model = _bridge("llama")
    edit = (("blocks.0.ln1.hook_normalized", lambda t, hook: t * 1.0),)
    with pytest.raises(RuntimeError, match="'ln' LRP rule did not run in blocks.0.ln1"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _input_grad(model, ["ln"], fwd_hooks=edit)


def test_yields_the_components_each_rule_covers() -> None:
    with lrp_rules(_bridge("llama")) as sites:
        assert "ln_final" in sites["ln"] and all(sites.values())
        assert all(site.startswith("blocks.") for site in sites["half"] + sites["ah"])
    with lrp_rules(_bridge("gpt2"), ["ln", "half"]) as sites:
        assert sites["half"] == []  # No gated MLP: the rule has nothing to act on.


def _frozen_rms_norm(self: Any, x: torch.Tensor) -> torch.Tensor:
    scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    return self.weight * (x * scale.detach())


def _frozen_layer_norm(self: Any, x: torch.Tensor) -> torch.Tensor:
    std = (x.var(-1, keepdim=True, unbiased=False) + self.eps).sqrt()
    return (x - x.mean(-1, keepdim=True)) / std.detach() * self.weight + self.bias


def _constant_gate_mlp(self: Any, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(self.act_fn(self.gate_proj(x).detach()) * self.up_proj(x))


def _lxt_cp_twin(arch: str) -> nn.Module:
    """LXT's CP-LRP on the HF modules: a frozen norm scale, a constant MLP gate and the
    Identity-rule on plain MLP activations (attention is detached by the caller)."""
    twin = _hf_twin(arch, _attn_implementation="eager")
    patches = {"LlamaRMSNorm": _frozen_rms_norm, "LayerNorm": _frozen_layer_norm}
    patches["LlamaMLP"] = _constant_gate_mlp
    for module in twin.modules():
        patch = patches.get(type(module).__name__)
        if patch is not None:
            setattr(module, "forward", types.MethodType(patch, module))
        act = getattr(module, "act", None) if type(module).__name__ == "GPT2MLP" else None
        if isinstance(act, nn.Module):
            act.register_forward_hook(lambda m, a, y: a[0] * (y / a[0]).detach())
    return twin


@pytest.mark.parametrize("arch", ["llama", "gpt2"])
def test_lxt_cp_lrp_preset_matches_lxt(arch: str, monkeypatch: pytest.MonkeyPatch) -> None:
    rules: list[LRPRule] = ["ln", "ah", "gate", "identity"]
    _, grad, _ = _input_grad(_bridge(arch, _attn_implementation="eager"), rules)
    twin = _lxt_cp_twin(arch)
    softmax = torch.nn.functional.softmax
    monkeypatch.setattr(torch.nn.functional, "softmax", lambda *a, **k: softmax(*a, **k).detach())
    oracle = _hf_input_grad(twin)
    # HF RMSNorm computes in float32 even for a float64 model.
    torch.testing.assert_close(grad, oracle, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("arch", ["mixtral", "qwen2_moe"])
def test_constant_gates_conserve_relevance_on_moe(arch: str) -> None:
    # Constant routing weights and gates leave every MoE block linear in its input.
    model = _bridge(arch, bias_free=True, _experts_implementation="eager")
    x, grad, logit = _input_grad(model, ["ln", "ah", "gate", "identity"])
    torch.testing.assert_close((x * grad).sum(), logit, rtol=1e-6, atol=1e-7)
    x, grad, logit = _input_grad(model, "all")
    assert abs((x * grad).sum() - logit) > 1e-5  # The Half-rule lets the router absorb some.


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_forward_only_work_inside_the_block(arch: str) -> None:
    model = _bridge(arch)
    with lrp_rules(model):
        with torch.no_grad():
            model(_tokens())
        with torch.inference_mode():
            model(_tokens())
        model.generate(_tokens(), max_new_tokens=3, do_sample=False, verbose=False)
    model.requires_grad_(False)  # Nothing in the graph needs a gradient.
    with lrp_rules(model):
        model(_tokens())


def test_cached_decoding_inside_the_block() -> None:
    # On later steps T5's cross-attention takes K and V from the cache filled in the block.
    model = _bridge("t5")
    encoder, decoder = _tokens(0), _tokens(1)
    with lrp_rules(model, _ATTN_LRP):
        _, cache = model(
            encoder, decoder_input=decoder[:, :3], use_cache=True, return_type="logits_and_cache"
        )
        logits = model(encoder, decoder_input=decoder[:, 3:4], past_key_values=cache)
        logits[0, -1, 7].backward()
        model.generate(encoder, max_new_tokens=3, verbose=False)


def test_double_backward_raises() -> None:
    model = _bridge("llama")
    x: dict[str, torch.Tensor] = {}
    with lrp_rules(model), model.hooks(
        fwd_hooks=[("blocks.0.hook_in", lambda t, hook: x.update(x=t))]
    ):
        (grad,) = torch.autograd.grad(model(_tokens())[0, -1, 7], x["x"], create_graph=True)
        with pytest.raises(RuntimeError, match="differentiate twice"):
            grad.sum().backward()


def test_shared_modules_get_one_rule() -> None:
    # A norm (Llama) or projection (GPT-J, delegated) shared by two layers: same gradients as
    # two separate copies with the same weights; a scale applied twice would compound.
    for arch, rules in (("llama", ["ln"]), ("gptj", ["attn_half"])):
        model = _bridge(arch)
        first, second = model.blocks[0], model.blocks[1]
        part = "ln1" if arch == "llama" else "attn"
        with torch.no_grad():
            for a, b in zip(getattr(first, part).parameters(), getattr(second, part).parameters()):
                b.copy_(a)
        _, separate, _ = _input_grad(model, rules)
        if arch == "llama":
            second.ln1.set_original_component(first.ln1.original_component)
        else:
            second.attn._modules["q"] = first.attn.q
            setattr(second.attn.original_component, "q_proj", first.attn.q)
        _, shared, _ = _input_grad(model, rules)
        torch.testing.assert_close(shared, separate, rtol=0, atol=1e-12, msg=arch)


def test_padded_batches() -> None:
    # Per row, the relevance conserves that row's logit and padding gets none.
    model = _bridge("gpt2", bias_free=True)
    tokens = torch.cat([_tokens(0), _tokens(1)])
    mask = torch.ones_like(tokens)
    mask[0, :2] = 0  # Left padding.
    mask[1, -2:] = 0  # Right padding.
    last = torch.tensor([5, 3])
    store: dict[str, torch.Tensor] = {}
    keep = [("blocks.0.hook_in", lambda t, hook: store.update(x=t))]
    with lrp_rules(model), model.hooks(fwd_hooks=keep):
        logits = model(tokens, attention_mask=mask)[torch.arange(2), last, 7]
        (grad,) = torch.autograd.grad(logits.sum(), store["x"])
    relevance = (store["x"] * grad).sum(-1)
    torch.testing.assert_close(relevance.sum(-1), logits.detach(), rtol=1e-6, atol=1e-10)
    assert relevance[0, :2].abs().max() < 1e-12 and relevance[1, 4:].abs().max() < 1e-12


def _llava() -> TransformerBridge:
    torch.manual_seed(0)
    text = transformers.LlamaConfig(**_DIMS, **_COMMON, **_GQA)
    vision = transformers.CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
    )
    config = transformers.LlavaConfig(
        text_config=text, vision_config=vision, image_token_index=49, vision_feature_layer=-1
    )
    hf_model = transformers.LlavaForConditionalGeneration(config).double().eval()
    return build_bridge_from_module(
        hf_model, "LlavaForConditionalGeneration", hf_config=config, tokenizer=None, device="cpu"
    )


def test_vision_language_model() -> None:
    model = _llava()
    tokens = _tokens()
    tokens[0, 1:5] = 49
    pixels = torch.randn(1, 3, 28, 28, dtype=torch.float64, requires_grad=True)
    # CLIPVisionEncoderBridge.forward returns a ModelOutput under a Tensor annotation,
    # which the test suite's runtime type checker rejects with or without the rules.
    jaxtyping.config.update("jaxtyping_disable", True)
    try:
        with lrp_rules(model) as sites:
            logit = model(tokens, pixel_values=pixels)[0, -1, 7]
            (grad,) = torch.autograd.grad(logit, pixels)
    finally:
        jaxtyping.config.update("jaxtyping_disable", False)
    assert any(site.startswith("vision_encoder") for site in sites["ah"])
    assert "vision_projector" in sites["identity"] and torch.isfinite(grad).all()


def test_constant_gate_on_gated_attention_outputs() -> None:
    # HRM-Text multiplies each attention output by sigmoid(gate_proj(x)); with that gate and the
    # MLP gates constant, the bias-free model is linear in its input.
    model = _bridge("hrm_text", bias_free=True)
    store: dict[str, torch.Tensor] = {}
    keep = [("embed.hook_out", lambda t, hook: store.update(x=t))]
    with lrp_rules(model, ["ln", "ah", "gate", "identity"]) as sites, model.hooks(fwd_hooks=keep):
        logit = model(_tokens())[0, -1, 7]
        (grad,) = torch.autograd.grad(logit, store["x"])
    assert any(site.endswith("attn") for site in sites["gate"])
    torch.testing.assert_close((store["x"] * grad).sum(), logit.detach(), rtol=1e-6, atol=1e-7)
    with pytest.raises(NotImplementedError, match="gated output"):
        with lrp_rules(model):
            pass


def test_bypass_with_frozen_input_raises() -> None:
    # Frozen weights and input, but a gradient-requiring leaf inside the MLP: the processed
    # gated MLP calls its activation as a function, which the guard must still catch.
    model = _bridge("llama", compat="processed")
    model.requires_grad_(False)
    leaf = [("blocks.1.mlp.gate.hook_out", lambda t, hook: t.detach().requires_grad_())]
    with pytest.raises(RuntimeError, match="'identity' LRP rule did not run"):
        with lrp_rules(model), model.hooks(fwd_hooks=leaf):
            model(_tokens())


def test_attention_projection_bypass_raises() -> None:
    # GPT-J runs attention in HF; a k projection that skips the bridged module is caught,
    # although HF passes a KV cache to every layer.
    model = _bridge("gptj")
    hf_attn = model.blocks[0].attn.original_component
    weight = hf_attn.k_proj.original_component.weight
    bypass = nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype)
    bypass.weight = weight  # Same projection, but not the bridged module.
    setattr(hf_attn, "k_proj", bypass)
    for rules in (["ah"], ["attn_half"]):
        with pytest.raises(RuntimeError, match="did not run in blocks.0.attn"):
            _input_grad(model, rules)


def test_one_module_with_two_roles_raises() -> None:
    # An activation shared by a gated text MLP (constant gate) and a plain vision MLP
    # (Identity-rule) cannot take both rules.
    model = _llava()
    text_mlp: Any = model.blocks[0].mlp.original_component
    vision_mlp: Any = model.vision_encoder.encoder_layers[0].mlp.original_component
    text_mlp.act_fn = vision_mlp.activation_fn
    with pytest.raises(NotImplementedError, match="two rules"):
        with lrp_rules(model, ["ln", "ah", "gate", "identity"]):
            pass
