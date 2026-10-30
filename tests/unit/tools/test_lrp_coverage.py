"""Cases from the previous relevance-rule suite that test_lrp.py does not cover."""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable

import pytest
import torch
import torch.nn.functional as F
import transformers
from torch import nn

from tests.unit.tools.test_lrp import (
    _VOCAB,
    _bridge,
    _divide_gradient,
    _input_grad,
    _tokens,
)
from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import NormalizationBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)
from transformer_lens.tools.analysis import lrp_rules
from transformer_lens.tools.analysis.lrp import _FrozenScaleNorm, _ratio, _Rewire

# Rebuilt attention not in test_lrp.py: ALiBi (Bloom, MPT) and CodeGen's partial rotary.
_FAMILIES: dict[str, tuple[str, Callable[[], Any]]] = {
    "bloom": (
        "BloomForCausalLM",
        lambda: transformers.BloomConfig(vocab_size=_VOCAB, hidden_size=32, n_layer=2, n_head=4),
    ),
    "mpt": (
        "MptForCausalLM",
        lambda: transformers.MptConfig(
            vocab_size=_VOCAB, max_seq_len=32, d_model=32, expansion_ratio=2, n_layers=2, n_heads=4
        ),
    ),
    "codegen": (
        "CodeGenForCausalLM",
        lambda: transformers.CodeGenConfig(
            vocab_size=_VOCAB,
            n_positions=32,
            n_ctx=32,
            n_embd=32,
            n_layer=2,
            n_head=4,
            rotary_dim=4,
        ),
    ),
}


def _family_bridge(arch: str, bias_free: bool = False) -> TransformerBridge:
    """``_bridge`` for the families above (float64, eval mode)."""
    torch.manual_seed(0)
    architecture, make_config = _FAMILIES[arch]
    hf_config = make_config()
    hf_model = getattr(transformers, architecture)(hf_config).double().eval()
    model = build_bridge_from_module(
        hf_model, architecture, hf_config=hf_config, tokenizer=None, device="cpu"
    )
    if bias_free:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith("bias"):
                    param.zero_()
    return model.eval()


def _per_layer(model: TransformerBridge, point: str, fn: Callable[..., Any]) -> tuple:
    return tuple((f"blocks.{i}.attn.{point}", fn) for i in range(len(model.blocks)))


@pytest.mark.parametrize("rules", [None, ["ln", "ah", "gate", "identity"]])
@pytest.mark.parametrize("arch", _FAMILIES)
def test_rebuilt_attention_families_conserve_the_logit(arch: str, rules: Any) -> None:
    model = _family_bridge(arch, bias_free=True)
    x, grad, logit = _input_grad(model, "all" if rules is None else rules)
    torch.testing.assert_close((x * grad).sum(), logit, rtol=1e-6, atol=1e-7)
    tokens = _tokens()
    plain = model(tokens)
    with lrp_rules(model, rules):
        assert torch.equal(model(tokens), plain)


@pytest.mark.parametrize("arch", _FAMILIES)
def test_rebuilt_attention_families_match_attention_oracles(arch: str) -> None:
    model = _family_bridge(arch)
    # AH-rule: the pattern is a constant.
    _, grad, _ = _input_grad(model, ["ah"])
    detach = _per_layer(model, "hook_pattern", lambda t, hook: t.detach())
    _, oracle, _ = _input_grad(model, "plain", fwd_hooks=detach)
    torch.testing.assert_close(grad, oracle, rtol=0, atol=1e-12)
    # AttnLRP: pattern gradient / 4 (QK^T and pattern @ V), values / 2.
    _, grad, _ = _input_grad(model, ["attn_half"])
    quarter = _per_layer(model, "hook_pattern", lambda t, hook: _divide_gradient(t, 4))
    half = _per_layer(model, "hook_v", lambda t, hook: _divide_gradient(t, 2))
    _, oracle, _ = _input_grad(model, "plain", fwd_hooks=quarter + half)
    torch.testing.assert_close(grad, oracle, rtol=1e-10, atol=1e-12)


def _native() -> TransformerBridge:
    config = TransformerBridgeConfig(
        d_model=32,
        d_head=8,
        n_heads=4,
        n_layers=2,
        n_ctx=16,
        d_vocab=_VOCAB,
        d_mlp=64,
        act_fn="silu",
        normalization_type="LN",
        seed=0,
    )
    model: TransformerBridge = getattr(TransformerBridge, "boot_native")(config)
    return model.eval()


def test_native_source_attention_gets_the_rules_through_its_pattern_seam() -> None:
    # Native attention applies hook_pattern (as pattern_fn) before its own pattern @ V.
    model = _native()
    _, grad, _ = _input_grad(model, ["ah"])
    detach = _per_layer(model, "hook_pattern", lambda t, hook: t.detach())
    _, oracle, _ = _input_grad(model, "plain", fwd_hooks=detach)
    torch.testing.assert_close(grad, oracle, rtol=1e-5, atol=1e-7)
    _, grad, _ = _input_grad(model, ["attn_half"])
    quarter = _per_layer(model, "hook_pattern", lambda t, hook: _divide_gradient(t, 4))
    half = _per_layer(model, "hook_v", lambda t, hook: _divide_gradient(t, 2))
    _, oracle, _ = _input_grad(model, "plain", fwd_hooks=quarter + half)
    torch.testing.assert_close(grad, oracle, rtol=1e-5, atol=1e-7)
    # The native MLP calls its activation as a function, so the Identity-rule cannot reach it.
    with pytest.raises(NotImplementedError, match="no activation module.*without 'identity'"):
        with lrp_rules(model):
            pass
    with lrp_rules(model, ["ln", "ah"]) as sites:
        assert len(sites["ah"]) == 2 and sites["ln"]


def _gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


@pytest.mark.parametrize("act, ratio", [(F.silu, torch.sigmoid), (F.gelu, _gaussian_cdf)])
def test_identity_rule_passes_the_gradient_as_f_over_x(act: Callable, ratio: Callable) -> None:
    base = torch.randn(3, 5, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    base[0, 0, :2] = 0.0
    x = base.transpose(0, 2).requires_grad_()  # Batched and non-contiguous.
    y = act(x)
    out = _Rewire.apply(y.detach(), x, _ratio(y, x))
    assert torch.equal(out, y.detach())
    upstream = torch.randn_like(out)
    (grad,) = torch.autograd.grad((out * upstream).sum(), x)
    # At x == 0 the multiplier is 0 (as LRPforTL's stabilised ratio), not the limit f'(0).
    expected = torch.where(x == 0, 0.0, upstream * ratio(x.detach()))
    torch.testing.assert_close(grad, expected, rtol=1e-12, atol=1e-14)
    assert (grad[x == 0] == 0).all() and (x == 0).sum() == 2


class _OffsetRMSNorm(nn.Module):
    """Gemma-style RMSNorm: scales by ``1 + weight``."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float64))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1 + self.weight)


@pytest.mark.parametrize("kind", ["layernorm", "offset_rmsnorm", "parameter_free_rmsnorm"])
def test_ln_rule_correction_matches_the_closed_form(kind: str) -> None:
    torch.manual_seed(0)
    dim, eps = 16, 1e-5
    norm: nn.Module
    if kind == "layernorm":
        norm, centered = nn.LayerNorm(dim, eps=eps).double(), True
    elif kind == "offset_rmsnorm":
        norm, centered = _OffsetRMSNorm(dim, eps), False
    else:
        norm = nn.RMSNorm(dim, eps=eps, elementwise_affine=False).double()
        centered = False
    with torch.no_grad():
        for param in norm.parameters():
            param.copy_(torch.randn_like(param))
    bias = getattr(norm, "bias", None)
    x = torch.randn(2, 3, dim, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(2, 3, dim, dtype=torch.float64)
    out = _FrozenScaleNorm.apply(norm(x), x, None if bias is None else bias.detach(), centered, eps)
    assert torch.equal(out, norm(x).detach())
    params = list(norm.parameters())
    grad_x, *grad_params = torch.autograd.grad((out * upstream).sum(), [x, *params])

    xc = x.detach() - x.detach().mean(-1, keepdim=True) if centered else x.detach()
    scale = (xc.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    weight = getattr(norm, "weight", None)
    effective = 1.0 if weight is None else (1 + weight if kind == "offset_rmsnorm" else weight)
    weighted = upstream * effective.detach() if isinstance(effective, torch.Tensor) else upstream
    projected = weighted - weighted.mean(-1, keepdim=True) if centered else weighted
    torch.testing.assert_close(grad_x, projected / scale, rtol=1e-12, atol=1e-13)
    # The norm's own parameters keep their ordinary gradients.
    ordinary = torch.autograd.grad((norm(x) * upstream).sum(), params) if params else ()
    for got, want in zip(grad_params, ordinary):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-12)


# gpt2's norms take the python path; llama's run HF's own RMSNorm (native path).
@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_ln_rule_keeps_norm_parameter_gradients_ordinary(arch: str) -> None:
    model = _bridge(arch)
    hf_norm = model.blocks[0].ln1.original_component
    params = [p for p in (hf_norm.weight, getattr(hf_norm, "bias", None)) if p is not None]
    store: dict[str, torch.Tensor] = {}
    fwd = [("blocks.0.ln1.hook_in", lambda t, hook: store.update(x=t))]
    bwd = [("blocks.0.ln1.hook_out", lambda g, hook: store.update(g=g))]
    with lrp_rules(model, ["ln"]), model.hooks(fwd_hooks=fwd, bwd_hooks=bwd):
        logit = model(_tokens())[0, -1, 7]
        grads = torch.autograd.grad(logit, params)
    x, upstream = store["x"].detach(), store["g"]
    eps = getattr(hf_norm, "eps", None) or hf_norm.variance_epsilon
    centered = isinstance(hf_norm, nn.LayerNorm)
    xc = x - x.mean(-1, keepdim=True) if centered else x
    normalized = xc / (xc.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    expected = [(upstream * normalized).sum((0, 1)), upstream.sum((0, 1))]
    # HF RMSNorm computes in float32 even for a float64 model.
    for got, want in zip(grads, expected):
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-9)


def _scale_names(model: TransformerBridge) -> list[str]:
    return [m.hook_scale.name for m in model.modules() if isinstance(m, NormalizationBridge)]


def test_ln_rule_holds_an_edited_python_path_scale_constant() -> None:
    model = _bridge("gpt2")
    edited = "blocks.0.ln1.hook_scale"
    _, grad, _ = _input_grad(model, ["ln"], fwd_hooks=((edited, lambda t, hook: 2 * t),))
    oracle_hooks = tuple(
        (
            name,
            (lambda t, hook: (2 * t).detach()) if name == edited else (lambda t, hook: t.detach()),
        )
        for name in _scale_names(model)
    )
    _, oracle, _ = _input_grad(model, "plain", fwd_hooks=oracle_hooks)
    torch.testing.assert_close(grad, oracle, rtol=0, atol=1e-12)


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_backward_hooks_on_norm_internals_leave_the_rule_in_place(arch: str) -> None:
    # On llama these hooks make the norm fall back to the python path; the LN-rule follows.
    model = _bridge(arch)
    _, expected, _ = _input_grad(model, ["ln"])
    seen: dict[str, torch.Tensor] = {}
    points = ["blocks.0.ln1.hook_scale", "blocks.0.ln1.hook_normalized"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with model.hooks(
            bwd_hooks=[(p, lambda g, hook: seen.update({hook.name: g})) for p in points]
        ):
            _, grad, _ = _input_grad(model, ["ln"])
    assert set(seen) == set(points) and seen[points[0]].abs().sum() > 0
    # The fallback moves plain gradients by the same ~7e-8 (HF RMSNorm runs in float32).
    torch.testing.assert_close(grad, expected, rtol=1e-6, atol=2e-7)
