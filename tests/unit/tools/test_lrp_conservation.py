"""Relevance conservation under the LRP rules, beyond the input-level check in test_lrp.py.

Layer-wise: on a bias-free model the residual stream carries the explained logit at every
block boundary, so errors that cancel between layers cannot hide behind a correct input
sum. Per component: each norm, attention and MLP passes on exactly the relevance it
receives. With biases: the input relevance plus the relevance each bias absorbs (bias times
its gradient) is the logit, so conservation is checked on unmodified models too.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
import torch

from tests.unit.tools.test_lrp import _CONSERVATION_ARCHS, _bridge, _tokens
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    MLPBridge,
    NormalizationBridge,
)
from transformer_lens.tools.analysis import lrp_rules
from transformer_lens.tools.analysis.lrp import DEFAULT_RULES, LRPRule

_CP_LRP: list[LRPRule] = ["ln", "ah", "gate", "identity"]
_PRESETS = {"default": DEFAULT_RULES, "cp_lrp": _CP_LRP}
_DECODERS = [arch for arch in _CONSERVATION_ARCHS if arch not in ("bert", "roberta")]
# Decoders with biases in their norms, attention and MLPs (and GPT-J in its unembedding).
_BIASED = ["gpt2", "opt", "neox", "gptj"]


def _relevance_at(
    model: TransformerBridge, rules: Any, names: list[str], fwd_hooks: tuple = ()
) -> tuple[list[Optional[torch.Tensor]], torch.Tensor]:
    """Relevance (activation x gradient, summed) at each hook point, and the target logit.

    ``fwd_hooks`` run first, so a hook that replaces the tensor is what gets measured. A hook
    point whose tensor does not reach the logit (a mirrored, observation-only point) is None.
    """
    store: dict[str, torch.Tensor] = {}
    keep = [(name, lambda t, hook: store.update({hook.name: t})) for name in names]
    with lrp_rules(model, rules), model.hooks(fwd_hooks=[*fwd_hooks, *keep]):
        logit = model(_tokens())[0, -1, 7]
        grads = torch.autograd.grad(logit, [store[name] for name in names], allow_unused=True)
    relevances = [
        None if grad is None else (store[name] * grad).sum() for name, grad in zip(names, grads)
    ]
    return relevances, logit.detach()


def _residual_points(model: TransformerBridge) -> list[str]:
    names = []
    for index in range(len(model.blocks)):
        names += [f"blocks.{index}.hook_in", f"blocks.{index}.hook_out"]
    if "ln_final.hook_in" in model.hook_dict:
        names.append("ln_final.hook_in")
    return names


@pytest.mark.parametrize("preset", _PRESETS)
@pytest.mark.parametrize("arch", _CONSERVATION_ARCHS)
def test_relevance_is_the_logit_at_every_block_boundary(arch: str, preset: str) -> None:
    model = _bridge(arch, bias_free=True)
    names = _residual_points(model)
    relevances, logit = _relevance_at(model, _PRESETS[preset], names)
    for name, relevance in zip(names, relevances):
        assert relevance is not None, name
        # HF RMSNorm computes in float32 even for a float64 model.
        torch.testing.assert_close(relevance, logit, rtol=1e-6, atol=1e-7, msg=name)


def _components(model: TransformerBridge) -> list[Any]:
    kinds = (NormalizationBridge, AttentionBridge, MLPBridge)
    found = [
        child
        for block in model.blocks
        for child in block.children()
        if isinstance(child, kinds) and child.hook_in.name and child.hook_out.name
    ]
    return found + [model.ln_final]


def _clone(t: torch.Tensor, hook: Any) -> torch.Tensor:
    # A component-private tensor, so its gradient comes only through that component.
    return t.clone()


@pytest.mark.parametrize("arch", _DECODERS)
def test_every_component_passes_on_the_relevance_it_receives(arch: str) -> None:
    model = _bridge(arch, bias_free=True)
    components = _components(model)
    names = [name for c in components for name in (c.hook_in.name, c.hook_out.name)]
    private = tuple((name, _clone) for name in names)
    relevances, logit = _relevance_at(model, None, names, fwd_hooks=private)
    checked = []
    for component, relevance_in, relevance_out in zip(
        components, relevances[0::2], relevances[1::2]
    ):
        name = component.hook_in.name.rsplit(".", 1)[0]
        if relevance_in is None or relevance_out is None:
            continue  # OPT's ln2 and MLP only mirror hook points from the layer's own code.
        torch.testing.assert_close(
            relevance_in, relevance_out, rtol=1e-6, atol=1e-7 * abs(float(logit)), msg=name
        )
        checked.append(name)
    assert {"blocks.0.ln1", "blocks.0.attn", "ln_final"} <= set(checked)


@pytest.mark.parametrize("preset", _PRESETS)
@pytest.mark.parametrize("arch", _BIASED)
def test_input_and_bias_relevance_add_up_to_the_logit(arch: str, preset: str) -> None:
    model = _bridge(arch)
    biases = [
        param
        for name, param in model.named_parameters()
        if name.endswith("bias") and "embed" not in name
    ]
    assert biases
    # HF initialises biases to zero; random ones make the absorbed relevance visible.
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for bias in biases:
            bias.copy_(0.1 * torch.randn(bias.shape, generator=generator, dtype=bias.dtype))
    store: dict[str, torch.Tensor] = {}
    keep = [("blocks.0.hook_in", lambda t, hook: store.update(x=t))]
    with lrp_rules(model, _PRESETS[preset]), model.hooks(fwd_hooks=keep):
        logit = model(_tokens())[0, -1, 7]
        x_grad, *bias_grads = torch.autograd.grad(logit, [store["x"], *biases], allow_unused=True)
    absorbed = sum((b * g).sum() for b, g in zip(biases, bias_grads) if g is not None)
    total = (store["x"] * x_grad).sum() + absorbed
    torch.testing.assert_close(total, logit.detach(), rtol=1e-6, atol=1e-7)
    assert abs(float(absorbed)) > 1e-6  # The biases do absorb relevance here.
