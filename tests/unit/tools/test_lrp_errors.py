"""Errors and warnings of lrp_rules: types, messages and remedies."""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

from tests.unit.tools.test_lrp import _bridge, _input_grad
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AltUpBlockBridge,
    GatedRMSNormBridge,
    OpaqueBlockBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.tools.analysis import (
    LRPConfigError,
    LRPError,
    LRPRuntimeError,
    LRPUnsupportedError,
    LRPWarning,
    lrp_rules,
)


@pytest.mark.parametrize(
    "error, builtin",
    [
        (LRPConfigError, ValueError),
        (LRPUnsupportedError, NotImplementedError),
        (LRPRuntimeError, RuntimeError),
    ],
)
def test_errors_are_lrp_errors_and_their_builtins(error: type, builtin: type) -> None:
    assert issubclass(error, LRPError) and issubclass(error, builtin)


def _enter(model: TransformerBridge, rules: Any = None) -> None:
    with lrp_rules(model, rules):
        pass


@pytest.mark.parametrize(
    "rules, match",
    [
        ("ln", r"not the string 'ln'; for a single rule pass \['ln'\]"),
        ((), "nonempty.*pass None for the default"),
        (["ln", "softmax"], r"unknown: \['softmax'\]"),
        (["ah", "attn_half"], "'ah' and 'attn_half' rewrite the same attention layers"),
        (["half", "gate"], "'half' and 'gate' rewrite the same gated products"),
    ],
)
def test_invalid_rule_sets_name_the_problem(rules: Any, match: str) -> None:
    model = _bridge("gpt2")
    with pytest.raises(LRPConfigError, match=match):
        _enter(model, rules)
    # A refused entry leaves nothing behind.
    _, grad, _ = _input_grad(model, "plain")
    _, plain, _ = _input_grad(_bridge("gpt2"), "plain")
    assert torch.equal(grad, plain)


def test_training_mode_and_nesting() -> None:
    model = _bridge("gpt2")
    model.blocks[0].mlp.original_component.train()
    with pytest.raises(LRPConfigError, match="evaluation mode"):
        _enter(model)
    model.eval()
    with lrp_rules(model):
        with pytest.raises(LRPRuntimeError, match="already active.*outer block"):
            _enter(model)


def _with_stub(component: nn.Module) -> TransformerBridge:
    model = _bridge("gpt2")
    model.add_module("stub", component.eval())
    return model


@pytest.mark.parametrize(
    "component, rules, match",
    [
        # Stubs added after setup lack a component prefix; match on the class name.
        (OpaqueBlockBridge("stub"), None, r"\(OpaqueBlockBridge\): not supported"),
        (AltUpBlockBridge("stub"), ["ln"], r"\(AltUpBlockBridge\): not supported"),
        (GatedRMSNormBridge("stub"), None, "gated norm: use rules without 'ln', 'half' or 'gate'"),
    ],
)
def test_unsupported_components_raise_on_entry(component: Any, rules: Any, match: str) -> None:
    with pytest.raises(LRPUnsupportedError, match=match):
        _enter(_with_stub(component), rules)


def test_gated_norm_is_only_refused_by_the_rules_it_breaks() -> None:
    _enter(_with_stub(GatedRMSNormBridge("stub")), ["ah", "identity"])


def test_gated_attention_output_refuses_attn_half() -> None:
    # HRM-Text multiplies each attention output by sigmoid(gate_proj(x)).
    with pytest.raises(LRPUnsupportedError, match="gated output: use rules without"):
        _enter(_bridge("hrm_text"), ["ln", "attn_half"])


def test_unknown_norm_class_raises() -> None:
    model = _bridge("llama")
    hf_norm = model.blocks[0].ln1.original_component
    hf_norm.__class__ = type("ScaleShift", (type(hf_norm),), {})
    with pytest.raises(LRPUnsupportedError, match="ScaleShift is not a known norm.*without 'ln'"):
        _enter(model, ["ln"])
    _enter(model, ["ah"])


def test_hf_attention_without_separate_projections_raises() -> None:
    # GPT-J runs attention in HF; shared K/V cannot be scaled separately.
    model = _bridge("gptj")
    model.blocks[0].attn._modules["v"] = model.blocks[0].attn._modules["k"]
    for rules in (["ah"], ["attn_half"]):
        with pytest.raises(LRPUnsupportedError, match="no separate q, k and v projections"):
            _enter(model, rules)


def test_keyword_called_norm_raises() -> None:
    model = _bridge("llama")
    hf_norm = model.blocks[0].ln1.original_component
    x = torch.randn(1, 3, 32, dtype=torch.float64, requires_grad=True)
    with lrp_rules(model, ["ln"]):
        with pytest.raises(LRPRuntimeError, match="keyword arguments.*without 'ln'"):
            hf_norm(hidden_states=x)


def test_experts_called_with_keywords_raise() -> None:
    model = _bridge("mixtral", dtype=torch.float32)
    experts: Any = model.blocks[0].mlp.original_component.experts
    if isinstance(experts, GeneralizedComponent):
        experts = experts.original_component
    with lrp_rules(model, ["half"]):
        with pytest.raises(LRPRuntimeError, match="not called positionally.*without 'half'"):
            experts(hidden_states=torch.zeros(3, 32))


def test_bypassed_rules_name_the_component_and_the_remedy() -> None:
    # A down projection that skips the bridged module never reaches hook_post.
    model = _bridge("llama")
    hf_mlp = model.blocks[0].mlp.original_component
    weight = hf_mlp.down_proj.original_component.weight
    bypass = nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype).eval()
    bypass.weight = weight
    setattr(hf_mlp, "down_proj", bypass)
    with pytest.raises(
        LRPRuntimeError,
        match="'half' LRP rule did not run in blocks.0.mlp.*" "Use rules without 'half'",
    ):
        _input_grad(model, ["half"])
    # Processed weights call the activation as a function.
    with pytest.raises(LRPRuntimeError, match="'identity'.*no_processing=True"):
        _input_grad(_bridge("llama", compat="processed"), "all")


def test_activations_outside_covered_components_raise_during_forward() -> None:
    # GLM-4-MoE's adapter does not bridge the block's shared experts (a gated MLP).
    model = _bridge("glm4_moe", dtype=torch.float32)
    with pytest.raises(LRPRuntimeError, match="shared_experts.*without 'half', 'gate' and"):
        _input_grad(model, ["ln", "ah", "half"])


def _lrp_warnings(model: TransformerBridge, rules: Any) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _enter(model, rules)
    return [w for w in caught if issubclass(w.category, LRPWarning)]


def test_explicit_rule_without_sites_warns() -> None:
    model = _bridge("gpt2")  # No gated MLP.
    caught = _lrp_warnings(model, ["ln", "half"])
    assert [str(w.message) for w in caught] == [
        "The LRP rule 'half' found no gated products (gated MLPs, MoE experts, gated "
        "attention outputs) in this model, so it has no effect."
    ]
    assert caught[0].filename == __file__  # Points at the caller's with statement.
    assert not _lrp_warnings(_bridge("llama"), ["ln", "half"])


@pytest.mark.parametrize(
    "rules",
    [
        None,
        ["ln", "identity", "half"],  # RelP
        ["ln", "attn_half", "half", "identity"],  # AttnLRP
        ["ln", "ah", "gate", "identity"],  # CP-LRP, which the docs recommend for GPT-2
    ],
)
def test_documented_presets_do_not_warn_on_models_lacking_an_operation(rules: Any) -> None:
    assert not _lrp_warnings(_bridge("gpt2"), rules)  # No gated MLP for 'half' or 'gate'.


@pytest.mark.parametrize("dtype, warned", [(torch.float16, True), (torch.bfloat16, False)])
def test_ln_rule_on_float16_warns(dtype: torch.dtype, warned: bool) -> None:
    model = _bridge("gpt2", dtype=dtype)
    caught = [w for w in _lrp_warnings(model, None) if "float16" in str(w.message)]
    assert bool(caught) == warned
    assert not [w for w in _lrp_warnings(model, ["ah"]) if "float16" in str(w.message)]
