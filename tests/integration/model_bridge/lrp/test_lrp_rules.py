"""Each rule changes the backward pass in exactly the intended way."""

import pytest
import torch

from tests.integration.model_bridge.lrp.helpers import (
    assert_close,
    norm_eps,
    norm_weight,
    rms_norm_input_grad,
    run_backward,
)
from transformer_lens.model_bridge.lrp.rules import stabilize


def _patched(bridge, *rules):
    bridge.patch_lrp()
    bridge.lrp_on()
    bridge.set_lrp_rules(*rules)
    return bridge


def _component(bridge, path):
    component = bridge
    for part in path.split("."):
        component = component[int(part)] if part.isdigit() else getattr(component, part)
    return component


def assert_norm_scale_detached(bridge, tokens, norm_path, detached):
    """The local input gradient of ``norm_path`` matches RMSNorm with/without detached scale."""
    norm = _component(bridge, norm_path)
    _, captured, cache = run_backward(
        bridge, tokens, capture=[f"{norm_path}.hook_in"], cache=[f"{norm_path}.hook_out"]
    )
    x = captured[f"{norm_path}.hook_in"]
    upstream = cache[f"{norm_path}.hook_out_grad"]
    weight, eps = norm_weight(norm), norm_eps(norm)
    with_detach = rms_norm_input_grad(x, weight, eps, upstream, detach_scale=True)
    without = rms_norm_input_grad(x, weight, eps, upstream, detach_scale=False)
    assert not torch.allclose(with_detach, without, rtol=1e-3, atol=1e-6), "test not sensitive"
    expected, other = (with_detach, without) if detached else (without, with_detach)
    assert_close(x.grad, expected, rtol=1e-4, atol=1e-6)
    assert not torch.allclose(x.grad, other, rtol=1e-3, atol=1e-6)


def _last_block(bridge):
    # The rules act in every block, so an earlier block also sees a changed upstream gradient.
    # Only the last block receives the same upstream gradient with and without the rule.
    return f"blocks.{bridge.cfg.n_layers - 1}"


# ---------------------------------------------------------------------------------- ln rules


@pytest.mark.parametrize("norm_path", ["blocks.0.ln1", "blocks.1.ln2", "ln_final"])
def test_ln_rule_detaches_block_norm_scale(tiny_bridge, tokens, norm_path):
    assert_norm_scale_detached(_patched(tiny_bridge, "ln-rule"), tokens, norm_path, True)


def test_ln_rule_leaves_qk_norms_attached(tiny_bridge, tokens):
    bridge = _patched(tiny_bridge, "ln-rule")
    assert_norm_scale_detached(bridge, tokens, "blocks.0.attn.q_norm", False)


@pytest.mark.parametrize("norm_path", ["blocks.0.attn.q_norm", "blocks.1.attn.k_norm"])
def test_ln_rule_qk_detaches_qk_norm_scale(tiny_bridge, tokens, norm_path):
    assert_norm_scale_detached(_patched(tiny_bridge, "ln-rule-qk"), tokens, norm_path, True)


def test_ln_rule_qk_leaves_block_norms_attached(tiny_bridge, tokens):
    bridge = _patched(tiny_bridge, "ln-rule-qk")
    assert_norm_scale_detached(bridge, tokens, "blocks.0.ln1", False)


def test_no_rule_keeps_full_norm_jacobian(tiny_bridge, tokens):
    bridge = _patched(tiny_bridge, "ln-rule")
    bridge.lrp_off()
    assert_norm_scale_detached(bridge, tokens, "blocks.0.ln1", False)


@pytest.mark.parametrize("norm_path", ["blocks.0.ln1_post", "blocks.1.ln2_post"])
def test_ln_rule_post_detaches_post_norms_on_gemma(tiny_gemma_bridge, tokens, norm_path):
    bridge = _patched(tiny_gemma_bridge, "ln-rule-post")
    assert_norm_scale_detached(bridge, tokens, norm_path, True)


def test_ln_rule_post_leaves_pre_norms_attached_on_gemma(tiny_gemma_bridge, tokens):
    bridge = _patched(tiny_gemma_bridge, "ln-rule-post")
    assert_norm_scale_detached(bridge, tokens, "blocks.0.ln1", False)
    assert_norm_scale_detached(bridge, tokens, "blocks.1.ln2", False)


def test_ln_rule_leaves_post_norms_attached_on_gemma(tiny_gemma_bridge, tokens):
    bridge = _patched(tiny_gemma_bridge, "ln-rule")
    assert_norm_scale_detached(bridge, tokens, "blocks.0.ln1", True)
    assert_norm_scale_detached(bridge, tokens, "blocks.0.ln1_post", False)
    assert_norm_scale_detached(bridge, tokens, "blocks.1.ln2_post", False)


# ------------------------------------------------------------------------------ attention


def test_ah_rule_blocks_gradient_into_attention_scores(tiny_bridge, tokens):
    bridge = _patched(tiny_bridge, "ah-rule")
    name = "blocks.0.attn.hook_attn_scores"
    _, captured, _ = run_backward(bridge, tokens, capture=[name])
    assert captured[name].grad is None
    bridge.lrp_off()
    _, captured, _ = run_backward(bridge, tokens, capture=[name])
    assert captured[name].grad is not None and captured[name].grad.abs().sum() > 0


def test_ah_rule_keeps_value_path(tiny_bridge, tokens):
    bridge = _patched(tiny_bridge, "ah-rule")
    name = "blocks.0.attn.v.hook_out"
    _, _, cache = run_backward(bridge, tokens, cache=[name])
    assert cache[f"{name}_grad"].abs().sum() > 0


@pytest.mark.parametrize("kind", ["qwen3", "gpt2"])
def test_half_rule_attn_halves_attention_input_gradient(make_bridge, tokens, kind):
    bridge = _patched(make_bridge(kind), "half-rule-attn")
    name = f"{_last_block(bridge)}.ln1.hook_out"  # feeds only the q/k/v projections
    _, _, with_rule = run_backward(bridge, tokens, cache=[name])
    bridge.lrp_off()
    _, _, plain = run_backward(bridge, tokens, cache=[name])
    assert_close(with_rule[f"{name}_grad"], 0.5 * plain[f"{name}_grad"], rtol=1e-6, atol=0)


@pytest.mark.parametrize(("rule", "norm"), [("half-rule-attn", "ln1"), ("half-rule-mlp", "ln2")])
def test_half_rules_under_compatibility_mode_without_processing(tiny_bridge, tokens, rule, norm):
    """Compatibility mode renames hook points to their HookedTransformer aliases and reshapes
    ``v.hook_out`` per head through a hook conversion. The survey must still find the
    bridge-native points, and the halving commutes with the linear reshape."""
    tiny_bridge.enable_compatibility_mode(no_processing=True)
    bridge = _patched(tiny_bridge, rule)
    assert bridge.get_lrp_coverage().keys_with(rule, "applied") == ["blocks.0", "blocks.1"]
    name = f"{_last_block(bridge)}.{norm}.hook_out"
    _, _, with_rule = run_backward(bridge, tokens, cache=[name])
    bridge.lrp_off()
    _, _, plain = run_backward(bridge, tokens, cache=[name])
    assert_close(with_rule[f"{name}_grad"], 0.5 * plain[f"{name}_grad"], rtol=1e-6, atol=0)


# ------------------------------------------------------------------------------------ mlp


def test_half_rule_mlp_halves_mlp_input_gradient(tiny_bridge, tokens):
    bridge = _patched(tiny_bridge, "half-rule-mlp")
    name = f"{_last_block(bridge)}.ln2.hook_out"  # feeds only the MLP
    _, _, with_rule = run_backward(bridge, tokens, cache=[name])
    bridge.lrp_off()
    _, _, plain = run_backward(bridge, tokens, cache=[name])
    assert_close(with_rule[f"{name}_grad"], 0.5 * plain[f"{name}_grad"], rtol=1e-6, atol=0)


@pytest.mark.parametrize("kind", ["qwen3", "gemma2"])
def test_identity_rule_act_in_model(make_bridge, tokens, kind):
    bridge = _patched(make_bridge(kind), "identity-rule-act")
    gate_name, up_name, product_name = (
        "blocks.0.mlp.gate.hook_out",
        "blocks.0.mlp.in.hook_out",
        "blocks.0.mlp.out.hook_in",
    )
    _, captured, cache = run_backward(
        bridge, tokens, capture=[gate_name, up_name], cache=[product_name]
    )
    gate, up = captured[gate_name], captured[up_name]
    upstream = cache[f"{product_name}_grad"]
    act = _component(bridge, "blocks.0").mlp.original_component.act_fn.act  # the wrapped one
    ratio = act(gate.detach()) / stabilize(gate.detach())
    assert_close(gate.grad, upstream * up.detach() * ratio, rtol=1e-4, atol=1e-6)
    assert_close(up.grad, upstream * act(gate.detach()), rtol=1e-4, atol=1e-6)
    # Without the rule the activation's true derivative is used instead.
    bridge.lrp_off()
    _, captured, cache = run_backward(
        bridge, tokens, capture=[gate_name, up_name], cache=[product_name]
    )
    gate, up = captured[gate_name], captured[up_name]
    upstream = cache[f"{product_name}_grad"]
    ratio = act(gate.detach()) / stabilize(gate.detach())
    assert not torch.allclose(gate.grad, upstream * up.detach() * ratio, rtol=1e-3, atol=1e-6)


def test_identity_rule_keeps_mlp_internal_gradients_cacheable(tiny_bridge, tokens):
    """Every MLP-internal hook point still reports a gradient under the identity rule."""
    bridge = _patched(tiny_bridge, "identity-rule-act")
    names = ["blocks.0.mlp.gate.hook_out", "blocks.0.mlp.in.hook_out", "blocks.0.mlp.out.hook_in"]
    _, _, cache = run_backward(bridge, tokens, cache=names)
    for name in names:
        assert f"{name}_grad" in cache, name
        assert cache[f"{name}_grad"].abs().sum() > 0, name


def test_identity_rule_rejected_on_processed_weights_path(tiny_bridge):
    bridge = _patched(tiny_bridge, "half-rule-mlp")
    bridge.blocks[0].mlp._use_processed_weights = True
    try:
        with pytest.raises(ValueError, match="identity-rule-act"):
            bridge.set_lrp_rules("identity-rule-act")
        bridge.set_lrp_rules("half-rule-mlp", "ah-rule")  # other rules unaffected
    finally:
        bridge.blocks[0].mlp._use_processed_weights = False


def test_identity_rule_rejected_under_real_compatibility_mode(tiny_bridge):
    """Weight processing makes the gated MLP use a functional activation that bypasses the
    wrapped ``act_fn``; the rule is refused rather than silently skipped."""
    tiny_bridge.patch_lrp()
    tiny_bridge.enable_compatibility_mode()
    with pytest.raises(ValueError, match="compatibility mode with weight processing"):
        tiny_bridge.set_lrp_rules("identity-rule-act")
    tiny_bridge.set_lrp_rules("ah-rule", "ln-rule")


def test_lrp_on_warns_when_compatibility_mode_followed_rule_selection(tiny_bridge):
    tiny_bridge.patch_lrp()
    tiny_bridge.set_lrp_rules("identity-rule-act")
    tiny_bridge.enable_compatibility_mode()
    with pytest.warns(UserWarning, match="no longer applies"):
        tiny_bridge.lrp_on()
