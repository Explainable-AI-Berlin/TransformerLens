"""Model-free checks of the LRP primitives, hook factories, coverage table and rule names."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.model_bridge.lrp.coverage import (
    STATUSES,
    LRPCoverage,
    classify_scale_hook,
    compress_keys,
)
from transformer_lens.model_bridge.lrp.hooks import (
    LRPActivation,
    LRPState,
    make_mlp_product_hook,
    make_pattern_hook,
    make_scale_hook,
    make_value_hook,
)
from transformer_lens.model_bridge.lrp.mixin import _normalize_rule
from transformer_lens.model_bridge.lrp.rules import (
    LRPRules,
    half_gradient,
    identity_act,
    stabilize,
)


def _state(*rules: str, enabled: bool = True) -> LRPState:
    state = LRPState(enabled=enabled)
    state.rules = set(rules)
    return state


class TestPrimitives:
    def test_identity_act_function(self):
        torch.manual_seed(1)
        x = torch.randn(3, 5, requires_grad=True)
        upstream = torch.randn(3, 5)
        out = identity_act(x, F.silu)
        torch.testing.assert_close(out, F.silu(x), rtol=1e-5, atol=1e-5)
        out.backward(upstream)
        ratio = F.silu(x.detach()) / stabilize(x.detach())
        torch.testing.assert_close(x.grad, upstream * ratio, rtol=1e-5, atol=1e-6)

    def test_half_gradient_keeps_value_and_halves_gradient(self):
        x = torch.randn(4, requires_grad=True)
        y = half_gradient(x)
        assert torch.equal(y, x.detach())
        y.sum().backward()
        assert torch.equal(x.grad, torch.full_like(x, 0.5))

    def test_stabilize_moves_values_away_from_zero(self):
        z = torch.tensor([0.0, 1.0, -1.0])
        s = stabilize(z)
        assert s[0] > 0 and s[1] > 1 and s[2] < -1
        assert torch.isfinite(z / s).all()


class TestHookFactories:
    def test_hooks_return_none_when_off_or_unselected(self):
        x = torch.randn(2, 3, requires_grad=True)
        for state in (_state("ln-rule", "ah-rule", enabled=False), _state()):
            hooks = (
                make_scale_hook(state, LRPRules.LN_RULE),
                make_pattern_hook(state),
                make_value_hook(state),
                make_mlp_product_hook(state),
            )
            for hook in hooks:
                assert hook(x, hook=None) is None, hook.__name__

    def test_scale_hook_detaches_only_under_its_own_rule(self):
        x = torch.randn(2, 3, requires_grad=True)
        state = _state("ln-rule")
        out = make_scale_hook(state, LRPRules.LN_RULE)(x)
        assert out is not None and not out.requires_grad and torch.equal(out, x.detach())
        assert make_scale_hook(state, LRPRules.LN_RULE_QK)(x) is None

    def test_pattern_hook_detaches_under_ah_rule_and_halves_under_half_rule(self):
        x = torch.randn(2, 3, requires_grad=True)
        detached = make_pattern_hook(_state("ah-rule", "half-rule-attn"))(x)
        assert detached is not None and not detached.requires_grad
        halved = make_pattern_hook(_state("half-rule-attn"))(x)
        assert halved is not None
        halved.sum().backward()
        assert torch.equal(x.grad, torch.full_like(x, 0.5))

    def test_value_and_product_hooks_halve_the_gradient(self):
        for factory, rule in (
            (make_value_hook, "half-rule-attn"),
            (make_mlp_product_hook, "half-rule-mlp"),
        ):
            x = torch.randn(2, 3, requires_grad=True)
            out = factory(_state(rule))(x)
            assert out is not None and torch.equal(out, x.detach())
            out.sum().backward()
            assert torch.equal(x.grad, torch.full_like(x, 0.5))

    def test_lrp_activation_matches_wrapped_module_and_swaps_jacobian(self):
        torch.manual_seed(2)
        act = nn.SiLU()
        state = _state("identity-rule-act", enabled=False)
        wrapper = LRPActivation(state, act)
        upstream = torch.randn(3, 4)

        x = torch.randn(3, 4, requires_grad=True)
        wrapper(x).backward(upstream)
        x_plain = x.detach().clone().requires_grad_(True)
        act(x_plain).backward(upstream)
        assert torch.equal(wrapper(x), act(x))
        torch.testing.assert_close(x.grad, x_plain.grad)

        state.enabled = True
        x = torch.randn(3, 4, requires_grad=True)
        wrapper(x).backward(upstream)
        ratio = act(x.detach()) / stabilize(x.detach())
        torch.testing.assert_close(x.grad, upstream * ratio, rtol=1e-5, atol=1e-6)

    def test_state_active_requires_enabled_and_selected(self):
        state = _state("ah-rule", enabled=False)
        assert not state.active(LRPRules.AH_RULE)
        state.enabled = True
        assert state.active(LRPRules.AH_RULE)
        assert not state.active(LRPRules.LN_RULE)


class TestCoverageTable:
    def test_scale_hook_classification(self):
        assert classify_scale_hook("blocks.3.ln1.hook_scale") is LRPRules.LN_RULE
        assert classify_scale_hook("blocks.3.ln2.hook_scale") is LRPRules.LN_RULE
        assert classify_scale_hook("ln_final.hook_scale") is LRPRules.LN_RULE
        assert classify_scale_hook("blocks.3.ln1_post.hook_scale") is LRPRules.LN_RULE_POST
        assert classify_scale_hook("blocks.3.ln2_post.hook_scale") is LRPRules.LN_RULE_POST
        assert classify_scale_hook("blocks.3.attn.q_norm.hook_scale") is LRPRules.LN_RULE_QK
        assert classify_scale_hook("blocks.3.attn.k_norm.hook_scale") is LRPRules.LN_RULE_QK

    def test_compress_keys(self):
        assert compress_keys(["blocks.0", "blocks.1", "blocks.2", "blocks.5", "ln_final"]) == (
            "blocks.0-2, blocks.5, ln_final"
        )
        assert compress_keys(["blocks.7"]) == "blocks.7"
        assert compress_keys([]) == ""

    def test_repr_is_summary_and_detail_compresses_ranges(self):
        cov = LRPCoverage()
        for i in range(26):
            cov.set(LRPRules.AH_RULE, f"blocks.{i}", "applied", "AttentionBridge: hook_pattern")
            cov.set(LRPRules.LN_RULE_POST, f"blocks.{i}", "absent", "no post-sublayer norm")
        cov.set(LRPRules.LN_RULE, "ln_final", "applied", "hook_scale")
        cov.set(LRPRules.HALF_RULE_MLP, "blocks.3", "unsupported", "no out.hook_in")

        # A bare value in a notebook must not show the raw dict.
        assert repr(cov) == str(cov)
        assert "CoverageEntry" not in repr(cov)
        assert "ah-rule            applied 26" in str(cov)
        assert "absent on blocks.0-25: no post-sublayer norm" in str(cov)
        for rule in LRPRules:
            assert rule.value in str(cov)
        detail = cov.detail(LRPRules.AH_RULE)
        assert detail.splitlines()[0] == "ah-rule:"
        assert "blocks.0-25" in detail and len(detail.splitlines()) == 2
        assert "half-rule-mlp:" in cov.detail() and "blocks.3" in cov.detail()

    def test_keys_with_and_counts(self):
        cov = LRPCoverage()
        cov.set(LRPRules.LN_RULE, "blocks.0", "applied", "ln1, ln2")
        cov.set(LRPRules.LN_RULE, "blocks.1", "partial", "hooked ln1; cannot hook x")
        counts = cov.counts(LRPRules.LN_RULE)
        assert set(counts) == set(STATUSES)
        assert counts["applied"] == 1 and counts["partial"] == 1 and counts["absent"] == 0
        assert cov.keys_with("ln-rule", "partial", "unsupported") == ["blocks.1"]
        assert cov.table(LRPRules.AH_RULE) == {}


class TestRuleNames:
    def test_normalize_rule_accepts_names_and_members(self):
        assert _normalize_rule("ah-rule") is LRPRules.AH_RULE
        assert _normalize_rule(LRPRules.LN_RULE_QK) is LRPRules.LN_RULE_QK

    def test_unknown_rule_lists_the_valid_names(self):
        with pytest.raises(ValueError, match="valid rules: 'identity-rule-act'"):
            _normalize_rule("no-such-rule")
