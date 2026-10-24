"""The coverage table and the checks set_lrp_rules derives from it."""

import pytest

from transformer_lens.model_bridge.lrp.coverage import STATUSES
from transformer_lens.model_bridge.lrp.rules import LRPRules


def _statuses(bridge, rule):
    return {key: entry.status for key, entry in bridge.get_lrp_coverage().table(rule).items()}


def test_qwen3_everything_applied_except_post_norms(tiny_bridge):
    tiny_bridge.patch_lrp()
    blocks = {"blocks.0", "blocks.1"}
    for rule in LRPRules:
        table = _statuses(tiny_bridge, rule)
        if rule is LRPRules.LN_RULE_POST:
            assert table == {b: "absent" for b in blocks}
        elif rule is LRPRules.LN_RULE:
            assert table == {**{b: "applied" for b in blocks}, "ln_final": "applied"}
        else:
            assert table == {b: "applied" for b in blocks}, rule


def test_vacuous_rule_only_warns(tiny_bridge):
    tiny_bridge.patch_lrp()
    tiny_bridge.lrp_on()
    with pytest.warns(UserWarning, match="'ln-rule-post' applies to no component"):
        tiny_bridge.set_lrp_rules("ln-rule-post", "ah-rule")
    assert tiny_bridge._lrp_state.rules == {"ln-rule-post", "ah-rule"}


def test_gemma_post_norms_applied_and_qk_norms_absent(tiny_gemma_bridge):
    tiny_gemma_bridge.patch_lrp()
    assert set(_statuses(tiny_gemma_bridge, LRPRules.LN_RULE_POST).values()) == {"applied"}
    assert set(_statuses(tiny_gemma_bridge, LRPRules.LN_RULE_QK).values()) == {"absent"}
    table = tiny_gemma_bridge.get_lrp_coverage().table(LRPRules.LN_RULE_POST)
    assert table["blocks.0"].detail == "ln1_post, ln2_post"


def test_gpt2_coverage(make_bridge):
    bridge = make_bridge("gpt2")
    bridge.patch_lrp()
    bridge.lrp_on()
    assert set(_statuses(bridge, LRPRules.AH_RULE).values()) == {"applied"}
    assert set(_statuses(bridge, LRPRules.HALF_RULE_ATTN).values()) == {"applied"}
    assert set(_statuses(bridge, LRPRules.LN_RULE).values()) == {"applied"}  # LayerNorm too
    assert set(_statuses(bridge, LRPRules.HALF_RULE_MLP).values()) == {"absent"}
    assert set(_statuses(bridge, LRPRules.IDENTITY_RULE_ACT).values()) == {"unsupported"}
    with pytest.warns(UserWarning, match="'half-rule-mlp' applies to no component"):
        bridge.set_lrp_rules("half-rule-mlp", "ah-rule")
    with pytest.raises(ValueError, match="identity-rule-act cannot be applied correctly on 2 of 2"):
        bridge.set_lrp_rules("identity-rule-act")
    with pytest.warns(UserWarning, match="ungated MLP"):
        bridge.set_lrp_rules("identity-rule-act", "ah-rule", strict=False)
    assert bridge._lrp_state.rules == {"identity-rule-act", "ah-rule"}


def test_moe_mlp_rules_unsupported(make_bridge):
    bridge = make_bridge("qwen3_moe")
    bridge.patch_lrp()
    bridge.lrp_on()
    for rule in (LRPRules.HALF_RULE_MLP, LRPRules.IDENTITY_RULE_ACT):
        assert set(_statuses(bridge, rule).values()) == {"unsupported"}, rule
    for rule in (LRPRules.AH_RULE, LRPRules.HALF_RULE_ATTN, LRPRules.LN_RULE, LRPRules.LN_RULE_QK):
        assert set(_statuses(bridge, rule).values()) >= {"applied"}, rule
    with pytest.raises(ValueError, match="experts are fused"):
        bridge.set_lrp_rules("half-rule-mlp")
    bridge.set_lrp_rules("ah-rule", "ln-rule", "ln-rule-qk")


def test_coverage_report_lists_every_rule(make_bridge):
    bridge = make_bridge("gpt2")
    bridge.patch_lrp()
    report = str(bridge.get_lrp_coverage())
    for rule in LRPRules:
        assert rule.value in report
    assert "identity-rule-act" in report and "unsupported" in report
    counts = bridge.get_lrp_coverage().counts(LRPRules.LN_RULE)
    assert set(counts) == set(STATUSES) and counts["applied"] == 3  # ln1, ln2 per block + ln_final


def test_coverage_requires_patching(tiny_bridge):
    with pytest.raises(RuntimeError, match="not installed"):
        tiny_bridge.get_lrp_coverage()
