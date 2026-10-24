"""The forward pass of a patched model must stay (numerically) the same."""

import pytest
import torch

from tests.integration.model_bridge.lrp.helpers import all_rules_except


def _pair(make_bridge, kind):
    plain = make_bridge(kind)
    patched = make_bridge(kind)
    patched.patch_lrp()
    return plain, patched


@pytest.mark.parametrize("kind", ["qwen3", "gemma2"])
def test_lrp_off_is_bit_identical(make_bridge, tokens, kind):
    plain, patched = _pair(make_bridge, kind)
    patched.lrp_off()
    assert torch.equal(plain(tokens), patched(tokens))


@pytest.mark.parametrize("kind", ["qwen3", "gemma2"])
def test_rules_without_identity_keep_forward(make_bridge, tokens, kind):
    plain, patched = _pair(make_bridge, kind)
    patched.lrp_on()
    patched.set_lrp_rules(*all_rules_except("identity-rule-act", "half-rule-attn"))
    torch.testing.assert_close(patched(tokens), plain(tokens), rtol=1e-5, atol=1e-5)


def test_half_rule_attn_keeps_forward(make_bridge, tokens):
    plain, patched = _pair(make_bridge, "qwen3")
    patched.lrp_on()
    patched.set_lrp_rules("half-rule-attn", "half-rule-mlp", "ln-rule")
    torch.testing.assert_close(patched(tokens), plain(tokens), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["qwen3", "gemma2"])
def test_identity_rule_perturbs_forward_below_1e4(make_bridge, tokens, kind):
    plain, patched = _pair(make_bridge, kind)
    patched.lrp_on()
    patched.set_lrp_rules(*all_rules_except("half-rule-attn"))
    assert (patched(tokens) - plain(tokens)).abs().max() < 1e-4


def test_lrp_off_after_on_restores_exact_forward(make_bridge, tokens):
    plain, patched = _pair(make_bridge, "qwen3")
    patched.lrp_on()
    patched.set_lrp_rules(*all_rules_except("half-rule-attn"))
    patched(tokens)
    patched.lrp_off()
    assert torch.equal(plain(tokens), patched(tokens))
