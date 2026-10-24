"""End-to-end input relevance on a tiny model: the whole workflow from the quick start."""

import torch

RULES = ["identity-rule-act", "half-rule-mlp", "ah-rule", "ln-rule"]


def _input_relevance(bridge, tokens):
    _, cache = bridge.run_with_cache(
        tokens, return_type="loss", names_filter=["embed.hook_out"], incl_bwd=True
    )
    return (cache["embed.hook_out"] * cache["embed.hook_out_grad"]).sum(-1)


def test_input_relevance_workflow(tiny_bridge, tokens):
    plain_logits = tiny_bridge(tokens)
    tiny_bridge.patch_lrp()
    tiny_bridge.lrp_off()
    assert torch.equal(tiny_bridge(tokens), plain_logits)
    gxi = _input_relevance(tiny_bridge, tokens)
    tiny_bridge.lrp_on()
    tiny_bridge.set_lrp_rules(*RULES)
    assert (tiny_bridge(tokens) - plain_logits).abs().max() < 1e-4
    lrp = _input_relevance(tiny_bridge, tokens)
    assert torch.isfinite(lrp).all()
    assert lrp.shape == gxi.shape == tokens.shape
    assert not torch.allclose(lrp, gxi)
    tiny_bridge.unpatch_lrp()
    assert torch.equal(tiny_bridge(tokens), plain_logits)
