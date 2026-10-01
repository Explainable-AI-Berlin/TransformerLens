"""Attention relevance rules on the base AttentionBridge protocol and its reconstructing roster.

A tiny one-block GPT-2 (``JointQKVAttentionBridge``, built offline from a programmatic
config like the sibling joint-QKV tests) pins the two rules' local VJPs against a
closed-form recompute of GPT-2 attention: the AH-rule's gradient into the attention
input equals the recompute with the post-softmax pattern detached, and the Half-rule's
equals half of the ordinary gradient exactly. The roster test freezes which
``AttentionBridge`` subclasses claim to reconstruct attention in Python, so a new
subclass must decide explicitly instead of inheriting a flag that would turn the rule
into a silent no-op on a delegating forward.
"""

import math
from typing import Iterator, Optional

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRules,
    use_relevance_rules,
)
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

TOKENS = torch.tensor([[3, 1, 4, 1, 5]])


@pytest.fixture(scope="module")
def gpt2_bridge() -> TransformerBridge:
    hf_config = GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=8, vocab_size=16)
    torch.manual_seed(0)
    return build_bridge_from_module(
        GPT2LMHeadModel(hf_config).eval(),
        "GPT2LMHeadModel",
        hf_config=hf_config,
        tokenizer=None,
        device="cpu",
    )


def _run_capturing_attention(
    bridge: TransformerBridge, rules: RelevanceRules
) -> dict[str, torch.Tensor]:
    """One rule-active backward; returns the attention input, its gradient, and the
    gradient the attention output received (the local VJP's three ingredients)."""
    attn = bridge.blocks[0].attn
    captured: dict[str, torch.Tensor] = {}

    def _capture(key: str):
        def _hook(tensor: torch.Tensor, hook: Optional[HookPoint] = None) -> None:
            captured[key] = tensor.detach().clone()

        return _hook

    attn.hook_in.add_hook(_capture("x"))
    attn.hook_in.add_hook(_capture("grad_in"), dir="bwd")
    attn.hook_out.add_hook(_capture("grad_out"), dir="bwd")
    attn.hook_attn_scores.add_hook(_capture("grad_scores"), dir="bwd")
    attn.v.hook_out.add_hook(_capture("grad_v"), dir="bwd")
    try:
        with use_relevance_rules(bridge, rules):
            bridge(TOKENS).sum().backward()
    finally:
        for hook in (attn.hook_in, attn.hook_out, attn.hook_attn_scores, attn.v.hook_out):
            hook.remove_hooks(dir="both")
        for parameter in bridge.parameters():
            parameter.grad = None
    return captured


def _gpt2_attention(
    bridge: TransformerBridge, x: torch.Tensor, detach_pattern: bool
) -> torch.Tensor:
    """GPT-2's eager attention from its own Conv1D weights, optionally with the
    post-softmax pattern held constant (the AH-rule's backward)."""
    hf_attn = bridge.blocks[0].attn.original_component
    n_heads = int(bridge.cfg.n_heads)
    batch, seq, d_model = x.shape
    head_dim = d_model // n_heads
    qkv = x @ hf_attn.c_attn.weight.detach() + hf_attn.c_attn.bias.detach()
    q, k, v = qkv.split(d_model, dim=-1)

    def _heads(t: torch.Tensor) -> torch.Tensor:
        return t.view(batch, seq, n_heads, head_dim).transpose(1, 2)

    scores = _heads(q) @ _heads(k).transpose(-1, -2) / math.sqrt(head_dim)
    causal = torch.ones(seq, seq, dtype=torch.bool).tril()
    pattern = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
    if detach_pattern:
        pattern = pattern.detach()
    z = (pattern @ _heads(v)).transpose(1, 2).reshape(batch, seq, d_model)
    return z @ hf_attn.c_proj.weight.detach() + hf_attn.c_proj.bias.detach()


def _oracle_grad_in(
    bridge: TransformerBridge, captured: dict[str, torch.Tensor], detach_pattern: bool
) -> torch.Tensor:
    x = captured["x"].detach().requires_grad_(True)
    out = _gpt2_attention(bridge, x, detach_pattern)
    (grad_in,) = torch.autograd.grad(out, x, grad_outputs=captured["grad_out"])
    return grad_in


class TestForwardIdentity:
    @pytest.mark.parametrize("kind", ["attention", "attention_half"])
    def test_active_forward_matches_inactive_forward(self, gpt2_bridge, kind):
        with torch.no_grad():
            baseline = gpt2_bridge(TOKENS)
        with use_relevance_rules(gpt2_bridge, RelevanceRules(**{kind: True})) as coverage:
            with torch.no_grad():
                active = gpt2_bridge(TOKENS)
        assert coverage.installed == ("blocks.0.attn",)
        assert torch.equal(active, baseline)


class TestVJPMatchesClosedFormOracle:
    def test_oracle_reproduces_the_ordinary_gradient(self, gpt2_bridge):
        """Control: with nothing detached the recompute must agree with the bridge's
        own ordinary backward, or the rule checks below would prove nothing."""
        captured = _run_capturing_attention(gpt2_bridge, RelevanceRules())
        expected = _oracle_grad_in(gpt2_bridge, captured, detach_pattern=False)
        torch.testing.assert_close(captured["grad_in"], expected, atol=1e-5, rtol=1e-4)

    def test_ah_rule_matches_recompute_with_detached_pattern(self, gpt2_bridge):
        captured = _run_capturing_attention(gpt2_bridge, RelevanceRules(attention=True))
        expected = _oracle_grad_in(gpt2_bridge, captured, detach_pattern=True)
        torch.testing.assert_close(captured["grad_in"], expected, atol=1e-5, rtol=1e-4)
        # Nothing reaches the scores, so their backward hook never fires; the value
        # path is untouched.
        assert "grad_scores" not in captured
        assert captured["grad_v"].abs().sum() > 0

    def test_ah_rule_disagrees_with_ordinary_autodiff(self, gpt2_bridge):
        plain = _run_capturing_attention(gpt2_bridge, RelevanceRules())
        assert "grad_scores" in plain
        detached = _oracle_grad_in(gpt2_bridge, plain, detach_pattern=True)
        assert not torch.allclose(plain["grad_in"], detached)

    def test_half_rule_halves_the_gradient_into_attention_exactly(self, gpt2_bridge):
        """Every path from the attention input to its output crosses the pattern or
        the values, each of which the rule halves, so the input gradient is exactly
        half of the ordinary one (scaling by one half is exact in floating point)."""
        plain = _run_capturing_attention(gpt2_bridge, RelevanceRules())
        halved = _run_capturing_attention(gpt2_bridge, RelevanceRules(attention_half=True))
        torch.testing.assert_close(halved["grad_out"], plain["grad_out"], rtol=0, atol=0)
        torch.testing.assert_close(halved["grad_in"], 0.5 * plain["grad_in"], rtol=1e-6, atol=0)
        torch.testing.assert_close(halved["grad_v"], 0.5 * plain["grad_v"], rtol=1e-6, atol=0)


class TestMutualExclusion:
    def test_nested_scope_with_the_other_kind_raises_and_keeps_the_outer_rule(self, gpt2_bridge):
        attn = gpt2_bridge.blocks[0].attn
        with use_relevance_rules(gpt2_bridge, RelevanceRules(attention=True)):
            with pytest.raises(ValueError, match="mutually exclusive"):
                with use_relevance_rules(gpt2_bridge, RelevanceRules(attention_half=True)):
                    pass
            assert attn._relevance_rule_attention_active is True
            assert attn._relevance_rule_attention_half_active is False
        assert attn._relevance_rule_attention_active is False
        assert attn._relevance_rule_attention_half_active is False


# Bridges whose own forward applies hook_pattern before their pattern @ V product and
# route both factors through _apply_attention_relevance_rules (directly or by
# inheriting an unchanged reconstruction).
RECONSTRUCTING_ATTENTION_BRIDGES = frozenset(
    {
        "PositionEmbeddingsAttentionBridge",
        "JointQKVAttentionBridge",
        "JointQKVPositionEmbeddingsAttentionBridge",
        "ALiBiJointQKVAttentionBridge",
        "MPTALiBiAttentionBridge",
        "BloomAttentionBridge",
        "CodeGenAttentionBridge",
        "MLAAttentionBridge",
        "GlmMoeDsaAttentionBridge",
        "_BaichuanAttentionBridge",
        "_InternLM2AttentionBridge",
        "_OpenAIGPTJointQKVAttentionBridge",
        "_Exaone4AttentionBridge",
        "_SmolLM3AttentionBridge",
        "_Cohere2AttentionBridge",
        "_BitNetAttentionBridge",
        "_LLaDAAttentionBridge",
    }
)
# Bridges that hand attention to the wrapped module: the rules are unsupported there.
DELEGATING_ATTENTION_BRIDGES = frozenset(
    {
        "NativeForwardAttentionBridge",
        "T5Gemma2MergedAttentionBridge",
        "_JetMoeAttentionBridge",
        "_LLaDA2FusedAttentionBridge",
    }
)


def _all_subclasses(cls: type[AttentionBridge]) -> Iterator[type[AttentionBridge]]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _all_subclasses(subclass)


class TestReconstructingRoster:
    def test_every_attention_bridge_subclass_declares_its_side(self):
        """A new subclass must land in exactly one of the two sets above: inheriting
        the reconstruction flag onto a delegating forward would make the attention
        rules a silent no-op, and leaving it off a reconstructing forward would make
        them raise for no reason."""
        # Importing the package registers every private per-architecture subclass.
        import transformer_lens.model_bridge.supported_architectures  # noqa: F401

        # Only library classes are policed; other test modules define their own
        # AttentionBridge doubles, which show up here once the whole suite is loaded.
        subclasses = {
            subclass.__name__: subclass
            for subclass in _all_subclasses(AttentionBridge)
            if subclass.__module__.startswith("transformer_lens.")
        }
        flagged = {name for name, cls in subclasses.items() if cls._reconstructs_attention}
        assert flagged == RECONSTRUCTING_ATTENTION_BRIDGES
        assert set(subclasses) - flagged == DELEGATING_ATTENTION_BRIDGES
        assert AttentionBridge._reconstructs_attention is False
