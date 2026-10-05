"""Tests for the scoped relevance-rule context: forward-identity, cleanup, and nesting.

The fixture component below installs the real LN-rule primitive
(``transformer_lens.model_bridge._relevance_rules.ln_rule``) through the protocol
``use_relevance_rules`` relies on to find and toggle rule-capable components. That keeps
these tests focused on the scoping mechanics -- forward identity, gradient restoration,
nested contexts, exception safety, and positional (not class-based) targeting -- rather
than on any concrete NormalizationBridge or gated-MLP integration, which land in later
commits. ``_FakeAttention`` covers the same mechanics for the hook rules the attention
kinds use, plus their mutual exclusion and their check that the rewritten HookPoints
feed the output.

The scoped context does not exist yet, so this fails collection with a single
ImportError -- the expected red state before the context is implemented.
"""

import dataclasses

import pytest
import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import _relevance_rules
from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRuleCoverage,
    RelevanceRules,
    RelevanceRuleUnsupportedError,
    ln_rule,
    use_relevance_rules,
)


class _FakeNormComponent(nn.Module):
    """A minimal ln1/ln2-style target: installs the real LN-rule primitive on request."""

    _relevance_rule_kinds = ("normalization",)

    def __init__(self, eps: float = 1e-2):
        super().__init__()
        # Composing this division with a plain second normalization and a linear
        # readout makes the LN-rule's denom-as-constant correction shrink linearly
        # with eps, so a realistic normalization epsilon (1e-6) would leave the
        # correction too small for torch.allclose to detect reliably end to end.
        self.eps = eps
        self._rule_active = False

    def _enable_relevance_rule(self, kind: str) -> None:
        self._rule_active = True

    def _disable_relevance_rule(self, kind: str) -> None:
        self._rule_active = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denom = x.abs().mean(dim=-1, keepdim=True) + self.eps
        if self._rule_active:
            return ln_rule(x, denom)
        return x / denom


class _PlainMount(nn.Module):
    """Occupies a targeted mount name but implements no relevance-rule protocol."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TinyBlock(nn.Module):
    """Mimics a block's ln1/ln2 mount points plus a same-class q_norm and an MLP.

    ``q_norm`` uses the identical fake-normalization class as ``ln1``/``ln2`` so tests
    can assert that targeting is positional (by mount name) rather than class-based.
    """

    def __init__(self):
        super().__init__()
        self.ln1: nn.Module = _FakeNormComponent()
        self.q_norm = _FakeNormComponent()
        self.ln2: nn.Module = _FakeNormComponent()
        self.mlp = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = self.q_norm(x)
        x = self.ln2(x)
        return self.mlp(x)


def _tiny_block() -> _TinyBlock:
    block = _TinyBlock()
    with torch.no_grad():
        block.mlp.weight.copy_(torch.eye(4) * 0.5)
        block.mlp.bias.zero_()
    return block


class _SandwichBlock(nn.Module):
    """Mimics a sandwich-norm block: pre-norms ln1/ln2 plus post-norms ln1_post/ln2_post.

    Sandwich-norm architectures mount a second normalization after attention and after
    the MLP at ``ln1_post``/``ln2_post``, so the LN-rule must reach those mounts as well
    as the pre-norms rather than leaving them silently out of coverage.
    """

    def __init__(self):
        super().__init__()
        self.ln1: nn.Module = _FakeNormComponent()
        self.ln1_post: nn.Module = _FakeNormComponent()
        self.ln2: nn.Module = _FakeNormComponent()
        self.ln2_post: nn.Module = _FakeNormComponent()
        self.mlp = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = self.ln1_post(x)
        x = self.ln2(x)
        x = self.ln2_post(x)
        return self.mlp(x)


class _FakeGatedMLPComponent(nn.Module):
    """A minimal mlp-mount target that answers to two rule kinds independently.

    A real gated-MLP node answers to both "activation" (Identity-rule on its
    activation function) and "multiplicative_gate" (Half-rule on its gate*up
    product) at the same mount, and either can be requested without the other,
    so this fixture tracks the two kinds as separate booleans rather than one.
    """

    _relevance_rule_kinds = ("activation", "multiplicative_gate")

    def __init__(self):
        super().__init__()
        self._activation_rule_active = False
        self._gate_rule_active = False

    def _enable_relevance_rule(self, kind: str) -> None:
        if kind == "activation":
            self._activation_rule_active = True
        elif kind == "multiplicative_gate":
            self._gate_rule_active = True

    def _disable_relevance_rule(self, kind: str) -> None:
        if kind == "activation":
            self._activation_rule_active = False
        elif kind == "multiplicative_gate":
            self._gate_rule_active = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Boom(Exception):
    """Marker exception raised inside a context to test cleanup on failure."""


def test_relevance_rules_defaults_to_no_rules():
    rules = RelevanceRules()
    assert rules.normalization is False
    assert rules.activation is False
    assert rules.multiplicative_gate is False
    assert rules.attention is False
    assert rules.attention_half is False
    assert rules.qk_normalization is False


def test_relevance_rules_is_frozen():
    rules = RelevanceRules(normalization=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rules.normalization = False  # type: ignore[misc]


def test_forward_is_identical_while_rule_active():
    block = _tiny_block()
    x = torch.randn(2, 3, 4)
    baseline = block(x)
    with use_relevance_rules(block, RelevanceRules(normalization=True)):
        active = block(x)
    assert torch.equal(active, baseline)


def test_positional_targeting_excludes_same_class_component_at_other_mount():
    block = _tiny_block()
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
        assert block.ln1._rule_active is True
        assert block.ln2._rule_active is True
        assert block.q_norm._rule_active is False
        assert set(coverage.installed) == {"ln1", "ln2"}
    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False


def test_normalization_rule_installs_on_sandwich_post_norm_mounts():
    block = _SandwichBlock()
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
        assert block.ln1._rule_active is True
        assert block.ln1_post._rule_active is True
        assert block.ln2._rule_active is True
        assert block.ln2_post._rule_active is True
        assert set(coverage.installed) == {"ln1", "ln1_post", "ln2", "ln2_post"}
        assert coverage.skipped == ()
    assert block.ln1_post._rule_active is False
    assert block.ln2_post._rule_active is False


def test_sandwich_post_norm_on_python_path_is_reported_skipped_never_absent():
    block = _SandwichBlock()
    # A post-norm mount whose occupant takes the python-norm path (no rule protocol)
    # must surface as skipped, never vanish from coverage as it did before the mount
    # names were recognized.
    block.ln2_post = _PlainMount()
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
        assert set(coverage.installed) == {"ln1", "ln1_post", "ln2"}
        assert set(coverage.skipped) == {"ln2_post"}


def test_requesting_a_kind_with_no_canonical_mount_raises(monkeypatch):
    block = _tiny_block()
    # A RelevanceRules field without a canonical mount must raise rather than install
    # nothing and return empty coverage.
    monkeypatch.delitem(_relevance_rules._CANONICAL_MOUNTS, "attention")
    with pytest.raises(ValueError, match="attention"):
        with use_relevance_rules(block, RelevanceRules(attention=True)):
            pass


def test_unsupported_component_at_targeted_mount_is_skipped():
    block = _tiny_block()
    block.ln2 = _PlainMount()
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
        assert isinstance(coverage, RelevanceRuleCoverage)
        assert set(coverage.installed) == {"ln1"}
        assert set(coverage.skipped) == {"ln2"}


def test_requesting_unsupported_kind_on_capable_component_raises():
    block = _tiny_block()
    block.mlp = _FakeGatedMLPComponent()
    # This mount implements the protocol and currently only supports "activation",
    # but -- unlike a mount that never deals with "multiplicative_gate" at all --
    # names it in _relevance_rule_unsupported_kinds as one it is expected to honor
    # here and currently cannot, so requesting it raises instead of being skipped.
    block.mlp._relevance_rule_kinds = ("activation",)
    block.mlp._relevance_rule_unsupported_kinds = ("multiplicative_gate",)
    with pytest.raises(RelevanceRuleUnsupportedError, match="mlp"):
        with use_relevance_rules(block, RelevanceRules(multiplicative_gate=True)):
            pass
    assert block.mlp._gate_rule_active is False


def test_no_rules_requested_installs_nothing():
    block = _tiny_block()
    with use_relevance_rules(block, RelevanceRules()) as coverage:
        assert coverage.installed == ()
        assert coverage.skipped == ()
        assert block.ln1._rule_active is False
        assert block.ln2._rule_active is False


def test_ordinary_gradients_restored_after_exit():
    block = _tiny_block()
    x = torch.randn(2, 3, 4, requires_grad=True)

    with use_relevance_rules(block, RelevanceRules(normalization=True)):
        y_rule = block(x)
        (grad_rule,) = torch.autograd.grad(y_rule.sum(), x)

    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False

    x_plain = x.detach().clone().requires_grad_(True)
    y_plain = block(x_plain)
    (grad_plain,) = torch.autograd.grad(y_plain.sum(), x_plain)

    # The LN-rule treats the denominator as constant, so its VJP differs from
    # ordinary autodiff through the same division wherever the denominator
    # actually depends on the input -- true for every row of this fixture.
    assert not torch.allclose(grad_rule, grad_plain)

    # A second plain pass confirms the exit left no residual rule state: it must
    # reproduce grad_plain exactly rather than drifting toward grad_rule.
    x_plain_again = x.detach().clone().requires_grad_(True)
    y_plain_again = block(x_plain_again)
    (grad_plain_again,) = torch.autograd.grad(y_plain_again.sum(), x_plain_again)
    torch.testing.assert_close(grad_plain_again, grad_plain)


def test_nested_contexts_restore_outer_state_on_inner_exit():
    block = _tiny_block()
    x = torch.randn(2, 3, 4)
    baseline = block(x)

    with use_relevance_rules(block, RelevanceRules(normalization=True)) as outer_coverage:
        assert block.ln1._rule_active is True
        with use_relevance_rules(block, RelevanceRules(normalization=True)) as inner_coverage:
            assert block.ln1._rule_active is True
            assert torch.equal(block(x), baseline)
        # The inner exit must not disable the rule the outer context still needs.
        assert block.ln1._rule_active is True
        assert block.ln2._rule_active is True
        assert torch.equal(block(x), baseline)

    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False
    assert set(outer_coverage.installed) == {"ln1", "ln2"}
    assert set(inner_coverage.installed) == {"ln1", "ln2"}


def test_exception_inside_context_leaves_no_rule_state():
    block = _tiny_block()
    x = torch.randn(2, 3, 4)
    baseline = block(x)

    with pytest.raises(_Boom):
        with use_relevance_rules(block, RelevanceRules(normalization=True)):
            assert block.ln1._rule_active is True
            raise _Boom("failure inside the scoped context")

    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False
    assert torch.equal(block(x), baseline)


def test_exception_during_nested_context_restores_outer_state():
    block = _tiny_block()
    x = torch.randn(2, 3, 4)
    baseline = block(x)

    with use_relevance_rules(block, RelevanceRules(normalization=True)):
        with pytest.raises(_Boom):
            with use_relevance_rules(block, RelevanceRules(normalization=True)):
                raise _Boom("failure inside the nested scoped context")
        # The outer context is still active after the inner one unwinds.
        assert block.ln1._rule_active is True
        assert torch.equal(block(x), baseline)

    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False


def test_one_kind_requested_on_a_two_kind_mount_leaves_the_other_kind_inactive():
    block = _tiny_block()
    block.mlp = _FakeGatedMLPComponent()
    with use_relevance_rules(block, RelevanceRules(multiplicative_gate=True)) as coverage:
        assert block.mlp._gate_rule_active is True
        assert block.mlp._activation_rule_active is False
        assert coverage.installed == ("mlp",)
    assert block.mlp._gate_rule_active is False


def test_both_kinds_requested_together_both_activate_on_the_same_mount():
    block = _tiny_block()
    block.mlp = _FakeGatedMLPComponent()
    with use_relevance_rules(
        block, RelevanceRules(activation=True, multiplicative_gate=True)
    ) as coverage:
        assert block.mlp._activation_rule_active is True
        assert block.mlp._gate_rule_active is True
        assert set(coverage.installed) == {"mlp"}
    assert block.mlp._activation_rule_active is False
    assert block.mlp._gate_rule_active is False


def test_nested_scopes_over_different_kinds_on_one_mount_refcount_independently():
    block = _tiny_block()
    block.mlp = _FakeGatedMLPComponent()
    with use_relevance_rules(block, RelevanceRules(activation=True)):
        assert block.mlp._activation_rule_active is True
        with use_relevance_rules(block, RelevanceRules(multiplicative_gate=True)):
            assert block.mlp._activation_rule_active is True
            assert block.mlp._gate_rule_active is True
        # The inner (multiplicative_gate-only) scope's exit must not disable
        # the outer scope's independently refcounted activation rule.
        assert block.mlp._activation_rule_active is True
        assert block.mlp._gate_rule_active is False
    assert block.mlp._activation_rule_active is False


class _FakeAttention(nn.Module):
    """An attn-mount target exposing the HookPoints the attention hook rules rewrite.

    Like a reconstructing attention bridge, it runs ``hook_pattern`` and ``hook_v`` on
    the two factors of its ``pattern @ v`` product. With ``delegating=True`` it instead
    fires ``hook_pattern`` only after its output exists, like a bridge that hands the
    computation to the wrapped module and hooks the attention weights it returns.
    """

    def __init__(self, delegating: bool = False, with_value_hook: bool = True):
        super().__init__()
        self.delegating = delegating
        self.hook_pattern = HookPoint()
        if with_value_hook:
            self.hook_v = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.delegating:
            output = self.hook_out(scores.softmax(-1) @ v)
            self.hook_pattern(scores.softmax(-1))
            return output
        pattern = self.hook_pattern(scores.softmax(-1))
        if "hook_v" in self._modules:
            v = self.hook_v(v)
        return self.hook_out(pattern @ v)


class _AttentionBlock(nn.Module):
    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn

    def forward(self, scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.attn(scores, v)


def _attention_inputs() -> tuple:
    generator = torch.Generator().manual_seed(0)
    scores = torch.randn(2, 3, 3, generator=generator, requires_grad=True)
    v = torch.randn(2, 3, 4, generator=generator, requires_grad=True)
    grad_output = torch.randn(2, 3, 4, generator=generator)
    return scores, v, grad_output


def _attention_grads(block: nn.Module) -> tuple:
    """Output plus the gradients reaching the scores and the values (None if unreached)."""
    scores, v, grad_output = _attention_inputs()
    output = block(scores, v)
    grads = torch.autograd.grad(output, (scores, v), grad_outputs=grad_output, allow_unused=True)
    return output.detach(), grads


def _rule_hook_count(attn: nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in attn.modules())


def test_attention_rule_holds_the_pattern_constant_and_keeps_the_value_path():
    block = _AttentionBlock(_FakeAttention())
    plain_output, (_, plain_grad_v) = _attention_grads(block)
    with use_relevance_rules(block, RelevanceRules(attention=True)) as coverage:
        assert coverage.installed == ("attn",)
        output, (grad_scores, grad_v) = _attention_grads(block)
    assert torch.equal(output, plain_output)
    assert grad_scores is None
    assert torch.equal(grad_v, plain_grad_v)


def test_attention_half_rule_halves_the_gradient_reaching_each_factor():
    block = _AttentionBlock(_FakeAttention())
    plain_output, (plain_grad_scores, plain_grad_v) = _attention_grads(block)
    with use_relevance_rules(block, RelevanceRules(attention_half=True)):
        output, (grad_scores, grad_v) = _attention_grads(block)
    assert torch.equal(output, plain_output)
    assert torch.equal(grad_scores, 0.5 * plain_grad_scores)
    assert torch.equal(grad_v, 0.5 * plain_grad_v)


def test_hook_rule_leaves_no_hooks_behind_and_restores_ordinary_gradients():
    block = _AttentionBlock(_FakeAttention())
    _, plain_grads = _attention_grads(block)
    with use_relevance_rules(block, RelevanceRules(attention_half=True)):
        assert _rule_hook_count(block.attn) > 0
    assert _rule_hook_count(block.attn) == 0
    _, grads = _attention_grads(block)
    for grad, plain_grad in zip(grads, plain_grads):
        assert torch.equal(grad, plain_grad)


def test_nested_requests_of_a_hook_rule_do_not_stack():
    block = _AttentionBlock(_FakeAttention())
    _, (plain_grad_scores, _) = _attention_grads(block)
    with use_relevance_rules(block, RelevanceRules(attention_half=True)):
        with use_relevance_rules(block, RelevanceRules(attention_half=True)):
            _, (grad_scores, _) = _attention_grads(block)
        # Halved once, not quartered; and the inner exit keeps the outer rule.
        assert torch.equal(grad_scores, 0.5 * plain_grad_scores)
        _, (grad_scores, _) = _attention_grads(block)
        assert torch.equal(grad_scores, 0.5 * plain_grad_scores)
    assert _rule_hook_count(block.attn) == 0


def test_mutually_exclusive_kinds_in_one_scope_raise_and_leave_no_state():
    block = _AttentionBlock(_FakeAttention())
    with pytest.raises(ValueError, match="attention_half"):
        with use_relevance_rules(block, RelevanceRules(attention=True, attention_half=True)):
            pass
    assert _rule_hook_count(block.attn) == 0
    assert not any(block.attn._relevance_rule_refcounts.values())


def test_mutually_exclusive_kind_is_refused_inside_an_outer_scope():
    block = _AttentionBlock(_FakeAttention())
    _, (plain_grad_scores, _) = _attention_grads(block)
    with use_relevance_rules(block, RelevanceRules(attention_half=True)):
        with pytest.raises(ValueError, match="'attention_half' is active"):
            with use_relevance_rules(block, RelevanceRules(attention=True)):
                pass
        _, (grad_scores, _) = _attention_grads(block)
        assert torch.equal(grad_scores, 0.5 * plain_grad_scores)
    assert _rule_hook_count(block.attn) == 0


def test_hook_rule_on_a_forward_that_hooks_the_pattern_after_its_output_raises():
    block = _AttentionBlock(_FakeAttention(delegating=True))
    scores, v, _ = _attention_inputs()
    with use_relevance_rules(block, RelevanceRules(attention=True)) as coverage:
        assert coverage.installed == ("attn",)
        with pytest.raises(RelevanceRuleUnsupportedError, match="hook_pattern"):
            block(scores, v)
    assert _rule_hook_count(block.attn) == 0


def test_hook_rule_needing_a_hook_point_the_mount_lacks_raises_at_entry():
    block = _AttentionBlock(_FakeAttention(with_value_hook=False))
    with pytest.raises(RelevanceRuleUnsupportedError, match="attn"):
        with use_relevance_rules(block, RelevanceRules(attention_half=True)):
            pass
    # The AH-rule rewrites only hook_pattern, which this mount does expose.
    with use_relevance_rules(block, RelevanceRules(attention=True)) as coverage:
        assert coverage.installed == ("attn",)


def test_attn_mount_without_attention_hook_points_is_skipped():
    block = _AttentionBlock(_PlainMount())
    with use_relevance_rules(block, RelevanceRules(attention=True)) as coverage:
        assert coverage.installed == ()
        assert coverage.skipped == ("attn",)


def test_hook_rule_survives_user_hook_removal_and_composes_with_user_edits():
    block = _AttentionBlock(_FakeAttention())
    hook_pattern = block.attn.hook_pattern
    hook_pattern.add_hook(lambda pattern, hook: pattern * 0.5)
    try:
        edited_output, _ = _attention_grads(block)
        with use_relevance_rules(block, RelevanceRules(attention=True)):
            # Clearing every user hook, permanent ones included, keeps the rule.
            hook_pattern.remove_hooks(including_permanent=True)
            hook_pattern.add_hook(lambda pattern, hook: pattern * 0.5)
            output, (grad_scores, _) = _attention_grads(block)
        assert torch.equal(output, edited_output)
        assert grad_scores is None
    finally:
        hook_pattern.remove_hooks(including_permanent=True)


def test_backward_hooks_on_the_factors_see_the_half_rule_gradient():
    """The gradient is scaled before the HookPoint's own backward hooks run, so they
    observe the rule's gradient, and one that returns a gradient keeps the rule."""
    block = _AttentionBlock(_FakeAttention())
    _, (plain_grad_scores, _) = _attention_grads(block)
    seen = []
    block.attn.hook_v.add_hook(lambda grad, hook: seen.append(grad), dir="bwd")
    block.attn.hook_pattern.add_hook(lambda grad, hook: grad, dir="bwd")
    try:
        _attention_grads(block)
        plain_seen = seen.pop()
        with use_relevance_rules(block, RelevanceRules(attention_half=True)):
            _, (grad_scores, _) = _attention_grads(block)
    finally:
        block.attn.hook_v.remove_hooks(dir="both")
        block.attn.hook_pattern.remove_hooks(dir="both")
    assert torch.equal(seen.pop(), 0.5 * plain_seen)
    assert torch.equal(grad_scores, 0.5 * plain_grad_scores)


def test_mount_with_rule_hook_points_but_no_hook_out_raises_at_entry_without_leaking():
    attn = _FakeAttention()
    del attn.hook_out
    block = _AttentionBlock(attn)
    with pytest.raises(RelevanceRuleUnsupportedError, match="attn"):
        with use_relevance_rules(block, RelevanceRules(attention=True)):
            pass
    assert _rule_hook_count(attn) == 0
