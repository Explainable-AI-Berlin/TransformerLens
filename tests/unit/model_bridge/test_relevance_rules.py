"""Tests for the scoped relevance-rule context: targeting, rewrites, cleanup, and nesting.

Every rule attaches through hooks on modules a component exposes, so the fixtures
below are minimal stand-ins exposing the same HookPoints as the bridge components (a
norm's ``hook_scale``, a gated MLP's ``hook_pre``/``hook_pre_linear``/``hook_post`` and
activation module, an attention's ``hook_pattern``/``hook_v``). That keeps these tests
on the scoping mechanics -- forward identity, each rewrite's VJP, gradient restoration,
nested contexts, exception safety, positional (not class-based) targeting, and the
check that rewritten modules feed the output -- rather than on any concrete bridge,
which the component and integration tests cover.
"""

import dataclasses
from typing import Optional

import pytest
import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge._relevance_rules import (
    _RULES,
    RelevanceRuleCoverage,
    RelevanceRules,
    RelevanceRuleUnsupportedError,
    use_relevance_rules,
)


class _FakeNorm(nn.Module):
    """An ln1/ln2-style target dividing by a denominator it runs through ``hook_scale``."""

    def __init__(self, eps: float = 1e-2):
        super().__init__()
        # Composing this division with a plain second normalization and a linear
        # readout makes the LN-rule's denom-as-constant correction shrink linearly
        # with eps, so a realistic normalization epsilon (1e-6) would leave the
        # correction too small for torch.allclose to detect reliably end to end.
        self.eps = eps
        self.hook_scale = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denom = self.hook_scale(x.abs().mean(dim=-1, keepdim=True) + self.eps)
        return self.hook_out(x / denom)


class _PlainMount(nn.Module):
    """Occupies a targeted mount name but exposes none of the rules' hook points."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TinyBlock(nn.Module):
    """Mimics a block's ln1/ln2 mounts plus a same-class q_norm and an MLP.

    ``q_norm`` uses the identical fake-normalization class as ``ln1``/``ln2`` so tests
    can assert that targeting is positional (by mount name) rather than class-based.
    """

    def __init__(self):
        super().__init__()
        self.ln1: nn.Module = _FakeNorm()
        self.q_norm = _FakeNorm()
        self.ln2: nn.Module = _FakeNorm()
        self.mlp = nn.Linear(4, 4)
        with torch.no_grad():
            self.mlp.weight.copy_(torch.eye(4) * 0.5)
            self.mlp.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln2(self.q_norm(self.ln1(x))))


class _SandwichBlock(nn.Module):
    """Mimics a sandwich-norm block: pre-norms ln1/ln2 plus post-norms ln1_post/ln2_post."""

    def __init__(self):
        super().__init__()
        self.ln1: nn.Module = _FakeNorm()
        self.ln1_post: nn.Module = _FakeNorm()
        self.ln2: nn.Module = _FakeNorm()
        self.ln2_post: nn.Module = _FakeNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln2_post(self.ln2(self.ln1_post(self.ln1(x))))


class _FakeMLP(nn.Module):
    """An mlp-mount target: ``down(act(gate(x)) * up(x))``, or ``down(act(in(x)))``.

    Exposes the gated MLP bridge's hook points: ``hook_pre`` (the activation input),
    ``hook_pre_linear`` (up; gated only), ``hook_post`` (the down projection's input)
    and its ``activation_fn`` module.
    """

    def __init__(self, gated: bool = True, activation: Optional[nn.Module] = None):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.gated = gated
        self.w_gate = nn.Parameter(torch.randn(4, 8, generator=generator))
        self.w_up = nn.Parameter(torch.randn(4, 8, generator=generator))
        self.w_down = nn.Parameter(torch.randn(8, 4, generator=generator))
        self.activation_fn = nn.SiLU() if activation is None else activation
        self.hook_pre = HookPoint()
        if gated:
            self.hook_pre_linear = HookPoint()
        self.hook_post = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = self.activation_fn(self.hook_pre(x @ self.w_gate))
        if self.gated:
            activated = activated * self.hook_pre_linear(x @ self.w_up)
        return self.hook_out(self.hook_post(activated) @ self.w_down)

    def oracle_input_grad(
        self, x: torch.Tensor, grad_output: torch.Tensor, identity: bool, half: bool
    ) -> torch.Tensor:
        """The input gradient with each rule written out as in the pinned RelP reference."""
        x = x.detach().clone().requires_grad_(True)
        gate = x @ self.w_gate
        activated = gate * (torch.sigmoid(gate)).detach() if identity else nn.functional.silu(gate)
        product = activated * (x @ self.w_up) if self.gated else activated
        if half:
            product = product / 2 + (product / 2).detach()
        (grad,) = torch.autograd.grad(product @ self.w_down, x, grad_outputs=grad_output)
        return grad


class _FakeAttention(nn.Module):
    """An attn-mount target exposing the HookPoints the attention rules rewrite.

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


class _Mount(nn.Module):
    """Mounts one component under ``name`` and forwards to it."""

    def __init__(self, name: str, component: nn.Module):
        super().__init__()
        self.name = name
        self.add_module(name, component)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return getattr(self, self.name)(*args)


class _Boom(Exception):
    """Marker exception raised inside a context to test cleanup on failure."""


def _rule_hook_count(model: nn.Module) -> int:
    return sum(
        len(module._forward_hooks)
        + len(module._forward_pre_hooks)
        + len(module._backward_pre_hooks)
        for module in model.modules()
    )


def _input_grad(model: nn.Module, x: torch.Tensor, grad_output=None) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    y = model(x)
    (grad,) = torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y) if grad_output is None else grad_output
    )
    return grad


def _attention_inputs() -> tuple:
    generator = torch.Generator().manual_seed(0)
    scores = torch.randn(2, 3, 3, generator=generator, requires_grad=True)
    v = torch.randn(2, 3, 4, generator=generator, requires_grad=True)
    grad_output = torch.randn(2, 3, 4, generator=generator)
    return scores, v, grad_output


def _attention_grads(model: nn.Module) -> tuple:
    """Output plus the gradients reaching the scores and the values (None if unreached)."""
    scores, v, grad_output = _attention_inputs()
    output = model(scores, v)
    grads = torch.autograd.grad(output, (scores, v), grad_outputs=grad_output, allow_unused=True)
    return output.detach(), grads


def test_relevance_rules_defaults_to_no_rules():
    rules = RelevanceRules()
    assert not any(getattr(rules, field.name) for field in dataclasses.fields(rules))


def test_relevance_rules_is_frozen():
    rules = RelevanceRules(normalization=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rules.normalization = False  # type: ignore[misc]


def test_every_rule_kind_has_a_rule():
    assert {field.name for field in dataclasses.fields(RelevanceRules)} == set(_RULES)


def test_forward_is_identical_while_rule_active():
    block = _TinyBlock()
    x = torch.randn(2, 3, 4)
    baseline = block(x)
    with use_relevance_rules(block, RelevanceRules(normalization=True, qk_normalization=True)):
        active = block(x)
    assert torch.equal(active, baseline)


def test_ln_rule_holds_the_denominator_constant():
    norm = _FakeNorm()
    block = _Mount("ln1", norm)
    x = torch.randn(2, 3, 4)
    with use_relevance_rules(block, RelevanceRules(normalization=True)):
        grad = _input_grad(block, x)
    x_oracle = x.clone().requires_grad_(True)
    denom = (x_oracle.abs().mean(dim=-1, keepdim=True) + norm.eps).detach()
    (expected,) = torch.autograd.grad((x_oracle / denom).sum(), x_oracle)
    torch.testing.assert_close(grad, expected)
    assert not torch.allclose(grad, _input_grad(block, x))


def test_positional_targeting_excludes_same_class_component_at_other_mount():
    block = _TinyBlock()
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
        assert set(coverage.installed) == {"ln1", "ln2"}
        assert _rule_hook_count(block.q_norm) == 0
    with use_relevance_rules(block, RelevanceRules(qk_normalization=True)) as coverage:
        assert coverage.installed == ("q_norm",)
        assert _rule_hook_count(block.ln1) == 0
    assert _rule_hook_count(block) == 0


def test_normalization_rule_installs_on_sandwich_post_norm_mounts():
    block = _SandwichBlock()
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
        assert set(coverage.installed) == {"ln1", "ln1_post", "ln2", "ln2_post"}
        assert coverage.skipped == ()


def test_mount_without_the_rules_hook_points_is_skipped():
    block = _TinyBlock()
    block.ln2 = _PlainMount()
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
        assert isinstance(coverage, RelevanceRuleCoverage)
        assert coverage.installed == ("ln1",)
        assert coverage.skipped == ("ln2",)


def test_mount_with_the_rules_hook_points_but_no_hook_out_is_skipped():
    norm = _FakeNorm()
    del norm.hook_out
    with use_relevance_rules(_Mount("ln1", norm), RelevanceRules(normalization=True)) as coverage:
        assert coverage.skipped == ("ln1",)
    assert _rule_hook_count(norm) == 0


def test_no_rules_requested_installs_nothing():
    block = _TinyBlock()
    with use_relevance_rules(block, RelevanceRules()) as coverage:
        assert coverage.installed == ()
        assert coverage.skipped == ()
        assert _rule_hook_count(block) == 0


def test_ordinary_gradients_restored_after_exit():
    block = _TinyBlock()
    x = torch.randn(2, 3, 4)
    plain = _input_grad(block, x)
    with use_relevance_rules(block, RelevanceRules(normalization=True)):
        assert not torch.allclose(_input_grad(block, x), plain)
    assert _rule_hook_count(block) == 0
    torch.testing.assert_close(_input_grad(block, x), plain)


def test_nested_contexts_restore_outer_state_on_inner_exit():
    block = _TinyBlock()
    x = torch.randn(2, 3, 4)
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as outer_coverage:
        rule_grad = _input_grad(block, x)
        hooks = _rule_hook_count(block)
        with use_relevance_rules(block, RelevanceRules(normalization=True)) as inner_coverage:
            # Refcounted: the nested request adds no second set of hooks.
            assert _rule_hook_count(block) == hooks
        # The inner exit must not remove the rule the outer context still needs.
        torch.testing.assert_close(_input_grad(block, x), rule_grad)
    assert _rule_hook_count(block) == 0
    assert set(outer_coverage.installed) == set(inner_coverage.installed) == {"ln1", "ln2"}


def test_exception_inside_context_leaves_no_rule_state():
    block = _TinyBlock()
    with pytest.raises(_Boom):
        with use_relevance_rules(block, RelevanceRules(normalization=True)):
            raise _Boom("failure inside the scoped context")
    assert _rule_hook_count(block) == 0


def test_exception_during_nested_context_restores_outer_state():
    block = _TinyBlock()
    x = torch.randn(2, 3, 4)
    with use_relevance_rules(block, RelevanceRules(normalization=True)):
        rule_grad = _input_grad(block, x)
        with pytest.raises(_Boom):
            with use_relevance_rules(block, RelevanceRules(normalization=True)):
                raise _Boom("failure inside the nested scoped context")
        torch.testing.assert_close(_input_grad(block, x), rule_grad)
    assert _rule_hook_count(block) == 0


@pytest.mark.parametrize(
    ("identity", "half"), [(True, False), (False, True), (True, True)], ids=["act", "gate", "both"]
)
def test_mlp_rules_match_reference_formulas_alone_and_combined(identity, half):
    mlp = _FakeMLP()
    block = _Mount("mlp", mlp)
    x = torch.randn(3, 4)
    grad_output = torch.randn(3, 4)
    baseline = block(x)
    rules = RelevanceRules(activation=identity, multiplicative_gate=half)
    with use_relevance_rules(block, rules) as coverage:
        assert coverage.installed == ("mlp",) * (identity + half)
        assert torch.equal(block(x), baseline)
        grad = _input_grad(block, x, grad_output)
    torch.testing.assert_close(grad, mlp.oracle_input_grad(x, grad_output, identity, half))


def test_mlp_rule_kinds_refcount_independently_on_one_mount():
    mlp = _FakeMLP()
    block = _Mount("mlp", mlp)
    x = torch.randn(3, 4)
    grad_output = torch.randn(3, 4)
    with use_relevance_rules(block, RelevanceRules(activation=True)):
        with use_relevance_rules(block, RelevanceRules(multiplicative_gate=True)):
            both = _input_grad(block, x, grad_output)
        # The inner (gate-only) exit must not remove the outer activation rule.
        activation_only = _input_grad(block, x, grad_output)
    torch.testing.assert_close(both, mlp.oracle_input_grad(x, grad_output, True, True))
    torch.testing.assert_close(activation_only, mlp.oracle_input_grad(x, grad_output, True, False))
    assert _rule_hook_count(block) == 0


def test_multiplicative_gate_skips_an_ungated_mlp_and_activation_covers_it():
    mlp = _FakeMLP(gated=False)
    block = _Mount("mlp", mlp)
    with use_relevance_rules(block, RelevanceRules(multiplicative_gate=True)) as coverage:
        assert coverage.skipped == ("mlp",)
    x = torch.randn(3, 4)
    grad_output = torch.randn(3, 4)
    with use_relevance_rules(block, RelevanceRules(activation=True)) as coverage:
        assert coverage.installed == ("mlp",)
        grad = _input_grad(block, x, grad_output)
    torch.testing.assert_close(grad, mlp.oracle_input_grad(x, grad_output, True, False))


@pytest.mark.parametrize("activation", [nn.Sigmoid(), nn.GLU()], ids=["sigmoid", "glu"])
def test_identity_rule_refuses_an_activation_not_elementwise_through_zero(activation):
    block = _Mount("mlp", _FakeMLP(activation=activation))
    with pytest.raises(RelevanceRuleUnsupportedError, match="mlp"):
        with use_relevance_rules(block, RelevanceRules(activation=True)):
            pass
    assert _rule_hook_count(block) == 0
    with use_relevance_rules(block, RelevanceRules(multiplicative_gate=True)) as coverage:
        assert coverage.installed == ("mlp",)


def test_identity_rule_fills_the_ratio_at_zero_with_the_activations_slope():
    mlp = _FakeMLP(gated=False, activation=nn.Tanh())
    with torch.no_grad():
        mlp.w_gate.zero_()  # every activation input is exactly 0
    block = _Mount("mlp", mlp)
    x = torch.randn(3, 4)
    with use_relevance_rules(block, RelevanceRules(activation=True)):
        grad = _input_grad(block, x)
    # Every path through the activation runs at x == 0, where tanh's ratio limit is 1.
    torch.testing.assert_close(grad, _input_grad(block, x))


def test_attention_rule_holds_the_pattern_constant_and_keeps_the_value_path():
    block = _Mount("attn", _FakeAttention())
    plain_output, (_, plain_grad_v) = _attention_grads(block)
    with use_relevance_rules(block, RelevanceRules(attention=True)) as coverage:
        assert coverage.installed == ("attn",)
        output, (grad_scores, grad_v) = _attention_grads(block)
    assert torch.equal(output, plain_output)
    assert grad_scores is None
    assert torch.equal(grad_v, plain_grad_v)


def test_attention_half_rule_halves_the_gradient_reaching_each_factor():
    block = _Mount("attn", _FakeAttention())
    plain_output, (plain_grad_scores, plain_grad_v) = _attention_grads(block)
    with use_relevance_rules(block, RelevanceRules(attention_half=True)):
        output, (grad_scores, grad_v) = _attention_grads(block)
    assert torch.equal(output, plain_output)
    assert torch.equal(grad_scores, 0.5 * plain_grad_scores)
    assert torch.equal(grad_v, 0.5 * plain_grad_v)


def test_nested_requests_of_a_halving_rule_do_not_stack():
    block = _Mount("attn", _FakeAttention())
    _, (plain_grad_scores, _) = _attention_grads(block)
    with use_relevance_rules(block, RelevanceRules(attention_half=True)):
        with use_relevance_rules(block, RelevanceRules(attention_half=True)):
            _, (grad_scores, _) = _attention_grads(block)
        assert torch.equal(grad_scores, 0.5 * plain_grad_scores)
    assert _rule_hook_count(block) == 0


def test_mutually_exclusive_kinds_in_one_scope_raise_and_leave_no_state():
    block = _Mount("attn", _FakeAttention())
    with pytest.raises(ValueError, match="attention_half"):
        with use_relevance_rules(block, RelevanceRules(attention=True, attention_half=True)):
            pass
    assert _rule_hook_count(block) == 0
    assert not any(block.attn._relevance_rule_refcounts.values())


def test_mutually_exclusive_kind_is_refused_inside_an_outer_scope():
    block = _Mount("attn", _FakeAttention())
    _, (plain_grad_scores, _) = _attention_grads(block)
    with use_relevance_rules(block, RelevanceRules(attention_half=True)):
        with pytest.raises(ValueError, match="'attention_half' is active"):
            with use_relevance_rules(block, RelevanceRules(attention=True)):
                pass
        _, (grad_scores, _) = _attention_grads(block)
        assert torch.equal(grad_scores, 0.5 * plain_grad_scores)
    assert _rule_hook_count(block) == 0


def test_rule_on_a_forward_that_hooks_its_target_after_the_output_raises():
    block = _Mount("attn", _FakeAttention(delegating=True))
    scores, v, _ = _attention_inputs()
    with use_relevance_rules(block, RelevanceRules(attention=True)) as coverage:
        assert coverage.installed == ("attn",)
        with pytest.raises(RelevanceRuleUnsupportedError, match="hook_pattern"):
            block(scores, v)
    assert _rule_hook_count(block) == 0


def test_target_lacking_one_of_the_rules_hook_points_raises_at_entry():
    block = _Mount("attn", _FakeAttention(with_value_hook=False))
    with pytest.raises(RelevanceRuleUnsupportedError, match="attn"):
        with use_relevance_rules(block, RelevanceRules(attention_half=True)):
            pass
    # The AH-rule rewrites only hook_pattern, which this mount does expose.
    with use_relevance_rules(block, RelevanceRules(attention=True)) as coverage:
        assert coverage.installed == ("attn",)


def test_rule_survives_user_hook_removal_and_composes_with_user_edits():
    block = _Mount("attn", _FakeAttention())
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


def test_backward_hooks_on_a_halved_module_see_the_rules_gradient():
    """The gradient is halved before the HookPoint's own backward hooks run, so they
    observe the rule's gradient, and one that returns a gradient keeps the rule."""
    block = _Mount("attn", _FakeAttention())
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
