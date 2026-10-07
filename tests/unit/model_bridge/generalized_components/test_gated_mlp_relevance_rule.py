"""Identity-/Half-rule on GatedMLPBridge's opaque native-forward path.

The HF module's own forward runs unchanged; the rules attach through hooks for a
``use_relevance_rules`` scope: the Identity-rule on the module's activation (an
``ACT2FN``-style module), the Half-rule on ``hook_post``, the gate*up product entering
the down projection. Anything the real forward does beyond the core
``down(act(gate) * up)`` shape -- a post-product multiplier, the module's own hooks --
stays in the backward graph, and the rules attach regardless of the projection weight
class. An activation the Identity-rule does not apply to (``f(0) != 0``) is refused,
and one that is a bare function rather than a module cannot be hooked, so it is
skipped.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRules,
    RelevanceRuleUnsupportedError,
    use_relevance_rules,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.gated_mlp import (
    GatedMLPBridge,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class _Cfg:
    def __init__(self, hidden_act: str = "silu"):
        self.hidden_act = hidden_act


_ACTIVATIONS = {"silu": nn.SiLU, "sigmoid": nn.Sigmoid}


class _OpaqueProj(nn.Module):
    """Weight-backed projection that is neither nn.Linear nor Conv1D.

    The rules attach through hooks, so they never read the projection's weights.
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in))
        self.bias = nn.Parameter(torch.randn(d_out)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight.T
        return out if self.bias is None else out + self.bias


class _TinyGatedMLP(nn.Module):
    """Mirrors the Qwen2/Llama/Gemma gated-MLP: one opaque call over its own submodules.

    The activation is an ``nn.Module`` attribute the forward calls, matching the
    ``ACT2FN`` shape the Identity-rule hooks.
    """

    def __init__(self, gate_proj, up_proj, down_proj, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = _ACTIVATIONS[hidden_act]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _MultiplierGatedMLP(_TinyGatedMLP):
    """A gated MLP whose forward does strictly more than ``down(act(gate) * up)``.

    The constant post-product multiplier stands in for the Falcon-H1 / Gemma3n
    families whose native forward applies extra scaling. A rule VJP taken through a
    reconstruction of only the core gated shape would omit the multiplier and
    disagree with the VJP through this real forward.
    """

    def __init__(self, gate_proj, up_proj, down_proj, multiplier: float = 1.7):
        super().__init__(gate_proj, up_proj, down_proj)
        self.multiplier = multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * self.multiplier


class _Block(nn.Module):
    """Mounts a gated-MLP bridge at the canonical mlp position."""

    def __init__(self, mlp: GeneralizedComponent):
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _make_projections(backing_class: str, d_model: int = 4, d_mlp: int = 8, bias: bool = True):
    torch.manual_seed(0)
    if backing_class == "nn.Linear":
        return (
            nn.Linear(d_model, d_mlp, bias=bias),
            nn.Linear(d_model, d_mlp, bias=bias),
            nn.Linear(d_mlp, d_model, bias=bias),
        )
    if backing_class == "Conv1D":
        return (
            Conv1D(d_mlp, d_model),
            Conv1D(d_mlp, d_model),
            Conv1D(d_model, d_mlp),
        )
    return (
        _OpaqueProj(d_model, d_mlp, bias=bias),
        _OpaqueProj(d_model, d_mlp, bias=bias),
        _OpaqueProj(d_mlp, d_model, bias=bias),
    )


def _wire_bridge(hf_mlp: nn.Module, config: _Cfg) -> GatedMLPBridge:
    bridge = GatedMLPBridge(name="mlp", config=config)
    gate_bridge = LinearBridge(name="gate_proj")
    in_bridge = LinearBridge(name="up_proj")
    out_bridge = LinearBridge(name="down_proj")
    bridge.add_module("gate", gate_bridge)
    bridge.add_module("in", in_bridge)
    bridge.add_module("out", out_bridge)
    bridge.set_original_component(hf_mlp)
    gate_bridge.set_original_component(hf_mlp.gate_proj)
    in_bridge.set_original_component(hf_mlp.up_proj)
    out_bridge.set_original_component(hf_mlp.down_proj)
    # As at bridge boot: the HF module calls the projection bridges in place of its own
    # projections, and hook_pre/hook_pre_linear/hook_post resolve.
    hf_mlp.gate_proj, hf_mlp.up_proj, hf_mlp.down_proj = gate_bridge, in_bridge, out_bridge
    bridge._register_aliases()
    return bridge


def _make_bridge(
    backing_class: str, bias: bool = True, hidden_act: str = "silu"
) -> tuple[_Block, _TinyGatedMLP]:
    gate_proj, up_proj, down_proj = _make_projections(backing_class, bias=bias)
    hf_mlp = _TinyGatedMLP(gate_proj, up_proj, down_proj, hidden_act=hidden_act)
    bridge = _wire_bridge(hf_mlp, _Cfg(hidden_act))
    return _Block(bridge), hf_mlp


def _make_multiplier_bridge(multiplier: float = 1.7) -> tuple[_Block, _MultiplierGatedMLP]:
    gate_proj, up_proj, down_proj = _make_projections("nn.Linear", bias=True)
    hf_mlp = _MultiplierGatedMLP(gate_proj, up_proj, down_proj, multiplier=multiplier)
    bridge = _wire_bridge(hf_mlp, _Cfg("silu"))
    return _Block(bridge), hf_mlp


def _oracle_grads(hf_mlp, x, activation_active, gate_active, multiplier: float = 1.0):
    # hf_mlp's parameters are shared with the bridge under test, so a prior backward
    # already left gradients on them; reset first or this second backward would
    # accumulate on top instead of producing an independently comparable oracle. The
    # oracle runs the module's real submodules -- including the multiplier -- so it is
    # the VJP through the actual forward, not through the core gated shape alone.
    for p in hf_mlp.parameters():
        p.grad = None
    x_oracle = x.detach().clone().requires_grad_(True)
    gate_output = hf_mlp.gate_proj(x_oracle)
    up_output = hf_mlp.up_proj(x_oracle)
    # The rules as written in the pinned RelP reference (gate outputs are nonzero here).
    activated = hf_mlp.act_fn(gate_output)
    if activation_active:
        activated = gate_output * (activated / gate_output).detach()
    gated = activated * up_output
    if gate_active:
        gated = gated / 2 + (gated / 2).detach()
    down = hf_mlp.down_proj(gated) * multiplier
    down.sum().backward()
    grads = {n: p.grad.clone() for n, p in hf_mlp.named_parameters()}
    return x_oracle.grad.clone(), grads


BACKING_CLASSES = ["nn.Linear", "Conv1D", "opaque"]


class TestGatedMLPRelevanceRuleCapability:
    @pytest.mark.parametrize("backing_class", BACKING_CLASSES)
    def test_both_kinds_install_regardless_of_weight_backing(self, backing_class):
        block, _ = _make_bridge(backing_class)
        rules = RelevanceRules(activation=True, multiplicative_gate=True)
        with use_relevance_rules(block, rules) as coverage:
            assert coverage.installed == ("mlp", "mlp")

    def test_identity_rule_unsupported_for_sigmoid_activation(self):
        block, _ = _make_bridge("nn.Linear", hidden_act="sigmoid")
        with pytest.raises(RelevanceRuleUnsupportedError, match="mlp"):
            with use_relevance_rules(block, RelevanceRules(activation=True)):
                pass

    def test_activation_skipped_when_no_activation_module_is_exposed(self):
        # A gated MLP whose forward uses a bare function has no activation module for
        # the Identity-rule to hook, so the mount is reported skipped. The Half-rule
        # needs no activation access and stays available.
        class _FunctionalActMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(4, 8)
                self.up_proj = nn.Linear(4, 8)
                self.down_proj = nn.Linear(8, 4)

            def forward(self, x):
                return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        torch.manual_seed(0)
        bridge = _wire_bridge(_FunctionalActMLP(), _Cfg("silu"))
        block = _Block(bridge)

        with use_relevance_rules(block, RelevanceRules(activation=True)) as coverage:
            assert coverage.skipped == ("mlp",)
        with use_relevance_rules(block, RelevanceRules(multiplicative_gate=True)) as coverage:
            assert coverage.installed == ("mlp",)

    def test_half_rule_remains_available_under_sigmoid_activation(self):
        block, _ = _make_bridge("nn.Linear", hidden_act="sigmoid")
        x = torch.randn(2, 4)
        baseline = block(x)
        with use_relevance_rules(block, RelevanceRules(multiplicative_gate=True)) as coverage:
            assert coverage.installed == ("mlp",)
            assert torch.equal(block(x), baseline)


class TestGatedMLPRelevanceRuleForwardIdentity:
    @pytest.mark.parametrize("backing_class", BACKING_CLASSES)
    def test_forward_identical_while_rule_active(self, backing_class):
        block, _ = _make_bridge(backing_class)
        x = torch.randn(3, 4)
        baseline = block(x)
        with use_relevance_rules(block, RelevanceRules(activation=True, multiplicative_gate=True)):
            active = block(x)
        assert torch.equal(active, baseline)

    def test_forward_identical_while_rule_active_with_extra_forward_ops(self):
        block, _ = _make_multiplier_bridge()
        x = torch.randn(3, 4)
        baseline = block(x)
        with use_relevance_rules(block, RelevanceRules(activation=True, multiplicative_gate=True)):
            active = block(x)
        assert torch.equal(active, baseline)

    def test_activation_module_left_unhooked_after_scope(self):
        block, hf_mlp = _make_bridge("nn.Linear")
        with use_relevance_rules(block, RelevanceRules(activation=True)):
            assert hf_mlp.act_fn._forward_hooks
        assert not hf_mlp.act_fn._forward_hooks


class TestGatedMLPRelevanceRuleVJP:
    @pytest.mark.parametrize("backing_class", BACKING_CLASSES)
    def test_both_rules_active_matches_real_forward_oracle(self, backing_class):
        block, hf_mlp = _make_bridge(backing_class)
        x = torch.randn(3, 4, requires_grad=True)

        with use_relevance_rules(block, RelevanceRules(activation=True, multiplicative_gate=True)):
            out = block(x)
            out.sum().backward()

        grad_x = x.grad.clone()
        grads = {n: p.grad.clone() for n, p in hf_mlp.named_parameters()}

        expected_grad_x, expected_grads = self._expected(hf_mlp, x, True, True)
        torch.testing.assert_close(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)
        for name, grad in grads.items():
            # The down projection's own weight gradient stays ordinary: the Half-rule
            # only halves the gradient reaching the product, not the parameter grads.
            torch.testing.assert_close(grad, expected_grads[name], atol=1e-5, rtol=1e-5)

    def test_both_rules_match_vjp_through_extra_forward_ops(self):
        block, hf_mlp = _make_multiplier_bridge()
        x = torch.randn(3, 4, requires_grad=True)

        with use_relevance_rules(block, RelevanceRules(activation=True, multiplicative_gate=True)):
            out = block(x)
            out.sum().backward()

        grad_x = x.grad.clone()
        grads = {n: p.grad.clone() for n, p in hf_mlp.named_parameters()}

        expected_grad_x, expected_grads = _oracle_grads(
            hf_mlp, x, activation_active=True, gate_active=True, multiplier=hf_mlp.multiplier
        )
        torch.testing.assert_close(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)
        for name, grad in grads.items():
            torch.testing.assert_close(grad, expected_grads[name], atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("backing_class", BACKING_CLASSES)
    def test_activation_only_leaves_gate_split_ordinary(self, backing_class):
        block, hf_mlp = _make_bridge(backing_class)
        x = torch.randn(3, 4, requires_grad=True)

        with use_relevance_rules(block, RelevanceRules(activation=True)):
            out = block(x)
            out.sum().backward()

        grad_x = x.grad.clone()
        expected_grad_x, _ = self._expected(hf_mlp, x, True, False)
        torch.testing.assert_close(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("backing_class", BACKING_CLASSES)
    def test_multiplicative_gate_only_leaves_activation_ordinary(self, backing_class):
        block, hf_mlp = _make_bridge(backing_class)
        x = torch.randn(3, 4, requires_grad=True)

        with use_relevance_rules(block, RelevanceRules(multiplicative_gate=True)):
            out = block(x)
            out.sum().backward()

        grad_x = x.grad.clone()
        expected_grad_x, _ = self._expected(hf_mlp, x, False, True)
        torch.testing.assert_close(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)

    def test_rule_inactive_gradients_are_ordinary(self):
        block, hf_mlp = _make_bridge("nn.Linear")
        x = torch.randn(3, 4, requires_grad=True)

        out = block(x)
        out.sum().backward()
        grad_x = x.grad.clone()
        grads = {n: p.grad.clone() for n, p in hf_mlp.named_parameters()}

        for p in hf_mlp.parameters():
            p.grad = None
        x_plain = x.detach().clone().requires_grad_(True)
        hf_mlp(x_plain).sum().backward()

        torch.testing.assert_close(grad_x, x_plain.grad)
        for name, param in hf_mlp.named_parameters():
            torch.testing.assert_close(grads[name], param.grad)

    def test_bias_free_projections_are_handled(self):
        block, hf_mlp = _make_bridge("nn.Linear", bias=False)
        x = torch.randn(3, 4, requires_grad=True)

        with use_relevance_rules(block, RelevanceRules(activation=True, multiplicative_gate=True)):
            out = block(x)
            out.sum().backward()

        grad_x = x.grad.clone()
        expected_grad_x, expected_grads = self._expected(hf_mlp, x, True, True)
        torch.testing.assert_close(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)
        for name, param in hf_mlp.named_parameters():
            torch.testing.assert_close(param.grad, expected_grads[name], atol=1e-5, rtol=1e-5)

    @staticmethod
    def _expected(hf_mlp, x, activation_active, gate_active):
        return _oracle_grads(hf_mlp, x, activation_active, gate_active, multiplier=1.0)


class TestGatedMLPRelevanceRulePreservesRealForward:
    def test_modules_own_backward_hook_fires_during_rule_active_backward(self):
        # The rules attach to the live module and leave its native forward in the
        # graph, so a backward hook registered on an HF submodule still fires during
        # a rule-active backward rather than being bypassed by a separate graph.
        block, hf_mlp = _make_bridge("nn.Linear")
        fired = {"count": 0}

        def _record(module, grad_input, grad_output):
            fired["count"] += 1

        handle = hf_mlp.down_proj.register_full_backward_hook(_record)
        x = torch.randn(3, 4, requires_grad=True)
        try:
            with use_relevance_rules(
                block, RelevanceRules(activation=True, multiplicative_gate=True)
            ):
                block(x).sum().backward()
        finally:
            handle.remove()

        assert fired["count"] > 0


class TestGatedMLPRelevanceRuleCompatibilityMode:
    """The processed-weights (compatibility) path reconstructs the forward from folded
    weights but calls the module's own activation, so the same hooks apply."""

    def _make_compat_bridge(self, bias: bool = True):
        block, hf_mlp = _make_bridge("nn.Linear", bias=bias)
        bridge = block.mlp
        bridge._use_processed_weights = True
        # Plain (non-Parameter) tensors so the bridge's custom __getattr__ resolves
        # them; nn.Linear stores weight as [out, in], the layout functional linear
        # expects, so the folded-weight forward reproduces the native projections.
        bridge._processed_W_gate = hf_mlp.gate_proj.weight.detach()
        bridge._processed_b_gate = None if not bias else hf_mlp.gate_proj.bias.detach()
        bridge._processed_W_in = hf_mlp.up_proj.weight.detach()
        bridge._processed_b_in = None if not bias else hf_mlp.up_proj.bias.detach()
        bridge._processed_W_out = hf_mlp.down_proj.weight.detach()
        bridge._processed_b_out = None if not bias else hf_mlp.down_proj.bias.detach()
        return block, hf_mlp

    def test_compat_forward_identical_and_rule_vjp_matches_oracle(self):
        block, hf_mlp = self._make_compat_bridge()
        x = torch.randn(3, 4, requires_grad=True)

        baseline = block(x.detach())
        with use_relevance_rules(block, RelevanceRules(activation=True, multiplicative_gate=True)):
            active = block(x)
        assert torch.equal(active.detach(), baseline)

        active.sum().backward()
        grad_x = x.grad.clone()
        expected_grad_x, _ = _oracle_grads(hf_mlp, x, True, True)
        torch.testing.assert_close(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)
