"""LN-rule on NormalizationBridge's native-autograd and python-norm paths.

The rule detaches ``hook_scale``. On the python-norm path that holds the divisor
constant directly; on the native-autograd path the detach changes no value, so the
bridge keeps HF's own output and takes the gradient from its reconstruction, after
checking once whether the module scales by ``weight`` or ``1 + weight``. Either way
the forward is unchanged and the backward is the LN-rule (denominator treated as
constant, the pinned RelP reference's ``x / scale.detach()``) while weight and bias
keep their ordinary gradient, targeting is positional (the same class at
``attn.q_norm`` answers to "qk_normalization" only), and user hooks compose with it.
"""

import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRules,
    use_relevance_rules,
)
from transformer_lens.model_bridge.generalized_components.normalization import (
    LayerNormPreBridge,
    NormalizationBridge,
    RMSNormPreBridge,
)

PATHS = pytest.mark.parametrize("native", [True, False], ids=["native_autograd", "python_norm"])


class _Cfg:
    def __init__(
        self,
        uses_rms_norm: bool = False,
        eps: float = 1e-5,
        rmsnorm_uses_offset: bool = False,
        layer_norm_folding: bool = False,
    ):
        self.uses_rms_norm = uses_rms_norm
        self.eps = eps
        self.rmsnorm_uses_offset = rmsnorm_uses_offset
        self.layer_norm_folding = layer_norm_folding


class _TinyRMSNorm(nn.Module):
    """Minimal RMSNorm mirroring LlamaRMSNorm's forward."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d) * 0.1 + 1.0)
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.variance_epsilon)


class _TinyGemmaRMSNorm(nn.Module):
    """Minimal Gemma-style RMSNorm: weight is stored as an offset from 1."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d) * 0.1)
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.variance_epsilon)
        return x_normed * (1.0 + self.weight)


class _TinyOlmoLayerNorm(nn.Module):
    """Param-free centered LayerNorm mirroring OLMo's OlmoLayerNorm: no weight, no bias."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d,)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, None, None, self.eps)


def _layernorm(d: int) -> nn.LayerNorm:
    layer = nn.LayerNorm(d, eps=1e-5)
    nn.init.normal_(layer.weight, std=0.1)
    nn.init.normal_(layer.bias, std=0.1)
    return layer


class _Block(nn.Module):
    """Mounts a normalization bridge at the canonical ln1 position."""

    def __init__(self, norm: NormalizationBridge):
        super().__init__()
        self.ln1 = norm


class _QkNormBlock(nn.Module):
    """Mounts a normalization bridge at the per-head attn.q_norm position."""

    def __init__(self, norm: NormalizationBridge):
        super().__init__()
        self.attn = nn.Module()
        self.attn.q_norm = norm


def _make_bridge(
    native: bool,
    rms: bool = False,
    offset: bool = False,
    d: int = 16,
    layer_norm_folding: bool = False,
) -> NormalizationBridge:
    layer: nn.Module
    if offset:
        layer = _TinyGemmaRMSNorm(d)
    elif rms:
        layer = _TinyRMSNorm(d)
    else:
        layer = _layernorm(d)
    bridge = NormalizationBridge(
        name="ln1",
        config=_Cfg(
            uses_rms_norm=rms or offset,
            rmsnorm_uses_offset=offset,
            layer_norm_folding=layer_norm_folding,
        ),
        use_native_layernorm_autograd=native,
    )
    bridge.set_original_component(layer)
    return bridge


def _denom_detached_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    uses_rms: bool,
    offset: bool,
    eps: float,
) -> torch.Tensor:
    """Manual recompute with the denominator detached: the LN-rule's target VJP."""
    x_centered = x if uses_rms else x - x.mean(-1, keepdim=True)
    denom = (x_centered.pow(2).mean(-1, keepdim=True) + eps).sqrt().detach()
    w_eff = (1.0 + weight) if offset else weight
    out = (x_centered / denom) * w_eff
    if bias is not None:
        out = out + bias
    return out


class TestForwardValue:
    @PATHS
    @pytest.mark.parametrize("rms", [False, True], ids=["layernorm", "rmsnorm"])
    def test_active_forward_matches_inactive_forward(self, rms, native):
        bridge = _make_bridge(native=native, rms=rms)
        x = torch.randn(2, 5, 16)
        baseline = bridge(x)
        with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)):
            active = bridge(x)
        assert torch.equal(active, baseline)

    def test_rule_with_observing_hooks_emits_no_fallback_warning(self):
        bridge = _make_bridge(native=True)
        bridge.hook_scale.add_hook(lambda scale, hook: None)
        try:
            with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    bridge(torch.randn(2, 5, 16))
        finally:
            bridge.hook_scale.remove_hooks()


class TestUnflaggedOffsetWeight:
    """Some norms scale by ``1 + weight`` without the config's ``rmsnorm_uses_offset``
    (Qwen3-Next, Nemotron's LayerNorm1P); the bridge must find that form itself."""

    def test_offset_form_found_from_the_module(self):
        bridge = NormalizationBridge(
            name="ln1", config=_Cfg(uses_rms_norm=True), use_native_layernorm_autograd=True
        )
        bridge.set_original_component(_TinyGemmaRMSNorm(16))
        x = torch.randn(2, 5, 16, requires_grad=True)
        baseline = bridge(x.detach())
        with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)):
            active = bridge(x)
            active.sum().backward()
        assert torch.equal(active.detach(), baseline)

        x_oracle = x.detach().clone().requires_grad_(True)
        _denom_detached_oracle(
            x_oracle, bridge.original_component.weight, None, uses_rms=True, offset=True, eps=1e-5
        ).sum().backward()
        torch.testing.assert_close(x.grad, x_oracle.grad)

    def test_unreproducible_module_is_refused(self):
        class _SquaredWeightRMSNorm(_TinyRMSNorm):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x) * self.weight

        bridge = NormalizationBridge(
            name="ln1", config=_Cfg(uses_rms_norm=True), use_native_layernorm_autograd=True
        )
        bridge.set_original_component(_SquaredWeightRMSNorm(16))
        with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)):
            with pytest.raises(RuntimeError, match="cannot reproduce"):
                bridge(torch.randn(2, 5, 16))


class TestVJPMatchesDetachedDenomOracle:
    @PATHS
    def test_layernorm(self, native):
        bridge = _make_bridge(native=native)
        x = torch.randn(2, 5, 16, requires_grad=True)
        with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)):
            bridge(x).sum().backward()

        x_oracle = x.detach().clone().requires_grad_(True)
        _denom_detached_oracle(
            x_oracle,
            bridge.original_component.weight,
            bridge.original_component.bias,
            uses_rms=False,
            offset=False,
            eps=1e-5,
        ).sum().backward()
        torch.testing.assert_close(x.grad, x_oracle.grad)

    @PATHS
    @pytest.mark.parametrize("offset", [False, True], ids=["plain_rms", "gemma_offset"])
    def test_rmsnorm(self, offset, native):
        bridge = _make_bridge(native=native, rms=not offset, offset=offset)
        x = torch.randn(2, 5, 16, requires_grad=True)
        with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)):
            bridge(x).sum().backward()

        x_oracle = x.detach().clone().requires_grad_(True)
        _denom_detached_oracle(
            x_oracle,
            bridge.original_component.weight,
            None,
            uses_rms=True,
            offset=offset,
            eps=1e-5,
        ).sum().backward()
        torch.testing.assert_close(x.grad, x_oracle.grad)

    @PATHS
    def test_rule_grad_disagrees_with_ordinary_autodiff(self, native):
        """The whole point of the rule: it must actually change the gradient."""
        bridge = _make_bridge(native=native)
        x = torch.randn(2, 5, 16, requires_grad=True)
        with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)):
            (grad_rule,) = torch.autograd.grad(bridge(x).sum(), x)
        (grad_plain,) = torch.autograd.grad(bridge(x).sum(), x)
        assert not torch.allclose(grad_rule, grad_plain)


class TestParameterGradientsPreserved:
    @PATHS
    def test_weight_and_bias_receive_ordinary_gradient(self, native):
        bridge = _make_bridge(native=native)
        x = torch.randn(2, 5, 16, requires_grad=True)
        with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)):
            bridge(x).sum().backward()
        weight_grad = bridge.original_component.weight.grad
        bias_grad = bridge.original_component.bias.grad

        # d(output)/d(weight) = normalized value (pre-weight); d(output)/d(bias) = 1.
        normalized = _denom_detached_oracle(
            x.detach(),
            torch.ones_like(bridge.original_component.weight),
            None,
            uses_rms=False,
            offset=False,
            eps=1e-5,
        )
        reduce_dims = tuple(range(normalized.dim() - 1))
        torch.testing.assert_close(
            weight_grad, normalized.sum(dim=reduce_dims), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            bias_grad, torch.full_like(bias_grad, x.shape[0] * x.shape[1]), atol=1e-4, rtol=1e-4
        )


class TestCoverage:
    @PATHS
    def test_installed_on_both_dispatch_paths(self, native):
        bridge = _make_bridge(native=native)
        with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)) as coverage:
            assert coverage.installed == ("ln1",)
            assert coverage.skipped == ()

    @pytest.mark.parametrize(
        ("bridge_cls", "uses_rms"),
        [(LayerNormPreBridge, False), (RMSNormPreBridge, True)],
        ids=["ln_pre", "rms_pre"],
    )
    def test_param_free_pre_norm_applies_the_rule_with_unit_weight(self, bridge_cls, uses_rms):
        bridge = bridge_cls(name="ln1", config=_Cfg())
        bridge.set_original_component(nn.LayerNorm(16, eps=1e-5, elementwise_affine=False))
        x = torch.randn(2, 5, 16, requires_grad=True)
        baseline = bridge(x)
        with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)) as coverage:
            assert coverage.installed == ("ln1",)
            active = bridge(x)
            active.sum().backward()
        assert torch.equal(active, baseline)

        x_oracle = x.detach().clone().requires_grad_(True)
        _denom_detached_oracle(
            x_oracle, torch.ones(16), None, uses_rms=uses_rms, offset=False, eps=1e-5
        ).sum().backward()
        torch.testing.assert_close(x.grad, x_oracle.grad)

    @PATHS
    def test_q_norm_mount_answers_only_to_qk_normalization(self, native):
        bridge = _make_bridge(native=native, rms=True)
        block = _QkNormBlock(bridge)
        with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
            assert coverage.installed == ()
        x = torch.randn(2, 4, 5, 16, requires_grad=True)
        with use_relevance_rules(block, RelevanceRules(qk_normalization=True)) as coverage:
            assert coverage.installed == ("attn.q_norm",)
            bridge(x).sum().backward()

        x_oracle = x.detach().clone().requires_grad_(True)
        _denom_detached_oracle(
            x_oracle, bridge.original_component.weight, None, uses_rms=True, offset=False, eps=1e-5
        ).sum().backward()
        torch.testing.assert_close(x.grad, x_oracle.grad)


class TestMissingWeightTreatedAsIdentity:
    """A native-autograd bridge over a parameter-free norm (OLMo's OlmoLayerNorm has no
    ``weight``) must treat the missing weight as a unit scale when it reconstructs its
    output from the rule-edited scale."""

    def test_forward_and_vjp_match_detached_denom_oracle(self):
        bridge = NormalizationBridge(
            name="ln1", config=_Cfg(uses_rms_norm=False), use_native_layernorm_autograd=True
        )
        bridge.set_original_component(_TinyOlmoLayerNorm(16))
        x = torch.randn(2, 5, 16, requires_grad=True)
        baseline = bridge.original_component(x.detach())
        with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)):
            active = bridge(x)
            active.sum().backward()
        assert torch.equal(active.detach(), baseline)

        x_oracle = x.detach().clone().requires_grad_(True)
        _denom_detached_oracle(
            x_oracle, torch.ones(16), None, uses_rms=False, offset=False, eps=1e-5
        ).sum().backward()
        torch.testing.assert_close(x.grad, x_oracle.grad)


class TestUserHooksComposeWithRule:
    @PATHS
    def test_forward_edit_of_hook_scale_is_held_constant_by_the_rule(self, native):
        bridge = _make_bridge(native=native)
        x = torch.randn(2, 5, 16, requires_grad=True)
        bridge.hook_scale.add_hook(lambda scale, hook: scale * 2.0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                edited = bridge(x.detach())
                with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)):
                    active = bridge(x)
                    active.sum().backward()
        finally:
            bridge.hook_scale.remove_hooks()
        torch.testing.assert_close(active.detach(), edited)

        x_oracle = x.detach().clone().requires_grad_(True)
        _denom_detached_oracle(
            x_oracle,
            bridge.original_component.weight / 2.0,
            bridge.original_component.bias,
            uses_rms=False,
            offset=False,
            eps=1e-5,
        ).sum().backward()
        torch.testing.assert_close(x.grad, x_oracle.grad)

    @PATHS
    def test_bwd_hook_on_hook_normalized_fires_and_keeps_the_rule(self, native):
        bridge = _make_bridge(native=native)
        x = torch.randn(2, 5, 16, requires_grad=True)
        fired = []
        bridge.hook_normalized.add_hook(lambda grad, hook: fired.append(grad), dir="bwd")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with use_relevance_rules(_Block(bridge), RelevanceRules(normalization=True)):
                    bridge(x).sum().backward()
        finally:
            bridge.hook_normalized.remove_hooks(dir="bwd")
        assert len(fired) == 1

        x_oracle = x.detach().clone().requires_grad_(True)
        _denom_detached_oracle(
            x_oracle,
            bridge.original_component.weight,
            bridge.original_component.bias,
            uses_rms=False,
            offset=False,
            eps=1e-5,
        ).sum().backward()
        torch.testing.assert_close(x.grad, x_oracle.grad)

    def test_user_edit_without_rule_still_warns(self):
        """Control: hooks added through add_hook keep warning about the fallback."""
        bridge = _make_bridge(native=True)
        bridge.hook_scale.add_hook(lambda scale, hook: scale * 2.0)
        try:
            with pytest.warns(UserWarning, match="reconstructed from the hooked values"):
                bridge(torch.randn(2, 5, 16))
        finally:
            bridge.hook_scale.remove_hooks()

    def test_user_bwd_hook_without_rule_still_warns(self):
        """Control: unchanged behavior when no rule is active."""
        bridge = _make_bridge(native=True)
        bridge.hook_normalized.add_hook(lambda grad, hook: grad, dir="bwd")
        try:
            with pytest.warns(UserWarning, match="Backward hooks"):
                bridge(torch.randn(2, 5, 16, requires_grad=True))
        finally:
            bridge.hook_normalized.remove_hooks(dir="bwd")
