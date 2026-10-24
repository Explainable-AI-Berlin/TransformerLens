"""The LRP rules expressed as TransformerLens forward hooks, plus the shared runtime state.

Every rule is a forward hook on a ``TransformerBridge`` hook point. A hook returns ``None``
(no edit) whenever LRP is switched off or its rule is inactive, so the bridge stays on its
native numerics path; when active it returns a tensor whose value equals the original but
whose autograd graph is modified (a detached factor, a halved gradient). A plain
gradient-times-input backward pass through the hooked model then yields LRP relevances.

A hook point that carries a ``hook_conversion`` (compatibility mode reshapes ``v.hook_out``
per head) hands the hook the converted tensor and reverts the returned one. Detaching and
halving commute with any linear, invertible conversion, so the rules keep their meaning
there.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.lrp.coverage import LRPCoverage
from transformer_lens.model_bridge.lrp.rules import (
    LRPRules,
    half_gradient,
    identity_act,
)

HookFn = Callable[..., Optional[torch.Tensor]]


@dataclass
class LRPState:
    """Runtime LRP state attached to a bridge by ``patch_lrp``."""

    enabled: bool = False
    rules: Set[str] = field(default_factory=set)
    # Per-rule, per-block support table built by ``coverage.survey``.
    coverage: LRPCoverage = field(default_factory=LRPCoverage)
    # Canonical names of the hook points we attached to, and the hook callables we attached.
    hook_names: List[str] = field(default_factory=list)
    hook_fn_ids: Set[int] = field(default_factory=set)
    # HuggingFace MLP modules whose activation was swapped for ``LRPActivation``, with the
    # original activation module, so ``unpatch_lrp`` can restore them.
    swapped_activations: List[Tuple[nn.Module, nn.Module]] = field(default_factory=list)
    # The ``warnings.filters`` entry installed while LRP is on, so it can be removed exactly.
    warning_filter: Optional[Tuple[Any, ...]] = None

    def active(self, rule: LRPRules) -> bool:
        """Whether ``rule`` currently modifies the backward pass."""
        return self.enabled and rule.value in self.rules


def make_scale_hook(state: LRPState, rule: LRPRules) -> HookFn:
    """Hook for a normalization ``hook_scale`` point: detach the scale under ``rule``."""

    def scale_hook(
        tensor: torch.Tensor, hook: Optional[HookPoint] = None
    ) -> Optional[torch.Tensor]:
        if state.active(rule):
            return tensor.detach()
        return None

    scale_hook.__name__ = f"lrp_{rule.value}"
    return scale_hook


def make_pattern_hook(state: LRPState) -> HookFn:
    """Hook for ``attn.hook_pattern``: detach (ah-rule) or halve (half-rule-attn)."""

    def pattern_hook(
        tensor: torch.Tensor, hook: Optional[HookPoint] = None
    ) -> Optional[torch.Tensor]:
        if state.active(LRPRules.AH_RULE):
            return tensor.detach()
        if state.active(LRPRules.HALF_RULE_ATTN):
            return half_gradient(tensor)
        return None

    pattern_hook.__name__ = "lrp_attention_pattern"
    return pattern_hook


def make_value_hook(state: LRPState) -> HookFn:
    """Hook for ``attn.v.hook_out``: halve the value gradient under half-rule-attn."""

    def value_hook(
        tensor: torch.Tensor, hook: Optional[HookPoint] = None
    ) -> Optional[torch.Tensor]:
        if state.active(LRPRules.HALF_RULE_ATTN):
            return half_gradient(tensor)
        return None

    value_hook.__name__ = "lrp_attention_value"
    return value_hook


def make_mlp_product_hook(state: LRPState) -> HookFn:
    """Hook for ``mlp.out.hook_in`` (the ``act(gate) * up`` product): half-rule-mlp."""

    def product_hook(
        tensor: torch.Tensor, hook: Optional[HookPoint] = None
    ) -> Optional[torch.Tensor]:
        if state.active(LRPRules.HALF_RULE_MLP):
            return half_gradient(tensor)
        return None

    product_hook.__name__ = "lrp_mlp_product"
    return product_hook


class LRPActivation(nn.Module):
    """Drop-in replacement for a HuggingFace MLP activation module.

    Computes the wrapped activation unchanged, except under identity-rule-act where the
    activation's Jacobian is replaced by the ratio ``act(x) / x``. Installed in place of
    ``mlp.act_fn`` on the HuggingFace module so that every bridge hook point stays on the
    gradient path.
    """

    def __init__(self, state: LRPState, act: nn.Module):
        super().__init__()
        self.state = state
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.state.active(LRPRules.IDENTITY_RULE_ACT):
            return identity_act(x, self.act)
        return self.act(x)


__all__ = [
    "HookFn",
    "LRPActivation",
    "LRPState",
    "make_mlp_product_hook",
    "make_pattern_hook",
    "make_scale_hook",
    "make_value_hook",
]
