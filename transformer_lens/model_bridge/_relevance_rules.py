"""Relevance rules (LRP) as scoped rewrites of a model's backward pass.

Each rule kind changes how the gradient passes one kind of operation, so that
gradient x input yields relevance: the LN-rule holds a norm's denominator constant,
the Identity-rule passes an activation's gradient as ``y / x``, the Half-rule splits
a product's gradient evenly between its factors, and the AH-rule holds the attention
pattern constant. ``use_relevance_rules`` attaches the requested rules for the
duration of a ``with`` block through hooks on modules the bridge components already
expose (mostly HookPoints), targeting components positionally by mount name and
reporting which mounts were installed versus skipped. Components carry no rule code.
"""

import dataclasses
import enum
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from transformer_lens.hook_points import HookPoint


class _IdentityRule(torch.autograd.Function):
    """Forward returns ``y = f(x)`` unchanged; the VJP is ``grad * y / x``.

    At ``x == 0`` the ratio takes its limit ``f'(0)`` (``0.5`` for SiLU and the GELUs).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor, slope_at_zero: float) -> torch.Tensor:
        ctx.save_for_backward(x, y)
        ctx.slope_at_zero = slope_at_zero
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        x, y = ctx.saved_tensors
        safe_x = torch.where(x == 0, torch.ones_like(x), x)
        ratio = torch.where(x == 0, torch.full_like(x, ctx.slope_at_zero), y / safe_x)
        return grad_output * ratio, None, None


def identity_rule(x: torch.Tensor, y: torch.Tensor, slope_at_zero: float = 0.5) -> torch.Tensor:
    """Apply the Identity-rule to an elementwise activation's input ``x`` and output ``y``."""
    result: torch.Tensor = _IdentityRule.apply(x, y, slope_at_zero)
    return result


def _slope_at_zero(activation: nn.Module) -> Optional[float]:
    """``f'(0)``, the limit of ``f(x) / x``, if ``f`` is elementwise with ``f(0) == 0``.

    Those are the activations whose relevance the Identity-rule can pass through
    unchanged; any other (``f(0) != 0`` like sigmoid, or shape-changing like GLU) is None.
    """
    parameter = next(activation.parameters(), None)
    zero = torch.zeros(
        2,
        device=None if parameter is None else parameter.device,
        dtype=torch.float32 if parameter is None else parameter.dtype,
        requires_grad=True,
    )
    with torch.enable_grad():
        # forward, not __call__, so no hooks (an outer scope's rule) fire; a clone, so an
        # in-place activation does not write into a leaf.
        value = activation.forward(zero.clone())
    if value.shape != zero.shape or bool(value.detach().abs().max() != 0):
        return None
    (slope,) = torch.autograd.grad(value.sum(), zero)
    return float(slope[0])


class RelevanceRuleUnsupportedError(RuntimeError):
    """A requested relevance rule cannot be applied to a component it targets.

    Raised at ``use_relevance_rules`` entry when a targeted component lacks some of the
    modules the rule rewrites (a fused-QKV attention without ``hook_v`` under the
    attention Half-rule) or whose activation the Identity-rule does not apply to (one
    with ``f(0) != 0``, like sigmoid). Raised in the forward pass when a rewritten module does not
    feed the component's output, as in an attention bridge that delegates to the
    wrapped HF module. A mount that is not the kind of component a rule applies to
    (a recurrent mixer at ``attn``) is reported skipped instead.
    """


@dataclasses.dataclass(frozen=True)
class RelevanceRules:
    """Which relevance rules to request for the duration of a ``use_relevance_rules`` scope.

    Each field names a rule kind; ``True`` requests it on every component at the kind's
    canonical mounts. Unset fields (the default) leave the corresponding components
    untouched.

    ``normalization``: LN-rule (the denominator is a constant in the backward pass) on
    the block norms ``ln1``/``ln2``, the sandwich post-norms ``ln1_post``/``ln2_post``
    and ``ln_final``.
    ``qk_normalization``: the same LN-rule on the per-head norms ``attn.q_norm``/
    ``attn.k_norm``, requested separately so block-level and per-head norms can be
    studied independently.
    ``activation``: Identity-rule on an MLP's elementwise activation ``f`` with ``f(0) == 0``.
    ``multiplicative_gate``: Half-rule on a gated MLP's ``act(gate) * up`` product.
    ``attention``: AH-rule -- the attention pattern is a constant in the backward pass,
    so relevance reaches the residual stream only through the values.
    ``attention_half``: Half-rule on the ``pattern @ V`` product -- the gradient reaching
    the pattern and the gradient reaching the values are each halved.

    ``attention`` and ``attention_half`` rewrite the same product, so they are mutually
    exclusive. Backward hooks see the rule's gradient; one on a value a rule holds
    constant never fires: ``hook_scale`` under the LN-rule, and ``hook_pattern``,
    ``hook_attn_scores`` and the query/key hooks under the AH-rule. The forward value is
    unchanged.
    """

    normalization: bool = False
    activation: bool = False
    multiplicative_gate: bool = False
    attention: bool = False
    attention_half: bool = False
    qk_normalization: bool = False


@dataclasses.dataclass
class RelevanceRuleCoverage:
    """Which canonical mounts a ``use_relevance_rules`` scope installed versus skipped.

    ``installed`` holds the dotted path of every mount where a requested rule kind was
    enabled. ``skipped`` holds the dotted path of every mount that matched a requested
    kind's canonical mount name but is not the kind of component the rule applies to.
    """

    installed: Tuple[str, ...]
    skipped: Tuple[str, ...]


class _Rewrite(enum.Enum):
    """How a rule rewrites the backward pass at the output of one module."""

    # The output is a constant: nothing upstream of it is differentiated.
    DETACH = enum.auto()
    # The gradient reaching the output is halved, before the module's backward hooks.
    HALVE = enum.auto()
    # The output y = f(x) of an elementwise activation passes gradient as y / x.
    IDENTITY = enum.auto()


# What torch passes to and expects from a backward pre-hook.
_Grads = Union[Tuple[torch.Tensor, ...], torch.Tensor]


def _module_at(component: nn.Module, attr: str) -> Optional[nn.Module]:
    value = getattr(component, attr, None)
    return value if isinstance(value, nn.Module) else None


@dataclasses.dataclass(frozen=True)
class _Rule:
    """How one rule kind attaches to the components at its canonical mounts.

    ``rewrites`` maps attributes of the component, each resolving to a module (mostly a
    HookPoint), to the rewrite applied at that module's output. A bridge component with
    every attribute in ``identified_by`` (default: all rewritten ones) is a target; any
    other mount is skipped. A target must expose every rewritten module, and an
    ``IDENTITY`` target must be an activation the Identity-rule applies to, or the scope
    raises.

    A rewrite is only correct if the module feeds the component's output, so the rule
    also checks at the component's ``hook_out`` that each rewritten module fired earlier
    in the same call; if one did not, the forward raises instead of silently keeping the
    ordinary gradient.
    """

    mounts: Tuple[str, ...]
    rewrites: Mapping[str, _Rewrite]
    identified_by: Tuple[str, ...] = ()

    def support(self, component: nn.Module) -> Optional[bool]:
        """True to install, False if a target cannot honor the rule, None to skip."""
        identifiers = self.identified_by or tuple(self.rewrites)
        if not isinstance(getattr(component, "hook_out", None), HookPoint) or any(
            _module_at(component, attr) is None for attr in identifiers
        ):
            return None
        for attr, rewrite in self.rewrites.items():
            module = _module_at(component, attr)
            if module is None or (rewrite is _Rewrite.IDENTITY and _slope_at_zero(module) is None):
                return False
        return True

    def install(self, name: str, kind: str, component: nn.Module) -> List[RemovableHandle]:
        """Register the rule's hooks on ``component``; return their handles."""
        fired: Set[str] = set()

        def _rewrite_at(attr: str, module: nn.Module) -> Callable[..., Optional[torch.Tensor]]:
            rewrite = self.rewrites[attr]
            slope = _slope_at_zero(module) if rewrite is _Rewrite.IDENTITY else None

            def _hook(module: nn.Module, args: Tuple[Any, ...], output: torch.Tensor):
                fired.add(attr)
                if rewrite is _Rewrite.DETACH:
                    return output.detach()
                if slope is not None:
                    return identity_rule(args[0], output, slope)
                return None

            return _hook

        def _halve_grad(module: nn.Module, grad_output: _Grads) -> _Grads:
            return tuple(0.5 * grad for grad in grad_output)

        def _check_fired_before_output(module: nn.Module, args: Any, output: Any) -> None:
            missing = [attr for attr in self.rewrites if attr not in fired]
            fired.clear()
            if missing:
                raise RelevanceRuleUnsupportedError(
                    f"{name!r} ({type(component).__name__}) cannot apply the {kind!r} "
                    f"relevance rule: its forward did not run {', '.join(missing)} before "
                    "computing its output (for example because it delegates to the wrapped "
                    "module), so the rule would silently not apply. To leave such mounts "
                    "out, scope the request to a capable subtree, for example "
                    "use_relevance_rules(bridge.blocks, rules)."
                )

        # Count only modules fired within the current call.
        handles = [component.register_forward_pre_hook(lambda *_: fired.clear())]
        for attr, rewrite in self.rewrites.items():
            module = cast(nn.Module, _module_at(component, attr))
            handles.append(module.register_forward_hook(_rewrite_at(attr, module)))
            if rewrite is _Rewrite.HALVE:
                handles.append(module.register_full_backward_pre_hook(_halve_grad))
        hook_out = cast(HookPoint, component.hook_out)
        handles.append(hook_out.register_forward_hook(_check_fired_before_output))
        return handles


# Targeting is positional: a component is only considered for a kind when it sits at
# one of the kind's mount names, never by class, so a q_norm sharing NormalizationBridge's
# class is reached only through "qk_normalization". The normalization mounts match the
# pinned RelP reference: pre-norms, sandwich post-norms and the final norm.
_NORM_MOUNTS = ("ln1", "ln2", "ln1_post", "ln2_post", "ln_final")
_ATTENTION_MOUNTS = ("attn", "self_attn", "cross_attn")

# In every attention bridge that computes pattern @ V itself, hook_pattern runs on the
# pattern it multiplies (activation patching relies on that) and hook_v's output reaches
# the output only through that product, so these are the product's two factors. A gated
# MLP's hook_post is its act(gate) * up product; hook_pre_linear (up) marks it as gated.
_RULES: Mapping[str, _Rule] = {
    "normalization": _Rule(_NORM_MOUNTS, {"hook_scale": _Rewrite.DETACH}),
    "qk_normalization": _Rule(("q_norm", "k_norm"), {"hook_scale": _Rewrite.DETACH}),
    "activation": _Rule(("mlp",), {"activation_fn": _Rewrite.IDENTITY}),
    "multiplicative_gate": _Rule(
        ("mlp",), {"hook_post": _Rewrite.HALVE}, identified_by=("hook_pre_linear",)
    ),
    "attention": _Rule(_ATTENTION_MOUNTS, {"hook_pattern": _Rewrite.DETACH}),
    "attention_half": _Rule(
        _ATTENTION_MOUNTS,
        {"hook_pattern": _Rewrite.HALVE, "hook_v": _Rewrite.HALVE},
        identified_by=("hook_pattern",),
    ),
}

# Kinds that rewrite the same backward edge, so at most one of each group may be active
# on a component at a time, whether requested in one scope or in nested scopes.
_MUTUALLY_EXCLUSIVE_KINDS: Tuple[FrozenSet[str], ...] = (
    frozenset({"attention", "attention_half"}),
)


def _iter_canonical_mount_candidates(
    model: nn.Module, mount_names: Tuple[str, ...]
) -> Iterator[Tuple[str, nn.Module]]:
    """Yield each distinct module reachable at one of ``mount_names``, once.

    A bridge component reachable at a canonical mount name (for example
    ``blocks.0.ln1``) is also reachable, under the same parent, through the
    raw HF module tree the bridge wraps in place (for example
    ``blocks.0._original_component.input_layernorm``) -- both names resolve to
    the identical object. ``nn.Module.named_modules()`` deduplicates by object
    identity and keeps only whichever path it visits first, which is the raw
    HF-attribute path (registered before the canonical alias), so on a real
    assembled model the canonical name is silently never seen. Walking with
    ``remove_duplicate=False`` restores every path so the canonical name is
    visible, and picking the fewest-dot-separated-segments path per object
    (breaking a tie between two paths that both happen to end in a mount name,
    such as ``mlp``, which HF's own attribute name also frequently matches)
    reports the shallower, canonical-looking path rather than an internal one.
    """
    best_by_id: Dict[int, Tuple[str, nn.Module]] = {}
    for name, module in model.named_modules(remove_duplicate=False):
        if name.rsplit(".", 1)[-1] not in mount_names:
            continue
        existing = best_by_id.get(id(module))
        if existing is None or name.count(".") < existing[0].count("."):
            best_by_id[id(module)] = (name, module)
    yield from best_by_id.values()


def _acquire_rule(name: str, component: nn.Module, kind: str) -> None:
    """Install ``kind``'s hooks on ``component`` only for the outermost scope requesting it.

    Refcounted per kind, so nested scopes never stack a second set of hooks (two Half-rule
    hooks would quarter the gradient) and one kind's inner exit never removes another's.
    """
    counts: Dict[str, int] = getattr(component, "_relevance_rule_refcounts", None) or {}
    count = counts.get(kind, 0)
    if count == 0:
        active_exclusive = [
            other
            for group in _MUTUALLY_EXCLUSIVE_KINDS
            if kind in group
            for other in sorted(group - {kind})
            if counts.get(other, 0) > 0
        ]
        if active_exclusive:
            raise ValueError(
                f"Cannot enable the {kind!r} relevance rule on {name!r} while "
                f"{', '.join(repr(other) for other in active_exclusive)} is active: they "
                "rewrite the same backward edge, so at most one of them may be requested."
            )
        handles: Dict[str, List[RemovableHandle]] = (
            getattr(component, "_relevance_rule_handles", None) or {}
        )
        handles[kind] = _RULES[kind].install(name, kind, component)
        setattr(component, "_relevance_rule_handles", handles)
    counts[kind] = count + 1
    setattr(component, "_relevance_rule_refcounts", counts)


def _release_rule(component: nn.Module, kind: str) -> None:
    """Remove ``kind``'s hooks from ``component`` once its innermost scope exits."""
    counts: Dict[str, int] = getattr(component, "_relevance_rule_refcounts")
    counts[kind] -= 1
    if counts[kind] == 0:
        for handle in getattr(component, "_relevance_rule_handles").pop(kind):
            handle.remove()


@contextmanager
def use_relevance_rules(model: nn.Module, rules: RelevanceRules) -> Iterator[RelevanceRuleCoverage]:
    """Install the requested relevance rules on ``model`` only for this scope.

    Each requested kind targets the components at its canonical mount names (never by
    class). A mount that is not the kind of component the rule applies to is reported
    skipped; a targeted component that cannot honor the rule raises
    ``RelevanceRuleUnsupportedError`` (see there), so analysis never proceeds as if a
    rule applied where it does not. Enabling two mutually exclusive kinds on one
    component (``attention`` and ``attention_half``), in one scope or nested ones,
    raises ``ValueError``. Scopes over the same model are reference-counted, so an inner
    scope's exit never removes a rule an outer scope still needs. No model configuration
    is mutated; only hooks are added, and only for the scope's duration.
    """
    requested_kinds = [
        field.name for field in dataclasses.fields(rules) if getattr(rules, field.name)
    ]

    installed: List[Tuple[str, nn.Module, str]] = []
    skipped: List[str] = []
    for kind in requested_kinds:
        rule = _RULES[kind]
        for name, component in _iter_canonical_mount_candidates(model, rule.mounts):
            support = rule.support(component)
            if support is None:
                skipped.append(name)
            elif not support:
                raise RelevanceRuleUnsupportedError(
                    f"{name!r} ({type(component).__name__}) cannot install the {kind!r} "
                    "relevance rule: unsupported configuration for this component."
                )
            else:
                installed.append((name, component, kind))

    # Acquire one by one so a refused acquisition (a mutually exclusive kind already
    # active) leaves none of this scope's earlier acquisitions behind.
    acquired: List[Tuple[nn.Module, str]] = []
    try:
        for name, component, kind in installed:
            _acquire_rule(name, component, kind)
            acquired.append((component, kind))
        yield RelevanceRuleCoverage(
            installed=tuple(name for name, _, _ in installed),
            skipped=tuple(skipped),
        )
    finally:
        for component, kind in acquired:
            _release_rule(component, kind)
