"""Forward-equivalent relevance-rule primitives, plus the scoped context that installs them.

Each primitive is a ``torch.autograd.Function`` that reproduces its native forward
value exactly while replacing the backward pass with the rule's closed-form VJP.
``use_relevance_rules`` installs these rules on a model's canonical mount points only
for the duration of a ``with`` block, targeting components positionally (by mount
name, never by class) and reporting which mounts were installed versus skipped.
Most rules are toggled through the component's own ``_RelevanceRuleCapable`` code;
the attention rules instead rewrite tensors at HookPoints every attention bridge
already exposes (``_HOOK_RULES``), so attention implementations carry no rule code.
"""

import dataclasses
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
    Protocol,
    Set,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from transformer_lens.hook_points import HookPoint


def ln_rule_grad(grad_output: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """Core LN-rule VJP: divide by ``denom`` without differentiating through it.

    Shared by the ``ln_rule`` primitive below and by any integration (such as
    ``NormalizationBridge``) that wraps a component's own native forward call
    instead of reproducing the division itself.
    """
    return grad_output / denom


class _LNRule(torch.autograd.Function):
    """LN-rule: forward is ``numerator / denom``; the VJP treats ``denom`` as constant."""

    @staticmethod
    def forward(ctx: Any, numerator: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(denom)
        return numerator / denom

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (denom,) = ctx.saved_tensors
        return ln_rule_grad(grad_output, denom), None


def ln_rule(numerator: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """Apply the LN-rule: native division forward, denom-as-constant backward."""
    result: torch.Tensor = _LNRule.apply(numerator, denom)
    return result


class _IdentityRule(torch.autograd.Function):
    """Identity-rule: forward is the native activation; the VJP is ``grad_out * phi(x)``.

    ``phi`` is ``f(x) / x``, the removable singularity at ``x == 0`` filled in with its
    limit ``0.5``. This holds for any elementwise activation with ``f(0) == 0`` and a
    well-defined derivative at zero, which covers SiLU and both GELU variants.
    """

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, act_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        y = act_fn(x)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        x, y = ctx.saved_tensors
        safe_x = torch.where(x == 0, torch.ones_like(x), x)
        phi = torch.where(x == 0, torch.full_like(x, 0.5), y / safe_x)
        return grad_output * phi, None


def identity_rule(x: torch.Tensor, act_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Apply the Identity-rule for an elementwise activation with ``f(0) == 0``."""
    result: torch.Tensor = _IdentityRule.apply(x, act_fn)
    return result


class _HalfRule(torch.autograd.Function):
    """Half-rule: forward is ``u * v``; the VJP halves each ordinary product-rule term."""

    @staticmethod
    def forward(ctx: Any, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(u, v)
        return u * v

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u, v = ctx.saved_tensors
        return 0.5 * grad_output * v, 0.5 * grad_output * u


def half_rule(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply the Half-rule: native product forward, evenly split backward."""
    result: torch.Tensor = _HalfRule.apply(u, v)
    return result


class _ScaleGradient(torch.autograd.Function):
    """Identity forward; the VJP scales the incoming gradient by a constant factor."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, factor: float) -> torch.Tensor:
        ctx.factor = factor
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return ctx.factor * grad_output, None


def scale_gradient(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Pass ``x`` through unchanged while scaling its gradient by ``factor``.

    The Half-rule on a product ``u * v`` halves each ordinary product-rule term,
    which is the same as halving the single gradient that enters the product before
    it splits. When the product is computed inside an opaque module the bridge cannot
    reach term by term (its native forward is called as one unit), scaling the
    gradient entering the product by ``0.5`` reproduces the Half-rule at that point
    without altering the native forward value.
    """
    result: torch.Tensor = _ScaleGradient.apply(x, factor)
    return result


class RelevanceRuleConflictError(RuntimeError):
    """A hook would silently break a rule-active forward/backward invariant.

    Raised instead of the ordinary warn-and-fall-back a component would use when
    no rule is active, since falling back while a rule is active would compose the
    rule with the hook edit and break the bit-identical-forward guarantee.
    """


class RelevanceRuleUnsupportedError(RuntimeError):
    """A requested relevance rule cannot be installed on an otherwise-capable component.

    Raised at ``use_relevance_rules`` entry, before any forward or backward pass, when
    a component reports the requested kind in its own ``_relevance_rule_unsupported_kinds``
    -- for example a gated-MLP recompute path backed by an unrecognized weight-orientation
    class, or an activation form the Identity-rule does not support -- or exposes only
    some of the HookPoints a hook rule uses (a fused-QKV attention without ``hook_v``
    under the attention Half-rule). Raised in the forward pass instead when a hook
    rule's HookPoints do not feed the component's output (see ``_HookRule``).
    Distinct from a kind that is simply absent from
    ``_relevance_rule_kinds`` without being named there (reported ``skipped``, not
    raised): that covers a component not implementing the protocol at all, or one whose
    mount genuinely never deals with the kind (for example a recurrent mixer mounted at
    ``attn``, which has no attention pattern), both benign non-applicability rather than
    a rule request the component was expected to honor.
    """


@dataclasses.dataclass(frozen=True)
class RelevanceRules:
    """Which relevance rules to request for the duration of a ``use_relevance_rules`` scope.

    Each field names a rule kind. Setting it ``True`` requests that rule wherever a
    component at that kind's canonical mount point implements ``_RelevanceRuleCapable``
    (or, for a hook rule, exposes the HookPoints it rewrites). Unset fields (the default)
    leave the corresponding components untouched.

    ``normalization``: LN-rule (the denominator is a constant in the backward pass) on
    the block norms ``ln1``/``ln2``, the sandwich post-norms ``ln1_post``/``ln2_post``
    and ``ln_final``.
    ``qk_normalization``: the same LN-rule on the per-head norms ``attn.q_norm``/
    ``attn.k_norm``, requested separately so block-level and per-head norms can be
    studied independently.
    ``activation``: Identity-rule on a gated MLP's activation function.
    ``multiplicative_gate``: Half-rule on a gated MLP's ``act(gate) * up`` product.
    ``attention``: AH-rule -- the attention pattern is a constant in the backward pass,
    so relevance reaches the residual stream only through the values.
    ``attention_half``: Half-rule on the ``pattern @ V`` product -- the gradient reaching
    the pattern and the gradient reaching the values are each halved.
    ``attention`` and ``attention_half`` rewrite the same product, so they are mutually
    exclusive. Backward hooks see the rule's gradient; one on a value a rule holds
    constant never fires: ``hook_scale`` of a python-path norm under the LN-rule, and
    ``hook_pattern``, ``hook_attn_scores`` and the query/key hooks under the AH-rule.
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
    actually enabled. ``skipped`` holds the dotted path of every mount that matched a
    requested kind's canonical mount name but did not implement the relevance-rule
    protocol there (or, for a hook rule, exposes none of its HookPoints), so no rule
    could be installed.
    """

    installed: Tuple[str, ...]
    skipped: Tuple[str, ...]


@runtime_checkable
class _RelevanceRuleCapable(Protocol):
    """Structural contract a component must satisfy to accept a relevance rule.

    ``_relevance_rule_kinds`` names every ``RelevanceRules`` field the component
    answers to at its current mount -- a gated-MLP node answers to both
    "activation" (Identity-rule on its activation function) and
    "multiplicative_gate" (Half-rule on its gate*up product) independently, since
    either can be requested without the other. ``_enable_relevance_rule``/
    ``_disable_relevance_rule`` take the specific kind being toggled and touch
    only that kind's state, without touching model configuration, so the
    component's own state is the only thing that changes and only for the
    scope's duration.

    A component may optionally also define ``_relevance_rule_unsupported_kinds``
    (a ``Tuple[str, ...]``, not part of this structural protocol so components that
    omit it stay isinstance-compatible) naming kinds it is expected to honor at its
    mount but currently cannot -- ``use_relevance_rules`` raises
    ``RelevanceRuleUnsupportedError`` for those instead of reporting them skipped.
    """

    _relevance_rule_kinds: Tuple[str, ...]

    def _enable_relevance_rule(self, kind: str) -> None:
        ...

    def _disable_relevance_rule(self, kind: str) -> None:
        ...


# What torch passes to and expects from a backward pre-hook.
_Grads = Union[Tuple[torch.Tensor, ...], torch.Tensor]


@dataclasses.dataclass(frozen=True)
class _HookRule:
    """A rule applied through hooks on HookPoints the mounted component already has.

    ``grad_scales`` maps each HookPoint to the factor on the gradient reaching its output.
    Factor 0 detaches the value in the forward pass, so nothing upstream is
    differentiated. Any other factor scales the gradient in a backward pre-hook, which
    runs before the HookPoint's own backward hooks, so those see the rule's gradient.

    This is only correct if the HookPoints feed the component's output. So the rule
    checks at ``hook_out`` that each of them fired earlier in the same call. If one did
    not (a bridge delegating to the wrapped module fires ``hook_pattern`` only on the
    returned weights), the forward raises ``RelevanceRuleUnsupportedError`` instead of
    silently keeping the ordinary gradient.
    """

    grad_scales: Mapping[str, float]

    def support(self, module: nn.Module) -> Optional[bool]:
        """True if installable on ``module``; None if it exposes none of the rule's
        HookPoints (the rule does not apply there); False if it exposes only some."""
        exposed = [
            isinstance(getattr(module, hook_name, None), HookPoint)
            for hook_name in (*self.grad_scales, "hook_out")
        ]
        if not any(exposed[:-1]):
            return None
        return all(exposed)

    def install(self, name: str, kind: str, module: nn.Module) -> List[RemovableHandle]:
        """Register the rule's hooks on ``module``; return their handles."""
        fired: Set[str] = set()

        def _mark_fired(hook_name: str, factor: float) -> Callable[..., Optional[torch.Tensor]]:
            def _hook(hook_point: nn.Module, args: Any, output: torch.Tensor):
                fired.add(hook_name)
                return output.detach() if factor == 0.0 else None

            return _hook

        def _scale_grad(factor: float) -> Callable[[nn.Module, _Grads], _Grads]:
            def _hook(hook_point: nn.Module, grad_output: _Grads) -> _Grads:
                return tuple(factor * grad for grad in grad_output)

            return _hook

        def _check_fired_before_output(hook_point: nn.Module, args: Any, output: Any) -> None:
            missing = [hook_name for hook_name in self.grad_scales if hook_name not in fired]
            fired.clear()
            if missing:
                raise RelevanceRuleUnsupportedError(
                    f"{name!r} ({type(module).__name__}) cannot apply the {kind!r} "
                    f"relevance rule: its forward did not run {', '.join(missing)} before "
                    "computing its output (for example because it delegates to the wrapped "
                    "module), so the rule would silently not apply. To leave such mounts "
                    "out, scope the request to a capable subtree, for example "
                    "use_relevance_rules(bridge.blocks, rules)."
                )

        # Count only HookPoints fired within the current call.
        handles = [module.register_forward_pre_hook(lambda *_: fired.clear())]
        for hook_name, factor in self.grad_scales.items():
            hook_point = cast(HookPoint, getattr(module, hook_name))
            handles.append(hook_point.register_forward_hook(_mark_fired(hook_name, factor)))
            if factor != 0.0:
                handles.append(hook_point.register_full_backward_pre_hook(_scale_grad(factor)))
        hook_out = cast(HookPoint, module.hook_out)
        handles.append(hook_out.register_forward_hook(_check_fired_before_output))
        return handles


# Rule kinds applied through _HookRule instead of the _RelevanceRuleCapable protocol.
# In every attention bridge that computes pattern @ V itself, hook_pattern runs on the
# pattern it multiplies (activation patching relies on that) and hook_v's output reaches
# the output only through that product, so these are the product's two factors.
_HOOK_RULES: Mapping[str, _HookRule] = {
    # AH-rule: the pattern is a constant in the backward pass.
    "attention": _HookRule({"hook_pattern": 0.0}),
    # Half-rule: halve the gradient reaching each factor of the product.
    "attention_half": _HookRule({"hook_pattern": 0.5, "hook_v": 0.5}),
}

# Canonical mount name per rule kind. Targeting is positional: a component is only
# considered for a kind when it sits at that kind's mount name, never by isinstance,
# so a same-class component mounted elsewhere is left untouched -- a q_norm sharing
# NormalizationBridge's class is reached only through "qk_normalization", never through
# "normalization". The normalization kind lists the pre-norm mounts (ln1, ln2), the
# sandwich post-norm mounts (ln1_post, ln2_post) that sandwich-norm architectures
# mount there, and the final norm, matching the pinned RelP reference. The attention
# kinds also list the encoder-decoder mounts (self_attn, cross_attn).
_CANONICAL_MOUNTS: Mapping[str, Tuple[str, ...]] = {
    "normalization": ("ln1", "ln2", "ln1_post", "ln2_post", "ln_final"),
    "qk_normalization": ("q_norm", "k_norm"),
    "activation": ("mlp",),
    "multiplicative_gate": ("mlp",),
    "attention": ("attn", "self_attn", "cross_attn"),
    "attention_half": ("attn", "self_attn", "cross_attn"),
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


def _enable_rule(name: str, module: nn.Module, kind: str) -> None:
    """Install a hook rule's hooks, or toggle a protocol rule on the component."""
    hook_rule = _HOOK_RULES.get(kind)
    if hook_rule is None:
        cast(_RelevanceRuleCapable, module)._enable_relevance_rule(kind)
        return
    handles: Dict[str, List[RemovableHandle]] = (
        getattr(module, "_relevance_rule_hook_handles", None) or {}
    )
    handles[kind] = hook_rule.install(name, kind, module)
    setattr(module, "_relevance_rule_hook_handles", handles)


def _disable_rule(module: nn.Module, kind: str) -> None:
    """Remove a hook rule's hooks, or toggle a protocol rule off on the component."""
    if kind not in _HOOK_RULES:
        cast(_RelevanceRuleCapable, module)._disable_relevance_rule(kind)
        return
    for handle in getattr(module, "_relevance_rule_hook_handles").pop(kind):
        handle.remove()


def _acquire_rule(name: str, module: nn.Module, kind: str) -> None:
    """Enable ``module``'s ``kind`` rule only on the outermost scope that requests it.

    Refcounted per kind, not per module: a gated-MLP node can have its
    "activation" rule and "multiplicative_gate" rule independently nested to
    different depths, so one kind's inner exit must never disable the other.
    Refcounting also keeps a nested request of a hook rule from stacking a second
    rewrite on the first (two Half-rule hooks would quarter the gradient).
    """
    counts: Dict[str, int] = getattr(module, "_relevance_rule_refcounts", None) or {}
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
        _enable_rule(name, module, kind)
    counts[kind] = count + 1
    setattr(module, "_relevance_rule_refcounts", counts)


def _release_rule(module: nn.Module, kind: str) -> None:
    """Disable ``module``'s ``kind`` rule only once its innermost scope exits."""
    counts: Dict[str, int] = getattr(module, "_relevance_rule_refcounts", None) or {}
    count = counts.get(kind, 0) - 1
    counts[kind] = max(count, 0)
    setattr(module, "_relevance_rule_refcounts", counts)
    if count <= 0:
        _disable_rule(module, kind)


@contextmanager
def use_relevance_rules(model: nn.Module, rules: RelevanceRules) -> Iterator[RelevanceRuleCoverage]:
    """Install the requested relevance rules on ``model`` only for this scope.

    Targeting is positional: a component is considered for a rule kind only when it
    sits at that kind's canonical mount name (never by class). A canonical mount
    occupied by a component that does not implement ``_RelevanceRuleCapable``, or
    whose ``_relevance_rule_kinds`` simply omits the requested kind, is reported as
    skipped -- both are benign non-applicability, covering a structurally different
    architecture or a mount whose current dispatch path the rule does not wrap. A
    component that additionally names the requested kind in its own
    ``_relevance_rule_unsupported_kinds`` raises ``RelevanceRuleUnsupportedError``
    instead: that names a kind the component is expected to honor at this mount but
    cannot given its current configuration, so silently skipping it would let
    analysis proceed as if the caller had never asked. A hook rule (``_HOOK_RULES``)
    splits the same way by the HookPoints it uses: a mount exposing none of them is
    skipped, one exposing only some raises, and one whose forward does not feed them
    into its output raises on that forward. Requesting a kind that has no canonical
    mount at all raises ``ValueError`` before the install loop, since there is no mount
    to target and the scope would otherwise return empty coverage as though the
    request had succeeded. Enabling two mutually exclusive kinds on one component
    (``attention`` and ``attention_half``), in one scope or nested ones, also raises
    ``ValueError``. Scopes over the same model are reference-counted, so an inner
    scope's exit never disables a rule an outer scope still needs. No model configuration is mutated; the only state that
    changes lives on the participating components, and only for the scope's
    duration.
    """
    requested_kinds = [
        field.name for field in dataclasses.fields(rules) if getattr(rules, field.name)
    ]

    unmapped_kinds = [kind for kind in requested_kinds if kind not in _CANONICAL_MOUNTS]
    if unmapped_kinds:
        joined = ", ".join(repr(kind) for kind in unmapped_kinds)
        raise ValueError(
            f"No canonical mount is defined for relevance-rule kind(s) {joined}. "
            "Such a kind has no mount to target, so it would install nothing and "
            "silently return an empty coverage report instead of applying the rule."
        )

    installed: List[Tuple[str, nn.Module, str]] = []
    skipped: List[str] = []
    for kind in requested_kinds:
        mount_names = _CANONICAL_MOUNTS.get(kind, ())
        hook_rule = _HOOK_RULES.get(kind)
        for name, module in _iter_canonical_mount_candidates(model, mount_names):
            # True: install; False: expected here but cannot be honored; None: skip.
            support: Optional[bool]
            if hook_rule is not None:
                support = hook_rule.support(module)
            elif not isinstance(module, _RelevanceRuleCapable):
                support = None
            elif kind in module._relevance_rule_kinds:
                support = True
            elif kind in getattr(module, "_relevance_rule_unsupported_kinds", ()):
                support = False
            else:
                support = None
            if support:
                installed.append((name, module, kind))
                continue
            if support is False:
                raise RelevanceRuleUnsupportedError(
                    f"{name!r} ({type(module).__name__}) cannot install the {kind!r} "
                    "relevance rule: unsupported configuration for this component."
                )
            skipped.append(name)

    # Acquire one by one so a refused acquisition (a mutually exclusive kind already
    # active) leaves none of this scope's earlier acquisitions behind.
    acquired: List[Tuple[nn.Module, str]] = []
    try:
        for name, module, kind in installed:
            _acquire_rule(name, module, kind)
            acquired.append((module, kind))
        yield RelevanceRuleCoverage(
            installed=tuple(name for name, _, _ in installed),
            skipped=tuple(skipped),
        )
    finally:
        for module, kind in acquired:
            _release_rule(module, kind)
