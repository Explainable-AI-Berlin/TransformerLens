"""``TransformerBridge`` methods that install and control the hook-based LRP rules.

``patch_lrp`` surveys the model (see ``coverage``), attaches permanent forward hooks for
the rules wherever they are supported and wraps each gated MLP's activation module. The
hooks are no-ops until ``lrp_on`` is called and rules are selected, so the forward pass of
a patched model is unchanged; ``reset_hooks()``, ``run_with_cache`` and ``run_with_hooks``
leave the hooks in place, and ``unpatch_lrp`` removes them again.
"""

import logging
import re
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.gated_mlp import (
    GatedMLPBridge,
)
from transformer_lens.model_bridge.generalized_components.normalization import (
    NATIVE_PATH_EDIT_FALLBACK_WARNING,
)
from transformer_lens.model_bridge.lrp.coverage import (
    PROBLEM_STATUSES,
    CoverageEntry,
    LRPCoverage,
    survey,
)
from transformer_lens.model_bridge.lrp.hooks import (
    LRPActivation,
    LRPState,
    make_mlp_product_hook,
    make_pattern_hook,
    make_scale_hook,
    make_value_hook,
)
from transformer_lens.model_bridge.lrp.rules import LRPRules

log = logging.getLogger(__name__)

STATE_ATTR = "_lrp_state"
RuleLike = Union[str, LRPRules]


# runtime_checkable so the test suite's beartype instrumentation can isinstance-check it.
@runtime_checkable
class _LRPHost(Protocol):
    """The slice of the bridge surface the LRP methods rely on."""

    @property
    def hook_dict(self) -> Dict[str, HookPoint]:
        ...

    def add_perma_hook(self, name: str, hook_fn: Any) -> None:
        ...

    def get_hook_point(self, hook_name: str) -> Optional[HookPoint]:
        ...


def _get_state(host: _LRPHost) -> Optional[LRPState]:
    # The bridge's __init__ never runs mixin initialisers, so the state lives in __dict__
    # (bypassing nn.Module registration), the same way the bridge stores original_model.
    return host.__dict__.get(STATE_ATTR)


def _require_state(host: _LRPHost) -> LRPState:
    state = _get_state(host)
    if state is None:
        raise RuntimeError("LRP is not installed on this model; call model.patch_lrp() first.")
    return state


def _resolves_structurally(root: Any, name: str, target: HookPoint) -> bool:
    """Whether ``name`` reaches ``target`` through registered child modules alone.

    Bridge-native names (``blocks.0.attn.v.hook_out``) are module paths; HookedTransformer
    aliases (``blocks.0.attn.hook_v``) are attributes resolved on the fly and never appear
    in a module's ``_modules``.
    """
    module: Any = root
    for part in name.split("."):
        if module is None:
            return False
        module = getattr(module, "_modules", {}).get(part)
    return module is target


def _canonical_hook_points(host: _LRPHost) -> Dict[str, HookPoint]:
    """Bridge-native name -> HookPoint, with HookedTransformer aliases collapsed onto one entry.

    ``HookPoint.name`` cannot serve as the key: compatibility mode renames hook points to
    their HookedTransformer aliases (``mlp.hook_post`` for ``mlp.out.hook_in``), and the
    survey matches bridge-native suffixes.
    """
    by_point: Dict[int, Tuple[HookPoint, List[str]]] = {}
    for name, hook_point in host.hook_dict.items():
        by_point.setdefault(id(hook_point), (hook_point, []))[1].append(name)
    points: Dict[str, HookPoint] = {}
    for hook_point, names in by_point.values():
        native = [name for name in names if _resolves_structurally(host, name, hook_point)]
        points[native[0] if native else (hook_point.name or names[0])] = hook_point
    return points


def _swap_activation(state: LRPState, mlp: GatedMLPBridge) -> None:
    """Replace the HuggingFace MLP's activation module with an ``LRPActivation`` wrapper."""
    hf_mlp = mlp.original_component
    assert hf_mlp is not None
    act = getattr(hf_mlp, "act_fn")
    setattr(hf_mlp, "act_fn", LRPActivation(state, act))
    state.swapped_activations.append((hf_mlp, act))


def _normalize_rule(rule: RuleLike) -> LRPRules:
    if isinstance(rule, LRPRules):
        return rule
    try:
        return LRPRules(rule)
    except ValueError:
        valid = ", ".join(repr(r.value) for r in LRPRules)
        raise ValueError(f"Unknown LRP rule {rule!r}; valid rules: {valid}.") from None


def _identity_rule_table(host: _LRPHost, state: LRPState) -> Dict[str, CoverageEntry]:
    """Coverage of identity-rule-act, downgraded where compatibility mode bypasses the wrapper.

    ``enable_compatibility_mode()`` with weight processing makes the bridge compute gated
    MLPs with a functional activation, so the wrapped ``act_fn`` module never runs. That can
    happen after ``patch_lrp``, so it is checked whenever rules are selected or enabled.
    """
    table = state.coverage.table(LRPRules.IDENTITY_RULE_ACT)
    blocks = getattr(host, "blocks")
    for key, entry in table.items():
        if entry.status != "applied" or not key.startswith("blocks."):
            continue
        mlp = blocks[int(key.split(".")[1])]._modules.get("mlp")
        if getattr(mlp, "_use_processed_weights", False):
            table[key] = CoverageEntry(
                "unsupported",
                "compatibility mode with weight processing bypasses the wrapped act_fn; "
                "use enable_compatibility_mode(no_processing=True)",
            )
    return table


def _silence_fallback_warning(state: LRPState) -> None:
    """Hide the norm bridge's edited-scale warning while the LN rules intentionally edit it."""
    if state.warning_filter is not None:
        return
    warnings.filterwarnings(
        "ignore", message=re.escape(NATIVE_PATH_EDIT_FALLBACK_WARNING), category=UserWarning
    )
    # filterwarnings inserts at the front; remember the exact entry so lrp_off removes only it.
    state.warning_filter = tuple(warnings.filters[0])


def _restore_fallback_warning(state: LRPState) -> None:
    entry, state.warning_filter = state.warning_filter, None
    if entry is None:
        return
    # typeshed types warnings.filters as a Sequence; CPython's is the list filterwarnings edits.
    filters = cast(List[Any], warnings.filters)
    try:
        filters.remove(entry)
    except ValueError:
        pass  # a warnings.catch_warnings() block already discarded the copy holding it


class LRPMixin:
    """Hook-based layer-wise relevance propagation on a ``TransformerBridge``.

    Mixed into ``TransformerBridge``; every method takes the bridge as ``self``. The rule
    names are the values of :class:`~transformer_lens.model_bridge.lrp.rules.LRPRules`.
    """

    def patch_lrp(self: _LRPHost) -> LRPCoverage:
        """Survey the model and attach the LRP rule hooks wherever they are supported.

        The hooks are permanent, so ``reset_hooks()`` and ``run_with_cache`` leave them
        installed; remove them with ``unpatch_lrp``. The returned coverage table (also
        available as ``get_lrp_coverage``) records which rules apply where and why not.

        Raises:
            RuntimeError: If no rule can be attached anywhere on this model.
        """
        existing = _get_state(self)
        if existing is not None:
            warnings.warn(
                "LRP is already installed on this model; patch_lrp() ignored.", stacklevel=2
            )
            return existing.coverage

        points = _canonical_hook_points(self)
        plan, coverage = survey(self, points)
        if len(plan) == 0:
            raise RuntimeError("No LRP rule can be attached to this model:\n" + str(coverage))
        state = LRPState(coverage=coverage)

        installed: List[Tuple[str, Callable[..., Any]]] = []
        installed += [(name, make_scale_hook(state, rule)) for name, rule in plan.scale_hooks]
        installed += [(name, make_pattern_hook(state)) for name in plan.pattern_hooks]
        installed += [(name, make_value_hook(state)) for name in plan.value_hooks]
        installed += [(name, make_mlp_product_hook(state)) for name in plan.product_hooks]
        for name, hook_fn in installed:
            self.add_perma_hook(name, hook_fn)
            state.hook_names.append(name)
            state.hook_fn_ids.add(id(hook_fn))
        for _, mlp in plan.activation_swaps:
            _swap_activation(state, mlp)

        log.info("LRP coverage:\n%s", coverage)
        self.__dict__[STATE_ATTR] = state
        return coverage

    def unpatch_lrp(self: _LRPHost) -> None:
        """Remove the hooks installed by ``patch_lrp``, restore activations, drop the state."""
        state = _require_state(self)
        _restore_fallback_warning(state)
        points = _canonical_hook_points(self)
        for name in state.hook_names:
            hook_point = points.get(name) or self.get_hook_point(name)
            if hook_point is None:
                continue
            keep = []
            for handle in hook_point.fwd_hooks:
                if id(handle.user_hook) in state.hook_fn_ids:
                    handle.hook.remove()
                else:
                    keep.append(handle)
            hook_point.fwd_hooks = keep
        for hf_mlp, act in state.swapped_activations:
            setattr(hf_mlp, "act_fn", act)
        state.swapped_activations.clear()
        del self.__dict__[STATE_ATTR]

    def lrp_on(self: _LRPHost) -> None:
        """Enable LRP: the selected rules modify the backward pass from the next forward on.

        Warns if a selected rule can no longer be applied, which happens when
        ``enable_compatibility_mode()`` ran after ``set_lrp_rules``.
        """
        state = _require_state(self)
        state.enabled = True
        _silence_fallback_warning(state)
        if LRPRules.IDENTITY_RULE_ACT.value in state.rules:
            bad = [
                f"{key}: {entry.detail}"
                for key, entry in _identity_rule_table(self, state).items()
                if entry.status in PROBLEM_STATUSES
            ]
            if bad:
                warnings.warn(
                    f"'{LRPRules.IDENTITY_RULE_ACT.value}' is selected but no longer applies on "
                    f"{len(bad)} component(s) ({bad[0]}); call set_lrp_rules again to re-validate.",
                    stacklevel=2,
                )

    def lrp_off(self: _LRPHost) -> None:
        """Disable LRP: the model computes standard gradients again."""
        state = _require_state(self)
        state.enabled = False
        _restore_fallback_warning(state)

    def set_lrp_rules(self: _LRPHost, *rules: RuleLike, strict: bool = True) -> None:
        """Select the active LRP rules by name (replaces the previous selection).

        Every requested rule is validated against the model's coverage table. A rule that
        would be applied incorrectly somewhere (``partial`` or ``unsupported`` on a
        component that has the relevant module) raises ``ValueError``; with
        ``strict=False`` it warns and is applied where it is supported. A rule that applies
        to no component at all only warns, so rule lists can be shared across models.

        Args:
            *rules: Rule names (``LRPRules`` values) or ``LRPRules`` members.
            strict: Raise instead of warning when a rule cannot be applied correctly.

        Raises:
            ValueError: For an unknown rule name, or a rule that cannot be applied correctly
                when ``strict`` is True.
        """
        state = _require_state(self)
        resolved = {_normalize_rule(rule) for rule in rules}

        problems: List[str] = []
        vacuous: List[str] = []
        for rule in sorted(resolved, key=lambda r: r.value):
            if rule is LRPRules.IDENTITY_RULE_ACT:
                table = _identity_rule_table(self, state)
            else:
                table = state.coverage.table(rule)
            bad = {key: e for key, e in table.items() if e.status in PROBLEM_STATUSES}
            if bad:
                shown = list(bad.items())[:3]
                examples = "; ".join(f"{key}: {e.status}, {e.detail}" for key, e in shown)
                more = f" (+{len(bad) - len(shown)} more)" if len(bad) > len(shown) else ""
                problems.append(
                    f"{rule.value} cannot be applied correctly on {len(bad)} of {len(table)} "
                    f"components: {examples}{more}"
                )
            elif not any(e.status == "applied" for e in table.values()):
                vacuous.append(rule.value)

        if problems:
            message = "\n".join(problems)
            if strict:
                raise ValueError(
                    message + "\nPass strict=False to apply the rules where they are supported; "
                    "see model.get_lrp_coverage() for the full table."
                )
            warnings.warn(
                message + "\nApplying the rules where supported (strict=False).", stacklevel=2
            )
        for name in vacuous:
            warnings.warn(f"LRP rule {name!r} applies to no component of this model.", stacklevel=2)

        state.rules = {rule.value for rule in resolved}

    def get_lrp_rules(self: _LRPHost) -> List[str]:
        """The currently selected rule names, sorted."""
        return sorted(_require_state(self).rules)

    def get_lrp_coverage(self: _LRPHost) -> LRPCoverage:
        """The per-rule, per-block support table built by ``patch_lrp`` (``print`` it)."""
        return _require_state(self).coverage

    def is_lrp_patched(self: _LRPHost) -> bool:
        """Whether ``patch_lrp`` has installed the LRP hooks on this model."""
        return _get_state(self) is not None
