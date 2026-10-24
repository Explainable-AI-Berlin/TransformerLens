"""Per-component support survey for the LRP rules.

``survey`` walks a ``TransformerBridge`` once and decides, for every block and every
rule, whether the rule can be attached there. It returns the install plan (which hook
points to hook, which activation modules to wrap) together with an ``LRPCoverage`` table
that records the decision and its reason, so ``set_lrp_rules`` can refuse rules that
would be applied incorrectly and users can inspect what a model supports.

Statuses:

- ``applied``: the rule is fully attached to this component.
- ``partial``: only part of the rule could be attached; the result would be wrong.
- ``unsupported``: the component exists but the rule cannot be attached to it.
- ``absent``: the component this rule acts on does not exist here (harmless).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components import (
    ALiBiJointQKVAttentionBridge,
    BloomAttentionBridge,
    CodeGenAttentionBridge,
    GlmMoeDsaAttentionBridge,
    JointQKVAttentionBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    MLAAttentionBridge,
    MPTALiBiAttentionBridge,
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.gated_mlp import (
    GatedMLPBridge,
)
from transformer_lens.model_bridge.generalized_components.joint_gate_up_mlp import (
    JointGateUpMLPBridge,
)
from transformer_lens.model_bridge.generalized_components.mlp import MLPBridge
from transformer_lens.model_bridge.generalized_components.moe import MoEBridge
from transformer_lens.model_bridge.lrp.rules import LRPRules

Status = Literal["applied", "partial", "unsupported", "absent"]
STATUSES: Tuple[Status, ...] = ("applied", "partial", "unsupported", "absent")
PROBLEM_STATUSES: Tuple[Status, ...] = ("partial", "unsupported")
# Reporting order for the summary line: the worst status a rule has on any component.
_WORST_FIRST: Tuple[Status, ...] = ("unsupported", "partial", "absent")

NORM_RULES = (LRPRules.LN_RULE, LRPRules.LN_RULE_POST, LRPRules.LN_RULE_QK)
ATTENTION_RULES = (LRPRules.AH_RULE, LRPRules.HALF_RULE_ATTN)
MLP_RULES = (LRPRules.HALF_RULE_MLP, LRPRules.IDENTITY_RULE_ACT)

# Attention bridges that recompute attention in Python and run ``hook_pattern`` before the
# value multiplication, so an edit there reaches the output. The base ``AttentionBridge``
# and other delegating variants fire ``hook_pattern`` on HuggingFace's returned weights
# after the fact, where an edit is a silent no-op. Only listed classes (and their
# subclasses) are trusted.
RECONSTRUCTING_ATTENTION_BRIDGES: Tuple[type, ...] = (
    PositionEmbeddingsAttentionBridge,
    JointQKVAttentionBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    ALiBiJointQKVAttentionBridge,
    BloomAttentionBridge,
    CodeGenAttentionBridge,
    MLAAttentionBridge,
    MPTALiBiAttentionBridge,
    GlmMoeDsaAttentionBridge,
)

SCALE_SUFFIX = ".hook_scale"
PATTERN_SUFFIX = ".attn.hook_pattern"
VALUE_SUFFIX = ".attn.v.hook_out"
MLP_PRODUCT_SUFFIX = ".mlp.out.hook_in"

_ABSENT_NORM = {
    LRPRules.LN_RULE: "no block norm",
    LRPRules.LN_RULE_POST: "no post-sublayer norm",
    LRPRules.LN_RULE_QK: "no query/key norm",
}


def classify_scale_hook(name: str) -> LRPRules:
    """Map a ``hook_scale`` point to the norm rule that governs it, by its owner's path.

    Example:
        >>> classify_scale_hook("blocks.3.attn.q_norm.hook_scale")
        <LRPRules.LN_RULE_QK: 'ln-rule-qk'>
        >>> classify_scale_hook("blocks.3.ln1_post.hook_scale")
        <LRPRules.LN_RULE_POST: 'ln-rule-post'>
        >>> classify_scale_hook("ln_final.hook_scale")
        <LRPRules.LN_RULE: 'ln-rule'>
    """
    owner = name[: -len(SCALE_SUFFIX)]
    if ".attn." in owner:
        return LRPRules.LN_RULE_QK
    if owner.endswith("_post"):
        return LRPRules.LN_RULE_POST
    return LRPRules.LN_RULE


@dataclass(frozen=True)
class CoverageEntry:
    """One rule's decision for one component: a status and the reason."""

    status: Status
    detail: str


@dataclass
class LRPCoverage:
    """Rule name -> component key (``blocks.3``, ``ln_final``) -> entry."""

    entries: Dict[str, Dict[str, CoverageEntry]] = field(default_factory=dict)

    def set(self, rule: LRPRules, key: str, status: Status, detail: str) -> None:
        """Record ``rule``'s decision for component ``key``."""
        self.entries.setdefault(rule.value, {})[key] = CoverageEntry(status, detail)

    def table(self, rule: Union[LRPRules, str]) -> Dict[str, CoverageEntry]:
        """A copy of one rule's per-component entries."""
        name = rule.value if isinstance(rule, LRPRules) else rule
        return dict(self.entries.get(name, {}))

    def keys_with(self, rule: Union[LRPRules, str], *statuses: Status) -> List[str]:
        """The component keys on which ``rule`` has one of ``statuses``."""
        return [key for key, entry in self.table(rule).items() if entry.status in statuses]

    def counts(self, rule: Union[LRPRules, str]) -> Dict[Status, int]:
        """How many components fall under each status for ``rule``."""
        table = self.table(rule)
        return {s: sum(1 for e in table.values() if e.status == s) for s in STATUSES}

    def __str__(self) -> str:
        """One line per rule: status counts, plus the worst status's components and reason."""
        lines = []
        for rule in LRPRules:
            table, counts = self.table(rule), self.counts(rule)
            parts = [f"{status} {n}" for status, n in counts.items() if n]
            line = f"{rule.value:18s} {', '.join(parts) if parts else 'no component'}"
            worst = next((s for s in _WORST_FIRST if counts[s]), None)
            if worst is not None and (worst != "absent" or counts["applied"] == 0):
                keys = self.keys_with(rule, worst)
                line += f"   {worst} on {compress_keys(keys)}: {table[keys[0]].detail}"
            lines.append(line)
        lines.append("(.detail() lists every component)")
        return "\n".join(lines)

    # A bare `model.get_lrp_coverage()` in a notebook shows repr: keep it as readable as print.
    __repr__ = __str__

    def detail(self, rule: Optional[Union[LRPRules, str]] = None) -> str:
        """Per-component table, consecutive blocks collapsed into ranges (all rules by default)."""
        rules: List[Union[LRPRules, str]] = list(LRPRules) if rule is None else [rule]
        lines = []
        for r in rules:
            name = r.value if isinstance(r, LRPRules) else r
            lines.append(f"{name}:")
            groups: Dict[Tuple[str, str], List[str]] = {}
            for key, entry in self.table(name).items():
                groups.setdefault((entry.status, entry.detail), []).append(key)
            for (status, det), keys in groups.items():
                lines.append(f"  {status:12s} {compress_keys(keys):22s} {det}")
        return "\n".join(lines)


def compress_keys(keys: List[str]) -> str:
    """Collapse consecutive block indices into ranges.

    Example:
        >>> compress_keys(["blocks.0", "blocks.1", "blocks.2", "blocks.5", "ln_final"])
        'blocks.0-2, blocks.5, ln_final'
    """
    indices, other = [], []
    for key in keys:
        head, _, tail = key.partition(".")
        if head == "blocks" and tail.isdigit():
            indices.append(int(tail))
        else:
            other.append(key)
    indices.sort()
    parts, i = [], 0
    while i < len(indices):
        j = i
        while j + 1 < len(indices) and indices[j + 1] == indices[j] + 1:
            j += 1
        parts.append(f"blocks.{indices[i]}" if i == j else f"blocks.{indices[i]}-{indices[j]}")
        i = j + 1
    return ", ".join(parts + other)


@dataclass
class InstallPlan:
    """The hook points to hook and the activation modules to wrap, as decided by ``survey``."""

    scale_hooks: List[Tuple[str, LRPRules]] = field(default_factory=list)
    pattern_hooks: List[str] = field(default_factory=list)
    value_hooks: List[str] = field(default_factory=list)
    product_hooks: List[str] = field(default_factory=list)
    activation_swaps: List[Tuple[str, GatedMLPBridge]] = field(default_factory=list)

    def __len__(self) -> int:
        return (
            len(self.scale_hooks)
            + len(self.pattern_hooks)
            + len(self.value_hooks)
            + len(self.product_hooks)
        )


def _child(component: nn.Module, name: str) -> Optional[nn.Module]:
    """A registered sub-bridge by name, without falling through to the HF module."""
    return component._modules.get(name)


def _has_hook(component: nn.Module, name: str) -> bool:
    return isinstance(component._modules.get(name), HookPoint)


def survey(bridge: Any, points: Dict[str, HookPoint]) -> Tuple[InstallPlan, LRPCoverage]:
    """Decide per block which rules attach where; ``points`` maps canonical names to hooks."""
    blocks = getattr(bridge, "blocks", None)
    if blocks is None:
        raise RuntimeError("LRP needs a model with a `blocks` stack; this bridge has none.")
    plan, coverage = InstallPlan(), LRPCoverage()
    for i, block in enumerate(blocks):
        key = f"blocks.{i}"
        _survey_attention(key, block, points, plan, coverage)
        _survey_mlp(key, block, points, plan, coverage)
        _survey_norms(key, block, points, plan, coverage)
    for name in points:
        if name.endswith(SCALE_SUFFIX) and not name.startswith("blocks."):
            rule = classify_scale_hook(name)
            plan.scale_hooks.append((name, rule))
            coverage.set(rule, name[: -len(SCALE_SUFFIX)], "applied", "hook_scale")
    return plan, coverage


def _survey_attention(
    key: str,
    block: nn.Module,
    points: Dict[str, HookPoint],
    plan: InstallPlan,
    coverage: LRPCoverage,
) -> None:
    attn = _child(block, "attn")
    if attn is None:
        for rule in ATTENTION_RULES:
            coverage.set(rule, key, "absent", "no attention module")
        return
    cls = type(attn).__name__
    if not isinstance(attn, RECONSTRUCTING_ATTENTION_BRIDGES):
        detail = (
            f"{cls} is not a known reconstructing attention bridge; hook_pattern edits would "
            "not reach the output"
        )
        for rule in ATTENTION_RULES:
            coverage.set(rule, key, "unsupported", detail)
        return
    pattern, value = f"{key}{PATTERN_SUFFIX}", f"{key}{VALUE_SUFFIX}"
    if pattern not in points:
        for rule in ATTENTION_RULES:
            coverage.set(rule, key, "unsupported", f"{cls} exposes no hook_pattern")
        return
    plan.pattern_hooks.append(pattern)
    coverage.set(LRPRules.AH_RULE, key, "applied", f"{cls}: hook_pattern")
    if value in points:
        plan.value_hooks.append(value)
        coverage.set(LRPRules.HALF_RULE_ATTN, key, "applied", f"{cls}: hook_pattern, v.hook_out")
    else:
        coverage.set(
            LRPRules.HALF_RULE_ATTN,
            key,
            "partial",
            f"{cls} exposes hook_pattern but no attn.v.hook_out; values would not be halved",
        )


def _survey_mlp(
    key: str,
    block: nn.Module,
    points: Dict[str, HookPoint],
    plan: InstallPlan,
    coverage: LRPCoverage,
) -> None:
    mlp = _child(block, "mlp")
    if mlp is None:
        for rule in MLP_RULES:
            coverage.set(rule, key, "absent", "no MLP module")
        return
    cls = type(mlp).__name__
    if isinstance(mlp, MoEBridge):
        detail = f"{cls}: experts are fused inside the HuggingFace MoE block; no per-expert hooks"
        for rule in MLP_RULES:
            coverage.set(rule, key, "unsupported", detail)
        return
    if isinstance(mlp, GatedMLPBridge):
        product = f"{key}{MLP_PRODUCT_SUFFIX}"
        if product in points:
            plan.product_hooks.append(product)
            coverage.set(LRPRules.HALF_RULE_MLP, key, "applied", f"{cls}: out.hook_in")
        else:
            coverage.set(
                LRPRules.HALF_RULE_MLP, key, "unsupported", f"{cls} exposes no out.hook_in"
            )
        if isinstance(mlp, JointGateUpMLPBridge):
            coverage.set(
                LRPRules.IDENTITY_RULE_ACT,
                key,
                "unsupported",
                f"{cls} runs a captured activation; wrapping it is not implemented",
            )
            return
        act = getattr(mlp.original_component, "act_fn", None)
        if isinstance(act, nn.Module):
            plan.activation_swaps.append((key, mlp))
            coverage.set(
                LRPRules.IDENTITY_RULE_ACT, key, "applied", f"{cls}: act_fn ({type(act).__name__})"
            )
        else:
            coverage.set(
                LRPRules.IDENTITY_RULE_ACT,
                key,
                "unsupported",
                f"{cls}: the HuggingFace MLP has no act_fn module to wrap",
            )
        return
    if isinstance(mlp, MLPBridge):
        coverage.set(
            LRPRules.HALF_RULE_MLP, key, "absent", f"{cls}: ungated MLP has no gate product"
        )
        coverage.set(
            LRPRules.IDENTITY_RULE_ACT,
            key,
            "unsupported",
            f"{cls}: activation wrapping is not implemented for ungated MLPs",
        )
        return
    for rule in MLP_RULES:
        coverage.set(rule, key, "unsupported", f"{cls} is not a supported MLP bridge")


def _survey_norms(
    key: str,
    block: nn.Module,
    points: Dict[str, HookPoint],
    plan: InstallPlan,
    coverage: LRPCoverage,
) -> None:
    prefix = f"{key}."
    hooked: Dict[LRPRules, List[str]] = {}
    for name in points:
        if name.startswith(prefix) and name.endswith(SCALE_SUFFIX):
            rule = classify_scale_hook(name)
            hooked.setdefault(rule, []).append(name[len(prefix) : -len(SCALE_SUFFIX)])
            plan.scale_hooks.append((name, rule))
    unhookable: Dict[LRPRules, List[str]] = {}
    for path, module in block.named_modules(remove_duplicate=False):
        if not path or "original_component" in path or not isinstance(module, GeneralizedComponent):
            continue
        if "norm" in type(module).__name__.lower() and not _has_hook(module, "hook_scale"):
            rule = classify_scale_hook(f"{key}.{path}{SCALE_SUFFIX}")
            unhookable.setdefault(rule, []).append(
                f"{path} ({type(module).__name__} has no hook_scale)"
            )
    for rule in NORM_RULES:
        ok, bad = hooked.get(rule, []), unhookable.get(rule, [])
        if ok and bad:
            coverage.set(
                rule, key, "partial", f"hooked {', '.join(ok)}; cannot hook {'; '.join(bad)}"
            )
        elif bad:
            coverage.set(rule, key, "unsupported", "; ".join(bad))
        elif ok:
            coverage.set(rule, key, "applied", ", ".join(ok))
        else:
            coverage.set(rule, key, "absent", _ABSENT_NORM[rule])
