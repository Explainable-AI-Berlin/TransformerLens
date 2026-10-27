"""LRP rules for ``TransformerBridge`` backward passes.

Inside :func:`lrp_rules`, gradients through norms, attention, gated products and activations
are replaced so that ``activation * gradient`` at a hook point is an LRP relevance, with a
bit-identical forward: CP-LRP's AH-rule (Ali et al., 2022) or AttnLRP's uniform rule
(Achtibat et al., 2024) on attention, the LN-, Half- and Identity-rule as in RelP, and
CP-LRP's constant gates as in LXT. First-order and eager mode only. See
``docs/source/content/lrp.md``.

References:
    - Ali et al. (2022), "XAI for Transformers: Better Explanations through Conservative
      Propagation," ICML -- the LN-rule, the AH-rule and constant gates (CP-LRP).
    - Arras, Osman, Müller and Samek (2019), "Evaluating Recurrent Neural Network
      Explanations," BlackboxNLP -- the Half-rule (LRP-half) for multiplicative gates.
    - Achtibat et al. (2024), "AttnLRP: Attention-Aware Layer-Wise Relevance Propagation
      for Transformers," ICML -- the uniform rule on attention and other products.
    - Rezaei Jafari et al. (2024), "MambaLRP: Explaining Selective State Space Sequence
      Models," NeurIPS -- the Identity-rule for elementwise activations.
    - Rezaei Jafari, Eberle, Khakzar and Nanda (2025), "RelP: Faithful and Efficient
      Circuit Discovery in Language Models via Relevance Patching," arXiv:2508.21258.

Assumptions the runtime checks cannot see (each has an oracle test):
    - HF norms (classes named ``*Norm``) compute ``w * (x - mean) / sqrt(var + eps) + b`` over
      the last axis, with ``eps`` or ``variance_epsilon`` (else ``cfg.eps``) and ``bias``.
    - An MLP bridge with a ``gate`` projection computes ``act(gate) * up``; HF's default
      experts gate computes it per expert and multiplies by ``top_k_weights``.
    - Rebuilt attention softmaxes ``hook_attn_scores`` and projects ``pattern @ V`` through
      ``o``; HF-run attention uses q and k only for the logits and v only for the values.
    - The final-logit softcap acts on ``unembed``'s output, and MLM heads apply their
      activation between ``mlm_head`` and ``ln_final``.
"""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Iterator,
    Literal,
    Optional,
    Union,
)

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from transformer_lens.model_bridge.generalized_components import (
    AltUpBlockBridge,
    AttentionBridge,
    BertPoolerBridge,
    BlockBridge,
    GatedRMSNormBridge,
    LinearBridge,
    MLPBridge,
    MoEBridge,
    NormalizationBridge,
    OpaqueBlockBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.t5_block import T5BlockBridge
from transformer_lens.utilities.activation_functions import softcap_enabled

if TYPE_CHECKING:
    from transformer_lens.model_bridge import TransformerBridge

LRPRule = Literal["ln", "ah", "attn_half", "half", "gate", "identity"]
ALL_RULES: tuple[LRPRule, ...] = ("ln", "ah", "attn_half", "half", "gate", "identity")
DEFAULT_RULES: tuple[LRPRule, ...] = ("ln", "ah", "half", "identity")
_EXCLUSIVE = ({"ah", "attn_half"}, {"half", "gate"})
Scale = Union[float, torch.Tensor]

# Elementwise with f(0) = 0 (xIELU: up to its eps of 1e-6), so f(x) / x is a well-defined
# relevance multiplier. The Identity-rule drops the parameter gradients of PReLU and xIELU.
_TORCH_ACTIVATIONS: tuple[type[nn.Module], ...] = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU)
_TORCH_ACTIVATIONS += (nn.GELU, nn.SiLU, nn.Mish, nn.Hardswish, nn.Tanh)
_HF_ACTIVATIONS = set(
    "GELUActivation GELUTanh NewGELUActivation FastGELUActivation QuickGELUActivation "
    "AccurateGELUActivation ClippedGELUActivation SiLUActivation MishActivation "
    "ReLUSquaredActivation LinearActivation XIELUActivation BloomGelu".split()
)
_ACTIVATION_PATHS: tuple[str, ...] = ("act_fn", "act", "activation_fn", "activation", "gelu_impl")
_ACTIVATION_PATHS += ("intermediate.intermediate_act_fn",)  # BERT, RoBERTa.
_CACHE_KWARGS = ("past_key_values", "past_key_value", "layer_past")
_BLOCKS = (BlockBridge, T5BlockBridge)
# Block parts whose nonlinearities the rules cover; anything else is an unsupported mixer.
_BLOCK_PARTS = (AttentionBridge, MLPBridge, MoEBridge, NormalizationBridge, LinearBridge)
# Components whose one elementwise activation sits outside any product.
_ACTIVATION_COMPONENTS = (BertPoolerBridge, VisionProjectionBridge)
# Gradient scales at rebuilt attention's products (scores, z = o.hook_in) and at HF-run
# attention's projection outputs (q, k, v).
_ATTENTION_SCALES = {
    "ah": {"scores": 0.0, "q": 0.0, "k": 0.0},
    "attn_half": {"scores": 0.5, "z": 0.5, "q": 0.25, "k": 0.25, "v": 0.5},
}
_DROP_ACTIVATION_RULES = "Use rules without 'half', 'gate' and 'identity'."
_BYPASS_HINTS = {
    "ah": " This attention does not use hook_attn_scores inside its computation.",
    "attn_half": " This attention does not use hook_attn_scores and o inside its computation.",
    "ln": " Forward hooks that edit hook_scale or hook_normalized are not supported.",
    "identity": " Compatibility mode with processed weights calls the activation as a "
    "function; use enable_compatibility_mode(no_processing=True) or a raw bridge.",
}
_ACTIVE: set[int] = set()


class _Rewire(torch.autograd.Function):
    """Returns a copy of ``value``; the backward sends ``grad * multiplier`` to ``source``."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, source: torch.Tensor, multiplier: Scale) -> Any:
        if isinstance(multiplier, torch.Tensor):
            ctx.save_for_backward(multiplier)
        else:
            ctx.scale = multiplier
        # A copy, not a view, so user hooks can still edit it in place.
        return value.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[None, torch.Tensor, None]:
        multiplier = ctx.saved_tensors[0] if ctx.saved_tensors else ctx.scale
        # A zero tensor, not None, so backward hooks upstream still fire (edge attribution).
        return None, grad * multiplier, None


class _FrozenScaleNorm(torch.autograd.Function):
    """Returns the HF norm output ``y``; the backward removes the gradient through the scale.

    For ``y = w * xh + b`` with ``xh = (x - mean) / s`` (no mean for RMSNorm), the HF
    backward gives ``x`` the gradient ``C (w * g) / s - xh <xh, w * g> / (N s)``. Adding
    ``xh <y - b, g> / (N s)`` cancels the second term, which is the gradient through ``s``.
    Weights and ``1 + w`` offsets enter only through ``y``. ``y - b`` comes from the rounded
    output, so it loses precision when ``|b| >> |w * xh|``; in float16 the two terms can
    overflow where the plain gradient can.
    """

    @staticmethod
    def forward(
        ctx: Any, y: torch.Tensor, x: torch.Tensor, bias: Any, centered: bool, eps: Scale
    ) -> torch.Tensor:
        ctx.save_for_backward(y, x)
        ctx.bias, ctx.centered, ctx.eps = bias, centered, eps
        return y.clone()

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        y, x = ctx.saved_tensors
        dtype = torch.promote_types(x.dtype, torch.float32)
        centered_x = x.to(dtype)
        if ctx.centered:
            centered_x = centered_x - centered_x.mean(-1, keepdim=True)
        variance = centered_x.pow(2).mean(-1, keepdim=True) + ctx.eps
        y_minus_b = y.to(dtype) if ctx.bias is None else y.to(dtype) - ctx.bias.to(dtype)
        dot = (y_minus_b * grad.to(dtype)).sum(-1, keepdim=True)
        correction = centered_x * dot / (x.shape[-1] * variance)
        return grad, correction.to(x.dtype), None, None, None


def _needs_grad(tensor: Any) -> bool:
    return torch.is_grad_enabled() and getattr(tensor, "requires_grad", False)


def _name(component: nn.Module, module_name: str = "") -> str:
    hook_in = getattr(component, "hook_in", None)
    if hook_in is not None and hook_in.name:
        return hook_in.name.rsplit(".", 1)[0]
    return ".".join(part for part in module_name.split(".") if part != "_original_component")


def _ratio(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # At x == 0 the multiplier is 0: the relevance x * grad there is 0 either way, and no
    # gradient passes upstream through that entry (the limit f'(0) would pass some).
    return torch.where(x == 0, 0.0, y.detach() / x.detach())


def _unwrap(module: Any) -> Any:
    return module.original_component if isinstance(module, GeneralizedComponent) else module


def _is_activation(module: nn.Module) -> bool:
    # Softmax and nn.MultiheadAttention live in torch's activation module too.
    kind = type(module)
    return (
        kind.__module__ in ("transformers.activations", nn.ReLU.__module__)
        and "Softm" not in kind.__name__
        and not isinstance(module, nn.MultiheadAttention)
    )


def _activation(owner: nn.Module) -> Optional[nn.Module]:
    for path in _ACTIVATION_PATHS:
        candidate: Any = owner
        for attr in path.split("."):
            candidate = getattr(candidate, attr, None)
        if isinstance(candidate, nn.Module):
            return candidate
    return None


def _gated_attention(attn: AttentionBridge) -> bool:
    return attn._modules.get("gate") is not None or getattr(attn.config, "gated_q_proj", False)


def _gate_and_out(module: nn.Module) -> tuple[Any, Any]:
    """The gate and output projections of a gated MLP (also a dense layer in an MoE stack)."""
    if isinstance(module, MLPBridge):
        return getattr(module, "gate", None), getattr(module, "out", None)
    if isinstance(module, MoEBridge) and module.bound_dense:
        return module._modules.get("dense_gate"), module._modules.get("dense_out")
    return None, None


def _experts(module: nn.Module, rule: str) -> Optional[nn.Module]:
    """The experts module of a sparse MoE block, if ``module`` is one; raises if unsupported."""
    if not isinstance(module, MoEBridge) or module.bound_dense:
        return None
    hf = module.original_component
    experts = getattr(hf, "experts", None)
    # A bridge around the experts module itself (Qwen2-MoE): its MoE block covers it.
    if experts is None and hasattr(hf, "act_fn"):
        return None
    # HF's experts interface is recognised by its private default gate; anything else, or a
    # rename, fails closed.
    apply_gate = getattr(type(_unwrap(experts)), "_apply_gate", None)
    if getattr(apply_gate, "__name__", "") != "_default_apply_gate" or not getattr(
        _unwrap(experts), "has_gate", False
    ):
        raise NotImplementedError(
            f"{rule!r} on {_name(module)}: {type(hf).__name__} does not use the standard "
            f"gated HF experts module. {_DROP_ACTIVATION_RULES}"
        )
    return experts


def _activation_owner(module: nn.Module, rule: str) -> Optional[nn.Module]:
    """The HF module holding the activation of an MLP-like component, or None."""
    dense = isinstance(module, MoEBridge) and module.bound_dense
    if isinstance(module, (MLPBridge, *_ACTIVATION_COMPONENTS)) or dense:
        return _unwrap(module)
    experts = _experts(module, rule)
    return _unwrap(experts) if experts is not None else None


def _eps(norm: nn.Module) -> Optional[Scale]:
    eps = getattr(norm, "eps", None)
    return getattr(norm, "variance_epsilon", None) if eps is None else eps


class _Installer:
    """Registers rule hooks and the checks that they ran; methods skip modules they do not
    cover. Every site is a key; ``fired`` holds the keys that ran since their owners'
    ``hook_in``, and each (owner, rule) requires its keys at ``hook_out``."""

    def __init__(self, rules: Collection[LRPRule]):
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.sites: dict[LRPRule, list[str]] = {rule: [] for rule in rules}
        self.fired: set[int] = set()
        self.required: dict[tuple[int, str], set[int]] = {}
        # (module id, slot) -> (role, keys): one hook per module and slot ("out" or an input
        # argument), so shared modules apply it once and conflicting roles are caught.
        self.modules: dict[tuple[int, str], tuple[tuple, set[int]]] = {}
        self.filled_caches: set[int] = set()
        self.keys = itertools.count()

    def hook(self, module: nn.Module, fn: Callable[..., Any]) -> None:
        # Prepend, so user hooks on the same point see (and cache) the rule's output.
        self.handles.append(module.register_forward_hook(fn, prepend=True))

    def site(
        self,
        owner: GeneralizedComponent,
        rule: LRPRule,
        module: nn.Module,
        role: tuple,
        slot: str = "out",
    ) -> tuple[int, bool]:
        """A new site key that ``owner`` requires; returns (key, True if ``module`` is new)."""
        key = next(self.keys)
        if (id(owner), rule) not in self.required:
            required = self.required[(id(owner), rule)] = set()
            name = _name(owner)

            def reset(_: nn.Module, __: Any, x: Any) -> None:
                needs_grad[0] = _needs_grad(x)
                self.fired.difference_update(required)

            def check(_: nn.Module, __: Any, y: Any) -> None:
                # A frozen input with trainable parameters still needs the rules.
                cut = needs_grad[0] and not _needs_grad(y)
                if cut or (_needs_grad(y) and not required <= self.fired):
                    raise RuntimeError(
                        f"The {rule!r} LRP rule did not run in {name} on this forward path, "
                        f"so its gradients would be wrong.{_BYPASS_HINTS.get(rule, '')}"
                    )

            needs_grad = [False]
            self.hook(owner.hook_in, reset)
            self.hook(owner.hook_out, check)
        self.required[(id(owner), rule)].add(key)
        known = self.modules.get((id(module), slot))
        if known is not None and known[0] != role:
            raise NotImplementedError(
                f"{_name(owner)}: one module would need two rules ({known[0]} and {role}); "
                "use rules without one of them."
            )
        self.modules.setdefault((id(module), slot), (role, set()))[1].add(key)
        return key, known is None

    def mark(self, module: nn.Module, slot: str = "out") -> None:
        self.fired.update(self.modules[(id(module), slot)][1])

    def scale_grad_at(
        self, owner: GeneralizedComponent, rule: LRPRule, site: nn.Module, scale: Scale
    ) -> int:
        """Scale the gradient through ``site``'s output; ``owner`` checks that it ran."""
        key, new = self.site(owner, rule, site, ("scale", scale))

        def scale_hook(_: nn.Module, __: Any, out: torch.Tensor) -> Optional[torch.Tensor]:
            self.mark(site)
            return _Rewire.apply(out.detach(), out, scale) if _needs_grad(out) else None

        if new:
            self.hook(site, scale_hook)
        return key

    def ln(self, norm: nn.Module) -> None:
        if not isinstance(norm, NormalizationBridge):
            return
        hf_norm = norm.original_component
        if hf_norm is None or isinstance(hf_norm, nn.Identity):
            return
        if not type(hf_norm).__name__.endswith("Norm"):
            raise NotImplementedError(
                f"LN-rule on {_name(norm)}: {type(hf_norm).__name__} is not a known norm "
                "class. Use rules without 'ln'."
            )
        self.sites["ln"].append(_name(norm))
        key, _ = self.site(norm, "ln", norm.hook_scale, ("ln",))

        def scale_hook(_: nn.Module, __: Any, scale: torch.Tensor) -> Optional[torch.Tensor]:
            # Mirrors NormalizationBridge.forward: the native path returns HF's output (the
            # module hook applies the rule there) and takes any returned value as an edit.
            # Without user hooks it computes the scale without grad, so that case is safe
            # even if this copy goes stale.
            native = norm.use_native_layernorm_autograd or getattr(
                norm.config, "layer_norm_folding", False
            )
            if native and not (norm.hook_scale.bwd_hooks or norm.hook_normalized.bwd_hooks):
                return None
            self.fired.add(key)
            return _Rewire.apply(scale.detach(), scale, 0.0) if _needs_grad(scale) else None

        self.hook(norm.hook_scale, scale_hook)
        eps = _eps(hf_norm)
        eps = getattr(norm.config, "eps", 1e-5) if eps is None else eps
        bias = None if norm.uses_rms_norm else getattr(hf_norm, "bias", None)
        # Also covers HF code that calls the raw norm directly (per-head ln1 for edge attribution).
        self.norm_hook(_name(norm), hf_norm, bias, not norm.uses_rms_norm, eps, key)

    def norm_hook(
        self, name: str, hf_norm: nn.Module, bias: Any, centered: bool, eps: Scale, key: int
    ) -> None:
        """LN-rule on an HF norm module, hooked once and marking every owner's site."""
        if (id(hf_norm), "out") in self.modules:
            self.modules[(id(hf_norm), "out")][1].add(key)
            return
        self.modules[(id(hf_norm), "out")] = (("ln",), {key})
        bias = bias.detach() if isinstance(bias, torch.Tensor) else None

        def module_hook(_: nn.Module, args: Any, y: torch.Tensor) -> Optional[torch.Tensor]:
            if not args:
                raise RuntimeError(f"{name}: the HF norm was called with keyword arguments.")
            self.mark(hf_norm)
            if not _needs_grad(args[0]):
                return None
            return _FrozenScaleNorm.apply(y, args[0], bias, centered, eps)

        self.hook(hf_norm, module_hook)

    def attention(self, attn: nn.Module, rule: LRPRule) -> None:
        if not isinstance(attn, AttentionBridge):
            return
        name, scales = _name(attn), _ATTENTION_SCALES[rule]
        self.sites[rule].append(name)
        pattern_seam = getattr(attn.original_component, "accepts_pattern_fn", False)
        if type(attn).forward is not AttentionBridge.forward and not pattern_seam:
            # Bridges that rebuild attention softmax the scores returned by hook_attn_scores
            # and feed pattern @ V into o; scaling at these product outputs keeps every
            # cached attention gradient an LRP gradient.
            self.scale_grad_at(attn, rule, attn.hook_attn_scores, scales["scores"])
            if "z" in scales:
                self.scale_grad_at(attn, rule, attn.o.hook_in, scales["z"])
            return
        if pattern_seam and rule == "ah":
            self.scale_grad_at(attn, rule, attn.hook_pattern, 0.0)
            return
        # Otherwise HF computes the pattern itself (the bridge fires hook_pattern afterwards).
        # Queries and keys feed only the logits and values only pattern @ V, so scaling their
        # gradients is exact for everything upstream. The scale sits after each projection's
        # hook_out, so cached hook_q/k/v gradients are the scaled ones.
        projections = [attn._modules.get(key) for key in ("q", "k", "v")]
        if any(p is None for p in projections) or len({id(p) for p in projections}) < 3:
            raise NotImplementedError(
                f"{rule!r} on {name}: this attention runs inside HF and has no separate q, k "
                "and v projections to scale the gradient at (fused QKV, shared K/V, or K/V "
                f"reused from another layer). Use rules without {rule!r}."
            )
        keys = {
            role: self.scale_grad_at(attn, rule, proj, scales[role])
            for role, proj in zip("qkv", projections)
            if proj is not None and role in scales
        }
        kv = {keys[role] for role in "kv" if role in keys}
        # Cross-attention may take K and V from a cache instead of projecting them; that is
        # covered only if K and V went through the rule when the cache was filled.
        cache: list[Any] = [None]

        def note_cache(_: nn.Module, __: Any, kwargs: Any) -> None:
            cross = kwargs.get("key_value_states") is not None
            found = (kwargs.get(key) for key in _CACHE_KWARGS)
            cache[0] = next((c for c in found if c is not None), None) if cross else None

        def reuse_cache(*_: Any) -> None:
            if cache[0] is not None and kv <= self.fired:
                self.filled_caches.add(id(cache[0]))
            elif cache[0] is not None and id(cache[0]) in self.filled_caches:
                self.fired.update(kv)

        self.handles.append(attn.register_forward_pre_hook(note_cache, with_kwargs=True))
        self.hook(attn.hook_out, reuse_cache)  # Prepended: runs before the check.

    def products(self, module: nn.Module, rule: LRPRule) -> None:
        """Products of two activations: the Half-rule halves the gradient into the product,
        constant gates (CP-LRP) zero the gradient into the gating factor."""
        half = rule == "half"
        # (site, scale, input argument or None for the output)
        sites: list[tuple[nn.Module, float, Optional[int]]] = []
        attn_gate = module._modules.get("gate") if isinstance(module, AttentionBridge) else None
        experts = _experts(module, rule)
        gate, out = _gate_and_out(module)
        if attn_gate is not None and not half:
            sites = [(attn_gate, 0.0, None)]
        elif isinstance(module, MoEBridge) and experts is not None:
            hf: Any = module.original_component
            hf_experts, shared_gate = _unwrap(experts), hf._modules.get("shared_expert_gate")
            if half:
                # Router product sum_k w_k E_k(x): halve at the experts output (both factors).
                # act(gate) * up in each expert: both factors reach x linearly, so halving at
                # the experts' own copy of x gives LXT's activation gradients (not its
                # gate_up_proj parameter gradients, which do not cross x).
                sites = [(getattr(experts, "hook_out", experts), 0.5, None), (hf_experts, 0.5, 0)]
                if shared_gate is not None:  # sigmoid(gate(x)) * shared_expert(x), Qwen2-MoE.
                    sites += [(hf._modules["shared_expert"], 0.5, None), (shared_gate, 0.5, None)]
            else:  # Constant routing weights (top_k_weights) and expert gates.
                sites = [(hf_experts, 0.0, 2), (self.activation(module, rule), 0.0, None)]
                sites += [(shared_gate, 0.0, None)] if shared_gate is not None else []
        elif isinstance(module, GeneralizedComponent) and gate is not None and out is not None:
            sites = [
                (out.hook_in, 0.5, None) if half else (self.activation(module, rule), 0.0, None)
            ]
        if not sites or not isinstance(module, GeneralizedComponent):
            return
        self.sites[rule].append(_name(module))
        for site, scale, argument in sites:
            if argument is None:
                self.scale_grad_at(module, rule, site, scale)
            else:
                self.experts_input(module, rule, site, argument, scale)

    def experts_input(
        self, moe: GeneralizedComponent, rule: LRPRule, experts: nn.Module, index: int, scale: float
    ) -> None:
        """Scale the gradient into positional argument ``index`` of ``experts(hidden_states,
        top_k_index, top_k_weights)``."""
        key, _ = self.site(moe, rule, experts, ("scale", scale), slot=f"input {index}")

        def scale_input(_: nn.Module, args: Any) -> Any:
            if len(args) <= index:
                raise RuntimeError(f"{_name(moe)}: the experts were not called positionally.")
            self.fired.add(key)
            x = args[index]
            return (*args[:index], _Rewire.apply(x.detach(), x, scale), *args[index + 1 :])

        self.handles.append(experts.register_forward_pre_hook(scale_input, prepend=True))

    def activation(self, module: GeneralizedComponent, rule: str) -> nn.Module:
        owner = _activation_owner(module, rule)
        act = None if owner is None else _activation(owner)
        if act is None or getattr(act, "inplace", False):
            problem = "no activation module to hook" if act is None else "in-place activation"
            raise NotImplementedError(
                f"{rule!r} on {_name(module)}: {problem}. Use rules without {rule!r}."
            )
        return act

    def identity(self, module: nn.Module) -> None:
        owner = _activation_owner(module, "identity")
        if owner is None or not isinstance(module, GeneralizedComponent):
            return
        gated = _gate_and_out(module)[0] is not None or _experts(module, "identity") is not None
        if "gate" in self.sites and gated:
            return  # The gate rule holds this activation's input constant.
        act = self.activation(module, "identity")
        if not (isinstance(act, _TORCH_ACTIVATIONS) or type(act).__name__ in _HF_ACTIVATIONS):
            raise NotImplementedError(
                f"Identity-rule on {_name(module)}: {type(act).__name__} is not a known "
                "elementwise activation with f(0) = 0. Use rules without 'identity'."
            )
        self.sites["identity"].append(_name(module))
        _, new = self.site(module, "identity", act, ("identity",))

        # A second hook on a shared module gets the same x, y / x and gradient: idempotent.
        def act_hook(_: nn.Module, args: Any, y: torch.Tensor) -> Optional[torch.Tensor]:
            self.mark(act)
            if not _needs_grad(args[0]):
                return None
            return _Rewire.apply(y.detach(), args[0], _ratio(y, args[0]))

        if new:
            self.hook(act, act_hook)

    def between(self, producer: nn.Module, consumer: nn.Module, what: str, output: bool) -> None:
        """Identity-rule on an elementwise function HF applies between two points.

        Keeps the producer's output ``z`` and replaces ``y = f(z)``, the consumer's input
        (or its output's ``logits``), with a copy whose gradient goes to ``z`` as
        ``grad * y / z``. Fails closed when they do not pair up on a call that needs
        gradients, or when HF computes a loss from ``y`` itself.
        """
        raw: list[torch.Tensor] = []

        def rewire(y: Any, loss: Any = None) -> Any:
            z = raw.pop() if raw else None
            if not _needs_grad(y if isinstance(y, torch.Tensor) else z):
                return y
            if (
                z is None
                or not isinstance(y, torch.Tensor)
                or y.shape != z.shape
                or loss is not None
            ):
                raise RuntimeError(
                    f"The 'identity' LRP rule cannot reach the {what} on this call: call the "
                    "bridge (not original_model with labels or return_dict=False) and compute "
                    "any loss from the logits, or use rules without 'identity'."
                )
            return _Rewire.apply(y.detach(), z, _ratio(y, z))

        def on_output(_: nn.Module, __: Any, out: Any) -> None:
            y = getattr(out, "logits", None)
            if (rewired := rewire(y, getattr(out, "loss", None))) is not y:
                out.logits = rewired

        self.handles.append(consumer.register_forward_pre_hook(lambda *_: raw.clear()))
        self.hook(producer, lambda _, __, z: raw.__setitem__(slice(None), [z]))
        if output:
            self.hook(consumer, on_output)
        else:
            self.handles.append(
                consumer.register_forward_pre_hook(
                    lambda _, args: (rewire(args[0]), *args[1:]), prepend=True
                )
            )
        self.sites["identity"].append(what)

    def uncovered(self, model: TransformerBridge) -> None:
        """Apply the LN-rule to plain HF norms; raise if another norm or activation runs."""
        acts = {"half", "gate", "identity"} & set(self.sites)
        # Activations of covered MLPs are known; any other one sits in an uncovered product.
        owners = [_activation_owner(m, "") for m in model.modules()] if acts else []
        known = {id(_activation(owner)) for owner in owners if owner is not None}
        mlm_head, ln_final = model._modules.get("mlm_head"), model._modules.get("ln_final")
        if mlm_head is not None and ln_final is not None:
            if "identity" in self.sites:
                self.between(mlm_head, ln_final, "MLM head activation", output=False)
            for parent in model.original_model.modules():
                if any(child in (mlm_head, _unwrap(mlm_head)) for child in parent.children()):
                    known |= {id(m) for m in parent.modules() if _is_activation(m)}
        # The bridge tree and the HF tree: HF modules outside every bridge run too.
        trees = [*model.original_model.named_modules(), *model.named_modules()]
        for module_name, module in {id(m): (n, m) for n, m in trees}.values():
            if (id(module), "out") in self.modules or isinstance(module, GeneralizedComponent):
                continue
            kind, eps, label = type(module).__name__, _eps(module), _name(module, module_name)
            plain_rms = kind.endswith("RMSNorm") and not isinstance(module, nn.RMSNorm)
            if "ln" in self.sites and eps is not None and (plain_rms or kind == "LayerNorm"):
                bias = getattr(module, "bias", None) if kind == "LayerNorm" else None
                self.norm_hook(label, module, bias, kind == "LayerNorm", eps, -1)
                self.sites["ln"].append(label)
                continue
            if "Norm" in kind and "ln" in self.sites:
                remedy = "Use rules without 'ln'."
            elif _is_activation(module) and acts and id(module) not in known:
                remedy = _DROP_ACTIVATION_RULES
            else:
                continue

            def tripwire(
                _: nn.Module,
                __: Any,
                out: Any,
                label: str = f"{label} ({kind})",
                remedy: str = remedy,
            ) -> None:
                if _needs_grad(out):
                    raise RuntimeError(
                        f"{label} ran outside every component the LRP rules "
                        f"{sorted(self.sites)} cover, so its gradient (or its gated product's) "
                        f"would be ordinary. {remedy}"
                    )

            self.hook(module, tripwire)


def _unsupported(model: nn.Module, rules: Collection[LRPRule]) -> list[str]:
    """Components that a requested rule cannot cover, each with the rules to drop."""
    found, requested = [], set(rules)
    mixer = "token mixer: use rules without 'ah', 'attn_half', 'half' or 'gate'"
    for module_name, module in model.named_modules():
        label = f"{_name(module, module_name)} ({type(module).__name__})"
        if isinstance(module, (OpaqueBlockBridge, AltUpBlockBridge)):
            found.append(f"{label}: not supported")
        elif isinstance(module, GatedRMSNormBridge) and {"ln", "half", "gate"} & requested:
            found.append(f"{label}: gated norm: use rules without 'ln', 'half' or 'gate'")
        elif isinstance(module, AttentionBridge) and _gated_attention(module):
            q_gate = getattr(module.config, "gated_q_proj", False)
            if {"half", "attn_half"} & requested or (q_gate and "gate" in requested):
                found.append(f"{label}: gated output: use rules without 'half' or 'attn_half'")
        elif isinstance(module, _BLOCKS) and {"ah", "attn_half", "half", "gate"} & requested:
            # Token mixers other than attention: SSM, linear attention, gated convolution.
            found += [
                f"{_name(child, f'{module_name}.{key}')} ({type(child).__name__}): {mixer}"
                for key, child in module.named_children()
                if isinstance(child, GeneralizedComponent) and not isinstance(child, _BLOCK_PARTS)
            ]
    return found


@contextmanager
def lrp_rules(
    model: TransformerBridge, rules: Optional[Collection[LRPRule]] = None
) -> Iterator[dict[LRPRule, list[str]]]:
    """Use LRP rules in every backward pass through ``model`` inside the ``with`` block.

    The forward pass is bit-identical; only gradients change. Run the forward and the
    backward inside the block.

    Args:
        model: The ``TransformerBridge`` to explain.
        rules: ``"ln"``; ``"ah"`` or ``"attn_half"`` for attention; ``"half"`` or ``"gate"`` for
            products; ``"identity"``. ``None`` means ``["ln", "ah", "half", "identity"]``. A rule
            whose operation the model lacks has an empty site list; every component that has
            it gets the rule or raises.

    Yields:
        A mapping from each rule to the components it was installed on.

    Raises:
        ValueError: Empty, unknown or mutually exclusive rules.
        NotImplementedError: A rule meets a component it does not support (the message names
            the rules to drop).
        RuntimeError: Nested use, or a forward that needs gradients bypassed a rule.
    """
    requested: list[LRPRule] = list(dict.fromkeys(DEFAULT_RULES if rules is None else rules))
    if (
        not requested
        or set(requested) - set(ALL_RULES)
        or any(pair <= set(requested) for pair in _EXCLUSIVE)
    ):
        raise ValueError(
            f"rules must be a nonempty subset of {ALL_RULES} without both 'ah' and "
            f"'attn_half' or both 'half' and 'gate', got {rules!r}"
        )
    keys = {id(model), id(model.original_model)}
    if keys & _ACTIVE:
        raise RuntimeError("lrp_rules is already active on this model; nesting would stack rules.")
    unsupported = _unsupported(model, requested)
    if unsupported:
        raise NotImplementedError(f"LRP rules {requested} cannot cover: {'; '.join(unsupported)}")

    inst = _Installer(requested)
    _ACTIVE.update(keys)
    try:
        for module in model.modules():
            for rule in requested:
                install = {"ln": inst.ln, "identity": inst.identity}.get(rule)
                if install is not None:
                    install(module)
                else:
                    (inst.attention if rule in _ATTENTION_SCALES else inst.products)(module, rule)
        if "identity" in requested and softcap_enabled(
            getattr(model.cfg, "output_logits_soft_cap", None)
        ):
            # An extension: RelP and AttnLRP prescribe no rule for the softcap.
            inst.between(model.unembed, model.original_model, "final-logit softcap", output=True)
        inst.uncovered(model)
        yield inst.sites
    finally:
        _ACTIVE.difference_update(keys)
        for handle in inst.handles:
            handle.remove()
