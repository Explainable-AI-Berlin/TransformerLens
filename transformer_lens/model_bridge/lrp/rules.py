"""Rule names and the tensor-level primitives of the hook-based LRP rules.

Each rule keeps the forward value of the tensor it acts on and only rewires its
autograd graph, so a plain gradient-times-input backward pass through a hooked
``TransformerBridge`` yields layer-wise relevance propagation scores.
"""

from enum import Enum
from typing import Callable

import torch


class LRPRules(Enum):
    """The relevance rules ``TransformerBridge.set_lrp_rules`` accepts, by string value.

    The norm rules are split by mount position: ``ln-rule`` covers the block norms that
    feed a sublayer or the unembedding (``ln1``, ``ln2``, ``ln_final``), ``ln-rule-post``
    the norms applied to a sublayer output before the residual add (Gemma 2/3
    ``ln1_post``, ``ln2_post``) and ``ln-rule-qk`` the attention query/key norms
    (``attn.q_norm``, ``attn.k_norm`` on Qwen3, Gemma 3, OLMo 2).
    """

    IDENTITY_RULE_ACT = "identity-rule-act"
    HALF_RULE_MLP = "half-rule-mlp"
    AH_RULE = "ah-rule"
    HALF_RULE_ATTN = "half-rule-attn"
    LN_RULE = "ln-rule"
    LN_RULE_POST = "ln-rule-post"
    LN_RULE_QK = "ln-rule-qk"


def stabilize(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Push values away from zero so that ``z / stabilize(z)`` is finite.

    Zeros move to ``+eps``; every other value moves ``eps`` further from zero, keeping
    its sign.
    """
    return z + ((z == 0.0).to(z) + z.sign()) * eps


def half_gradient(x: torch.Tensor) -> torch.Tensor:
    """Same value as ``x``; gradient scaled by one half (the relevance split of a product).

    Example:
        >>> x = torch.tensor(3.0, requires_grad=True)
        >>> half_gradient(x).backward()
        >>> float(x.grad)
        0.5
    """
    return 0.5 * x + 0.5 * x.detach()


def identity_act(x: torch.Tensor, act: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """``act(x)`` with the activation treated as an identity in the backward pass.

    Returns ``x * (act(x) / x).detach()`` with ``x`` stabilized away from zero, so the
    Jacobian becomes the element-wise ratio ``act(x) / x`` instead of ``act'(x)``. The
    stabilizer perturbs the forward value by less than about ``1e-4``.
    """
    z = act(x)
    x = stabilize(x)
    return x * (z / x).detach()
