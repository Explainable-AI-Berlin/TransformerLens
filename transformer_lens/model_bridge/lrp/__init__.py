"""Layer-wise relevance propagation (LRP) for ``TransformerBridge`` models.

LRP attributes a model's output back to its input tokens. This package hooks a
``TransformerBridge`` so that a standard gradient-times-input backward pass computes LRP
relevance scores instead of ordinary gradients, by applying LRP rules to individual
components through TransformerLens forward hooks. The forward pass stays on the model's
native path whenever LRP is off or a rule is not selected.

The methods live on ``TransformerBridge`` (see :class:`LRPMixin`):

.. code-block:: python

    from transformer_lens.model_bridge import TransformerBridge

    model = TransformerBridge.boot_transformers("Qwen/Qwen3-0.6B")
    model.patch_lrp()  # install the hooks once; prints nothing, returns the coverage table
    model.lrp_on()
    model.set_lrp_rules("identity-rule-act", "half-rule-mlp", "ah-rule", "ln-rule")

    tokens = model.to_tokens("Hello, world")
    loss, cache = model.run_with_cache(tokens, return_type="loss", incl_bwd=True)
    # Relevance at any hook point is activation times gradient.
    relevance = cache["blocks.11.hook_out"] * cache["blocks.11.hook_out_grad"]

``print(model.get_lrp_coverage())`` shows, per rule and block, whether the rule was
attached and why not; ``set_lrp_rules`` refuses rules the table marks as incorrectly
applicable unless ``strict=False``.
"""

from transformer_lens.model_bridge.lrp.coverage import CoverageEntry, LRPCoverage
from transformer_lens.model_bridge.lrp.hooks import LRPActivation, LRPState
from transformer_lens.model_bridge.lrp.mixin import LRPMixin
from transformer_lens.model_bridge.lrp.rules import LRPRules

__all__ = [
    "CoverageEntry",
    "LRPActivation",
    "LRPCoverage",
    "LRPMixin",
    "LRPRules",
    "LRPState",
]
