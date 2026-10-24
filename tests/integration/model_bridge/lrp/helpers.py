"""Shared helpers: run a forward/backward pass while capturing activations and gradients."""

import torch


def _make_capture(name, captured):
    def capture(tensor, hook=None):
        # A fresh view separates this consumer's gradient from other uses of the same tensor
        # (e.g. the residual stream feeding both a norm and the residual add).
        view = tensor.view_as(tensor)
        view.retain_grad()
        captured[name] = view
        return view

    capture.__name__ = f"capture_{name}"
    return capture


def run_backward(bridge, tokens, capture=(), cache=()):
    """Forward + backward of the LM loss.

    ``capture`` names get a forward hook prepended to the hook point that keeps the tensor
    (with ``retain_grad``) so ``.grad`` holds the gradient flowing into that point.
    ``cache`` names are cached through ``run_with_cache(incl_bwd=True)``; their gradients
    sit under ``<name>_grad``. Non-permanent hooks are removed afterwards.
    """
    captured = {}
    for name in capture:
        bridge.hook_dict[name].add_hook(_make_capture(name, captured), prepend=True)
    try:
        bridge.zero_grad(set_to_none=True)
        names_filter = list(cache) if cache else (lambda name: False)
        loss, cached = bridge.run_with_cache(
            tokens, return_type="loss", names_filter=names_filter, incl_bwd=True
        )
    finally:
        bridge.reset_hooks()
    return loss, captured, cached


def rms_norm_input_grad(x, weight, eps, upstream, detach_scale):
    """Gradient w.r.t. the input of ``x / rms(x) * weight`` given the upstream gradient."""
    x = x.detach().clone().float().requires_grad_(True)
    scale = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    if detach_scale:
        scale = scale.detach()
    y = x / scale * weight.detach().float()
    y.backward(upstream.float())
    return x.grad


def norm_eps(norm_bridge):
    component = norm_bridge.original_component
    eps = getattr(component, "variance_epsilon", None)
    if eps is None:
        eps = getattr(component, "eps")
    return float(eps)


def norm_weight(norm_bridge):
    """Effective multiplicative weight of a bridged RMSNorm (Gemma stores ``1 + w``)."""
    weight = norm_bridge.original_component.weight
    if getattr(norm_bridge.config, "rmsnorm_uses_offset", False):
        return 1.0 + weight
    return weight


def all_rules_except(*excluded):
    from transformer_lens.model_bridge.lrp import LRPRules

    return [rule.value for rule in LRPRules if rule.value not in excluded]


def assert_close(actual, expected, rtol=1e-5, atol=1e-6):
    assert torch.allclose(
        actual, expected, rtol=rtol, atol=atol
    ), f"max abs diff {(actual - expected).abs().max().item():.3e}"
