"""Hook installation, permanence, removal, rule selection and warning scoping."""

import warnings

import pytest

from transformer_lens.model_bridge.generalized_components.normalization import (
    NATIVE_PATH_EDIT_FALLBACK_WARNING,
)
from transformer_lens.model_bridge.lrp.hooks import LRPActivation
from transformer_lens.model_bridge.lrp.rules import LRPRules

PER_BLOCK = [
    "ln1.hook_scale",
    "ln2.hook_scale",
    "attn.q_norm.hook_scale",
    "attn.k_norm.hook_scale",
    "attn.hook_pattern",
    "attn.v.hook_out",
    "mlp.out.hook_in",
]


def expected_names(bridge):
    names = ["ln_final.hook_scale"]
    for i in range(bridge.cfg.n_layers):
        names += [f"blocks.{i}.{suffix}" for suffix in PER_BLOCK]
    return sorted(names)


def our_hooks(bridge, name):
    ids = bridge._lrp_state.hook_fn_ids
    return [h for h in bridge.hook_dict[name].fwd_hooks if id(h.user_hook) in ids]


def _fallback_warnings(caught):
    return [w for w in caught if str(w.message) == NATIVE_PATH_EDIT_FALLBACK_WARNING]


def test_patch_installs_one_permanent_hook_per_point(tiny_bridge):
    coverage = tiny_bridge.patch_lrp()
    assert coverage is tiny_bridge.get_lrp_coverage()
    assert sorted(tiny_bridge._lrp_state.hook_names) == expected_names(tiny_bridge)
    for name in expected_names(tiny_bridge):
        hooks = our_hooks(tiny_bridge, name)
        assert len(hooks) == 1 and hooks[0].is_permanent, name
    assert tiny_bridge.is_lrp_patched()


def test_hooks_survive_run_with_cache_and_reset(tiny_bridge, tokens):
    tiny_bridge.patch_lrp()
    tiny_bridge.run_with_cache(tokens, return_type="loss", incl_bwd=True)
    tiny_bridge.reset_hooks()
    for name in expected_names(tiny_bridge):
        assert len(our_hooks(tiny_bridge, name)) == 1, name


def test_patch_twice_warns_and_keeps_single_hook(tiny_bridge):
    tiny_bridge.patch_lrp()
    with pytest.warns(UserWarning, match="already installed"):
        tiny_bridge.patch_lrp()
    for name in expected_names(tiny_bridge):
        assert len(our_hooks(tiny_bridge, name)) == 1, name


def _hf_mlps(bridge):
    return [block.mlp.original_component for block in bridge.blocks]


def test_patch_swaps_mlp_activation(tiny_bridge):
    originals = [mlp.act_fn for mlp in _hf_mlps(tiny_bridge)]
    tiny_bridge.patch_lrp()
    for mlp, original in zip(_hf_mlps(tiny_bridge), originals):
        assert isinstance(mlp.act_fn, LRPActivation)
        assert mlp.act_fn.act is original
        assert mlp.act_fn.state is tiny_bridge._lrp_state


def test_unpatch_removes_hooks_and_state(tiny_bridge):
    originals = [mlp.act_fn for mlp in _hf_mlps(tiny_bridge)]
    tiny_bridge.patch_lrp()
    names = expected_names(tiny_bridge)
    tiny_bridge.unpatch_lrp()
    assert not tiny_bridge.is_lrp_patched()
    for name in names:
        assert tiny_bridge.hook_dict[name].fwd_hooks == [], name
    for mlp, original in zip(_hf_mlps(tiny_bridge), originals):
        assert mlp.act_fn is original
    with pytest.raises(RuntimeError, match="not installed"):
        tiny_bridge.lrp_on()


def test_unpatch_keeps_other_users_hooks(tiny_bridge):
    tiny_bridge.patch_lrp()
    name = "blocks.0.attn.hook_pattern"
    tiny_bridge.add_hook(name, lambda tensor, hook=None: None)
    tiny_bridge.unpatch_lrp()
    assert len(tiny_bridge.hook_dict[name].fwd_hooks) == 1
    tiny_bridge.reset_hooks()


def test_methods_require_patching(tiny_bridge):
    with pytest.raises(RuntimeError, match="not installed"):
        tiny_bridge.set_lrp_rules("ah-rule")


def test_set_rules_accepts_names_and_enum(tiny_bridge):
    tiny_bridge.patch_lrp()
    tiny_bridge.lrp_on()
    tiny_bridge.set_lrp_rules("ah-rule", LRPRules.LN_RULE)
    assert tiny_bridge.get_lrp_rules() == ["ah-rule", "ln-rule"]
    tiny_bridge.set_lrp_rules("half-rule-mlp")
    assert tiny_bridge._lrp_state.rules == {"half-rule-mlp"}


def test_unknown_rule_raises(tiny_bridge):
    tiny_bridge.patch_lrp()
    with pytest.raises(ValueError, match="Unknown LRP rule 'no-such-rule'"):
        tiny_bridge.set_lrp_rules("no-such-rule")


def test_rules_with_problem_coverage_are_rejected_unless_not_strict(tiny_bridge):
    tiny_bridge.patch_lrp()
    tiny_bridge.lrp_on()
    tiny_bridge.get_lrp_coverage().set(LRPRules.AH_RULE, "blocks.0", "unsupported", "injected")
    with pytest.raises(ValueError, match="ah-rule cannot be applied correctly on 1 of 2"):
        tiny_bridge.set_lrp_rules("ah-rule", "ln-rule")
    assert tiny_bridge._lrp_state.rules == set()  # nothing applied on failure
    with pytest.warns(UserWarning, match="Applying the rules where supported"):
        tiny_bridge.set_lrp_rules("ah-rule", "ln-rule", strict=False)
    assert tiny_bridge._lrp_state.rules == {"ah-rule", "ln-rule"}
    tiny_bridge.set_lrp_rules("ln-rule")  # unaffected rules still fine


def test_hook_names_stay_bridge_native_under_compatibility_mode(tiny_bridge):
    """Compatibility mode renames HookPoints to HookedTransformer aliases; the installed
    names (and so the coverage survey) must still use the bridge-native paths."""
    tiny_bridge.enable_compatibility_mode(no_processing=True)
    tiny_bridge.patch_lrp()
    assert sorted(tiny_bridge._lrp_state.hook_names) == expected_names(tiny_bridge)
    assert tiny_bridge.hook_dict["blocks.0.attn.v.hook_out"].name == "blocks.0.attn.hook_v"


def test_gemma_post_norms_are_hooked(tiny_gemma_bridge):
    tiny_gemma_bridge.patch_lrp()
    names = set(tiny_gemma_bridge._lrp_state.hook_names)
    for i in range(tiny_gemma_bridge.cfg.n_layers):
        for norm in ("ln1", "ln2", "ln1_post", "ln2_post"):
            assert f"blocks.{i}.{norm}.hook_scale" in names
        assert f"blocks.{i}.mlp.out.hook_in" in names
        assert f"blocks.{i}.attn.hook_pattern" in names


def test_fallback_warning_is_silenced_only_while_lrp_is_on(tiny_bridge, tokens):
    """The LN rules edit hook_scale on purpose, so the norm bridge's edited-scale warning is
    hidden while LRP is on and audible again for a user's own edit once it is off."""
    tiny_bridge.patch_lrp()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tiny_bridge.lrp_on()
        tiny_bridge.set_lrp_rules("ln-rule")
        tiny_bridge(tokens)
        assert not _fallback_warnings(caught)
        tiny_bridge.lrp_off()
        tiny_bridge.hook_dict["blocks.0.ln1.hook_scale"].add_hook(
            lambda tensor, hook=None: tensor * 1.0
        )
        try:
            tiny_bridge(tokens)
        finally:
            tiny_bridge.reset_hooks()
        assert _fallback_warnings(caught)


def test_unpatch_while_on_removes_the_warning_filter(tiny_bridge):
    tiny_bridge.patch_lrp()
    tiny_bridge.lrp_on()
    entry = tiny_bridge._lrp_state.warning_filter
    assert entry is not None and entry in warnings.filters
    tiny_bridge.unpatch_lrp()
    assert entry not in warnings.filters
