"""Attention and LN relevance rules on one tiny offline bridge per attention family.

The attention rules rewrite ``hook_pattern`` and ``hook_v``, which is only correct if
those HookPoints are the factors of the ``pattern @ V`` product in each family's own
attention reconstruction. That is checked family by family on random-weight models
built from programmatic HF configs (no network, no checkpoint): forward stays
bit-identical, the AH-rule detaches the pattern while keeping the value path, the
Half-rule halves the gradient entering attention exactly, a ``hook_pattern`` edit
composes with the rule, a native-source model (whose wrapped attention runs
``hook_pattern`` inside its own computation) is covered, and a delegating attention
bridge (OPT) raises on its forward instead of silently ignoring the rule. The
plain-LayerNorm families also exercise the LN-rule on the python-norm path, which
the RMS-norm fixtures of ``test_relevance_lens.py`` never reach.
"""

from functools import lru_cache
from typing import Callable, Optional

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    BloomConfig,
    CodeGenConfig,
    DeepseekV3Config,
    GPT2Config,
    GPTNeoXConfig,
    MptConfig,
    OPTConfig,
    PretrainedConfig,
    Qwen2Config,
)

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRules,
    RelevanceRuleUnsupportedError,
    ln_rule,
    use_relevance_rules,
)
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

TOKENS = torch.tensor([[1, 5, 7, 42, 9]])
N_LAYERS = 2


def _qwen2() -> Qwen2Config:
    return Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=97,
        max_position_embeddings=64,
    )


def _gpt2() -> GPT2Config:
    return GPT2Config(
        vocab_size=97,
        n_positions=64,
        n_embd=32,
        n_layer=N_LAYERS,
        n_head=4,
    )


def _gpt_neox() -> GPTNeoXConfig:
    return GPTNeoXConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        vocab_size=97,
        max_position_embeddings=64,
    )


def _bloom() -> BloomConfig:
    return BloomConfig(vocab_size=97, hidden_size=32, n_layer=N_LAYERS, n_head=4)


def _mpt() -> MptConfig:
    return MptConfig(
        vocab_size=97,
        max_seq_len=64,
        d_model=32,
        expansion_ratio=2,
        n_layers=N_LAYERS,
        n_heads=4,
        no_bias=True,
    )


def _codegen() -> CodeGenConfig:
    # CodeGen splits its fused qkv over mp_num=4 groups, so n_head must be a multiple
    # of 4 and rotary_dim even and at most head_dim.
    return CodeGenConfig(
        vocab_size=97,
        n_positions=64,
        n_ctx=64,
        n_embd=32,
        n_layer=N_LAYERS,
        n_head=4,
        rotary_dim=4,
    )


def _deepseek_v3() -> DeepseekV3Config:
    # first_k_dense_replace == num_hidden_layers keeps every layer dense; the MoE
    # fields are still mandatory for the config to validate.
    return DeepseekV3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        first_k_dense_replace=N_LAYERS,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        moe_intermediate_size=16,
        vocab_size=97,
        max_position_embeddings=64,
    )


def _opt() -> OPTConfig:
    return OPTConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=N_LAYERS,
        ffn_dim=64,
        word_embed_proj_dim=32,
        do_layer_norm_before=True,
        vocab_size=97,
        max_position_embeddings=64,
    )


# family -> (config builder, architecture string); one per attention bridge class that
# reconstructs attention, plus OPT for the delegating base bridge.
FAMILIES: dict[str, tuple[Callable[[], PretrainedConfig], str]] = {
    "qwen2": (_qwen2, "Qwen2ForCausalLM"),  # PositionEmbeddingsAttentionBridge
    "gpt2": (_gpt2, "GPT2LMHeadModel"),  # JointQKVAttentionBridge
    "gpt_neox": (_gpt_neox, "GPTNeoXForCausalLM"),  # JointQKVPositionEmbeddingsAttentionBridge
    "bloom": (_bloom, "BloomForCausalLM"),  # BloomAttentionBridge
    "mpt": (_mpt, "MPTForCausalLM"),  # MPTALiBiAttentionBridge
    "codegen": (_codegen, "CodeGenForCausalLM"),  # CodeGenAttentionBridge
    "deepseek_v3": (_deepseek_v3, "DeepseekV3ForCausalLM"),  # MLAAttentionBridge
    "opt": (_opt, "OPTForCausalLM"),  # delegating AttentionBridge
}
RECONSTRUCTING = ["qwen2", "gpt2", "gpt_neox", "bloom", "mpt", "codegen", "deepseek_v3"]
# Plain NormalizationBridge families whose norms take the python-norm path.
LAYERNORM_PYTHON_PATH = ["gpt2", "bloom", "mpt", "codegen"]


@lru_cache(maxsize=None)
def _build(name: str) -> TransformerBridge:
    make_config, architecture = FAMILIES[name]
    config = make_config()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    return build_bridge_from_module(
        model, architecture, hf_config=config, tokenizer=None, device="cpu"
    )


@pytest.fixture(scope="module", params=RECONSTRUCTING)
def reconstructing_bridge(request: pytest.FixtureRequest) -> TransformerBridge:
    return _build(request.param)


@pytest.fixture(scope="module", params=LAYERNORM_PYTHON_PATH)
def layernorm_bridge(request: pytest.FixtureRequest) -> TransformerBridge:
    return _build(request.param)


@pytest.fixture(scope="module")
def opt_bridge() -> TransformerBridge:
    return _build("opt")


def _backward_grad_at(
    bridge: TransformerBridge, hook: HookPoint, rules: RelevanceRules
) -> Optional[torch.Tensor]:
    """The gradient ``hook`` receives in one rule-active backward, or None if it never fires."""
    seen: dict[str, torch.Tensor] = {}

    def _record(tensor: torch.Tensor, hook: Optional[HookPoint] = None) -> None:
        seen["grad"] = tensor.detach().clone()

    hook.add_hook(_record, dir="bwd")
    try:
        with use_relevance_rules(bridge, rules):
            bridge(TOKENS).sum().backward()
    finally:
        hook.remove_hooks(dir="both")
        for parameter in bridge.parameters():
            parameter.grad = None
    return seen.get("grad")


class TestAttentionRules:
    @pytest.mark.parametrize("kind", ["attention", "attention_half"])
    def test_forward_is_bit_identical_and_every_block_installs(
        self, reconstructing_bridge: TransformerBridge, kind: str
    ) -> None:
        with torch.no_grad():
            baseline = reconstructing_bridge(TOKENS)
        with use_relevance_rules(reconstructing_bridge, RelevanceRules(**{kind: True})) as coverage:
            with torch.no_grad():
                active = reconstructing_bridge(TOKENS)
        assert torch.equal(active, baseline)
        assert set(coverage.installed) == {f"blocks.{i}.attn" for i in range(N_LAYERS)}
        assert coverage.skipped == ()

    def test_ah_rule_detaches_the_pattern_and_keeps_the_value_path(
        self, reconstructing_bridge: TransformerBridge
    ) -> None:
        attn = reconstructing_bridge.blocks[-1].attn
        plain = _backward_grad_at(reconstructing_bridge, attn.hook_attn_scores, RelevanceRules())
        assert plain is not None and plain.abs().sum() > 0
        # Detached pattern: no gradient ever reaches the scores, so the hook never fires.
        detached = _backward_grad_at(
            reconstructing_bridge, attn.hook_attn_scores, RelevanceRules(attention=True)
        )
        assert detached is None
        # Nothing downstream of the last block's values changes, so their gradient does not.
        plain_value_grad = _backward_grad_at(reconstructing_bridge, attn.hook_v, RelevanceRules())
        value_grad = _backward_grad_at(
            reconstructing_bridge, attn.hook_v, RelevanceRules(attention=True)
        )
        assert value_grad is not None and plain_value_grad is not None
        assert value_grad.abs().sum() > 0
        assert torch.equal(value_grad, plain_value_grad)

    def test_half_rule_halves_the_gradient_entering_attention(
        self, reconstructing_bridge: TransformerBridge
    ) -> None:
        # attn.hook_in feeds only the q/k/v projections (unlike ln1.hook_out, which on
        # parallel-attention CodeGen also feeds the MLP), and every path from it to the
        # output crosses the pattern or the values, so the rule halves it exactly. The
        # last block is the only one whose downstream gradient the rule leaves unchanged.
        attn = reconstructing_bridge.blocks[-1].attn
        plain = _backward_grad_at(reconstructing_bridge, attn.hook_in, RelevanceRules())
        halved = _backward_grad_at(
            reconstructing_bridge, attn.hook_in, RelevanceRules(attention_half=True)
        )
        assert plain is not None and halved is not None
        torch.testing.assert_close(halved, 0.5 * plain, rtol=1e-6, atol=0)

    @pytest.mark.parametrize("hook_name", ["hook_pattern", "hook_v"])
    def test_backward_hooks_on_the_factors_see_the_half_rule_gradient(
        self, reconstructing_bridge: TransformerBridge, hook_name: str
    ) -> None:
        hook = getattr(reconstructing_bridge.blocks[-1].attn, hook_name)
        plain = _backward_grad_at(reconstructing_bridge, hook, RelevanceRules())
        halved = _backward_grad_at(reconstructing_bridge, hook, RelevanceRules(attention_half=True))
        assert plain is not None and halved is not None
        assert torch.equal(halved, 0.5 * plain)

    def test_pattern_hook_edit_composes_with_the_ah_rule(
        self, reconstructing_bridge: TransformerBridge
    ) -> None:
        """The rule acts on the pattern hook_pattern returns, so an edit there is
        honored first and then held constant -- no conflict, same forward."""
        attn = reconstructing_bridge.blocks[-1].attn

        def _zero_first_head(
            pattern: torch.Tensor, hook: Optional[HookPoint] = None
        ) -> torch.Tensor:
            mask = torch.ones_like(pattern)
            mask[:, 0] = 0.0
            return pattern * mask

        with torch.no_grad():
            unhooked = reconstructing_bridge(TOKENS)
        attn.hook_pattern.add_hook(_zero_first_head)
        try:
            with torch.no_grad():
                hooked = reconstructing_bridge(TOKENS)
            with use_relevance_rules(reconstructing_bridge, RelevanceRules(attention=True)):
                with torch.no_grad():
                    hooked_with_rule = reconstructing_bridge(TOKENS)
        finally:
            attn.hook_pattern.remove_hooks()
        assert not torch.equal(hooked, unhooked)
        assert torch.equal(hooked_with_rule, hooked)


class TestDelegatingAttention:
    @pytest.mark.parametrize("kind", ["attention", "attention_half"])
    def test_forward_raises_instead_of_silently_ignoring_the_rule(
        self, opt_bridge: TransformerBridge, kind: str
    ) -> None:
        """OPT's base AttentionBridge delegates to HF and fires hook_pattern only on
        the returned weights, after the output exists, so the rule could not apply."""
        with use_relevance_rules(opt_bridge, RelevanceRules(**{kind: True})):
            with pytest.raises(RelevanceRuleUnsupportedError, match=r"blocks\.0\.attn"):
                opt_bridge(TOKENS)
        # The refusal is specific to the attention kinds: the norms still install.
        with use_relevance_rules(opt_bridge, RelevanceRules(normalization=True)) as coverage:
            assert coverage.skipped == ()
            assert "ln_final" in coverage.installed
        opt_bridge(TOKENS)


class TestNativeSourceAttention:
    def test_ah_rule_applies_through_the_wrapped_attentions_pattern_seam(self) -> None:
        """The base AttentionBridge hands hook_pattern to a native-source attention,
        which applies it before its own pattern @ V product."""
        bridge = TransformerBridge.boot_native(
            TransformerBridgeConfig(
                d_model=32,
                d_head=8,
                n_heads=4,
                n_layers=N_LAYERS,
                n_ctx=16,
                d_vocab=97,
                d_mlp=64,
                act_fn="silu",
                normalization_type="LN",
                seed=0,
            )
        )
        attn = bridge.blocks[-1].attn
        assert _backward_grad_at(bridge, attn.hook_q, RelevanceRules()) is not None
        with torch.no_grad():
            baseline = bridge(TOKENS)
        with use_relevance_rules(bridge, RelevanceRules(attention=True)):
            with torch.no_grad():
                assert torch.equal(bridge(TOKENS), baseline)
        assert _backward_grad_at(bridge, attn.hook_q, RelevanceRules(attention=True)) is None
        value_grad = _backward_grad_at(bridge, attn.hook_v, RelevanceRules(attention=True))
        assert value_grad is not None and value_grad.abs().sum() > 0


class TestLayerNormPythonPath:
    def test_normalization_installs_on_every_norm_and_keeps_forward(
        self, layernorm_bridge: TransformerBridge
    ) -> None:
        assert not layernorm_bridge.blocks[0].ln1.use_native_layernorm_autograd
        expected = {"ln_final"}
        for index, block in enumerate(layernorm_bridge.blocks):
            # Parallel-attention CodeGen has a single ln1 feeding both sublayers.
            for mount in ("ln1", "ln2"):
                if mount in block._modules:
                    expected.add(f"blocks.{index}.{mount}")
        with torch.no_grad():
            baseline = layernorm_bridge(TOKENS)
        with use_relevance_rules(layernorm_bridge, RelevanceRules(normalization=True)) as coverage:
            with torch.no_grad():
                active = layernorm_bridge(TOKENS)
        assert set(coverage.installed) == expected
        assert coverage.skipped == ()
        assert torch.equal(active, baseline)

    def test_ln_rule_matches_local_oracle_with_centering_and_bias(
        self, layernorm_bridge: TransformerBridge
    ) -> None:
        """The gradient ln1 passes upstream equals a LayerNorm recompute (centering,
        weight, bias) whose division goes through ``ln_rule``, given the gradient
        ln1 actually received downstream."""
        ln1 = layernorm_bridge.blocks[0].ln1
        captured: dict[str, torch.Tensor] = {}

        def _capture(key: str):
            def _hook(tensor: torch.Tensor, hook: Optional[HookPoint] = None) -> None:
                captured[key] = tensor.detach().clone()

            return _hook

        ln1.hook_in.add_hook(_capture("x"))
        ln1.hook_in.add_hook(_capture("grad_in"), dir="bwd")
        ln1.hook_out.add_hook(_capture("grad_out"), dir="bwd")
        try:
            with use_relevance_rules(layernorm_bridge, RelevanceRules(normalization=True)):
                layernorm_bridge(TOKENS).sum().backward()
        finally:
            ln1.hook_in.remove_hooks(dir="both")
            ln1.hook_out.remove_hooks(dir="both")
            for parameter in layernorm_bridge.parameters():
                parameter.grad = None

        layer = ln1.original_component
        x = captured["x"].detach().requires_grad_(True)
        x_centered = x - x.mean(-1, keepdim=True)
        eps = getattr(ln1.config, "eps", 1e-5)
        denom = (x_centered.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        y = ln_rule(x_centered, denom) * layer.weight.detach()
        if getattr(layer, "bias", None) is not None:
            y = y + layer.bias.detach()
        (expected,) = torch.autograd.grad(y, x, grad_outputs=captured["grad_out"])
        torch.testing.assert_close(captured["grad_in"], expected, atol=1e-5, rtol=1e-4)
