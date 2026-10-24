# Layer-wise relevance propagation (LRP)

LRP attributes a model's output back to its input tokens. `TransformerBridge` can apply
LRP rules to its components through TransformerLens forward hooks, so that a standard
gradient-times-input (GxI) backward pass computes LRP relevance scores instead of ordinary
gradients. Nothing about the model's forward pass changes while LRP is off, and the hooks
coexist with `run_with_cache`, `run_with_hooks` and your own hooks.

## Quick start

```python
from transformer_lens.model_bridge import TransformerBridge

# 1. Load a model with raw HuggingFace weights
model = TransformerBridge.boot_transformers("Qwen/Qwen3-0.6B")

# 2. Install the LRP hooks once; the returned table says which rules apply where
coverage = model.patch_lrp()

# 3. Enable LRP and choose which rules to apply
model.lrp_on()
model.set_lrp_rules("identity-rule-act", "half-rule-mlp", "ah-rule", "ln-rule")

# 4. Run forward + backward
tokens = model.to_tokens("Hello, world")
loss, cache = model.run_with_cache(tokens, return_type="loss", incl_bwd=True)

# Relevance at any hook point = activation × gradient
relevance = cache["blocks.11.hook_out"] * cache["blocks.11.hook_out_grad"]
```

The scalar whose gradient is taken selects what is explained: the loss above, or a logit
difference obtained by running `model(tokens)` and calling `.backward()` on it yourself.

## How it works

`model.patch_lrp()` attaches one permanent forward hook per rule-relevant hook point of the
bridge (norm scales, attention pattern, value projection, gated-MLP product) and wraps each
gated MLP's activation module for `identity-rule-act`. While LRP is off, or a rule is not
selected, the hooks return nothing and the model runs on its native path. While a rule is
active, the hook returns a tensor with the same value but a modified autograd graph (a
detached factor, a halved gradient), so backpropagation through the model yields LRP
scores rather than standard gradients.

Because everything is a TransformerLens hook, `reset_hooks()`, `run_with_cache` and
`run_with_hooks` all work as usual and leave the LRP hooks in place; only
`reset_hooks(including_permanent=True)` or `unpatch_lrp()` removes them. `model.lrp_off()`
runs the hooked model with standard gradients, which is useful for sanity-checking that
hooking has not altered the forward pass.

## API

| Method | Description |
|---|---|
| `model.patch_lrp()` | Install the LRP hooks on every supported hook point (once after loading); returns the coverage table |
| `model.unpatch_lrp()` | Remove the LRP hooks again and restore the wrapped activations |
| `model.lrp_on()` | Enable LRP |
| `model.lrp_off()` | Disable LRP (standard GxI behaviour) |
| `model.set_lrp_rules(*rules, strict=True)` | Select rules by name; raises if a rule cannot be applied correctly on this model (`strict=False` warns and applies it where supported) |
| `model.get_lrp_rules()` | The currently selected rule names |
| `model.get_lrp_coverage()` | The per-rule, per-block support table built by `patch_lrp` (print it) |
| `model.is_lrp_patched()` | Whether the hooks are installed |

Rule names are the values of `transformer_lens.model_bridge.LRPRules`; members of the enum
are accepted too.

## Available rules

| Rule | Hook point(s) per block `i` | Effect | Reference |
|---|---|---|---|
| `identity-rule-act` | the HuggingFace MLP's `act_fn` module (wrapped, no hook point) | Treats the gate activation as an identity transformation | [1] |
| `half-rule-mlp` | `mlp.out.hook_in` | Splits the relevance equally during the gate × up multiplication | [2, 3] |
| `ah-rule` | `attn.hook_pattern` | Fully detaches the attention pattern before value aggregation | [4] |
| `half-rule-attn` | `attn.hook_pattern`, `attn.v.hook_out` | Splits the relevance equally during multiplication of attention weights and value embeddings | [2, 3] |
| `ln-rule` | `ln1.hook_scale`, `ln2.hook_scale`, `ln_final.hook_scale` | Detaches the scale factor (`x / scale.detach()`) of the block norms, treating normalisation as a linear rescaling | [4] |
| `ln-rule-post` | `ln1_post.hook_scale`, `ln2_post.hook_scale` | Same for the post-sublayer norms of Gemma 2/3 and similar architectures | [4] |
| `ln-rule-qk` | `attn.q_norm.hook_scale`, `attn.k_norm.hook_scale` | Same for the attention query/key norms | [4] |

`ah-rule` and `half-rule-attn` both act on the attention pattern; when both are selected
the pattern is detached and only the value gradient is halved.

## Coverage: which rules a model supports

`patch_lrp()` surveys the model block by block and records, for every rule, whether it
could be attached (`applied`), attached only partly (`partial`), not attached although the
component exists (`unsupported`), or has nothing to act on (`absent`). Print the table to
see what a model supports:

```python
print(model.get_lrp_coverage())
# identity-rule-act  applied 28
# half-rule-mlp      applied 28
# ah-rule            applied 28
# ...
# ln-rule-post       absent 28   absent on blocks.0-27: no post-sublayer norm
# (.detail() lists every component)
```

`set_lrp_rules` uses the same table: a rule that would be applied incorrectly somewhere
(`partial` or `unsupported`) raises, and `strict=False` downgrades that to a warning and
applies the rule where it is supported. A rule that applies to nothing only warns, so rule
lists can be shared across models.

### Bridge module support

`patch_lrp` decides per block from the bridge classes TransformerLens wrapped the model
with. The tables list every class the survey knows; anything else is reported
`unsupported`. "Tested" names the tiny random-weight models in the test suite that
exercise the row.

**Attention bridges** (`ah-rule`, `half-rule-attn`)

| Bridge class | Typical models | `ah-rule` | `half-rule-attn` | Tested |
|---|---|---|---|---|
| `PositionEmbeddingsAttentionBridge` | Llama, Qwen, Gemma, Mistral, Phi-3, OLMo 2 (rotary) | yes | yes | Qwen3, Gemma 2, Qwen3-MoE |
| `JointQKVAttentionBridge`, `JointQKVPositionEmbeddingsAttentionBridge`, `ALiBiJointQKVAttentionBridge` | GPT-2, GPT-NeoX / Pythia, StableLM, fused-QKV models | yes | yes, when the bridge exposes `attn.v.hook_out` | GPT-2 |
| `BloomAttentionBridge`, `CodeGenAttentionBridge`, `MPTALiBiAttentionBridge`, `GlmMoeDsaAttentionBridge` | Bloom, CodeGen, MPT, GLM-MoE-DSA | yes | yes, when the bridge exposes `attn.v.hook_out` | no |
| `MLAAttentionBridge` | DeepSeek V2 / V3 / R1 | yes | no (compressed KV has no value projection; reported `partial` and refused) | no |
| `AttentionBridge` (base) and every other class | T5, OPT, GPT-J, GPT-Neo, XGLM, BERT, ViT, Gemma 3n / 4, Llama 4, T5Gemma 2 | no | no | no |

The base class fires `hook_pattern` on the weights HuggingFace returns after the output
is computed, so an edit there is a silent no-op; only the listed classes recompute
attention in Python with the hook inside.

**MLP bridges** (`half-rule-mlp`, `identity-rule-act`)

| Bridge class | Typical models | `half-rule-mlp` | `identity-rule-act` | Tested |
|---|---|---|---|---|
| `GatedMLPBridge` | Llama, Qwen, Gemma, Mistral | yes | yes (wraps the HuggingFace `act_fn` module) | Qwen3, Gemma 2 |
| `JointGateUpMLPBridge` | Phi-3, GLM, OpenELM | yes, when the bridge exposes `mlp.out.hook_in` | no (the bridge runs a captured activation) | no |
| `MLPBridge`, `BloomMLPBridge` (ungated) | GPT-2, Pythia, OPT, BERT, Phi-1 / 2, Bloom | not applicable (no gate product; reported `absent`) | no | GPT-2 |
| `MoEBridge` | Mixtral, Qwen3-MoE, GPT-OSS, DeepSeek, Llama 4 | no (experts are fused inside the HuggingFace module) | no | Qwen3-MoE |

**Normalization bridges** (`ln-rule`, `ln-rule-post`, `ln-rule-qk`)

| Bridge class | Typical models | norm rules | Tested |
|---|---|---|---|
| `RMSNormalizationBridge`, `NormalizationBridge` (LayerNorm), `RMSNormPreBridge`, `LayerNormPreBridge` | all transformer families | yes | Qwen3, Gemma 2, GPT-2 |
| `GatedRMSNormBridge` | Mamba-2 family inner norm (NemotronH, Falcon-H1, Bamba, Zamba2) | no (no `hook_scale`) | no |

Blocks without attention or MLP components (Mamba and gated-delta-net layers in hybrid
models) are reported `absent` and left alone. The rules do not reach vision or audio
towers of multimodal models.

## Compatibility mode

Models are loaded with raw HuggingFace weights. `enable_compatibility_mode()` folds norm
weights and centers writing weights, which changes the coordinate system relevances are
expressed in; leave it off unless you need HookedTransformer-equivalent numerics. With
weight processing enabled the bridge also bypasses the wrapped activation module, so
`identity-rule-act` is refused in that mode (`enable_compatibility_mode(no_processing=True)`
is fine). Enable compatibility mode before selecting rules: if it is enabled afterwards,
`lrp_on()` warns that the selected identity rule no longer applies, and `set_lrp_rules`
re-validates.

## Numerics and caveats

- `identity-rule-act` stabilizes the activation input away from zero, which perturbs the
  forward pass by less than about `1e-4`. All other rules keep the forward value exactly.
- The norm rules edit `hook_scale`. On bridges that run the HuggingFace norm kernel
  natively (`RMSNormalizationBridge`, i.e. most RMSNorm models), the bridge then rebuilds
  the norm output from the edited scale, so logits differ from the unhooked model at
  float-rounding scale and the bridge's edited-scale warning is silenced while LRP is on.
  Your own `hook_scale` edits warn as usual once LRP is off.
- Under `ah-rule` no gradient reaches `hook_attn_scores`, so backward hooks or
  `incl_bwd` caches there see nothing, and `W_Q`/`W_K` receive no gradient.
- Do not combine these hooks with the LN rule of the autograd-based
  `use_relevance_rules` backend on the same model: that backend refuses edited
  `hook_scale` values while its rule is active.
- `patch_lrp` installs permanent hooks; a norm with a forward hook attached computes its
  hook-side statistics on every call even while LRP is off. The output is unchanged.

Rule lists written for the 2.x `lrp_for_tl` package map as follows: `ln-rule-pre` is now
`ln-rule`, and the old `ln-rule` is `ln-rule-post` plus `ln-rule-qk`.

## References

The implementation follows [RelP](https://github.com/FarnoushRJ/RelP), expressed as hooks
on the bridge instead of a replacement model.

[1] Rezaei Jafari et al., *MambaLRP: Explaining Selective State Space Sequence Models*, NeurIPS 2024. [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/d6d0e41e0b1ed38c76d13c9e417a8f1f-Paper-Conference.pdf)

[2] Arras et al., *Explaining and Interpreting LSTMs*, in Explainable AI: Interpreting, Explaining and Visualizing Deep Learning, Springer, 2022. [DOI](https://doi.org/10.1007/978-3-030-28954-6_11)

[3] Achtibat et al., *AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers*, ICML 2024. [Paper](https://proceedings.mlr.press/v235/achtibat24a.html)

[4] Ali et al., *XAI for Transformers: Better Explanations through Conservative Propagation*, ICML 2022. [Paper](https://proceedings.mlr.press/v162/ali22a.html)
