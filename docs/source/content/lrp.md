# LRP Rules

Layer-wise relevance propagation (LRP) explains an output by distributing it over the inputs
and intermediate activations, so that the relevance at each layer sums to the explained value.
For transformers this is usually computed as a modified gradient times the input: the forward
pass stays the same, and the backward pass through some nonlinear operations is replaced.

`lrp_rules(model)` is a context manager that installs these replacements on a
`TransformerBridge`. Inside the block, every gradient that TransformerLens or PyTorch computes
through the model is an LRP gradient, so `activation * gradient` at a hook point is its
relevance (with the exceptions listed under [Relevance at hook points](#relevance-at-hook-points)).
Nothing else changes: the forward outputs are bit-identical, and the rules are removed when
the block exits.

Two attention rules are available: the AH-rule of conservative propagation (CP-LRP, Ali et
al., 2022), which holds the attention pattern constant, and AttnLRP's uniform rule on both
attention products (Achtibat et al., 2024). Products of two activations use either the
Half-rule or CP-LRP's constant gates; norms use the LN-rule and activations the Identity-rule.

## The rules

| Rule | Literature name | Where it acts | Backward pass |
|---|---|---|---|
| `"ln"` | LN-rule (Ali et al., 2022) | Every LayerNorm and RMSNorm: pre-, post-, final and q/k norms, with or without weights, `1 + w` offsets | The scale (standard deviation or RMS) is a constant. |
| `"ah"` | AH-rule (Ali et al., 2022) | Every attention layer | The attention pattern is a constant, so relevance flows only through the values; queries and keys get none. |
| `"attn_half"` | AttnLRP (Achtibat et al., 2024) | Every attention layer | The uniform rule on `QKᵀ` and on `pattern @ V`: the gradient into each product is halved, so queries and keys get 1/4 and values 1/2 of the ordinary gradient. The softmax keeps its plain gradient, which in gradient × input form is AttnLRP's softmax rule (Deep Taylor decomposition with bias). An attention-logit softcap (Gemma 2) between the two also keeps its plain gradient. |
| `"gate"` | Constant gates of CP-LRP (Ali et al., 2022), as LXT implements them for gated MLPs | The gating branch of `act(gate) * up` in gated MLPs; as this module's extension also MoE experts and routing weights, Qwen2-MoE's `sigmoid(gate)` on the shared expert and attention output gates with their own `gate` projection | No gradient enters the gating branch, so the other factor receives all of the product's relevance. Exclusive with `"half"`; with `"identity"`, the gated activations are left to this rule. Not in the default. |
| `"half"` | Half-rule, LRP-half (Arras et al., 2019); uniform rule (Achtibat et al., 2024) | `act(gate) * up` in gated MLPs, shared experts and MoE experts; the routing product `weight * expert_output`; Qwen2-MoE's `sigmoid(gate) * shared_expert` | The gradient into the product is halved, so each factor receives half of its relevance. |
| `"identity"` | Identity-rule (Rezaei Jafari et al., 2024) | MLP and MoE-expert activations (SiLU, GELU variants, ReLU and its variants, squared ReLU, Mish, hard swish, tanh, xIELU); the BERT pooler and MLM-head activations; the activation of a vision projector (Llava); the final-logit softcap `c * tanh(z / c)` (Gemma 2; an extension of this module, not prescribed by RelP or AttnLRP) | The multiplier `f'(x)` becomes `f(x) / x`, so relevance passes through unchanged. Parametric activations (PReLU, xIELU) lose their parameter gradients. |

`lrp_rules(model)` uses `["ln", "ah", "half", "identity"]`. On a dense model without biases
that is complete: the input relevance sums to the explained logit. An explicit list such as
`lrp_rules(model, ["ln", "ah"])` uses only those rules; `"ah"` excludes `"attn_half"`, and
`"half"` excludes `"gate"`; a rule listed twice counts once. Every component that contains a
requested rule's operation gets the rule or raises. A rule whose operation the model does not
have (`"half"` on GPT-2) changes nothing, as the plain gradient is already its LRP gradient,
and its site list is empty. The context manager yields the components each rule covers:

```python
from transformer_lens.tools.analysis import lrp_rules

with lrp_rules(model) as sites:
    print(sites["identity"])  # GPT-2: ['blocks.0.mlp', 'blocks.1.mlp', ...]; sites["half"] is []
```

Run both the forward and the backward pass inside the block. Forward-only work (`no_grad`,
`inference_mode`, `generate`, frozen models) runs unchanged inside it. The rules are
first-order: a double backward through them raises. They are tested in eager mode only, not
with `torch.compile` or gradient checkpointing. Entering `lrp_rules` again on a model where it
is already active raises, because the rules would stack.

## Presets

| Preset | `rules=` | Attention | Products |
|---|---|---|---|
| RelP with the AH-rule (default) | `None`, that is `["ln", "ah", "half", "identity"]` | Constant pattern | Half-rule |
| RelP (paper default) | `["ln", "identity", "half"]` | Plain gradient | Half-rule |
| AttnLRP (as in LXT) | `["ln", "attn_half", "half", "identity"]` | Uniform rule | Half-rule |
| CP-LRP (as in LXT) | `["ln", "ah", "gate", "identity"]` | Constant pattern | Constant gate |

The default and CP-LRP conserve relevance exactly on dense models without biases (CP-LRP also
on MoE models). Which preset explains best depends on the model:
[A Close Look at Decomposition-based XAI-Methods for Transformer Language Models](https://arxiv.org/abs/2502.15886)
(Table 2, mean reciprocal rank) puts AttnLRP ahead of CP-LRP on Llama-3.2-1B (0.814 vs 0.533)
and BERT-base (0.705 vs 0.637), and CP-LRP ahead on GPT-2 small (0.706 vs 0.634) and GPT-2 XL;
[MambaLRP](https://arxiv.org/abs/2406.07592) found the Half-rule more faithful than a constant
gate. So start with AttnLRP for rotary, gated decoders and with the default or CP-LRP for GPT-2.

## Token attribution

```python
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import lrp_rules

model = TransformerBridge.boot_transformers("gpt2", device="cpu")
tokens = model.to_tokens("The Eiffel Tower is in the city of")
paris = model.to_single_token(" Paris")

store = {}
keep = [("blocks.0.hook_in", lambda t, hook: store.update(resid=t))]
with lrp_rules(model), model.hooks(fwd_hooks=keep):
    logit = model(tokens)[0, -1, paris]
    (grad,) = torch.autograd.grad(logit, store["resid"])
token_relevance = (store["resid"] * grad).sum(-1)  # [batch, pos]
```

`blocks.0.hook_in` is the first residual stream, which includes the positional embedding on
GPT-2. Use `hook_embed` for the token embedding only.

## Relevance at hook points

{func}`~transformer_lens.tools.analysis.attribution_patching.cache_activation_and_gradient`
runs one forward and backward for any scalar metric and returns activations and gradients in
the hook points' canonical shapes (per head at `hook_z`, `hook_q`, ...):

```python
from transformer_lens.tools.analysis import lrp_rules
from transformer_lens.tools.analysis.attribution_patching import cache_activation_and_gradient

def metric(logits):
    return logits[0, -1, paris]

# Include a point at the input (blocks.0.hook_in), so the backward pass reaches every layer.
names = ["blocks.0.hook_in", "blocks.3.attn.hook_z", "blocks.3.mlp.hook_post", "blocks.3.hook_out"]
with lrp_rules(model):
    cache = cache_activation_and_gradient(model, tokens, metric, names_filter=names)
relevance = {name: cache.activations[name] * cache.gradients[name] for name in names}
head_relevance = relevance["blocks.3.attn.hook_z"].sum(-1)  # [batch, pos, head]
neuron_relevance = relevance["blocks.3.mlp.hook_post"]  # [batch, pos, d_mlp]
```

Per-head relevance at `hook_z` is exact while the output projection is linear and has no
bias. Other seeds are one `metric` away: a logit difference `logits[0, -1, a] - logits[0, -1, b]`
explains a contrast, and on GPT-2, whose logits are mostly negative, LXT recommends seeding the
target with +1 and every other token with -1/N. For a batch, return the sum of each example's
target logit: the examples stay independent, and padded positions get no relevance. Compute
relevance as `act.float() * grad.float()` in low precision. Differentiate a logit, not
`return_type="loss"`: the log-softmax inside the loss has no rule, so the relevance would not
sum to the loss.

The rule hooks run before user hooks on the same hook point, so a cached activation and its
cached gradient are consistent: the cached gradient is the LRP gradient with respect to the
cached tensor. User hooks may still edit activations, also in place. On a model without
biases, the relevance of the residual stream is the same at every layer (for example at
`blocks.3.hook_in` and `blocks.3.hook_out`). Exceptions to "activation × gradient is the
relevance":

- Under the AH-rule everything that feeds the attention logits gets zero gradient (`hook_q`,
  `hook_k`, `hook_rot_q`), while `hook_attn_scores` and `hook_pattern` report their own
  gradients where the bridge rebuilds attention. Under `"attn_half"`, `hook_attn_scores` and
  `hook_z` hold the full relevance of the two products and `hook_pattern`, `hook_v`, `hook_q`
  and `hook_k` the share of each factor. Masked attention scores are `-inf`, so mask them
  before multiplying: `torch.where(scores.isfinite(), scores * grad, 0)`.
- Where HF runs attention itself, see the support table.
- Under the LN-rule the gradient into `hook_scale` stops there.

To explain through a sparse autoencoder, splice it in at a hook point and keep its error as a
separate leaf, as in [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647):

```python
store = {}

def splice(act, hook):
    latents = sae.encode(act)
    reconstruction = sae.decode(latents)
    error = (act - reconstruction).detach().requires_grad_()
    store.update(latents=latents, error=error)
    return reconstruction + error

with lrp_rules(model), model.hooks(fwd_hooks=[("blocks.3.hook_out", splice)]):
    logit = model(tokens)[0, -1, paris]
    grads = torch.autograd.grad(logit, [store["latents"], store["error"]])
latent_relevance = store["latents"] * grads[0]
error_relevance = store["error"] * grads[1]
```

Latent and error relevance sum to the hook point's relevance minus the decoder bias's share
(`b_dec · grad`); upstream of the splice, relevance flows only through the latents.

JumpReLU and TopK latents need no rule: their gradient already equals the Identity-rule's.

## Relevance patching (RelP)

Relevance patching is attribution patching with LRP gradients instead of plain gradients. Run
{func}`~transformer_lens.tools.analysis.attribution_patching.attribution_patch` inside the block,
at node or edge granularity:

```python
from transformer_lens.tools.analysis import EdgeAttributionConfig, attribution_patch, lrp_rules

with lrp_rules(model):
    nodes = attribution_patch(model, clean, corrupt, metric_fn)
    edges = attribution_patch(
        model, clean, corrupt, metric_fn, EdgeAttributionConfig(granularity="edge")
    )
```

The edge sweep normalizes per-head copies of the residual stream through the raw HF `ln1`
module; the LN-rule covers that call too, so the edge scores of each writer add up to its node
score. Under the AH-rule, edges into queries and keys score exactly zero.

## Supported models and what raises

The rules need no recomputation of HF modules: they attach to hook points and HF modules that
the model already runs, so architectures built from the standard bridge components need no
extra code. Conservation is tested on GPT-2, GPT-J, GPT-NeoX, OPT, Llama, Gemma 2, Qwen3,
OLMo, Phi-3, BERT, RoBERTa, T5 and BART, and bit-identical forwards in raw and compatibility
mode, float32 and bfloat16. In bfloat16 LRP gradients are as close to float64 as plain ones
(about 1%). In float16 the LN-rule gradient can overflow where the plain backward can; use
bfloat16 or float32. The module docstring lists the assumptions about HF and the bridge that
the runtime checks cannot see.

| Situation | Behavior |
|---|---|
| Attention that the bridge rebuilds (most decoders) | AH-rule as a zero gradient below `hook_attn_scores`; AttnLRP as a halved gradient below `hook_attn_scores` and below `hook_z` (the `o` projection input). |
| Attention that runs inside HF (GPT-J, OPT, GPT-Neo, T5, BART, ...) | Gradient scales at the q, k (and v) projection outputs: 0 for the AH-rule; 1/4, 1/4 and 1/2 for AttnLRP. Queries and keys feed only the attention logits and values only `pattern @ V`, so the input relevance is the same, but q/k norms and other HF-internal points after the projections report unscaled gradients. `NotImplementedError` without separate q, k and v projections (fused QKV, keys equal to values, keys from another layer). |
| Compatibility mode with processed weights, gated MLP | The bridge calls the activation as a function, so the Identity-rule raises `RuntimeError` on the first forward. Use `enable_compatibility_mode(no_processing=True)`, a raw bridge, or `rules=["ln", "ah", "half"]`. |
| BERT, RoBERTa; Llava-style vision-language models | The MLM head's activation (a module in BERT, a function in RoBERTa) gets the Identity-rule between `mlm_head` and `ln_final`; the vision tower gets the rules like the language model, and the projector's activation the Identity-rule. |
| MoE blocks with the standard HF experts module (Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE, DeepSeek V2/V3, Granite-MoE, Ernie 4.5 MoE, ...), in any experts implementation | Identity-rule on the experts' activation. Half-rule: halved gradient at the experts' output (routing product) and input (`act(gate) * up`, both factors linear in the input), and on both factors of Qwen2-MoE's shared-expert gate. Constant gates: zero gradient into the routing weights, the expert gates and the shared-expert gate. Shared experts get the gated-MLP rules; the router keeps its plain gradient. Input and hook-point gradients equal LXT's per-expert placement; expert parameter gradients (`gate_up_proj`) do not. |
| Other MoE implementations (GPT-OSS's clamped SwiGLU, Llama 4, JetMoE, ...) | `NotImplementedError` for the Half-, gate and Identity-rule; `rules=["ln", "ah"]` or `["ln", "attn_half"]` still works. |
| MoE blocks whose adapter leaves a gated MLP unbridged (GLM-4-MoE shared experts; the GLM-4-MoE-Lite and GLM-MoE-DSA adapters do not bridge them either) | `RuntimeError` on the forward for the Half-, gate and Identity-rule. |
| Gated attention outputs | With their own `gate` projection (HRM-Text): `"gate"` holds the gate constant. Inside `q_proj` (`gated_q_proj`; Qwen3-Next, Qwen3.5, which are hybrids anyway): `"gate"` raises. `"half"` and `"attn_half"` raise for both. |
| KV-cache decoding (`generate`, `past_key_values`) | Supported. Self-attention projects K and V of every new token. HF-run cross-attention (T5) may skip its k and v projections only when the same cache object got K and V through the rule earlier in the block; otherwise it raises. A cache filled outside the block keeps ordinary gradients for the cached positions. |
| SSM, linear-attention, RWKV and gated-convolution mixers | Hybrid stacks (Jamba, Falcon-H1, Bamba, LFM2, Qwen3-Next): `NotImplementedError` for the attention rules, `"half"` and `"gate"`; the LN- and Identity-rule alone still work. Pure SSM models (Mamba, Mamba-2, opaque blocks): every rule set raises. |
| Final-logit softcap (Gemma 2, `output_logits_soft_cap`) | Identity-rule, applied to the logits that the HF model returns. A loss that HF computes inside the model (`original_model(..., labels=...)`) or a tuple output (`return_dict=False`) raises `RuntimeError`; compute the loss from the logits, as the bridge's `return_type="loss"` does. |
| Gated norms; opaque and AltUp blocks | `NotImplementedError` on entry (gated norms: for the LN-, Half- and gate rule). |
| Native-source bridges (`TransformerBridge.boot_native`) | Both attention rules, through the `pattern_fn` seam. The MLP calls its activation as a function, so the Identity-rule raises `NotImplementedError` on entry; use `rules=["ln", "ah"]` or `["ln", "attn_half"]`. |
| Activations not known to be elementwise with `f(0) = 0` (sigmoid, softplus, GLU, unknown classes), or in-place activations | `NotImplementedError` for the Identity-rule. |
| A LayerNorm or `*RMSNorm` that the adapter wraps as a plain component (DeepSeek `q_a_layernorm`) | LN-rule on the HF module. |
| Any other norm or activation outside every covered component runs | `RuntimeError` on the forward, naming the rules to drop. |
| A forward hook edits `hook_scale` or `hook_normalized` of a native-path norm | `RuntimeError`: the bridge then rebuilds the norm output and the LN-rule does not apply. |

Each rule checks on every forward that needs gradients that it ran where it applies, so a
bypass raises instead of giving ordinary gradients. Run the model and all submodules in
evaluation mode; `lrp_rules` raises `LRPConfigError` (a `ValueError`) on entry otherwise.

### Errors and warnings

Every error is an `LRPError` and also the builtin named above, so existing `except ValueError`
code keeps working. Each message names the component and the rules to drop.

| Class | Builtin | Raised for |
|---|---|---|
| `LRPConfigError` | `ValueError` | An empty or unknown rule, a string instead of a list (`rules="ln"`; pass `["ln"]`), both rules of a conflicting pair (`"ah"` and `"attn_half"`, `"half"` and `"gate"`), or a module in training mode. On entry. |
| `LRPUnsupportedError` | `NotImplementedError` | A requested rule meets a component it cannot cover. On entry. |
| `LRPRuntimeError` | `RuntimeError` | A forward that needs gradients bypassed a rule, or `lrp_rules` is already active on the model. |

`LRPWarning` (a `UserWarning`) flags a rule of a custom set that finds nothing to act on
(`["ln", "half"]` on GPT-2) and the LN-rule on float16 models. The presets above never warn
about a missing operation (CP-LRP's `"gate"` on GPT-2).

## Conservation and interpretation

Conservation (input relevance summing to the explained output) holds exactly only when nothing
else absorbs relevance:

- Biases, including LayerNorm biases, absorb relevance. Zero them to check conservation.
- Under the Half-rule the MoE router receives relevance through the routing product and passes
  it on with its plain gradient, as in AttnLRP, so only CP-LRP conserves relevance on MoE.
- AttnLRP's softmax rule is not conservative (it has a bias term), and the attention-logit
  softcap (Gemma 2) sits on that path; under the AH-rule both get zero gradient. The RelP
  preset keeps the plain attention gradient. So on dense models only the default and CP-LRP
  conserve relevance exactly.
- HF RMSNorm computes in float32 even in a float64 model, so conservation holds there to about
  `1e-7` relative.
- For ReLU, the Identity-rule gives the ordinary gradient (`relu(x) / x = relu'(x)`). Where an
  activation input is exactly 0, the Identity-rule passes no gradient upstream.
- Only computations that run inside the block use the rules. A KV cache filled outside the
  block keeps its ordinary gradients, and other forwards on the same model that run at the same
  time (other threads) also see the rules.

LRP relevance is an attribution, not a causal effect: check important components with real
interventions before you make a causal claim.

## Cost

Without the context manager nothing changes: no code runs on any forward or backward path.
Entering and leaving it walks the model once (about 1 ms for a small model). Inside it, each
layer runs about 17 small Python hooks and a few elementwise autograd functions, so the
relative overhead is larger on small models: on CPU (random 8-layer Llama, batch 4 × 128),
forward plus backward took 4% longer at `d_model` 1024 and 8–21% longer at `d_model` 512,
for every preset; a Qwen3-MoE model with 8 experts took 8–10% longer. The extra memory is
about one activation-sized tensor per norm (its output, kept for the correction) and short-lived
copies at the other sites; the Identity-rule's saved ratio `f(x) / x` replaces the
activation's own saved input.

API: {func}`~transformer_lens.tools.analysis.lrp.lrp_rules`.

## References

- Ali et al. (2022).
  [XAI for Transformers: Better Explanations through Conservative Propagation](https://arxiv.org/abs/2202.07304).
  ICML. The LN-rule, the AH-rule and constant gates (CP-LRP).
- Arras et al. (2025).
  [A Close Look at Decomposition-based XAI-Methods for Transformer Language Models](https://arxiv.org/abs/2502.15886).
  Comparison of AttnLRP and CP-LRP across models.
- Arras, Osman, Müller and Samek (2019).
  [Evaluating Recurrent Neural Network Explanations](https://aclanthology.org/W19-4813/).
  BlackboxNLP. The Half-rule (LRP-half) for multiplicative gates.
- Achtibat et al. (2024).
  [AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers](https://arxiv.org/abs/2402.05602).
  ICML. The uniform rule for products of activations, and the softmax rule in gradient × input
  form.
- Rezaei Jafari et al. (2024).
  [MambaLRP: Explaining Selective State Space Sequence Models](https://arxiv.org/abs/2406.07592).
  NeurIPS. The Identity-rule for elementwise activations.
- Rezaei Jafari, Eberle, Khakzar and Nanda (2025).
  [RelP: Faithful and Efficient Circuit Discovery in Language Models via Relevance Patching](https://arxiv.org/abs/2508.21258).
  NeurIPS 2025 Mechanistic Interpretability Workshop.
  [Reference code](https://github.com/FarnoushRJ/RelP).
