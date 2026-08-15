# Multi-head Latent Attention (MLA) - a toy-scale build

A minimal implementation of **Multi-head Latent Attention**, the attention
variant introduced in DeepSeek-V2 (DeepSeek-AI, 2024) and carried forward
into DeepSeek-V3 — designed to shrink the KV cache dramatically without
changing what the attention mechanism actually computes.

This README explains the idea in plain words. `mla_mini.ipynb` builds it,
trains it on a toy dataset, and verifies it learns.

*(This is also the mechanism Kimi K3's "Gated MLA" layer is named after and
simplifies away — see `../kda` for that architecture, which drops the
latent compression entirely since it's a serving optimization rather than a
change to what the layer computes. This notebook builds the compression
itself, since that's the interesting part.)*

---

## 1. The problem MLA solves

During autoregressive generation, a transformer caches every past token's
keys and values so it never has to recompute them — the "KV cache." The
catch: the cache's size grows with `n_heads * d_head` per token, and for
models with many heads and long context, it can become the dominant memory
cost of serving the model — often bigger than the model's own weights.

**MLA's idea:** instead of caching a full-size key and value per head,
compress *all* heads' keys and values down into one small shared **latent
vector** per token, and cache *that* instead. When you actually need
per-head keys and values for attention, up-project from the latent on the
fly.

```
c_t = W_down x_t                # one small latent vector per token -- THIS gets cached
k_t^(h) = W_up_k^(h) c_t          # reconstructed per-head key, computed on demand
v_t^(h) = W_up_v^(h) c_t          # reconstructed per-head value, computed on demand
```

If the latent is much smaller than all heads' keys and values combined, the
cache shrinks by roughly that ratio — while the attention computation
itself, once the up-projection happens, is completely ordinary multi-head
attention. **MLA doesn't change what attention computes — it changes what
gets stored between tokens.**

## 2. The one wrinkle: positional encoding

Most modern transformers bake **RoPE** (rotary position embeddings)
directly into keys and queries. But RoPE only works cleanly if the key
you're rotating is stable and specific to a token — and MLA's whole point is
that the *cached* thing (the latent `c_t`) is shared across heads and gets a
fresh up-projection every time it's used, which doesn't play well with
rotating it once and caching the rotated version.

MLA's fix: split off a **small, separate "rope" head** that carries
positional information directly — computed from the raw token, RoPE
applied, *not* routed through the compressed latent — and concatenate it
onto each head's content key before computing attention scores. So each
head's effective key is `[reconstructed_content_key ; shared_rope_key]`:
content comes from the compressed latent, position comes from an
uncompressed side channel.

## 3. What this notebook simplifies

In production, the up-projection matrices are often algebraically folded
into the query/output projections so full-size keys and values are never
materialized at all — an "absorption" trick purely for serving efficiency.
This notebook materializes them explicitly with `k_up`/`v_up` instead, since
it computes the exact same attention output and is much easier to follow.

## 4. Running it

Open `mla_mini.ipynb` in Colab, run all cells top to bottom. You'll see the
model built, sanity-checked, trained for 300 steps on a repeating
`"0123456789ABCDEF"` string, then used to generate — a correctly-trained toy
model should generate visibly periodic output.

## 5. Reference

DeepSeek-AI, *"DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model,"* 2024.
