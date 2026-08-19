# Grouped-Query Attention (GQA) - a toy-scale build

A minimal implementation of **Grouped-Query Attention**, from Ainslie,
Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai, *"GQA: Training
Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"*
(2023) — the simple, widely-deployed alternative to MLA (`../mla`) for
shrinking the KV cache.

This README explains the idea in plain words. `gqa_mini.ipynb` builds it,
trains it on a toy dataset, and verifies it learns.

---

## 1. The problem, again

Same motivation as MLA (`../mla`): the KV cache — everything a transformer
stores between tokens during generation — grows with `n_heads * d_head` per
token, and can dominate the memory cost of serving a model. MLA solves this
by compressing all heads' K/V into one small shared latent vector. GQA
takes a much simpler route.

## 2. The idea: fewer key/value heads than query heads

Ordinary **multi-head attention (MHA)** gives every query head its own
private key head and value head — `n_heads` of each. **Multi-query
attention (MQA)**, an earlier and more aggressive idea, goes to the other
extreme: every query head shares the *exact same single* key head and
value head. MQA shrinks the cache a lot, but can hurt quality — one shared
K/V pair has to serve every query head's needs at once.

**GQA sits in between.** Split the query heads into a small number of
*groups*; every head within a group shares one key head and one value head,
but different groups get different K/V heads:

```
n_kv_heads = n_heads // group_size
head h's key/value = kv_head[h // group_size]
```

- `n_kv_heads == n_heads` → this is just ordinary MHA (every group has size 1).
- `n_kv_heads == 1` → this is MQA (one giant group, everyone shares).
- Anything in between → GQA, trading off cache size against quality.

The KV cache only needs to store `n_kv_heads` sets of keys and values, so
picking `n_kv_heads` well below `n_heads` shrinks the cache by roughly that
ratio, at a much smaller implementation cost than MLA's latent compression.

## 3. Where this fits in the "efficient attention" progression

This repo now has a small ladder of ideas for making attention's KV cache
cheaper:

```
MHA  ->  GQA  ->  MQA  ->  MLA
```

MHA is the baseline (this notebook, with `n_kv_heads = n_heads`). GQA and
MQA (same notebook, fewer KV heads) get there by literally **sharing** K/V
heads across groups of query heads. MLA (`../mla`) takes a different, more
involved approach — instead of sharing raw K/V heads, it **compresses** all
heads' content into one small latent vector and reconstructs per-head
keys/values from it on demand. Worth building both and comparing: GQA is a
few lines of code; MLA needs a whole compression + decoupled-RoPE scheme.

## 4. What this notebook simplifies

Not much — GQA is already about as simple as MHA itself. The one thing to
notice in the code is `repeat_interleave`, which is what actually
implements "every head in a group shares the same K/V head," by
duplicating each KV head across its group before the attention computation.

## 5. Running it

Open `gqa_mini.ipynb` in Colab, run all cells top to bottom. The notebook
first checks MHA, GQA, and MQA configurations all run correctly, then
trains a tiny LM with real GQA (`n_kv_heads=2` out of 4 query heads) for
300 steps on a repeating `"0123456789ABCDEF"` string, then generates.

## 6. Reference

Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai, *"GQA: Training
Generalized Multi-Query Transformer Models from Multi-Head Checkpoints,"*
2023.
