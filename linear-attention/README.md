# Linear Attention - a toy-scale build of the ungated original

A minimal implementation of **Linear Attention**, from Katharopoulos, Vyas,
Pappas, Fleuret, *"Transformers are RNNs: Fast Autoregressive Transformers
with Linear Attention"* (2020) — the true root of the entire
gated-linear-attention family in this repo (GLA, RetNet, DeltaNet, xLSTM,
and KDA all build on top of this exact idea).

This README explains the idea in plain words. `linear_attention_mini.ipynb`
builds it, trains it on a toy dataset, and verifies it learns.

---

## 1. The idea

Softmax attention computes, for every query, a weighted average of all
values, where the weights come from `softmax(q · k)`. The softmax is what
makes this expensive: it doesn't factor into separate query-only and
key-only pieces, so there's no way to avoid comparing every query against
every key.

Linear attention's core trick: **replace the softmax with a positive kernel
feature map** `phi`, and drop the exponential entirely:

```
attention_weight(q_t, k_s) ~ phi(q_t) . phi(k_s)     (instead of exp(q_t . k_s))
```

Because this new "attention weight" factors into a `phi(q_t)` piece and a
`phi(k_s)` piece multiplied together, the whole computation can be
rearranged: instead of comparing the query against every key one at a time,
you can accumulate `phi(k_s) (x) v_s` into a running sum as you go, and
read it out with `phi(q_t)` at the end. That's exactly the "state that gets
updated one token at a time" pattern used by every other architecture in
this repo — this is where the pattern originates.

```
S_t = S_{t-1} + phi(k_t) (x) v_t         # just accumulate -- no decay, no erase, nothing
Z_t = Z_{t-1} + phi(k_t)                  # running normalizer
o_t = (phi(q_t)^T S_t) / (phi(q_t)^T Z_t)   # normalized readout
```

This notebook uses `phi(x) = elu(x) + 1` (the paper's feature map), which
guarantees every "attention weight" stays non-negative, mimicking one
property of the softmax without needing the softmax itself.

## 2. Why every other notebook in this repo adds something on top

Notice there's no decay, no erase step, no gating of any kind — the state
just accumulates every key-value pair it's ever seen, forever, at equal
weight. This works, but has a real cost: with nothing to forget, the state
gets muddier as the sequence gets longer, since old and new associations
for similar keys all get blended together indiscriminately by the `+=`.

This turns out to be the central limitation the rest of this repo's
recurrent-state architectures are built to fix:

- **GLA** (`../gla`) adds a per-channel forget gate, so old information can
  fade.
- **DeltaNet** (`../deltanet`) adds the delta rule, so writing a new value
  for a key first erases the old value for that same key, rather than
  blending them.
- **KDA** (`../kda`), **RetNet** (`../retnet`), and **xLSTM** (`../xlstm`)
  each combine decay and/or the delta rule in their own way.

Reading this notebook first, then any of those, should make it much clearer
exactly what problem each one is solving.

## 3. What this notebook simplifies

Not much — this is already close to the simplest version of the mechanism.
The one choice worth noting is the feature map itself: `elu(x)+1` is the
paper's original choice, but plenty of later work (including some cited by
the GLA and DeltaNet papers) experiments with other feature maps, including
learned ones.

## 4. Running it

Open `linear_attention_mini.ipynb` in Colab, run all cells top to bottom.
You'll see the model built, sanity-checked, trained for 300 steps on a
repeating `"0123456789ABCDEF"` string, then used to generate. Because this
toy task is short and highly repetitive, plain accumulation (with no
forgetting at all) is still enough for it to learn cleanly — a longer,
less repetitive task is where the lack of a forget gate would start to
show.

## 5. Reference

Katharopoulos, Vyas, Pappas, Fleuret, *"Transformers are RNNs: Fast
Autoregressive Transformers with Linear Attention,"* 2020.
