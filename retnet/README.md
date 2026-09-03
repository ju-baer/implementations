# RetNet - a toy-scale build of Retentive Networks

A minimal implementation of **Retention**, from Sun et al., *"[Retentive
Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)"* (2023).

This README explains the idea in plain words. `retnet_mini.ipynb` builds it,
trains it on a toy dataset, and verifies it learns.

---

## 1. The idea

RetNet is another linear-attention-with-a-state layer, in the same family as
GLA and KDA (see `../gla` and `../kda`), a fixed-size state matrix, updated
one token at a time, read out with a query.

Where it differs: the decay applied to the state at every step is **not
predicted from the input at all**. It's a **fixed constant**, chosen before
training and never changed, one constant per attention head:

```
S_t = gamma_h * S_{t-1} + k_t (x) v_t     # gamma_h is fixed, not learned per-token
o_t = q_t^T S_t
```

No gate network computes `gamma`; it's just a number baked into the model
per head, e.g. `gamma_h = 1 - 2^(-5-h)`. Different heads get different
`gamma` values: some decay fast (short memory, good at local patterns), some
decay very slowly (long memory, good at long-range dependencies). This is
**multi-scale retention**, instead of one learned, input-dependent decay
rate (like GLA), RetNet spreads a fixed range of decay rates across its
heads and lets each head specialize by position in that range.

## 2. Why fix the decay instead of learning it?

A fixed decay makes the recurrence easy to reformulate in three equivalent
forms:

- a **parallel (matrix) form**, efficient for training - similar to softmax
  attention's parallelism, but without the softmax,
- a **recurrent form**, for `O(1)`-memory, `O(1)`-per-token inference,
- a **chunked form**, mixing both - parallel within a chunk, recurrent
  across chunks - for long sequences.

All three compute the *exact same result*, and this only works cleanly
because the decay factors are fixed in advance rather than depending on the
data. That's RetNet's pitch: transformer-like training efficiency, RNN-like
constant-cost inference.

## 3. What this notebook simplifies

The sequential recurrence is used directly instead of RetNet's parallel or
chunked forms, same math, much easier to read, much slower. There's no
custom kernel or chunking logic here, just the plain token-by-token update.

## 4. Running it

Open `retnet_mini.ipynb` in Colab, run all cells top to bottom. You'll see
the model built, sanity-checked, trained for 300 steps on a repeating
`"0123456789ABCDEF"` string, then used to generate. A correctly-trained toy
model should generate visibly periodic output.

## 5. Reference

Sun, Dong, Huang, Ma, Xia, Xue, Wang, Wei, *"[Retentive Network: A Successor
to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621),"* 2023.
