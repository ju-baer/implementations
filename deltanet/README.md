# DeltaNet — a toy-scale build of the pure delta rule

A minimal implementation of **DeltaNet**, from Yang, Wang, Yu, Kim,
*"Parallelizing Linear Transformers with the Delta Rule over Sequence
Length"* (2024) — the direct architectural ancestor of KDA (see `../kda`),
isolating the **delta rule** on its own, with no decay gate at all.

This README explains the idea in plain words. `deltanet_mini.ipynb` builds
it, trains it on a toy dataset, and verifies it learns.

---

## 1. The idea

This repo's `gla/` folder shows what happens when you add a **decay gate**
to plain linear attention: the state fades over time, controlled by the
input. This folder shows the *other* ingredient KDA combines with decay:
the **delta rule** — on its own, with the decay gate removed entirely.

Plain linear attention (`../linear-attention`) only ever *adds* new
key-value pairs to its state, so if the same key shows up twice with two
different values, the state ends up holding a blend of both, forever. The
delta rule fixes this differently than a decay gate does: at every step, it
explicitly **removes whatever was previously stored for this specific
key**, before writing the new value in.

```
kv_old = k_t^T S_{t-1}                                              # what the state currently says about this key
S_t = S_{t-1} - beta_t * k_t (x) kv_old + beta_t * k_t (x) v_t     # erase the old association, write the new one
o_t = q_t^T S_t
```

Note there's no `alpha_t * S_{t-1}` decay term anywhere — the state is only
ever modified at the exact key being written to, not uniformly faded
everywhere. It's called the "delta rule" because the update is proportional
to the *difference* (delta) between what the state currently says about
this key and what it should say; `beta_t` controls how much of that
difference to correct on this step — a learning-rate-like quantity, not a
decay.

## 2. How this relates to KDA

KDA (`../kda`) takes exactly this delta-rule update and adds a channel-wise
decay gate on top (the `alpha_t * S_{t-1}` term this folder omits).
DeltaNet is what's left with that decay removed — a good place to see the
delta rule in isolation, and to feel out what the decay gate actually buys
you by comparing training on the same toy task.

## 3. What this notebook simplifies

The paper's main contribution is a **chunkwise-parallel algorithm** for
computing this update efficiently on a GPU — delta-rule updates are
trickier to parallelize than plain decay, since each step's erase operation
depends on the exact state produced by the previous step. This notebook
uses the plain sequential recurrence instead — a Python loop over
timesteps — same math, much easier to read.

## 4. Running it

Open `deltanet_mini.ipynb` in Colab, run all cells top to bottom. You'll
see the model built, sanity-checked, trained for 300 steps on a repeating
`"0123456789ABCDEF"` string, then used to generate — a correctly-trained toy
model should generate visibly periodic output.

## 5. Reference

Yang, Wang, Yu, Kim, *"Parallelizing Linear Transformers with the Delta
Rule over Sequence Length,"* 2024.
