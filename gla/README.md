# Gated Linear Attention (GLA) — a toy-scale build

A minimal implementation of **Gated Linear Attention**, from Yang et al.,
*"Gated Linear Attention Transformers with Hardware-Efficient Training"*
(2023) — the direct architectural ancestor of KDA (see `../kda`), stripped
to its simplest form: a linear-attention layer with a per-channel forget
gate and nothing else.

This README explains the idea in plain words. `gla_mini.ipynb` builds it,
trains it on a toy dataset, and verifies it learns.

---

## 1. The idea

GLA is a **linear attention** layer. Instead of storing every past token and
attending over all of them (which is what makes normal attention's memory
grow with sequence length), it keeps a fixed-size running **state** — a
small matrix — and updates it one token at a time.

Plain linear attention has one problem: it only ever *adds* new information
to the state, so old information never fades — the state just gets muddier
the longer the sequence runs. GLA's fix is a **per-channel forget gate**:
before writing anything new in, the state is first multiplied by a decay
value between 0 and 1, so old information fades unless the model chooses to
keep it.

At every timestep:

```
alpha_t = sigmoid(W_alpha x_t)                  # per-channel forget gate, in (0,1)
S_t = diag(alpha_t) * S_{t-1} + k_t (x) v_t     # decay, then write (outer product)
o_t = q_t^T S_t                                  # read out with the query
```

That's the whole mechanism: decay the state, add the new key-value outer
product, read with the query.

## 2. How this relates to KDA

KDA (`../kda`) starts from exactly this same idea and adds one more
ingredient: the **delta rule**. Where GLA just *adds* the new key-value pair
on top of the decayed state, KDA first *erases* whatever was previously
written for a similar key before writing the new value in. GLA is what's
left when that erase step is removed — a good place to see the "forget gate
on a matrix state" idea in isolation, before KDA layers more on top of it.

## 3. What this notebook simplifies

The real implementation computes this recurrence with a
**chunkwise-parallel** algorithm for GPU efficiency (the paper's whole
"hardware-efficient training" contribution). This notebook uses the plain
sequential recurrence instead — a Python `for` loop over timesteps. Same
math, much slower, much easier to read.

## 4. Running it

Open `gla_mini.ipynb` in Colab, run all cells top to bottom. You'll see the
model built, sanity-checked, trained for 300 steps on a repeating
`"0123456789ABCDEF"` string, then used to generate — a correctly-trained toy
model should generate visibly periodic output.

## 5. Reference

Yang, Wang, Shen, Panda, Kim, *"Gated Linear Attention Transformers with
Hardware-Efficient Training,"* 2023.
