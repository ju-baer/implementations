# Hyena - a toy-scale build of an implicit long-convolution operator

A minimal implementation of the **Hyena operator**, from Poli et al.,
*"[Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866)"* (2023), a way to replace attention entirely with long convolutions, generated
on-the-fly by a small neural network instead of stored as raw weights.

This README explains the idea in plain words. `hyena_mini.ipynb` builds it,
trains it on a toy dataset, and verifies it learns.

---

## 1. The idea

Attention computes, for every pair of tokens, how much they should
interact; powerful, but quadratic in sequence length. Hyena asks: what if
instead we used a **convolution** , cheaper, with a built-in "nearby things
interact more" bias, but made it long enough to reach across the *entire*
sequence, and let the model *control* the convolution with gates, the way
attention lets the model control what to attend to?

Two ingredients make this work:

- **Implicit long convolutions.** A convolution filter that reaches across
  the whole sequence would normally mean storing a huge number of weights
  (one per position, per channel). Instead, Hyena **generates** the filter
  from a small MLP: feed the MLP a position (as a sinusoidal embedding, like
  a positional encoding), and it outputs what the filter's value should be
  at that position. The filter is computed on the fly, for whatever
  sequence length is needed, from a tiny number of MLP weights — not stored
  directly.
- **Data-controlled gating.** A single long convolution is still just a
  fixed linear operation, however it's parameterized — it doesn't look at
  the data to decide what to do. Hyena interleaves convolutions with
  **elementwise multiplication by a projection of the input itself**, which
  is what actually makes the operator data-dependent, the way attention's
  weights depend on the actual tokens being attended to.

## 2. The recurrence, order by order

A Hyena operator of order `N` projects the input into `N+1` branches, then
alternates "long convolution" and "gate by another branch," `N` times:

```
x0, x1, ..., x_{N-1}, v = split(project(input))
z = v
for n in 0 .. N-1:
    z = long_conv_n(z) * x_n
output = project(z)
```

This notebook builds the **order-2** case (3 branches: `x0`, `x1`, `v`):

```
z = v
z = long_conv_0(z) * x0
z = long_conv_1(z) * x1
output = project(z)
```

Each `long_conv_n` uses its *own* implicit filter (its own small MLP), and
because the filter only has non-zero values for lags `0..t`, the whole
operation only ever looks at the past.

## 3. Making the long convolution both causal and fast

Three details matter for a long convolution to actually work:

- **Causality**: the filter is only defined and only ever used for
  non-negative lags relative to the current position, so it never looks at
  the future.
- **An exponential decay envelope**: the raw MLP-produced filter is
  multiplied by a decaying envelope (`exp(-decay · t)`), so distant lags
  are naturally down-weighted rather than left free to take on arbitrary
  magnitude. This is a stability trick the paper also uses, since an
  unconstrained implicit filter can otherwise be numerically unstable during
  training.
- **Speed via FFT**: computing a convolution with a filter as long as the
  whole sequence naively costs `O(T^2)`, the same as attention. Using the
  convolution theorem (`conv(x, f) = ifft(fft(x) · fft(f))`), it can be
  computed in `O(T log T)` instead, this is Hyena's actual efficiency
  argument over attention at long sequence lengths.

## 4. What this notebook simplifies

The paper's Hyena operator typically includes short convolutions on the
projected branches too (mirroring the short conv used ahead of KDA's and
Mamba's recurrences elsewhere in this repo), along with more careful filter
regularization. This notebook keeps just the two essential ingredients (implicit long convolutions and data-controlled gating) to keep the core
idea legible.

## 5. Running it

Open `hyena_mini.ipynb` in Colab, run all cells top to bottom. You'll see
the model built, sanity-checked, trained for 300 steps on a repeating
`"0123456789ABCDEF"` string, then used to generate, a correctly-trained toy
model should generate visibly periodic output.

## 6. Reference

Poli, Massaroli, Nguyen, Fu, Dao, Baccus, Bengio, Ermon, Ré, *"[Hyena
Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866),"* 2023.
