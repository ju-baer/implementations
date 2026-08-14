# Mamba (S6) — a toy-scale build of a Selective State Space Model

A minimal, readable implementation of a **Mamba** block, from Gu & Dao's
*"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"* (2023),
built to run and train on a free Colab GPU (or CPU) in a couple of minutes.

This README explains the idea in plain words. `mamba_mini.ipynb` builds it
cell by cell, trains it on a toy dataset, and verifies it learns.

---

## 1. The big picture

Mamba is built on a **State Space Model (SSM)** — an old idea from control
theory repurposed as a sequence layer. There's a hidden state `h` that gets
updated at every timestep and read out to produce the output:

```
h_t = A_bar · h_{t-1} + B_bar · x_t
y_t = C_t · h_t + D · x_t
```

This looks like an RNN, and it is one. The "S6" / "selective" part is what
makes it different from a classic SSM: instead of fixed `A`, `B`, `C`
matrices, Mamba makes `B`, `C`, and a per-token step size `Δ` **functions of
the current input**. That's what lets the model decide, per token, whether
to let information into the state, keep it, or ignore it entirely — the SSM
version of "relevance."

A full Mamba block wraps this recurrence with:

- an **input projection** splitting into two branches — one goes through the
  SSM, the other is a plain multiplicative gate,
- a **short causal convolution** before the SSM so nearby tokens mix locally
  first,
- a **SiLU gate** from the second branch, multiplied into the SSM's output.

## 2. What "selective" changes, step by step

At every timestep `t`, three values are computed *from the token itself*:

- **`Δ_t`** — a positive step size (via `softplus`), one per inner channel.
  Think of it as "how much time has passed" for this token: large `Δ_t`
  means the state changes a lot; small means the token barely moves it.
- **`B_t`** — which parts of the incoming token get written into the state.
- **`C_t`** — which parts of the state get read out as output.

These combine with a fixed, learned, always-negative per-channel matrix `A`
(so the state naturally decays over time) into a **discretized**,
per-timestep recurrence:

```
A_bar_t = exp(Δ_t · A)         # decay factor, specific to this token
B_bar_t = Δ_t · B_t            # how much of x_t actually gets written in
h_t = A_bar_t · h_{t-1} + B_bar_t · x_t
y_t = C_t · h_t + D · x_t
```

Every one of `Δ_t`, `B_t`, `C_t` depends on the current token — that's the
entire selection mechanism. A classic (non-selective) SSM reuses the same
`A_bar`, `B_bar` at every timestep, no matter the input; Mamba recomputes
them fresh, every token.

## 3. Simplifications made here

- **Sequential recurrence instead of a parallel scan.** The real
  implementation computes this recurrence with a parallel-scan algorithm
  (plus a fused CUDA kernel) so it runs in `O(log T)` parallel steps. This
  notebook uses a plain Python `for` loop over timesteps instead — exact
  same math, far slower, much easier to read.
- **Diagonal state matrix (S4D-style).** `A` here is diagonal, initialized
  as `-1, -2, ..., -d_state` per channel. The original S4 paper uses a
  structured (HiPPO) matrix designed to be better at long-range
  dependencies; the diagonal version is what Mamba itself actually uses in
  practice, so this isn't really a simplification so much as following the
  paper's own choice.

## 4. What's left out

- Hardware-aware kernel fusion and the parallel scan (see above).
- Mixed precision / recomputation tricks used to keep memory low during
  training on long sequences — irrelevant at toy scale.

## 5. Running it

Open `mamba_mini.ipynb` in Colab, run all cells top to bottom. You'll see the
model built, sanity-checked (one forward + backward pass, no NaN grads),
trained for 300 steps on a repeating `"0123456789ABCDEF"` string, then used
to generate — a correctly-trained toy model should generate visibly
periodic output.

## 6. Reference

Gu & Dao, *"Mamba: Linear-Time Sequence Modeling with Selective State
Spaces,"* 2023.
