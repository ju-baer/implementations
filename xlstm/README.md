# xLSTM - a toy-scale build of the matrix-memory LSTM (mLSTM)

A minimal implementation of the **mLSTM** cell from Beck et al., *"[xLSTM:
Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)"* (2024), the LSTM redesigned with matrix
memory and exponential gating so it can be trained in parallel, while still
running as a constant-memory recurrence at inference time.

This README explains the idea in plain words. `xlstm_mini.ipynb` builds it,
trains it on a toy dataset, and verifies it learns.

---

## 1. Why revisit the LSTM?

The classic LSTM updates a single *vector* memory cell with sigmoid input
and forget gates. It works, but it's fundamentally sequential as every step
depends on the last and which is part of why transformers overtook it once
fully parallel training became the priority.

xLSTM keeps the "recurrent state that decays and gets written to" idea but
redesigns it so it *can* be reformulated for parallel training, the same
way the other recurrent-with-a-state layers in this repo (GLA, RetNet, KDA)
can. The paper introduces two variants; this notebook builds **mLSTM**, the
fully parallelizable one:

- **Matrix memory** instead of a vector - same idea GLA, RetNet, and KDA all
  use: a small matrix that gets read out with a query.
- **Exponential gating** instead of plain sigmoid gating - the gates can now
  exceed what a sigmoid allows, which the paper shows helps the model
  "revise" earlier storage decisions.
- **A running-maximum stabilizer** in log-space, so the exponential gates
  don't overflow over long sequences.

## 2. The update rule, step by step

At every timestep `t`:

1. Compute two **pre-activation gates**: `i_tilde_t` (input) and `f_tilde_t`
   (forget) - plain linear projections, not yet squashed.
2. **Track a running log-scale maximum** `m_t`:
   ```
   m_t = max(log_sigmoid(f_tilde_t) + m_{t-1}, i_tilde_t)
   ```
   This is the trick that keeps the exponentials from ever overflowing -
   everything downstream is expressed *relative to* this running max.
3. Derive the **stabilized gates**:
   ```
   i_t = exp(i_tilde_t - m_t)
   f_t = exp(log_sigmoid(f_tilde_t) + m_{t-1} - m_t)
   ```
4. **Update the matrix memory** with an outer-product write (same shape of
   update as GLA/RetNet, with these exponential gates):
   ```
   C_t = f_t * C_{t-1} + i_t * (k_t (x) v_t)
   n_t = f_t * n_{t-1} + i_t * k_t          # a normalizer vector
   ```
5. **Read out**, dividing by the normalizer to keep the output scaled
   sensibly:
   ```
   h_t = (C_t^T q_t) / max(|n_t · q_t|, 1)
   ```

## 3. What this notebook simplifies

- **mLSTM only, not sLSTM.** The paper's other variant, sLSTM, keeps a
  scalar (not matrix) memory with a "new memory mixing" mechanism, and
  isn't fully parallelizable the way mLSTM is. mLSTM was chosen here
  because it slots into the same "linear attention with a matrix state"
  family as the rest of this repo like GLA, RetNet, and KDA all live right
  next to it and are worth comparing side by side.
- **Sequential loop, not a parallel form.** Same choice made throughout
  this repo: a plain Python `for` loop over timesteps for readability, not
  the parallel formulation you'd actually want for training speed.

## 4. Running it

Open `xlstm_mini.ipynb` in Colab, run all cells top to bottom. You'll see
the model built, sanity-checked, trained for 300 steps on a repeating
`"0123456789ABCDEF"` string, then used to generate, a correctly-trained toy
model should generate visibly periodic output.

## 5. Reference

Beck, Pöppel, Spanring, Auer, Prudnikova, Kopp, Klambauer, Brandstetter,
Hochreiter, *"[xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517),"* 2024.
