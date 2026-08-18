# Native Sparse Attention (NSA) - a toy-scale build

A minimal implementation of **Native Sparse Attention**, from DeepSeek-AI,
*"Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse
Attention"* (2025) — a way to make attention cheaper by making it
*trainably sparse*, rather than dense-then-pruned.

This README explains the idea in plain words. `nsa_mini.ipynb` builds it,
trains it on a toy dataset, and verifies it learns.

---

## 1. The idea

Regular causal attention makes every token attend to *every* earlier
token — that's what makes it expensive at long context. Prior work often
sparsifies attention *after* training (only look at a subset of tokens at
inference time), but that tends to hurt quality, because the model was
never trained to work with a sparse pattern in the first place.

NSA's approach: build the sparsity **into the architecture from the
start**, so the model learns to use it well. Every token's attention output
is a learned combination of **three branches**, each attending differently:

| branch | what it attends to | good at |
|---|---|---|
| **compressed** | coarse summaries of *blocks* of tokens (mean-pooled) | cheap, global context |
| **selected** | full-resolution attention, but only within the *few* blocks the compressed branch found most relevant | precise long-range recall, without the full cost |
| **sliding window** | a plain local window of the most recent tokens | fine-grained local/syntactic patterns |

All three branches share the same Q/K/V projections, run independently, and
get combined with a **learned, per-token, per-head gate**.

## 2. Each branch, step by step

**Compressed branch:** split the sequence into fixed-size blocks and
mean-pool each block's keys and values into one summary vector per block.
Run ordinary attention from every query to these (far fewer) block
summaries. Cheap, because there are much fewer blocks than tokens, and it
gives every query at least a coarse view of the whole sequence.

**Selected branch:** reuse the *attention scores* the compressed branch just
computed as an importance signal — "which blocks did this query find most
relevant?" Pick the top-`n` blocks per query, then run **full-resolution**
attention against every individual token inside just those blocks. This is
where NSA gets precise, without paying for full-resolution attention against
every token in the whole sequence.

**Sliding window branch:** completely ordinary causal attention, capped to a
fixed window of the most recent tokens. This branch exists mostly so the
model always has cheap, reliable access to local context, which the other
two branches aren't well suited for on their own.

**Combining:** a small linear layer on the input predicts one gate value
per branch, per head, per token (via sigmoid); the final output is the
gate-weighted sum of the three branches' outputs.

## 3. What this notebook simplifies

The paper's actual speedup comes from a custom, hardware-aligned Triton
kernel that exploits the sparsity pattern directly — skipping the
un-selected blocks entirely, rather than computing and discarding them.
This notebook computes the **compressed** and **sliding window** branches
densely (over the full block grid / full window), and only genuinely
narrows compute in the **selected** branch, via `gather` — enough to
demonstrate the actual sparsity pattern and get the causal masking exactly
right at toy scale. A real implementation skips far more compute than this
one does.

## 4. Running it

Open `nsa_mini.ipynb` in Colab, run all cells top to bottom. You'll see the
model built, sanity-checked, trained for 300 steps on a repeating
`"0123456789ABCDEF"` string, then used to generate — a correctly-trained toy
model should generate visibly periodic output.

## 5. Reference

DeepSeek-AI, *"Native Sparse Attention: Hardware-Aligned and Natively
Trainable Sparse Attention,"* 2025.
