# Titans - a toy-scale build of neural memory learned at test time

A minimal implementation of the **neural memory module** from Behrouz,
Zhong, Mirrokni, *"[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)"* (2024), a
memory that is itself a small neural network, whose weights get updated on
the fly, one token at a time, instead of a fixed-size vector or matrix
state.

This README explains the idea in plain words. `titans_mini.ipynb` builds it,
trains it on a toy dataset, and verifies it learns.

---

## 1. The idea

Every other architecture in this repo (KDA, GLA, RetNet, Mamba) represents
"memory" as a fixed-shape tensor (a matrix or vector) updated by some
linear-ish rule every step. Titans asks a different question: **what if the
memory were itself a small neural network**, and "updating the memory" meant
literally *training that network a little*, at every token, while the model
runs?

Concretely, the memory here is a tiny 2-layer MLP. At every timestep, three
things happen:

1. **Test the memory.** Feed the current key `k_t` into the memory MLP and
   see what it predicts. Compare that to the current value `v_t` with a
   loss. This is the model's **associative recall loss**: "if I've already
   learned to associate keys with values, how wrong am I about this one?"
2. **Compute the "surprise."** Take the gradient of that loss with respect
   to the memory's own weights. This gradient *is* the surprise, it points
   toward whatever would make the memory less wrong about this particular
   key-value pair, right now, mid-sequence.
3. **Update the memory's weights using that gradient**, combined with a
   momentum term (so a surprising token's effect lingers, like "recent
   surprise") and a weight-decay term (an adaptive forget gate on the whole
   memory).

```
surprise_grad = d/dW [ loss(memory(k_t), v_t) ]      # how wrong is the memory right now
momentum = eta_t * momentum - lr_t * surprise_grad     # accumulate recent surprise
W = (1 - alpha_t) * W + momentum                       # decay old weights, apply the update
```

The output at each step: query the *freshly updated* memory with the query
vector `q_t`.

This is a genuinely different flavor of "state" from everything else in
this repo. Instead of writing values into slots of a fixed-size tensor, the
model does a tiny amount of **gradient descent on itself**, live, as it
reads the sequence. `eta_t`, `alpha_t`, and the step size are all predicted
from the current token, so the model controls its own learning rate and
forgetting rate per token.



## 2. Why this needs a second derivative

The "surprise" is a gradient of a loss with respect to the memory's weights, and then, when the *whole model* is trained, the outer optimizer needs to
compute a gradient of the final loss with respect to everything, including
those per-token surprise gradients. That means backpropagating through a
gradient computation: a gradient of a gradient.

In the code, this is what `torch.autograd.grad(..., create_graph=True)`
does, it computes the inner surprise gradient in a way that stays part of
the differentiable graph, so the outer `.backward()` call can flow through
the *entire* test-time learning process. Practically, this also means the
memory update has to stay numerically well-behaved, since it's effectively
running its own tiny training loop nested inside the outer one.

## 3. Keeping it numerically stable

A naive version of this update genuinely diverges (weights blow up, loss
turns to `NaN`) once you unroll it over more than a handful of tokens. This
notebook uses three guardrails that matter a lot in practice:

- **L2-normalized keys and queries**- keeps the memory's inputs on the unit
  sphere, so the associative-recall loss (and its gradient) can't scale up
  just because an embedding got large.
- **A clamped surprise gradient** - bounds how big a single token's update
  to the memory can be, regardless of how "surprised" the memory technically
  is.
- **A forget gate that's never fully off** (`alpha` is scaled to always stay
  above a small floor) - so the memory's weights can't just accumulate
  forever across a long sequence.

## 4. What this notebook simplifies

The paper explores several variants like, different memory depths, different
update rules, and ways to fold this memory into a full architecture
alongside attention (the paper's "MAC", "MAG", "MAL" variants, which combine
neural memory with attention rather than replacing it). This notebook
implements just the core neural-memory update rule as a standalone sequence
layer, with a shallow 2-layer MLP as the memory and one independent memory
per attention head.

## 5. Running it

Open `titans_mini.ipynb` in Colab, run all cells top to bottom. Because
every training step here involves a gradient-of-a-gradient, it's noticeably
slower per step than the other notebooks in this repo — the notebook uses a
shorter sequence length and a single layer to keep it fast even on CPU.
You'll see the model built, sanity-checked, trained on a repeating
`"0123456789ABCDEF"` string, then used to generate.

## 6. Reference

Behrouz, Zhong, Mirrokni, *"[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663),"*
2024.
