# Expert Choice Routing - a toy-scale build

A minimal implementation of **Expert Choice routing**, from Zhou, Lei, Liu,
Du, Huang, Zhao, Dai, Chen, Le, Laudon, *"Mixture-of-Experts with Expert
Choice Routing"* (2022) — a way to route tokens to experts that guarantees
perfectly even expert load *by construction*, instead of relying on an
auxiliary loss to encourage it.

This README explains the idea in plain words. `expert_choice_mini.ipynb`
builds it, trains it on a toy dataset, and verifies it learns.

*(See `../moe` for the more common token-choice routing this notebook
inverts.)*

---

## 1. Flipping who does the choosing

The `../moe` folder in this repo implements the standard MoE routing
recipe: every **token** looks at all the experts and picks its favorite
top-k. This works, but nothing about it guarantees experts end up with
balanced workloads — some experts can end up popular and overloaded, others
neglected, which is why that notebook needs an auxiliary load-balancing
loss to actively discourage the router from collapsing onto a few experts.

**Expert Choice routing flips the direction of choice.** Instead of each
token picking its experts, each **expert** picks its favorite tokens:

```
for each expert e:
    score every token's affinity for expert e
    expert e picks its top-`capacity` favorite tokens
    process exactly those tokens, weighted by their affinity score
```

Since every expert always picks exactly `capacity` tokens — no more, no
less — **every expert is guaranteed to process the same number of tokens**,
by construction. There's no possibility of load imbalance for an auxiliary
loss to fix, because the mechanism itself can't produce imbalance in the
first place.

## 2. What "capacity" means here

`capacity` is simply how many tokens each expert is allowed to pick,
computed from a **capacity factor**:

```
capacity = capacity_factor * (total_tokens / n_experts)
```

A `capacity_factor` of `1.0` means "on average, each expert gets exactly
its fair share of tokens" (the same average load you'd get with
token-choice top-1 routing). A `capacity_factor` above `1.0` (this notebook
uses `2.0`) gives each expert some slack to pick a few more tokens than its
exact fair share, since some tokens are more universally useful to route on
than others, and a hard 1x cap would waste some of that available compute
headroom.

## 3. What this means for individual tokens

There's a real trade-off buried in this design: because experts pick
tokens (not the other way around), a token can now be picked by **several**
experts, or by **none at all** — nothing guarantees every token gets
processed. In a full model, this token-level unevenness is usually fine,
since the surrounding residual connection means an un-routed token simply
skips the MoE layer's contribution for that step, rather than being
dropped from the sequence outright.

## 4. What this notebook simplifies

The paper's routing uses a slightly more involved auxiliary procedure for
very small `capacity_factor` settings (to keep training stable when
capacity is tight), and discusses batch-composition effects at inference
time — since "top-c tokens out of this expert's column of scores" depends
on which other tokens happen to be in the same batch, a wrinkle
token-choice routing doesn't have, since each token's routing decision
there doesn't depend on any other token in the batch. This notebook
implements the core selection mechanism directly, with a generous
`capacity_factor=2.0` that keeps training simple and stable at toy scale.

## 5. Running it

Open `expert_choice_mini.ipynb` in Colab, run all cells top to bottom. The
notebook first checks the balance guarantee directly (every expert handling
exactly `capacity` tokens), then trains a tiny LM for 300 steps on a
repeating `"0123456789ABCDEF"` string, then generates — no auxiliary loss
term anywhere in the training loop.

## 6. Reference

Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Chen, Le, Laudon,
*"Mixture-of-Experts with Expert Choice Routing,"* 2022.
