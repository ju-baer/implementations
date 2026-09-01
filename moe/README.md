# Mixture-of-Experts (MoE) - a toy-scale build, isolated from any one architecture

A minimal implementation of a **sparse Mixture-of-Experts feed-forward
layer** (Mixtral / Switch-Transformer style top-k routing, with real sparse
dispatch and a load-balancing auxiliary loss), built as a standalone
component rather than folded into a specific attention variant. So it can
be dropped into, or compared against, any of the other architectures in
this repo.

This README explains the idea in plain words. `moe_mini.ipynb` builds it,
trains it on a toy dataset, and verifies it learns.

*(K3's Stable LatentMoE, in `../kda`, is a more elaborate relative of this
same idea. latent-compressed routed experts, a different activation, and a
dense-compute simplification rather than the real sparse dispatch built
here.)*

---

## 1. The idea

A normal transformer feed-forward layer is one MLP, applied to every token.
Making that MLP bigger makes the model more capable, but also makes *every
single token* pay the full cost of the bigger MLP, whether or not that
token actually needed the extra capacity.

**Mixture-of-Experts (MoE)** breaks the one big MLP into many smaller
"expert" MLPs, and routes each token to only a handful of them:

1. A small **router** network looks at each token and scores every expert.
2. Only the **top-k highest-scoring experts** actually process that token
   (`k` is usually small (1 or 2) even with dozens of experts total).
3. The expert outputs are combined with a weighted sum, using the router's
   own scores (renormalized over just the selected experts) as weights.

The result: the model's *total* parameter count can grow enormously (more
experts), while the *compute* per token stays roughly fixed (still only `k`
experts run per token). More capacity without a proportional increase in
compute per token — that's the whole trade MoE is making.

## 2. The part that's easy to get wrong: load balancing

Nothing forces the router to spread tokens evenly across experts. Left
alone, it tends to **collapse**: a few experts get most of the traffic (and
therefore most of the useful gradient signal, and get even better at
attracting tokens), while the rest are barely used and never improve. A
rich-get-richer dynamic that shows up almost immediately in practice.

The fix used here is a **load-balancing auxiliary loss**, added on top of
the normal training loss:

```
frac_tokens[e]      = fraction of tokens actually routed to expert e
mean_router_prob[e] = average router probability assigned to expert e (before top-k)
aux_loss = n_experts * sum_e( frac_tokens[e] * mean_router_prob[e] )
```

This loss is minimized when both quantities are close to uniform across
experts (`1/n_experts` each), so the model is explicitly penalized for
letting the router concentrate on a few experts, and gets pushed to
actually spread load out.

## 3. Real sparse dispatch (not the "compute everyone, zero the rest" shortcut)

A common shortcut, used in the `../kda` notebook's Stable LatentMoE, for
simplicity, is to run *every* expert on *every* token and multiply the
unselected ones by zero afterward. Much simpler to write, but the "sparse"
layer still does dense compute under the hood.

This notebook does the real thing instead: for each expert, gather only the
tokens actually routed to it, and run the expert *only* on those tokens.
This is what a production MoE implementation does (usually with more
machinery for balancing token counts across GPUs), and it's worth seeing at
least once, since it's the part that actually delivers MoE's efficiency
promise.

## 4. What this notebook simplifies

Production MoE systems also need a **capacity factor** — a hard cap on how
many tokens any one expert can accept, with overflow tokens dropped or
rerouted, to keep compute perfectly balanced across hardware. This notebook
lets every selected token through regardless of how many other tokens
picked the same expert, which is fine at toy scale but would cause uneven
GPU utilization at production scale.

## 5. How it's wired into a model here

`moe_mini.ipynb` wraps the MoE layer in an otherwise completely ordinary
transformer block: standard causal self-attention, then the sparse MoE
instead of a plain MLP. Since the MoE layer returns both its output *and*
an auxiliary loss, the tiny model collects the auxiliary losses from every
layer and adds them (lightly weighted) into the main training loss.

## 6. Running it

Open `moe_mini.ipynb` in Colab, run all cells top to bottom. You'll see the
model built, sanity-checked, trained for 300 steps on a repeating
`"0123456789ABCDEF"` string, then used to generate, a correctly-trained toy
model should generate visibly periodic output.

## 7. References

Shazeer et al., *"[Outrageously Large Neural Networks: The Sparsely-Gated
Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538),"* 2017; 
Fedus, Zoph, Shazeer, *"[Switch Transformers: Scaling to Trillion Parameter Models with Simple and
Efficient Sparsity](https://arxiv.org/pdf/2101.03961),"* 2021; 
Jiang et al., *"[Mixtral of Experts](https://arxiv.org/pdf/2401.04088),"* 2024.
