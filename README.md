# implementations

**a workbench of toy-scale architectures, built to be read**

---

Every few months a new architecture paper shows up with a wall of equations, a diagram with six colors, and a claim that it changes everything. Most of the time, the only way to actually *know* whether you understood it is to build
it. Badly, at toy scale, on a laptop or a free Colab GPU, until the loss curve goes down and you can point at the exact line of code that corresponds to Equation 7.

That's what this repo is for.

Every folder here is one architecture, stripped down to something that:
- **runs on a free Colab GPU (or CPU)** in a few minutes
- **trains on a tiny synthetic dataset** just to prove the gradients flow end-to-end and the design isn't broken
- **keeps the paper's section/equation numbers in the code comments**, so you can read the report and the code side by side instead of trusting a black-box `import`
- **documents every simplification made**, and *why* it's safe to make at toy scale; no simplification is silent

Nothing here is meant to be fast, production-grade, or state-of-the-art. It's meant to be **legible**. If you've ever asked "okay but what does this *actually* compute," this is the repo that tries to answer that with running code instead of hand-waving.

---

## how each folder is organized

```
<architecture-name>/
├── README.md              # plain-words explanation of the architecture
└── <architecture>_mini.ipynb   # annotated, runnable Colab notebook
```

The README is the "read this first" version, concepts, intuition, what's simplified and why. The notebook is the "now watch it actually work" version with same architecture, built up piece by piece, trained on a toy sequence, and sanity-checked with a real forward/backward pass.

## implementations

| folder | architecture | idea in one line |
|---|---|---|
| [`kda/`](./kda) | Kimi Delta Attention (Kimi K3) | linear attention with a per-channel forget gate, wrapped in a full MoE transformer block |
| `mamba/` | Mamba / S6 | selective state-space model, the recurrence's forget gate depends on the input |
| `gla/` | Gated Linear Attention | the "plain" ancestor of KDA, linear attention plus a scalar/vector forget gate |
| `retnet/` | RetNet | linear attention with a *fixed* exponential decay instead of a learned gate | 
| `titans/` | Titans (test-time memory) | a small neural network as the memory itself, updated at inference time |
| `xlstm/` | xLSTM | the LSTM, redesigned with matrix memory and exponential gating for the 2024s | 
| `mla/` | Multi-head Latent Attention (DeepSeek) | compress K/V into a small latent vector to shrink the KV cache |
| `nsa/` | Native Sparse Attention (DeepSeek) | trainable sparse attention instead of dense; compressed, selected, and local branches |
| `hyena/` | Hyena | replace attention with long implicit convolutions | 
| `moe/` | Mixture-of-Experts (standalone) | routed sparse FFNs, isolated from any one architecture, with load-balancing tricks |

*(Have a favorite architecture that's missing? Open an issue, toy implementations welcome.)*

## philosophy

1. **Toy scale is a feature, not an excuse.** `d_model=64`, a handful of layers, a synthetic dataset. The point is to see the mechanism work, not to compete on a benchmark.
2. **The paper is the source of truth.** Every non-obvious line of code should trace back to an equation number in a comment.
3. **Every simplification is written down.** "I skipped the chunkwise-parallel form and used the sequential recurrence instead" is a sentence that belongs in the README, not a fact buried in the diff.
4. **Simple words over precise-sounding words.** If a paragraph needs a background in the sub-field to parse, it gets rewritten.

## how to use this repo
Each notebook is self-contained. Open it in Colab, run all cells top to bottom, and you'll see the model built, sanity-checked, trained on a toy periodic sequence, and used to generate a few tokens at the end. No external datasets, no downloads, no config files to hunt down.
