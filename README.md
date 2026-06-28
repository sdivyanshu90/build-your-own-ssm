# build-your-own-ssm — Mamba from first principles

A complete, from-scratch implementation of the **Mamba** selective state space
model [Gu & Dao, 2023] and its mathematical antecedents (HiPPO, S4), written in
**pure PyTorch with zero external SSM dependencies**. Every component is derived
in the docs, fully type-annotated, `mypy --strict`-clean, black-formatted, and
covered by an exhaustive test suite.

> Mamba matches or beats Transformers at language modeling while scaling
> **linearly** in sequence length and decoding in **constant memory per token** —
> no attention, no KV cache, no positional encoding.

## Why this repo

- **Pedagogical + correct.** The naive sequential scan is the executable ground
  truth; the parallel associative scan is proven equivalent to it (forward *and*
  gradients) by `torch.autograd.gradcheck` and property-based tests.
- **Every layer derived.** Eight standalone documents in [`docs/`](docs/) take
  you from continuous-time ODEs → discretization → HiPPO → S4 → the selective
  (S6) mechanism → the hardware-aware scan → training and benchmarks.
- **Real model.** The `MambaConfig(d_model=768, n_layers=24)` preset builds a
  ~130M-parameter language model with weight-tied embeddings and full
  greedy/top-k/top-p/streaming generation.

## Install

```bash
pip install -e ".[dev]"          # editable install with test/lint extras
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

## Quickstart

```python
import torch
from mamba import MambaConfig, MambaLMHeadModel

config = MambaConfig(d_model=256, n_layers=6, d_state=16, vocab_size=50277)
model = MambaLMHeadModel(config)

ids = torch.randint(0, config.vocab_size, (1, 32))
out = model(ids, labels=ids)          # CausalLMOutput(logits=..., loss=...)
print(out.loss)

prompt = torch.randint(0, config.vocab_size, (1, 8))
tokens = model.generate(prompt, max_new_tokens=50, temperature=0.8, top_k=40)
```

Runnable examples:

```bash
python examples/train_lm.py     # train a tiny LM on a synthetic copy task
python examples/generate.py     # greedy / top-k / top-p / streaming decoding
```

## Architecture

The package is layered so dependencies only ever point *downward* (enforced by a
no-circular-imports test):

```
ops ───────► core ───────► layers ───────► models
(scans,      (discretize,  (RMSNorm,        (MambaModel,
 conv1d)      hippo,        MambaBlock,       MambaLMHeadModel)
              selective_ssm) ResidualBlock)
```

| Module | Role |
| --- | --- |
| `mamba/core/discretize.py` | ZOH / bilinear / Euler + the batched selective ZOH |
| `mamba/core/hippo.py` | HiPPO-LegS/LegT/LagT, NPLR & DPLR factorizations |
| `mamba/core/ssm_base.py` | LTI SSM base with conv↔recurrent duality |
| `mamba/core/selective_ssm.py` | **S6**: input-dependent Δ, B, C (the heart of Mamba) |
| `mamba/ops/selective_scan_naive.py` | sequential reference scan (ground truth) |
| `mamba/ops/selective_scan_parallel.py` | log-depth associative scan |
| `mamba/ops/causal_conv1d.py` | causal depthwise conv (full + rolling-buffer step) |
| `mamba/layers/mamba_block.py` | the gated block: `in_proj → conv → SSM → gate → out_proj` |
| `mamba/models/` | the backbone and the tied LM head with `generate()` |

Build order (each piece builds on the last): `discretize → HiPPO → SSM →
selective_scan → MambaBlock → Mamba model`.

## The selective scan, in one line

The recurrence `hₜ = Āₜ hₜ₋₁ + B̄ₜ uₜ` is a first-order linear recurrence,
hence a prefix-scan under the associative operator
`(a₁,b₁) ⊕ (a₂,b₂) = (a₂a₁, a₂b₁ + b₂)`. That associativity (proved in the docs,
tested in `tests/property`) is what lets training run in `O(log L)` parallel
depth while inference steps in `O(1)`.

## Testing

```bash
pytest -m "not slow"                       # full suite (fast)
pytest tests/benchmark -m slow --benchmark-only   # scan micro-benchmarks
mypy mamba/                                 # strict type check
black --check .                             # formatting
```

Highlights the suite guarantees:

- parallel scan == naive scan to `atol=1e-4` (forward and gradients);
- recurrent single-step decode == parallel forward to `atol=1e-4`;
- discretization matches `scipy.signal.cont2discrete`;
- stability, causality, linearity, and HiPPO reconstruction properties.

## Documentation

1. [SSM Mathematical Foundations](docs/01_ssm_mathematical_foundations.md)
2. [HiPPO Theory](docs/02_hippo_theory.md)
3. [S4 Architecture](docs/03_s4_architecture.md)
4. [Mamba's Selective Mechanism](docs/04_mamba_selective_mechanism.md)
5. [The Hardware-Aware Algorithm](docs/05_hardware_aware_algorithm.md)
6. [Implementation Guide](docs/06_implementation_guide.md)
7. [Training and Evaluation](docs/07_training_and_evaluation.md)
8. [Benchmarks and Comparisons](docs/08_benchmarks_and_comparisons.md)

## References

- A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023.
- A. Gu, K. Goel, C. Ré. *Efficiently Modeling Long Sequences with Structured State Spaces (S4).* 2021.
- A. Gu et al. *HiPPO: Recurrent Memory with Optimal Polynomial Projections.* 2020.
- T. Dao et al. *FlashAttention.* 2022.
- G. Blelloch. *Prefix sums and their applications.* 1990.

## License

MIT.
