"""Autoregressive generation example for the from-scratch Mamba.

Loads a checkpoint produced by ``examples/train_lm.py`` (or builds a random
model if none exists) and demonstrates the recurrent, constant-memory decoder
in greedy, temperature, top-k, and nucleus modes, plus token streaming.

Run::

    python examples/generate.py
"""

from __future__ import annotations

import os

import torch

from mamba.config import MambaConfig
from mamba.models.lm_head import MambaLMHeadModel, load_pretrained


class PrintStreamer:
    """A trivial streamer that prints each generated token id as it arrives."""

    def put(self, token: torch.Tensor) -> None:
        print(token.tolist(), end=" ", flush=True)

    def end(self) -> None:
        print()


def main() -> None:
    """Generate continuations with several decoding strategies."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = "mamba_ckpt.pt"
    if os.path.exists(ckpt):
        model = load_pretrained(ckpt, device=device)
        print(f"loaded {ckpt}")
    else:
        torch.manual_seed(0)
        config = MambaConfig(
            d_model=128,
            n_layers=4,
            d_state=16,
            vocab_size=256,
            pad_vocab_size_multiple=8,
        )
        model = MambaLMHeadModel(config).to(device)
        print("no checkpoint found; using a randomly-initialised model")
    model.eval()

    prompt = torch.randint(0, model.config.vocab_size, (1, 8), device=device)
    print("prompt:", prompt.tolist())

    greedy = model.generate(prompt, max_new_tokens=16, do_sample=False)
    print("greedy:", greedy[0, 8:].tolist())

    torch.manual_seed(0)
    sampled = model.generate(
        prompt, max_new_tokens=16, do_sample=True, temperature=0.8, top_k=40
    )
    print("top-k :", sampled[0, 8:].tolist())

    torch.manual_seed(0)
    nucleus = model.generate(
        prompt, max_new_tokens=16, do_sample=True, temperature=1.0, top_p=0.9
    )
    print("top-p :", nucleus[0, 8:].tolist())

    print("stream:", end=" ")
    model.generate(prompt, max_new_tokens=16, do_sample=False, streamer=PrintStreamer())


if __name__ == "__main__":
    main()
