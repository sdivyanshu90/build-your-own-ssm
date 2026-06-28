"""Minimal language-model training loop for the from-scratch Mamba.

This is a self-contained, dependency-free example: it trains a small Mamba LM to
memorise a synthetic copy/repeat task so you can watch the loss fall and the
machinery (optimizer split, warmup-cosine schedule, gradient clipping,
checkpointing) work end to end. Replace :func:`synthetic_batch` with a real
tokenised dataset to train on text.

Run::

    python examples/train_lm.py
"""

from __future__ import annotations

import torch

from mamba.config import MambaConfig
from mamba.models.lm_head import MambaLMHeadModel
from mamba.utils.checkpoint import save_checkpoint
from mamba.utils.training import build_optimizer, build_scheduler, clip_grad_norm_


def synthetic_batch(
    batch_size: int, seqlen: int, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """Return a batch of ``[prefix, prefix]`` sequences (a copy task)."""
    half = seqlen // 2
    prefix = torch.randint(0, vocab_size, (batch_size, half), device=device)
    return torch.cat([prefix, prefix], dim=1)[:, :seqlen]


def main() -> None:
    """Train a tiny Mamba LM on the synthetic copy task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MambaConfig(
        d_model=128,
        n_layers=4,
        d_state=16,
        vocab_size=256,
        pad_vocab_size_multiple=8,
    )
    model = MambaLMHeadModel(config).to(device)
    model.train()

    total_steps = 300
    optimizer = build_optimizer(model, lr=3e-3, weight_decay=0.1)
    scheduler = build_scheduler(optimizer, warmup_steps=20, total_steps=total_steps)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"training a {n_params/1e6:.2f}M-parameter Mamba on {device}")

    for step in range(total_steps):
        batch = synthetic_batch(16, 64, config.vocab_size, device)
        optimizer.zero_grad()
        loss = model(batch, labels=batch).loss
        assert loss is not None
        loss.backward()
        grad_norm = clip_grad_norm_(model, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if step % 25 == 0 or step == total_steps - 1:
            lr = scheduler.get_last_lr()[0]
            print(
                f"step {step:4d} | loss {loss.item():.4f} | "
                f"ppl {torch.exp(loss).item():8.2f} | lr {lr:.2e} | "
                f"|g| {grad_norm:.2f}"
            )

    save_checkpoint(model, optimizer, scheduler, step=total_steps, path="mamba_ckpt.pt")
    print("saved checkpoint to mamba_ckpt.pt")


if __name__ == "__main__":
    main()
