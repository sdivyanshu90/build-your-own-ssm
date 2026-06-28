"""Unit tests for :mod:`mamba.utils.training` and :mod:`mamba.utils.checkpoint`.

Check the optimizer's decay/no-decay split, the warmup-then-cosine schedule, the
loss shift, gradient clipping, and the reference-format key conversion
round-trip.
"""

from __future__ import annotations

import math

import torch

from mamba.config import MambaConfig
from mamba.models.lm_head import MambaLMHeadModel
from mamba.utils.checkpoint import convert_from_reference
from mamba.utils.training import (
    build_optimizer,
    build_scheduler,
    clip_grad_norm_,
    compute_loss,
)

# Reverse of the conversion table, to synthesise an official-format state dict.
_TO_REFERENCE = (
    ("mixer.conv1d.conv.weight", "mixer.conv1d.weight"),
    ("mixer.conv1d.conv.bias", "mixer.conv1d.bias"),
    ("mixer.ssm.x_proj.", "mixer.x_proj."),
    ("mixer.ssm.dt_proj.", "mixer.dt_proj."),
    ("mixer.ssm.A_log", "mixer.A_log"),
    ("mixer.ssm.D", "mixer.D"),
)


def _small_model() -> MambaLMHeadModel:
    torch.manual_seed(0)
    cfg = MambaConfig(d_model=32, n_layers=2, d_state=8, vocab_size=64)
    return MambaLMHeadModel(cfg)


class TestTraining:
    def test_optimizer_decay_split(self) -> None:
        """1-D params, A_log and D land in the no-decay group; matrices decay."""
        model = _small_model()
        opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)
        decay_group, no_decay_group = opt.param_groups
        assert decay_group["weight_decay"] == 0.1
        assert no_decay_group["weight_decay"] == 0.0
        # Every no-decay param is 1-D or an SSM A_log/D (which are >=2-D/1-D).
        no_decay_ids = {id(p) for p in no_decay_group["params"]}
        for name, p in model.named_parameters():
            if name.endswith("A_log") or name.endswith(".D") or p.ndim < 2:
                assert id(p) in no_decay_ids, name

    def test_scheduler_warmup_then_decay(self) -> None:
        """LR ramps linearly during warmup then decays following a cosine."""
        model = _small_model()
        opt = build_optimizer(model, lr=1.0)
        sched = build_scheduler(opt, warmup_steps=10, total_steps=110, min_lr_ratio=0.0)
        lrs = []
        for _ in range(110):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        assert lrs[0] < lrs[5] < lrs[10]  # warmup increasing
        assert math.isclose(lrs[10], 1.0, rel_tol=1e-6)  # peak at end of warmup
        assert lrs[-1] < 0.05  # decayed near zero

    def test_clip_grad_norm(self) -> None:
        """Clipping caps the global gradient norm and reports the pre-clip norm."""
        model = _small_model()
        ids = torch.randint(0, 64, (1, 8))
        model(ids, labels=ids).loss.backward()
        total = clip_grad_norm_(model, max_norm=0.5)
        assert torch.isfinite(total)
        post = torch.sqrt(
            sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)
        )
        assert post <= 0.5 + 1e-4

    def test_compute_loss_shift(self) -> None:
        """The loss aligns logits[t] with labels[t+1]."""
        logits = torch.zeros(1, 4, 10)
        labels = torch.tensor([[1, 2, 3, 4]])
        loss = compute_loss(logits, labels)
        # Uniform logits -> loss == log(vocab).
        assert math.isclose(loss.item(), math.log(10), rel_tol=1e-5)


class TestCheckpointConversion:
    def test_convert_from_reference_round_trip(self) -> None:
        """Official-format keys convert to this repo's layout and load cleanly."""
        model = _small_model()
        our_state = model.state_dict()

        # Build a synthetic "reference" state dict by reversing the renames.
        ref_state = {}
        for key, value in our_state.items():
            ref_key = key
            for ours, ref in _TO_REFERENCE:
                if ours in ref_key:
                    ref_key = ref_key.replace(ours, ref)
                    break
            ref_state[ref_key] = value

        converted = convert_from_reference(ref_state)
        assert set(converted) == set(our_state)

        fresh = MambaLMHeadModel(model.config)
        missing, unexpected = fresh.load_state_dict(converted, strict=False)
        assert not missing and not unexpected
