"""Integration tests for the full Mamba language model.

Exercise the assembled stack end to end: loss computation, a short training
loop that must reduce perplexity, generation in several decoding modes, the
recurrent/parallel consistency guarantee, checkpoint round-trips, weight tying,
and the absence of any positional encoding.
"""

from __future__ import annotations

import pytest
import torch

from mamba.config import MambaConfig
from mamba.models.lm_head import MambaLMHeadModel
from mamba.utils.checkpoint import load_checkpoint, save_checkpoint
from mamba.utils.training import build_optimizer


@pytest.fixture
def model() -> MambaLMHeadModel:
    torch.manual_seed(0)
    cfg = MambaConfig(
        d_model=32, n_layers=2, d_state=8, vocab_size=64, pad_vocab_size_multiple=8
    )
    return MambaLMHeadModel(cfg)


class TestMambaModel:
    def test_language_model_loss(self, model: MambaLMHeadModel) -> None:
        """Forward with labels returns a positive finite scalar loss."""
        ids = torch.randint(0, 64, (2, 12))
        out = model(ids, labels=ids)
        assert out.loss is not None
        assert out.loss.ndim == 0
        assert torch.isfinite(out.loss) and out.loss > 0

    def test_perplexity_decreases_with_training(self, model: MambaLMHeadModel) -> None:
        """Overfitting a single batch reduces the loss."""
        ids = torch.randint(0, 64, (2, 16))
        opt = build_optimizer(model, lr=1e-2)
        model.train()
        first = model(ids, labels=ids).loss.item()
        for _ in range(40):
            opt.zero_grad()
            loss = model(ids, labels=ids).loss
            loss.backward()
            opt.step()
        last = model(ids, labels=ids).loss.item()
        assert last < first, f"loss did not drop: {first} -> {last}"

    def test_generate_greedy(self, model: MambaLMHeadModel) -> None:
        """Greedy generation yields the requested number of new tokens."""
        prompt = torch.randint(0, 64, (1, 5))
        out = model.generate(prompt, max_new_tokens=10, do_sample=False)
        assert out.shape == (1, 15)

    def test_generate_sampling(self, model: MambaLMHeadModel) -> None:
        """Temperature sampling stays within the vocabulary and finishes."""
        torch.manual_seed(1)
        prompt = torch.randint(0, 64, (2, 4))
        out = model.generate(prompt, max_new_tokens=8, do_sample=True, temperature=0.8)
        assert out.shape == (2, 12)
        assert out.max() < model.config.padded_vocab_size

    def test_generate_top_p(self, model: MambaLMHeadModel) -> None:
        """Nucleus sampling runs and respects the output shape."""
        torch.manual_seed(2)
        prompt = torch.randint(0, 64, (1, 4))
        out = model.generate(prompt, max_new_tokens=6, do_sample=True, top_p=0.9)
        assert out.shape == (1, 10)

    def test_recurrent_generation_matches_parallel_scoring(
        self, model: MambaLMHeadModel
    ) -> None:
        """Greedy recurrent decode == argmax of repeated full-forward scoring."""
        model.eval()
        prompt = torch.randint(0, 64, (2, 6))
        gen = model.generate(
            prompt, max_new_tokens=10, do_sample=False, temperature=0.0
        )

        seq = prompt.clone()
        for _ in range(10):
            logits = model(seq).logits[:, -1, :]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
        torch.testing.assert_close(gen, seq)

    def test_save_load_checkpoint(
        self, model: MambaLMHeadModel, tmp_path: object
    ) -> None:
        """Save -> load -> forward reproduces the original outputs exactly."""
        ids = torch.randint(0, 64, (1, 8))
        model.eval()
        before = model(ids).logits
        path = tmp_path / "ckpt.pt"  # type: ignore[operator]
        save_checkpoint(model, None, None, step=3, path=path)

        fresh = MambaLMHeadModel(model.config).eval()
        step = load_checkpoint(path, fresh)
        after = fresh(ids).logits
        assert step == 3
        torch.testing.assert_close(before, after)

    def test_weight_tying(self, model: MambaLMHeadModel) -> None:
        """Embedding and output projection share the same tensor."""
        assert model.lm_head.weight is model.backbone.embedding.weight

    def test_no_positional_encoding_needed(self, model: MambaLMHeadModel) -> None:
        """No parameter encodes absolute position."""
        names = [n.lower() for n, _ in model.named_parameters()]
        assert not any(
            ("pos" in n) or ("position" in n) or ("time_embed" in n) for n in names
        )

    def test_variable_length_sequences_no_padding(
        self, model: MambaLMHeadModel
    ) -> None:
        """Different sequence lengths run without any padding."""
        for length in (1, 5, 33, 64):
            ids = torch.randint(0, 64, (1, length))
            assert model(ids).logits.shape == (
                1,
                length,
                model.config.padded_vocab_size,
            )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fp16_bf16_numerical_stability(
        self, model: MambaLMHeadModel, dtype: torch.dtype
    ) -> None:
        """Forward is finite in float16 and bfloat16."""
        model = model.to(dtype)
        ids = torch.randint(0, 64, (1, 12))
        logits = model(ids).logits
        assert logits.dtype == dtype
        assert torch.isfinite(logits).all()
