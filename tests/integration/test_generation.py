"""Integration tests for the decoding controls in :mod:`mamba.utils.generation`.

Cover EOS stopping, token budgeting, the repetition penalty, the greedy/top-k/
top-p sampling paths, batch consistency, streaming, and -- crucially -- the
constant-size recurrent state that gives Mamba its :math:`O(1)` decode memory.
"""

from __future__ import annotations

import pytest
import torch

from mamba.config import MambaConfig
from mamba.models.lm_head import MambaLMHeadModel
from mamba.utils.generation import InferenceParams, generate


@pytest.fixture
def model() -> MambaLMHeadModel:
    torch.manual_seed(0)
    cfg = MambaConfig(
        d_model=32, n_layers=2, d_state=8, vocab_size=64, pad_vocab_size_multiple=8
    )
    return MambaLMHeadModel(cfg).eval()


class TestGeneration:
    def test_max_new_tokens_respected(self, model: MambaLMHeadModel) -> None:
        """Output length equals prompt length plus max_new_tokens."""
        prompt = torch.randint(0, 64, (1, 5))
        out = generate(model, prompt, max_new_tokens=12, do_sample=False)
        assert out.shape == (1, 17)

    def test_eos_stopping(self, model: MambaLMHeadModel) -> None:
        """Generation halts early once every sequence has emitted EOS.

        We force EOS by penalising every other token to ``-inf`` via a tiny
        wrapper; here we simply check the API path with a plausible eos id and a
        large budget terminates at or before the budget.
        """
        prompt = torch.randint(0, 64, (1, 4))
        # Greedy: find the token the model will emit, set it as EOS so it stops.
        first_logits = model(prompt).logits[:, -1, :]
        eos = int(first_logits.argmax(dim=-1).item())
        out = generate(
            model, prompt, max_new_tokens=20, do_sample=False, eos_token_id=eos
        )
        # The first generated token is EOS, so exactly one token is appended.
        assert out.shape[1] == prompt.shape[1] + 1
        assert int(out[0, -1].item()) == eos

    def test_repetition_penalty_reduces_repeats(self, model: MambaLMHeadModel) -> None:
        """A strong repetition penalty produces no fewer unique tokens."""
        prompt = torch.randint(0, 64, (1, 4))
        plain = generate(model, prompt, max_new_tokens=24, do_sample=False)
        penalized = generate(
            model, prompt, max_new_tokens=24, do_sample=False, repetition_penalty=2.0
        )
        uniq_plain = len(set(plain[0, 4:].tolist()))
        uniq_pen = len(set(penalized[0, 4:].tolist()))
        assert uniq_pen >= uniq_plain

    def test_temperature_zero_is_greedy(self, model: MambaLMHeadModel) -> None:
        """temperature=0 reproduces explicit greedy decoding."""
        prompt = torch.randint(0, 64, (1, 5))
        greedy = generate(model, prompt, max_new_tokens=8, do_sample=False)
        temp0 = generate(
            model, prompt, max_new_tokens=8, do_sample=True, temperature=0.0
        )
        torch.testing.assert_close(greedy, temp0)

    def test_top_k_restricts_vocabulary(self, model: MambaLMHeadModel) -> None:
        """top_k=1 sampling is deterministic and equals greedy."""
        torch.manual_seed(3)
        prompt = torch.randint(0, 64, (1, 5))
        out_k1 = generate(
            model, prompt, max_new_tokens=8, do_sample=True, temperature=1.0, top_k=1
        )
        greedy = generate(model, prompt, max_new_tokens=8, do_sample=False)
        torch.testing.assert_close(out_k1, greedy)

    def test_top_p_nucleus_sampling(self, model: MambaLMHeadModel) -> None:
        """Nucleus sampling produces valid in-vocabulary tokens."""
        torch.manual_seed(4)
        prompt = torch.randint(0, 64, (2, 4))
        out = generate(model, prompt, max_new_tokens=6, do_sample=True, top_p=0.5)
        assert out.shape == (2, 10)
        assert out.min() >= 0 and out.max() < model.config.padded_vocab_size

    def test_batch_generation_consistent(self, model: MambaLMHeadModel) -> None:
        """Greedy decoding is independent across identical batch rows."""
        prompt = torch.randint(0, 64, (1, 5))
        single = generate(model, prompt, max_new_tokens=8, do_sample=False)
        batched = generate(
            model, prompt.repeat(4, 1), max_new_tokens=8, do_sample=False
        )
        for row in range(4):
            torch.testing.assert_close(batched[row], single[0])

    def test_streamer_receives_all_tokens(self, model: MambaLMHeadModel) -> None:
        """The streamer receives exactly max_new_tokens batches, then end()."""

        class CollectStreamer:
            def __init__(self) -> None:
                self.tokens: list[torch.Tensor] = []
                self.ended = False

            def put(self, token: torch.Tensor) -> None:
                self.tokens.append(token)

            def end(self) -> None:
                self.ended = True

        streamer = CollectStreamer()
        prompt = torch.randint(0, 64, (1, 4))
        generate(model, prompt, max_new_tokens=7, do_sample=False, streamer=streamer)
        assert len(streamer.tokens) == 7
        assert streamer.ended

    def test_inference_state_does_not_grow(self, model: MambaLMHeadModel) -> None:
        """Recurrent state shapes are constant across decode steps (O(1) memory)."""
        prompt = torch.randint(0, 64, (1, 6))
        params = InferenceParams(max_seqlen=30, max_batch_size=1)
        model(prompt, inference_params=params)
        params.seqlen_offset = prompt.shape[1]
        shapes_initial = {
            k: tuple(t.shape for t in v)
            for k, v in params.key_value_memory_dict.items()
        }
        tok = prompt[:, -1:]
        for _ in range(10):
            model(tok, inference_params=params)
            params.seqlen_offset += 1
        shapes_after = {
            k: tuple(t.shape for t in v)
            for k, v in params.key_value_memory_dict.items()
        }
        assert shapes_initial == shapes_after
