"""Unit tests for the --save-logprobs path through _generate_batch and the
per-row dump. Uses a fake model + tokenizer to avoid loading Gemma."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeBatchEncoding(dict):
    """Mimics HF BatchEncoding: dict-like + has .to() that returns self."""

    def __init__(self, input_ids: torch.Tensor):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, _device):
        return self


def _make_tokenizer(prompt_len: int = 3, batch_size: int = 1):
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.apply_chat_template = lambda msgs, **_kw: "<chat>"
    ids = torch.tensor([[2, 3, 4][:prompt_len]] * batch_size)
    tok.return_value = FakeBatchEncoding(input_ids=ids)
    tok.decode = lambda ids, skip_special_tokens=False: (
        f"tok{ids[0]}" if isinstance(ids, list) else f"tok{ids}"
    )
    return tok


def test_generate_batch_returns_logprobs_when_requested() -> None:
    """Smoke-test the new logprobs path: shape correctness + EOS truncation."""
    from train import _generate_batch

    tok = _make_tokenizer(prompt_len=3, batch_size=2)

    # 2 rows; row 0 hits EOS at step 1, row 1 runs full 4 steps without EOS.
    sequences = torch.tensor([
        [2, 3, 4, 5, 1, 0, 0],
        [2, 3, 4, 6, 7, 8, 9],
    ])
    vocab = 10
    scores = []
    for step in range(4):
        s = torch.full((2, vocab), -1e3)
        for row, tok_id in enumerate(sequences[:, 3 + step]):
            s[row, tok_id.item()] = 0.0  # logit puts ~all mass on actual chosen token
            s[row, (tok_id.item() + 1) % vocab] = -2.0
        scores.append(s)
    model = MagicMock()
    model.generate.return_value = SimpleNamespace(sequences=sequences, scores=scores)

    completions, logprobs = _generate_batch(
        model, tok,
        [[{"role": "user", "content": "x"}], [{"role": "user", "content": "y"}]],
        max_completion_length=4, logprobs_top_k=3,
    )
    assert len(completions) == 2
    assert len(logprobs) == 2
    assert len(logprobs[0]) == 2, f"row 0 should truncate at EOS, got {len(logprobs[0])}"
    assert len(logprobs[1]) == 4, f"row 1 should keep all 4 steps, got {len(logprobs[1])}"

    for row in logprobs:
        for step in row:
            assert len(step) == 3, "top_k=3 entries per step"
            for tid, tstr, lp in step:
                assert isinstance(tid, int)
                assert isinstance(tstr, str)
                assert isinstance(lp, float)

    # Top-1 prediction at row-0/step-0 should be the actually-sampled token (id=5).
    assert logprobs[0][0][0][0] == 5


def test_generate_batch_back_compat_no_logprobs_arg() -> None:
    """Without logprobs_top_k, return type is still list[str] (back-compat for training callers)."""
    from train import _generate_batch

    tok = _make_tokenizer(prompt_len=3, batch_size=1)
    model = MagicMock()
    model.generate.return_value = torch.tensor([[2, 3, 4, 5, 6, 1]])

    out = _generate_batch(model, tok, [[{"role": "user", "content": "x"}]], max_completion_length=3)
    assert isinstance(out, list)
    assert all(isinstance(c, str) for c in out)


def test_generate_batch_empty_messages_list() -> None:
    """Empty input returns the right shape under both modes."""
    from train import _generate_batch

    tok = _make_tokenizer()
    model = MagicMock()
    assert _generate_batch(model, tok, [], 4) == []
    assert _generate_batch(model, tok, [], 4, logprobs_top_k=5) == ([], [])
