"""Integration test: gemma4's --save-logprobs JSONL → autoresearch.token_confidence.

Proves the per-row schema written by `scripts/two_stage_eval.py --save-logprobs N`
is consumed correctly by `autoresearch.token_confidence`'s loader and
summariser. Synthesises the JSONL inline (no model required) so the test
stays fast and deterministic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from autoresearch.token_confidence import (
    bucket_by_failure,
    load_per_row_logprobs,
    summarize_confidence,
)


def _make_per_row(
    out: Path,
    *,
    arm: str = "two_stage",
    rows: list[dict],
) -> Path:
    """Write a JSONL matching gemma4's per_row.jsonl shape (with --save-logprobs)."""
    path = out / "synthetic.per_row.jsonl"
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def _step(top1_lp: float, distractors: list[float] | None = None, top1_id: int = 100) -> list:
    """One step's top-K = [(id, str, logp), ...] with the sampled token at top-1."""
    distractors = distractors or [-3.0, -4.0, -5.0]
    out = [(top1_id, f"tok{top1_id}", top1_lp)]
    for k, lp in enumerate(distractors):
        out.append((top1_id + 100 + k, f"tok{top1_id + 100 + k}", lp))
    return out


def test_load_per_row_consumes_gemma4_save_logprobs_shape(tmp_path: Path) -> None:
    """End-to-end: gemma4-shaped row → load_per_row_logprobs → Sample with logprobs."""
    rows = [
        {
            "i": 0,
            "ground_truth_triggers": ["trig_a"],
            "stage1_triggers": ["trig_a"],
            "input_json": {"foo": "bar"},
            "completions": {
                "two_stage": '{"explanations": ["..."]}',
                "vanilla": "ignored",
            },
            "scores": {
                "two_stage": {
                    "schema_valid": 1.0,
                    "f1_triggers": 1.0,
                    "no_hallucinated_facts": 1.0,
                    "well_formed": 1.0,
                },
                "vanilla": {"schema_valid": 1.0},
            },
            "logprobs": {
                "two_stage": [_step(top1_lp=-0.01), _step(top1_lp=-0.02), _step(top1_lp=-0.05)],
                "vanilla": [_step(top1_lp=-0.5)],
            },
        }
    ]
    path = _make_per_row(tmp_path, rows=rows)
    samples = load_per_row_logprobs(path, arm="two_stage")
    assert len(samples) == 1
    s = samples[0]
    assert s.i == 0
    assert s.completion.startswith('{"explanations"')
    # arm-keyed scores extracted
    assert s.scores["f1_triggers"] == 1.0
    assert s.scores["schema_valid"] == 1.0
    # arm-keyed logprobs extracted
    assert len(s.logprobs) == 3
    assert s.logprobs[0][0][2] == pytest.approx(-0.01)
    # vanilla arm not bled in
    assert "ignored" not in s.completion


def test_load_per_row_arm_isolation(tmp_path: Path) -> None:
    """Loading one arm doesn't pull in another arm's logprobs."""
    rows = [
        {
            "i": 0,
            "completions": {"a": "x", "b": "y"},
            "scores": {"a": {"k": 1.0}, "b": {"k": 0.0}},
            "logprobs": {
                "a": [_step(top1_lp=-0.01)],
                "b": [_step(top1_lp=-2.0), _step(top1_lp=-3.0)],
            },
        }
    ]
    path = _make_per_row(tmp_path, rows=rows)
    arm_a = load_per_row_logprobs(path, arm="a")
    arm_b = load_per_row_logprobs(path, arm="b")
    assert len(arm_a) == 1 and len(arm_b) == 1
    assert len(arm_a[0].logprobs) == 1
    assert len(arm_b[0].logprobs) == 2
    assert arm_a[0].completion == "x"
    assert arm_b[0].completion == "y"


def test_load_per_row_drops_rows_missing_logprobs(tmp_path: Path) -> None:
    """Rows without a logprobs field for the requested arm are silently dropped
    (so a partial save-logprobs run doesn't crash the loader)."""
    rows = [
        {
            "i": 0,
            "completions": {"two_stage": "ok"},
            "scores": {"two_stage": {"k": 1}},
            "logprobs": {"two_stage": [_step(top1_lp=-0.1)]},
        },
        {
            "i": 1,
            "completions": {"two_stage": "no_lp"},
            "scores": {"two_stage": {"k": 0}},
            # no logprobs field at all
        },
    ]
    path = _make_per_row(tmp_path, rows=rows)
    samples = load_per_row_logprobs(path, arm="two_stage")
    assert len(samples) == 1  # row 1 dropped
    assert samples[0].i == 0


def test_summarize_confidence_on_high_confidence_sample(tmp_path: Path) -> None:
    """A sample with all near-zero logprobs (very confident) should report
    high mean prob + low entropy + low pct_low_prob."""
    rows = [
        {
            "i": 0,
            "completions": {"two_stage": "x"},
            "scores": {"two_stage": {"k": 1.0}},
            "logprobs": {
                "two_stage": [_step(top1_lp=-0.001) for _ in range(10)],
            },
        }
    ]
    samples = load_per_row_logprobs(_make_per_row(tmp_path, rows=rows), arm="two_stage")
    summary = summarize_confidence(samples[0], low_thresh=0.5)
    assert summary.n_tokens == 10
    assert summary.mean_prob > 0.99
    assert summary.mean_entropy < 0.5  # nats
    assert summary.pct_low_prob == 0.0
    assert summary.lowest_positions  # always returns *some* positions, even for confident


def test_summarize_confidence_on_low_confidence_sample(tmp_path: Path) -> None:
    """A sample with deeply negative logprobs (gambling at high entropy)
    should report low mean prob + non-zero pct_low_prob."""
    rows = [
        {
            "i": 0,
            "completions": {"two_stage": "x"},
            "scores": {"two_stage": {"k": 0.0}},
            "logprobs": {
                # logprob -2.5 → prob ≈ 0.082 (well below 0.5 threshold)
                "two_stage": [_step(top1_lp=-2.5, distractors=[-2.6, -2.7, -2.8]) for _ in range(5)],
            },
        }
    ]
    samples = load_per_row_logprobs(_make_per_row(tmp_path, rows=rows), arm="two_stage")
    summary = summarize_confidence(samples[0], low_thresh=0.5)
    assert summary.n_tokens == 5
    assert summary.mean_prob == pytest.approx(math.exp(-2.5), rel=1e-3)
    assert summary.pct_low_prob == 1.0  # all 5 below threshold
    assert summary.mean_entropy > 1.0  # 4 near-equal logprobs ≈ ln(4) ≈ 1.39 nats


def test_bucket_by_failure_groups_samples_by_gate_violations(tmp_path: Path) -> None:
    """Two samples, two failing different gates → two singleton buckets."""
    rows = [
        {
            "i": 0,
            "completions": {"two_stage": "ok"},
            "scores": {
                "two_stage": {"f1_triggers": 1.0, "no_hallucinated_facts": 0.5},
            },
            "logprobs": {"two_stage": [_step(top1_lp=-0.1)]},
        },
        {
            "i": 1,
            "completions": {"two_stage": "ok2"},
            "scores": {
                "two_stage": {"f1_triggers": 0.4, "no_hallucinated_facts": 1.0},
            },
            "logprobs": {"two_stage": [_step(top1_lp=-0.1)]},
        },
    ]
    samples = load_per_row_logprobs(_make_per_row(tmp_path, rows=rows), arm="two_stage")
    gates = {"f1_triggers": 0.5, "no_hallucinated_facts": 1.0}
    buckets = bucket_by_failure(samples, gates)
    # Sample 0 fails no_hallucinated_facts (0.5 < 1.0), passes f1_triggers (1.0 ≥ 0.5)
    # Sample 1 fails f1_triggers (0.4 < 0.5), passes no_hallucinated_facts (1.0 ≥ 1.0)
    assert frozenset({"no_hallucinated_facts"}) in buckets
    assert frozenset({"f1_triggers"}) in buckets
    assert len(buckets[frozenset({"no_hallucinated_facts"})]) == 1
    assert len(buckets[frozenset({"f1_triggers"})]) == 1
    assert buckets[frozenset({"no_hallucinated_facts"})][0].i == 0
    assert buckets[frozenset({"f1_triggers"})][0].i == 1
