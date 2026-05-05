# Token-confidence diagnostic — reference plots

Two reference plots from the first end-to-end run of the
`autoresearch.token_confidence` pipeline (PR #25 capture + autoresearch
v0.19.3 analysis), kept here for the writeup record.

## `smoke_n20_distribution.png`

Single-run histogram of per-row mean prob, one stack per failure bucket.
Generated from `data/smoke_logprobs_n20.per_row.jsonl` (PR-F + v4 + relaxed
rubric, n=20, captured live with `--save-logprobs 5`).

All 20 rows have at least one gate failure (strict 3-gate set including
perfect F1=10), and every bucket clusters at mean prob 0.94–1.00. Zero
density anywhere in the 0.0–0.94 range.

## `smoke_vs_n1000_comparison.png`

Two-panel side-by-side:

* **Left** — per-row mean-prob distribution overlaid for two runs:
  - PR-F + v4 smoke n=20 (live LMFE-constrained sampling)
  - 100-row teacher-forced sample of the PR-F + v4 n=1000 dump
  - Live sampling sits at ~0.95–0.97; teacher-forced at ~0.87–0.93.
    The ~0.05 offset reflects the methodological difference (LMFE jamming
    boosts sampled-token prob; teacher-forcing returns the unconstrained
    belief).
* **Right** — per-bucket mean prob, grouped bars across the two runs.
  At n=100 the `passes_all` bucket appears (12 rows, mean prob 0.911) and
  is **statistically indistinguishable** from the failing buckets (range
  0.87–0.92).

## Headline finding

**Confidence does not separate good from bad outputs.** A passing row
looks indistinguishable from a hallucinating row in terms of how sure
the model is. This rules out two would-be cheap fixes — confidence-based
abstention and sampling-temperature tweaks — and confirms the next move
must be either a Stage 2 GRPO retrain on the v2 rubric or a slot-aware
prose hallucination check.

## Reproducing

```bash
# Capture: produces data/smoke_logprobs_n20.per_row.jsonl
uv run python scripts/two_stage_eval.py \
    --classifier-path data/trigger_classifier_v4_mlp.pt \
    --lora-path gemma_4_lora/train_v2_80gb/exp_27 \
    --eval-heldout-n 20 --batch-size 4 \
    --constrain-facts --enforce-slots \
    --arms two_stage --dump-per-row --save-logprobs 5 \
    --out data/smoke_logprobs_n20.json

# Single-run analysis (smoke_n20_distribution.png)
uv run python -m autoresearch.token_confidence summary \
    --per-row data/smoke_logprobs_n20.per_row.jsonl \
    --arm two_stage \
    --gate well_formed=0.5 --gate no_hallucinated_facts=1.0 --gate f1_triggers=10.0 \
    --out reports/token_confidence_v0193/

# Multi-run comparison: requires teacher-forced backfill of the n=1000 dump.
# See /tmp/score_to_logprobs.py and /tmp/plot_confidence_comparison.py for
# the helper scripts used to render smoke_vs_n1000_comparison.png.
```
