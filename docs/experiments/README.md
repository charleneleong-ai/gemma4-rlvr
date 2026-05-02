# Per-sweep experiment writeups

Each markdown file in `docs/experiments/<config_name>/<schedule_name>.md` documents
one sweep — the hypothesis, the schedule yaml, the per-iter results, the verdict,
and what the next sweep should test. The cross-sweep narrative lives in
[`../ceiling-diagnosis-2026-04-27.md`](../ceiling-diagnosis-2026-04-27.md).

## Layout

```
docs/experiments/
└── <task>/
    └── <config_name>/
        └── <schedule_name>.md   # one writeup per sweep yaml in configs/schedules/
```

Mirrors `experiments/<task>/<config_name>/results.jsonl` 1:1 — `<task>`,
`<config_name>`, and `<schedule_name>` are the three keys you can trace from
any artefact (results row, schedule yaml, runtime chart, doc) to the others.

## What goes in a sweep writeup

The autoresearch package emits the skeleton via
`uv run autoresearch-report --schedule <name> --config <name>` (see
[`autoresearch.report`](https://github.com/charleneleong-ai/autoresearch/blob/main/src/autoresearch/report.py)).
The skeleton inlines the schedule yaml, fills in the per-iter results table
from `results.jsonl`, and **auto-detects the chassis** (model name, LoRA
rank, max_seq_length, num_generations) from `configs/<config>.yaml` —
walking the Hydra `defaults:` chain so inherited fields land in the header.

The author fills in:

1. **Hypothesis** — what mechanism are we testing, and what would falsify it?
2. **Schedule** — the configs/schedules YAML inline, with knobs annotated.
3. **Pre-launch comparisons** — which prior experiments are the baseline.
4. **Results** — table of per-iter rows from `results.jsonl`.
5. **Verdict** — did the hypothesis hold? What does this mean for the next move?
6. **Next move** — pointer to the next sweep (or a "ceiling reached, pivot" note).

The skeleton is short on purpose. Cross-sweep synthesis belongs in the
diagnosis doc, not in any single sweep writeup.

## Index

### `dd_explainer` / `train_v2_80gb` (Gemma 4 4B dense + new dataset on 80GB A100 PCIe)

- [`v2_data_regen.md`](dd_explainer/train_v2_80gb/v2_data_regen.md) — first test of the regenerated multi-trigger dataset (E15-E16). f1 ceiling broken at 7.523.
- [`v2_no_halluc_weighted.md`](dd_explainer/train_v2_80gb/v2_no_halluc_weighted.md) — bracket the f1 ↔ no_halluc trade ridge with reward weights ×{2,3} (E17-E20). E18 = branch champ at 7.745.
- [`v2_granular_no_halluc.md`](dd_explainer/train_v2_80gb/v2_granular_no_halluc.md) — replace binary +1/-3 with per-fact granular reward (E21-E22). Plateau confirmed; reward-side exhausted.

### `dd_explainer` / `encoder_outlier` (no Gemma training — frozen-encoder OOD gate)

- [`v0_gate.md`](dd_explainer/encoder_outlier/v0_gate.md) — pre-flight encoder gate that routes OOD account contexts to "insufficient context" before Gemma generates. **Verdict: falsified for `mean_total`** (the rubric weights f1 over no_halluc so substituting fallback always loses), but the gate works as a component (heldout AUROC 0.879 across 11 mutations, no_halluc lifts +0.336 at threshold=0.5). Branch `feat/encoder-outlier-gate` merged into main.

### `dd_explainer` / `train_v3_26b_a4b` (Gemma 4 26B-A4B MoE — base model swap)

**Verdict: falsified 2026-04-29.** Three iters across three LRs revealed a structural MoE-LoRA gradient pathology — any LR ≥ 5e-6 explodes within 2 steps (KL > 8000), and the only stable LR (2e-6) trains too slowly to catch v2 (E7 plateau'd at mean_total=7.625 vs E18's 9.324). Postmortem in [`docs/v3-26b-a4b-migration.md`](../v3-26b-a4b-migration.md#postmortem-added-2026-04-29-after-falsification).

- [`v3_baseline.md`](dd_explainer/train_v3_26b_a4b/v3_baseline.md) — full per-iter trajectory + verdict + LR-landscape diagnosis.

### `dd_explainer` / `two_stage` (architectural decoupling — Stage 1 classifier + Stage 2 LLM explainer) ⭐ NEW CHAMPION

**Verdict: confirmed 2026-04-29 — first config across 26 experiments to break mean_total = 10.** Stage 1 frozen `bge-small` + 9 numeric features + 6 trigger-discriminator features + linear head hits classifier rubric=8.77 (vs E18's f1=7.745). End-to-end A/B at n=1000 with Stage 1's trigger predictions injected into E18's prompt as a templated response skeleton lands **mean_total=10.293, +1.939 over vanilla E18 (4-bit inference, both passes)**. f1 lift is +1.97 (the architectural-decoupling effect); no_halluc essentially flat (-0.036). Net win is from f1 by construction, not from the trade-ridge being resolved.

- [`v0_pipeline.md`](dd_explainer/two_stage/v0_pipeline.md) — Stage 1 variant exploration (bge-small/base/qwen3-0.6B + explicit features), Stage 2 wiring, n=1000 A/B verdict, sub-claim breakdown, next-move options. v4_mlp + E18 hit mean_total=10.816.
- [`v1_constrained.md`](dd_explainer/two_stage/v1_constrained.md) — prompt-time fact-grounding constraint on top of v4_mlp + E18. **mean_total = 10.961 (new champion), no_halluc lifted -0.848 → -0.732** (first move past the plateau). Partial-win on no_halluc (didn't clear -0.5 huge-win threshold); logit masking is the next escalation if production needs it.
