# Per-sweep experiment writeups

Each markdown file in `docs/experiments/<config_name>/<schedule_name>.md` documents
one sweep — the hypothesis, the schedule yaml, the per-iter results, the verdict,
and what the next sweep should test. The cross-sweep narrative lives in
[`../ceiling-diagnosis-2026-04-27.md`](../ceiling-diagnosis-2026-04-27.md).

## Layout

```
docs/experiments/
└── <config_name>/
    └── <schedule_name>.md   # one writeup per sweep yaml in configs/schedules/
```

`<config_name>` mirrors `experiments/<task>/<config_name>/results.jsonl` and
`<schedule_name>` mirrors `configs/schedules/<schedule_name>.yaml` so you can
trace from any artefact (results row, schedule yaml, doc) to the other two.

## What goes in a sweep writeup

The autoresearch package emits the skeleton via
`uv run autoresearch-report --schedule <name> --config <name>` (see
[`autoresearch.report`](https://github.com/charleneleong-ai/autoresearch/blob/main/src/autoresearch/report.py)).
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

### `train_v2_80gb` (Gemma 4 + new dataset on 80GB A100 PCIe)

- [`v2_data_regen.md`](train_v2_80gb/v2_data_regen.md) — first test of the regenerated multi-trigger dataset (E15-E16). f1 ceiling broken at 7.523.
- [`v2_no_halluc_weighted.md`](train_v2_80gb/v2_no_halluc_weighted.md) — bracket the f1 ↔ no_halluc trade ridge with reward weights ×{2,3} (E17-E20). E18 = branch champ at 7.745.
- [`v2_granular_no_halluc.md`](train_v2_80gb/v2_granular_no_halluc.md) — replace binary +1/-3 with per-fact granular reward (E21-E22). Plateau confirmed; reward-side exhausted.
