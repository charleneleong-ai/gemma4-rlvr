# PR-G E28 Retrospective

- Iter: `pr_g_e28_20260505_143955` (2026-05-05T18:40:07.978187+00:00)
- WandB: https://wandb.ai/chaleong/gemma4-rlvr/runs/inafr9fs
- Steps: 160, runtime: 239.1 min, status: KEEP
- Train metrics: best_reward=19.500, final_kl=0.0800, final_loss=+0.0012

## Eval (n=1000, two_stage, --constrain-facts --enforce-slots)
- **mean_total: 14.396**  (rubric `2026-05-05-well-formed-relaxed-to-4`)
- pass_all: 258/1000 (25.8%)
- f1_triggers (mean): 9.067 | prev_amount_correct (mean): 2.000
- no_hallucinated_facts (row-level mean): 0.601
- no_hallucinated_facts_slots (slot-only mean): 1.000
- underpayment_ok (mean): 0.492 | well_formed (mean): 0.237

## vs PR-F E27 champion (mean_total=14.014, no_halluc=1.000, pass_all=11.7%)
- mean_total: 14.014 → **14.396**  (Δ +0.382)
- no_halluc (row-level): 1.000 → 0.601  (Δ -0.399)
- no_halluc (slot-only): 1.000 → **1.000**  (preserved)
- pass_all_pct: 11.7 → **25.8**  (Δ +14.1 pp)

## Trade-off
Slot-side hallucination is still perfect (`no_hallucinated_facts_slots_mean=1.000`) — LMFE force-populate guarantees that. The drop on row-level `no_hallucinated_facts` (0.601) reflects extra prose around the slots that introduces unsupported claims in 39.9% of rows. mean_total still nets +0.382 over PR-F because pass_all more than doubles (11.7% → 25.8%) — the constrain-facts retrain produced more rows that clear the full rubric.

## Findings (autoresearch.retrospective)

_No automated findings raised by the 7 builtin detectors — train run looked clean._

(Detectors run: bucketed_failure, eval_score_plateau, gradient_collapse, sign_flip_in_rubric, silent_kill, triage_threshold_mismatch, value_transform_mismatch)
