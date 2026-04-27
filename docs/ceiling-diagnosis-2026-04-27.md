# v2 ceiling diagnosis — 2026-04-27

**Status:** complete after 17 experiments across 5 sweeps (`v2_baseline`, `v2_lr_explore`, `v2_step_time_relax`, `v2_cap_neg_tails`, `v2_data_regen`).

**Two causes confirmed empirically — both load-bearing for the mean_total ~9.6 ceiling:**

1. **Data-side** — the original dataset's 87% 1-trigger skew capped f1's *absolute* ceiling at ~7.45. Multi-trigger gradient signal didn't exist. **Confirmed by E16:** the regenerated dataset (45/30/20/4/1 distribution) lifted f1 to **7.523** — first config across 17 experiments to break 7.5.
2. **Rubric-side** — softening the `no_halluc` / `prev_amount` penalties makes f1 a TRADE-OFF variable: the model retreats from f1 to gain ground on hallucination + structure. **Confirmed by E14:** half-capped rubric drove f1 down to 6.47 while no_halluc improved to its best-ever -0.156.

**Why mean_total still didn't break 9.6:** the harder dataset (E15/E16) introduced a NEW failure mode the model hasn't learned to handle — multi-fact contexts trigger MORE hallucination (no_halluc -0.33 → -0.92). Data lifts f1, then loses the gain on no_halluc.

The ceiling is a **multi-knob equilibrium**, not a single bottleneck.

## What we tried

| sweep | experiments | hypothesis | outcome |
|---|---|---|---|
| `v2_baseline` | E0-E3 (4 iters) | v2 stack delivers; explore lr / beta / lora_rank | E1 `lr=5e-6` champion, mean_total=**9.611**. ceiling first observed. |
| `v2_lr_explore` | E4-E9 (6 iters, 3 useful) | longer training / ng=24 / max_completion=512 / β=0.03 / kitchen-sink can break ceiling | None beat E1. `max_completion=512` *hurt*; `β=0.03` marginal; `ng=24` triage-killed twice (false positive at SLOW_MEAN_S=90). |
| `v2_step_time_relax` | E10-E12 (3 iters) | retry the missed long-run + seed-variance + ng=24 with relaxed triage | Long-run plateaued at 9.41 (lower than E1). seed=1337 dropped to 9.02 (variance ±0.6 across seeds). ng=24 finished cleanly at 8.98. |
| `v2_cap_neg_tails` | E13-E14 (4 iters, 2 useful) | rubric was masking learning gains; cap -3 fail penalties to -1 | E13 anchor: mean_total=9.463, f1=7.39, no_halluc=-0.42. E14 long-run: **mean_total=8.96, f1=6.47 (lowest), no_halluc=-0.156 (BEST EVER), well_formed=+0.144 (matches E10's high), pass_all=29.9%**. Bug: cap only patched tariff-fail path, pct-fail kept at -3 (fixed in 59d9ad3). Headline read: trade-off — model retreats from f1 to gain no_halluc + well_formed. |
| `v2_data_regen` | E15-E16 (2 iters) | new dataset (45/30/20/4/1 trigger distribution) under uncapped rubric — does multi-trigger gradient lift f1 past 7.5? | E15 anchor: mean_total=8.78, f1=7.34, **pass_all=7.3% (catastrophic)**, no_halluc=**-0.90 (worst)**. E16 long-run: **mean_total=9.08, f1=7.52 ⭐ FIRST TO BREAK 7.5, no_halluc=-0.92, pass_all=13.7%**. Verdict: data-side cause CONFIRMED for f1; harder data also tanks no_halluc (model hallucinates more on multi-fact contexts), explaining why mean_total stayed below E1's 9.611. |

## The two signals that nailed it

### Signal 1 — f1 clusters around 7.0-7.5 across most configs

- E0 (anchor)              → f1=6.88
- E1 (champion lr=5e-6)    → f1=7.45
- E10 (long-run)           → f1=6.92
- E12 (ng=24)              → f1=6.88
- E13 (cap-neg-tails)      → f1=7.39

When most RL configs cluster around the same f1 number, the *absolute* ceiling looks data-bound (see distribution analysis below).

### Signal 2 — f1 is also a TRADE-OFF VARIABLE (E14)

E14 (long-run + half-capped rubric) broke the cluster:

| metric | E1 (uncapped) | E13 (½-capped, anchor) | **E14 (½-capped, long)** |
|---|---|---|---|
| mean_total | 9.611 | 9.463 | **8.961** |
| f1_triggers | 7.45 | 7.389 | **6.473** ← traded down |
| no_halluc mean | -0.328 | -0.424 | **-0.156** ← BEST EVER |
| well_formed mean | -0.001 | -0.01 | **+0.144** ← matches E10's high |
| pass_all | 24.8% | 22.2% | **29.9%** |

When the no_halluc penalty was softened, the model retreated from f1 to gain ground on hallucination + structure. Per-rubric quality went **up**, but mean_total went **down** because f1's max=10 dominates the sum.

**So the ceiling has two causes:** (1) the dataset limits f1 *absolute* on 1-trigger, but (2) f1 also moves *relative* to other rubric weights. Reward shaping shifts the equilibrium without lifting the ceiling.

## The data-distribution finding

`data/dd_dataset_20260424T174610Z_5500rows.jsonl`:

```
trigger count distribution (n=5501)
  1 triggers: 4772 (86.7%)
  2 triggers:  728 (13.2%)
  3+ triggers:   0  (0.0%)
```

**86.7% of training rows are 1-trigger.** For 1-trigger examples, F1 is binary: 1.0 if the model picks the correct trigger, 0.0 otherwise. There is no in-between gradient signal.

### Why this caps f1 at exactly ~7.45

If 1-trigger F1 ≈ 0.85 (model has saturated single-trigger accuracy) and 2-trigger F1 ≈ 0.40 (typical for under-trained multi-label):

```
weighted F1 = 0.867 × 0.85 + 0.132 × 0.40 = 0.79
reward      = 0.79 × 12 - 2 = 7.48
```

Matches observed ceiling almost exactly. The model has learned what this dataset can teach.

### What reward shaping actually did (revised view, after E14)

Capping the -3 hallucination penalty *did* improve no_halluc + well_formed scores (E14: best of any run). It just moved the model along the f1↔(no_halluc, well_formed) trade ridge. That's a real per-rubric quality improvement that would matter for downstream use, but mean_total — dominated by f1's max=10 — drops.

Lesson: rubric softening is the right lever for **per-rubric quality**, not for **mean_total ceiling-breaking**. The mean_total ceiling needs *both* better data (raise f1's absolute) AND a rubric balance that doesn't let the model trade f1 away.

## What unlocking it should look like

Adding 3+ trigger examples does three things at once:

1. **Continuous F1 surface** — a 3-trigger example can score F1 in {0.0, 0.4, 0.67, 0.8, 1.0}. Real partial-credit gradient instead of binary.
2. **Recall transfer** — multi-trigger learning improves 2-trigger F1 too. The 13% slice currently underperforms; it should converge upward.
3. **Richer context for other rubrics** — multi-trigger examples typically have more contract changes / amount data / tariff names. May finally give `prev_amount_correct` (stuck at 0 across all 14 runs) something to attempt.

### Target distribution

| trigger count | current | target |
|---|---|---|
| 1 | 86.7% | 40% (sanity baseline) |
| 2 | 13.2% | 30% |
| 3 | 0% | 20% |
| 4+ | 0% | 10% |

Approx 5500 rows total (preserves dataset size; substitutes hard for easy).

## Decision (updated post-E16)

The data-side hypothesis was correct: f1 climbed from 7.45 (E1, old data) to **7.52 (E16, new data)** — first config across 17 experiments to break 7.5. But mean_total stayed at 9.08 (below E1's 9.611) because the harder dataset triggers MORE hallucination (no_halluc -0.33 → -0.92).

The mean_total ceiling has now been mapped to a **multi-knob equilibrium**:
- Data composition controls f1's absolute ceiling
- Rubric weights control where the model lands on the f1 ↔ {no_halluc, well_formed} trade ridge
- Multi-fact contexts in harder data introduce new hallucination failure modes the model hasn't learned to handle

**Next move — push hallucination out of the model.** The reward landscape can't simultaneously incentivize high recall AND low hallucination on multi-fact contexts. The encoder-outlier idea (see "Deferred future approaches" below) bypasses this by routing OOD inputs to a structured "insufficient context" response instead of running the LLM at all. That moves the no_halluc problem out of the equilibrium entirely.

Other deferrable levers if encoder approach blocked:
- LoRA `target_modules` expansion to MLP (capacity for arithmetic — `prev_amount` still stuck at 0)
- Base-model swap to a stronger-grounded foundation
- Train-time data augmentation: pair multi-fact contexts with explicit "do not invent facts not in context" instruction

## Reference rows

| experiment | rubric | data | mean_total | f1_triggers | no_halluc | well_formed | pass_all |
|---|---|---|---|---|---|---|---|
| **E1 (mean_total champ)** | 2026-04-26-soften-well-formed | old (87% 1-trigger) | **9.611** | 7.45 | -0.328 | -0.001 | 24.8% |
| E10 (long-run) | 2026-04-26-soften-well-formed | old | 9.41 | 6.92 | (—) | 0.155 | 34.5% |
| E13 (½-capped anchor) | 2026-04-27-cap-neg-tails | old | 9.463 | 7.389 | -0.424 | -0.01 | 22.2% |
| E14 (½-cap long-run) | 2026-04-27-cap-neg-tails | old | 8.961 | 6.473 | -0.156 | +0.144 | 29.9% |
| E15 (data-regen anchor) | 2026-04-26-soften-well-formed | new (45/30/20/4/1) | 8.778 | 7.343 | -0.896 | -0.123 | 7.3% |
| **E16 (data-regen long-run)** | 2026-04-26-soften-well-formed | new | 9.082 | **⭐ 7.523** | **-0.920** | +0.026 | 13.7% |

W&B project: <https://wandb.ai/chaleong/gemma4-rlvr>
