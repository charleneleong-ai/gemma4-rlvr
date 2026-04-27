# v2 80GB migration plan

**Status:** drafted 2026-04-26 on `feat/auto-research-loop`. Pick this up after spinning up a fresh A100-80GB on vast.ai.

**Budget assumption:** $57 across a few days, vast.ai 80GB at $0.812/hr ≈ 70 GPU-hours.

---

## Why we're switching

Seven non-trivial changes have stacked since the last clean baseline. None of the existing rows in `experiments/dd_explainer/results.jsonl` (`#0`–`#23`) reflect all of them:

1. Eval metrics now log to W&B (`eval/heldout/*`, `eval/regression/*`) — see `_log_eval_to_wandb` in `train.py`
2. `CompletionPreviewCallback` — periodic train+heldout completion samples to W&B as a scrubbable table
3. Per-experiment LoRA snapshots under `gemma_4_lora/exp_<N>/` — enables retro-eval
4. Option-A `pass_all` semantics (`prev_amount_correct ≥ 0` counts as pass — see `_PASS_ALL_THRESHOLD` in `train.py`)
5. Held-out-based promotion (`metrics.heldout.mean_total` instead of train reward — see `promotion_score()` in `experiments/experiment_progress.py`)
6. Softened `well_formed` rubric (row-mean instead of all-or-nothing AND — currently in PR #5 against `main`, mirrored to `feat/auto-research-loop`)
7. `RUBRIC_VERSION` constant recorded into eval aggregates — detects rubric-definition drift across runs

Combined with 80GB hardware enabling `num_generations=16`, `max_seq_length=8192`, and `eval_batch_size=64`, this is a clean cutover point for a "v2 baseline".

## Why 80GB specifically

Specific hardware swap: **1× A100 SXM4 40GB → 1× A100 PCIe 80GB** (vast.ai).

Per-iter savings + qualitative wins:

| capability | A100 SXM4 40GB | A100 PCIe 80GB | net |
|---|---|---|---|
| memory | 40 GB | 80 GB | unlocks larger batches/context/group sizes |
| HBM2e bandwidth | ~2.0 TB/s | ~1.94 TB/s | ~3% slower memory access |
| TDP | 400 W | 300 W | minor — could see 5% slower peak compute |
| eval batch (heldout) | 32 → ~25 min | 64 → ~13 min | -12 min/iter |
| `num_generations` (GRPO group size) | 8 max | 16 viable | tighter advantage estimates |
| `max_seq_length` | 4096 (trims regression) | 8192 (full coverage) | more failed-trace rows usable |
| `lora_rank` | 64 | 128 viable | marginal, rank=64 already plenty |
| training step time | ~70-90s baseline | ~75-95s expected (PCIe penalty) | **slight regression — important caveat** |

20% per-hour premium ($0.812 vs $0.6769) buys memory-driven qualitative wins, **not raw compute speed**. The PCIe form-factor and lower TDP mean per-step training is ~5-10% slower than the SXM4 40GB. The eval batch-size win still nets out positive overall (~5-8% faster per iter), but don't expect step-time numbers to drop on the new box.

If `step_time` per-iter logs jump above ~100s sustained on the new hardware, suspect throttling or a config issue — vast.ai PCIe instances are sometimes power-capped below the 300W spec.

---

## Pre-migration checklist

- [x] **Commit + push the in-flight changes** on `feat/auto-research-loop` — done (5 commits pushed 2026-04-26)
- [x] **PR #5** (`refactor/soften-well-formed-rubric`) merged to `main` — done; rubric softening is upstream
- [x] **In-flight work finished** on the 40GB box (2026-04-26 14:09 UTC):
  - Both `rerun_top5_remaining.sh` (PID 340857) and `retro_eval_optA` (PID 349153) exited cleanly
  - `experiments/dd_explainer/results.jsonl` populated through row `#23`
  - Surgical retro-eval patched `#20` with `rubric_version="2026-04-26-soften-well-formed"`
  - GPU idle (4 MiB / 0% util) — ready for migration
- [ ] **Push final progress.html screenshot** so the chart history is in git (`experiments/progress/<config>/progress.png`)

### v1 final scoreboard

Surgical retro-eval mode: only **#20** (the v1 best by train reward) was re-evaluated under the new rubric, to anchor v1↔v2 comparison without burning ~3 hrs. Other adapter snapshots remain on disk under `gemma_4_lora/exp_<N>/` for future expansion.

| exp | config | train score | heldout mean_total | heldout pass_all_pct | rubric_version |
|---|---|---|---|---|---|
| #19 | baseline (`train_fast`) | 11.500 | — | — | (snapshot only) |
| **#20** | **`num_generations=8` ← v1 ANCHOR** | **14.000** | **8.536** | **14.1%** (141/1000) | `2026-04-26-soften-well-formed` |
| #21 | `lr=5e-6 + num_gen=8` | 13.875 | — | — | (snapshot only) |
| #22 | `beta=0.02` (#5 retrain) | (in row) | (natural eval — old rubric) | — | (pre-soften) |
| #23 | `lr=2e-6` (#13 retrain) | 10.938 | 8.276 | 9.6% | `2026-04-26-soften-well-formed` |

**Headline v1-result observation:** `pass_all_pct` jumped from 0/1000 → 141/1000 (14.1%) on #20 under option-A semantics. The softened `well_formed` rubric also moved `mean_total` from 8.49 → 8.536 (small — this snapshot trained against the old binary AND, so it was already optimised for the strict version).

**v2 anchor needs to beat: `mean_total > 8.536` and/or `pass_all_pct > 14.1%`** on heldout to declare a win. If the v2 stack on 80gb hardware doesn't clear this, debug rubric_version mismatch first before assuming the v2 changes themselves regressed.

**Caveats from the retro-eval run:**
- Regression eval on #20 came back with n=52 (not 100) — some rows skipped or OOM'd at bs=16, and `mean_total=0.925` looks suspiciously low. Worth checking `_run_regression` behaviour on the new box before trusting regression numbers.
- `prev_amount_correct` still fires positively on only 1/1000 rows — the rubric is effectively dead under both old and new logic. The "change the prompt to ask the model to cite previous £ amounts" follow-up (open-question #1 below) is the real lever.

**Heldout-based promotion validated end-to-end:** #23 (heldout 8.276) was correctly DISCARDed against the heldout anchor 8.536, even though its train reward (10.94) was meaningful. The new `_decide_status` ladder works as designed.

---

## Setup on new 80GB box

Vast.ai instance specs:
- A100-80GB PCIe (or SXM4 — either works; PCIe is what we costed against)
- Image: `pytorch:2.6.0-cuda12.4` or similar — Unsloth supports CUDA 12.4+
- ≥ 100 GB disk (snapshots, datasets, wandb cache)
- ssh key configured

The repo now has `mise.toml` task definitions and a fully-tracked `pyproject.toml` + `uv.lock` (added 2026-04-26 in commits `b25ce0d` + `ee8a7da`), so setup is one-shot:

```bash
# 1. Clone + check out
cd /workspace
git clone git@github.com:charleneleong-ai/gemma4-rlvr.git
cd gemma4-rlvr
git checkout feat/auto-research-loop  # or main once feat/ merges

# 2. Install mise (one-time; if not already on the box)
curl https://mise.jdx.dev/install.sh | sh
exec $SHELL

# 3. Bootstrap the venv — creates .venv, runs uv sync from lockfile,
#    verifies torch/transformers/trl/unsloth import + CUDA detection
mise run init

# 4. Secrets — needed for wandb run resume + HF model download
cp .env.example .env
# Fill in: WANDB_API_KEY, HF_TOKEN

# 5. Smoke test — 2-step v2 train, detached, no eval, wandb disabled.
#    Survives ssh/cc death; tail the printed log path to watch progress.
mise run smoke

# 6. Verify GPU memory headroom (from smoke, then after first real train)
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
# expected after a full v2 train iter: ~50-60 GB used / 20-30 GB free
```

Manual fallback if `mise` isn't available:

```bash
python3.12 -m venv .venv
.venv/bin/pip install uv
.venv/bin/uv sync
WANDB_MODE=disabled .venv/bin/python -u train.py train \
  -c train_v2_80gb -d "[v2 smoke]" \
  --max-steps 2 --eval-heldout-n 0 --eval-regression-n 0 \
  --completion-preview-every 0
```

---

## v2 config — `configs/train_v2_80gb.yaml`

Knobs to change vs `train_fast.yaml` (current 40GB baseline):

```yaml
# train_v2_80gb.yaml — A100-80GB clean baseline
defaults:
  - base
  - _self_

train:
  # Memory: drop 4-bit, larger LoRA, longer context
  load_in_4bit: false
  lora_rank: 128                 # was 64 — minor capacity bump
  max_seq_length: 8192           # was 4096 — full regression coverage
  max_completion_length: 1024    # unchanged — already at headroom

  # GRPO: larger group → tighter advantage estimates
  num_generations: 16            # was 8 — main GRPO quality lever
  batch_size: 8                  # unchanged — keep grad-step semantics consistent

  # Eval: bigger batches now fit
  eval_batch_size: 64            # was 32 — heldout halves to ~13 min
  eval_heldout_n: 1000           # unchanged
  eval_regression_n: 100         # unchanged

  # Training schedule
  max_steps: 80
  save_steps: 50
  patience: 8
  plateau_window: 10
  plateau_delta: 0.05

  # Completion preview to wandb (already wired)
  completion_preview_every: 25
  completion_preview_n_train: 4
  completion_preview_n_heldout: 4

wandb:
  project: gemma4-rlvr
  tags: [v2, 80gb, soften-well-formed, heldout-promotion]
```

> **Caveat on `eval_batch_size=64` for regression**: long prompts may still OOM. If `_run_regression` OOMs at bs=64, drop to bs=32 (which fits cleanly on 80GB even with longer seqs). The eval flow already retries on OOM per batch.

---

## Sweep plan

Three sweeps, ~$22 total, leaves ~$25 buffer.

### Sweep 1 — v2 baseline validation (~9 hrs, $7.30)

Goal: prove the v2 stack actually moves the needle vs v1 best (#20 heldout mean_total = 8.49).

Schedule lives at `configs/schedules/v2_baseline.yaml` — pure YAML, edit there to tweak iters without touching `autoresearch.py`. Iters defined:

1. Anchor — pure v2 defaults (eval-gate verification)
2. `lr=5e-6` (proven strong on v1)
3. `num_generations=24` (push group size further if 16 already won)
4. `beta=0.02` (lower KL — more exploration, was v1 #5)
5. `lora_rank=64` (drop back — tests if rank=128 is wasted)

Launch (via mise — auto-detached, survives ssh/cc death):

```bash
mise run sweep v2_baseline
# Prints log path; follow with: tail -F logs/sweep_v2_baseline_<timestamp>.log
# Status: mise run ps  (lists detached training/sweep PIDs)
# GPU:    mise run gpu  (live nvidia-smi)
# Kill:   mise run kill-train  (graceful SIGINT to all detached training)
```

Manual fallback:

```bash
setsid nohup .venv/bin/python -u experiments/autoresearch.py \
  --schedule v2_baseline > logs/v2_sweep1_baseline.log 2>&1 < /dev/null &
disown
```

Stop conditions:
- If iter 1 (anchor) heldout `mean_total` < v1 anchor (**8.536**, see scoreboard above), **STOP** — something regressed in the migration. Debug rubric_version mismatch first, then transformers/unsloth versions, then hardware-specific Unsloth patches.
- If `pass_all_pct` < 14.1% on iter 1, similar — investigate before continuing.
- If iter 1 anchor > 9.0, the v2 stack is winning — proceed with HP search.

### Sweep 2 — HP search around v2 winner (~9 hrs, $7.30)

After sweep 1 picks a promotion baseline, vary one knob at a time:
1. `lr=2e-5` (2x — faster but risk divergence)
2. `lr=5e-6` (0.5x)
3. `beta=0.01` (very low KL)
4. `weight_decay=0.0`
5. `num_generations=8` (sanity — does dropping back to v1 group size hurt?)

### Sweep 3 — direction sweep (~9 hrs, $7.30)

Spend this only if sweeps 1+2 reveal a clear direction. Otherwise, save the budget for retro-evals and one-offs.

Likely candidates if `well_formed` is now strong but `f1_triggers` lags:
- Higher temperature during rollouts (currently 1.0 — try 1.2 for more exploration)
- Train on a fresh dataset re-generation (`dd_explainer_data_generator`) with more multi-trigger rows

Likely candidates if `prev_amount_correct` still dead:
- Add explicit prompt instruction to cite previous DD amount — see `dd_explainer_data_generator.build_chat_messages`
- Bump `prev_amount_correct` reward weight (currently fires +2/0/-3, low absolute scale)

---

## Things NOT to change (keep comparable across runs)

- **`RUBRIC_VERSION`** — only bump if you change a rubric definition. Soften-well-formed already bumped it; don't re-bump for HP changes.
- **Heldout split** (1000 rows, seed=42) — re-using the same split is what makes mean_total comparisons valid across runs.
- **Regression set** (`.error_analysis_cache/20260413T075447Z_20260420T075447Z`) — stable reference for prod failure tracking.
- **Promotion logic** (`promotion_score` in `experiments/experiment_progress.py`) — the heldout-vs-train-fallback ladder is the supported semantics now.

## Things to monitor during the migration

- **rubric_version drift** — every results.jsonl row should have the same `rubric_version` going forward. If you see mixed values in the chart, retro-eval the older rows.
- **wandb eval logging** — first v2 run should show `eval/heldout/mean_total` and `eval/heldout/pass_all_pct` panels. If missing, the wandb-resume bit broke (check `_log_eval_to_wandb` is firing).
- **Completion preview on W&B** — should appear under `train/completions_preview` table after the first 25 steps. If it never lands, `model.eval()`/`model.train()` toggling around generation may have regressed.
- **GPU utilization** — should be >70% during train, <50% during eval (CPU-bound `apply_chat_template`). If train util is <50%, batch size is too small.

## After 80GB sweeps complete

- Update `docs/v2-80gb-migration.md` with v2 final scoreboard for handoff to v3.
- Push final `progress.html` screenshot to `docs/`.
- If v2 settings became permanent, copy `train_v2_80gb.yaml` knobs into `base.py` defaults and retire `train_fast.yaml`.

---

## Open questions for the human (not me)

1. **`prev_amount_correct` rubric** — the current rubric requires the model to cite previous £ amounts, but the prompt never asks. Two paths: (a) change the prompt to ask, (b) drop the rubric entirely. Worth deciding before sweep 3.
2. **`well_formed` thresholds** — 10-word header / 1-3 sentences are arbitrary. After softening to fractional, are these the right targets? Worth one experiment with looser bounds (15 words / 1-5 sentences) to see if the model actually wants to be slightly more verbose.
3. **Regression set freshness** — the `.error_analysis_cache/` is from 2026-04-13 to 2026-04-20. If new prod failures have different patterns, regenerate the cache before sweep 3.

---

## Deferred refactor — schedule yaml v2 (do this once on the new box if there's bandwidth)

Schedule + sweep config moved to `configs/schedules/<name>.yaml` already (option 1). The remaining "fuller" refactor (option 2) — defer until v2 sweeps are running and you have real reason to vary triage gates per sweep:

- **Move triage thresholds into the schedule yaml** — `KL_LIMIT`, `LOSS_LIMIT`, `SLOW_MEAN_S`, `STEP_TIME_SPIKE_S`, `NO_LEARN_WINDOW`, `NO_LEARN_BASELINE_DELTA`. Today they're module-level constants in `experiments/autoresearch.py`. Different sweeps may want different gates (e.g., a "long-context" sweep should have a higher `STEP_TIME_SPIKE_S` since 8k seqs are slower per step).
- **Add a `triage:` section to schedule yaml** — same shape as `common_overrides:` so a schedule can override any subset.
- **Add `--list-schedules` to the autoresearch CLI** — auto-discovery of available schedules with their iter counts + descriptions. Saves `ls configs/schedules/`.
- **Convert `schedules/v1_explore.yaml` to a true v2 example** — currently it preserves the historical schedule for reproducibility, but it's not a useful template for new sweeps.

Estimated work: 60-80 min on the new box. Skip entirely if v2 sweeps work fine with the current shared triage config.

---

*Drafted by Claude session on `feat/auto-research-loop`. Paths assume `/workspace/gemma4_rl/`. Adjust if cloned elsewhere on the new box.*
