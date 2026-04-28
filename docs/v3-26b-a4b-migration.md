# v3 Gemma 4 26B-A4B (MoE) migration plan

**Status:** drafted 2026-04-28 on `feat/gemma-4-26b-a4b-swap`.
**Driving doc:** [`docs/experiments/dd_explainer/encoder_outlier/v0_gate.md`](experiments/dd_explainer/encoder_outlier/v0_gate.md) — the gated A/B verdict that exhausted the encoder-side approach and pointed the next move at base-model swap.

---

## Why we're switching

22 experiments + the encoder-outlier-gate work have mapped the v2 ceiling cleanly:

| ceiling | cause | what we tried | result |
|---|---|---|---|
| `f1_triggers ≈ 7.45` (old data) | 86.7% 1-trigger skew, no gradient signal | regen multi-trigger dataset | broken to 7.523 (E16), then 7.745 (E18) |
| `no_halluc ≈ -0.88..-0.92` | uniform across heldout, not OOD-concentrated | reward weighting, granular shape, OOD gate | plateau confirmed at every reward variant; gate lifts +0.336 but mean_total drops |
| `prev_amount_correct = 0` | 4B model can't traverse nested JSON + arithmetic | LoRA rank, MLP target modules | stays 0 across 22 experiments |
| `mean_total = 9.611` (E1) → `9.324` (E18) | multi-knob equilibrium of the above | every reward + data lever | exhausted |

The gated A/B (n=1000 heldout, [`v0_gate.md`](experiments/dd_explainer/encoder_outlier/v0_gate.md#gated-ab--verdict-falsified)) was the falsifying test: the gate works *as a component* (no_halluc finally moves) but the rubric weighting (f1 max=10 dominates no_halluc max=1) means substituting fallback always loses. **The hallucination is uniform across in-distribution rows, not concentrated on weird tail rows.**

That points the lever at the **base model itself**. 4B parameters are not enough to reliably ground 5-10 facts of nested JSON simultaneously. No reward shape, dataset tweak, or pre-flight gate fixes a capability gap.

## Why 26B-A4B specifically

The serving constraint pins the answer: **L4 GPU on Cloud Run** has 24 GB VRAM. That sets the upper bound on inference-time model size.

Gemma 4 family options vs. the L4 envelope:

| variant | total / active | L4 INT8 fit | training fit (A100 80GB) | quality vs current 4B | verdict |
|---|---|---|---|---|---|
| `gemma-4-E2B-it` | 2B / 2B | trivial | fits | smaller | not the move |
| `gemma-4-E4B-it` (current v2) | 4B / 4B | trivial | already running | baseline | — |
| **`gemma-4-26B-A4B-it`** ⭐ | **26B / 4B** (MoE) | **comfortable (~16 GB)** | fits w/ Unsloth 4-bit + LoRA | **6.5× total params, same decode latency** | **the move** |
| `gemma-4-31B-it` | 31B / 31B | tight (batch=1 only) | tight | best quality | won't serve L4 |

The MoE variant is the only Gemma 4 model that simultaneously (a) is meaningfully bigger than the current 4B, and (b) fits L4 with serving headroom.

### Why MoE is uniquely good for this task

- **Decoding cost** (latency, tok/s, watts) is set by **active params** = 4B. Same as today.
- **Capability cost** (quality) is set by **total params** = 26B. Massive bump.

Specifically for the dd_explainer ceilings:
- **`prev_amount_correct = 0`**: the 26B parameter pool gives the experts enough capacity to do the JSON-traversal arithmetic. 4B was the ceiling.
- **Uniform `no_halluc` -0.88..-0.92**: more parameters → larger working memory → fewer "lost track of context" hallucinations.
- **f1 trade ridge**: bigger model handles multi-trigger discrimination without the f1 ↔ no_halluc fight.

## Training-side changes

Single config swap: `configs/train_v3_26b_a4b.yaml`. Cumulative deltas vs. v2:

| field | v2 (4B dense) | v3 (26B-A4B MoE) | rationale |
|---|---|---|---|
| `model_name` | `unsloth/gemma-4-E4B-it` | `unsloth/gemma-4-26B-A4B-it` | the swap |
| `load_in_4bit` | false | **true** | 26B BF16 = 52 GB; 4-bit = 13 GB |
| `lora_rank` | 128 | **256** | more capacity for the experts' MLPs |
| `batch_size` | 8 | **4** | per-rollout memory ~2× w/ 26B forward |
| `grad_accum` | 2 | **4** | keeps effective grad batch = 16 unchanged |
| `num_generations` | 16 | 16 | unchanged — divides batch×grad_accum |
| `max_seq_length` | 8192 | 8192 | unchanged |
| `eval_batch_size` | 32 | **16** | 26B eval forward is 2× heavier |

Step-time expectation: **~150-200s on PCIe 80GB** (~2-3× slower than v2's ~75-95s). The bigger base + larger LoRA mean each forward + backward is heavier even with sparse activation. Plan around this:

- `--max-steps 80` autoresearch iters take ~3-4h instead of v2's ~70min.
- Full long-runs (`--max-steps 160`) will take ~5-7h.
- Eval at `eval_heldout_n=1000` will take ~30-45min (2× v2's ~15min).
- Total per-iter budget: ~4-5h.

Same `experiments/autoresearch.py` orchestration. Same triage thresholds (just adjust `SLOW_MEAN_S` from 90 → 200 for the GPU-hang detector).

## Inference / serving target

**L4 GPU on Cloud Run** with INT8 quantization:

| precision | weights | + KV/batch | total | L4 fit | batch | decode |
|---|---|---|---|---|---|---|
| BF16 | ~52 GB | — | — | won't fit | — | — |
| **INT8** | ~13 GB | ~3-4 GB | **~16-17 GB** | **comfortable** | **4-8** | ~4B-class |
| INT4 (AWQ/GPTQ) | ~6.5 GB | ~2-3 GB | ~9-10 GB | plenty | 16+ | fastest |

`unsloth/gemma-4-26B-A4B-it-GGUF` (3.2M downloads on HF) and `unsloth/gemma-4-26B-A4B-it-unsloth-bnb-4bit` already exist — both are valid serving artefacts.

**Recommended for production:** INT4 AWQ. Headroom for batch=16+, fastest decode, and the active-param pass means quality holds up at lower precision better than dense 26B would.

### Cold-start + cost (L4 Cloud Run)

- L4 Cloud Run: ~$0.71/hr.
- Cold start with 26B INT8: ~30-45s container + model load.
- Steady-state with min-instances=1: ~$510/month always-on.
- Scale-to-zero option: cheaper, but eat the 30-45s cold start.

## Pre-migration checklist

- [x] Branch off main (`feat/gemma-4-26b-a4b-swap`)
- [x] `configs/train_v3_26b_a4b.yaml` written, model + LoRA params sized for 80 GB A100
- [ ] Smoke test: single 2-step train run, verify model loads, LoRA attaches, GRPO group-size constraint passes
- [ ] First sweep: `v3_baseline.yaml` schedule with 2-4 anchor iters at `--max-steps 80` to establish v3's mean_total/f1/no_halluc
- [ ] Prep INT8 / INT4 quantization recipe (Unsloth's `save_pretrained_merged` + bnb)
- [ ] L4 Cloud Run validation: spin up, hit the inference endpoint with a representative dd_explainer prompt, verify decode latency

## Sweep plan (post-merge)

1. **`v3_baseline`** — 2 iters: anchor (`max_steps=80`) + long-run (`max_steps=160`). Direct comparison vs E18 (mean_total=9.324, f1=7.745, no_halluc=-0.888).
2. **If v3_baseline mean_total ≥ 10**: ship v3 as the new champion. Skip further sweeps; the base-model swap was the answer.
3. **If v3_baseline mean_total ∈ [9.5, 10]**: re-run the v2 reward sweeps (binary×2, granular) on top of v3 — they may hit different optima with bigger working memory.
4. **If v3_baseline mean_total < 9.5**: investigate. Either the LoRA rank is too small (try r=512), or 4-bit quantization at training time is too lossy for MoE experts (try BF16 base + smaller seq length to fit memory), or — surprisingly — 4B was actually right-sized and bigger model overfits faster.

The decision point lands inside iter 1 (~4h) — fast falsification.

## Out of scope for this branch

- Long-tail v2 sweeps that haven't converged (rubric reweighting, base-model fine-tune of E18 adapter onto 26B-A4B).
- Cloud Run deployment infrastructure (GCP project, Dockerfile, service config) — the migration plan documents the *target*; the actual deploy is a separate PR.
- Encoder-outlier gate integration with v3 — the gate's value proposition (route OOD to fallback) is orthogonal to the base model. Pick this back up if v3 still has a no_halluc tail problem on tail customers.
- Migration to non-Gemma family (Qwen 2.5/3.5, Llama 3.1) — explicitly out of scope per project constraint. If 26B-A4B doesn't move the needle, that's the next pivot.

## Reference rows

| | v2 champion (E18) | v3 expected (heuristic) |
|---|---|---|
| model | `unsloth/gemma-4-E4B-it` | `unsloth/gemma-4-26B-A4B-it` |
| total params | 4 B | 26 B |
| active params/tok | 4 B | 4 B |
| mean_total (heldout, n=1000) | 9.324 | ≥ 10.0 if the swap works |
| f1_triggers | 7.745 | ≥ 8.0 (multi-trigger discrimination clearer) |
| no_halluc | -0.888 | ≥ -0.5 (uniform hallucination should drop) |
| prev_amount_correct | 0 | > 0 if MoE arithmetic capacity activates |
| training step time | ~75-95s | ~150-200s |
| L4 INT8 serving | trivial fit | comfortable fit |
