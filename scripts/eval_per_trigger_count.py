"""Per-trigger-count heldout eval against E16's saved adapter.

Re-runs the same heldout split that train.py used (1000 rows, seed=42),
groups results by len(ground_truth_triggers), and reports per-rubric
metrics per group. Answers: is hallucination i.i.d. across trigger
counts (so dropping 4+ doesn't help much), or concentrated in 4+/5
(so dropping does help)?

Output: scripts/eval_per_trigger_count_results.json + stdout summary.

Run as a daemon (~15min on 80GB):
  setsid nohup uv run python scripts/eval_per_trigger_count.py \
    </dev/null >>logs/eval_per_trig.log 2>&1 & disown
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from datasets import Dataset
from peft import PeftModel
from unsloth import FastLanguageModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dd_explainer_rewards import score_completion  # noqa: E402

ADAPTER_DIR = str(ROOT / "gemma_4_lora" / "train_v2_80gb" / "exp_16")
DATA_DIR = ROOT / "data"
OUT_PATH = ROOT / "scripts" / "eval_per_trigger_count_results.json"
HELDOUT_N = 1000
SEED = 42
MAX_NEW_TOKENS = 1024


def main() -> None:
    print(f"adapter: {ADAPTER_DIR}", flush=True)

    print("loading model + adapter...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-4-E4B-it",
        max_seq_length=8192,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    FastLanguageModel.for_inference(model)

    print("loading + splitting dataset (seed=42, n=1000 heldout)...", flush=True)
    jsonl_files = sorted(DATA_DIR.glob("dd_dataset_*_*rows.jsonl"))
    path = jsonl_files[-1]
    print(f"  using {path.name}", flush=True)
    dataset = Dataset.from_json(str(path))
    if "__meta__" in dataset.column_names:
        dataset = dataset.filter(lambda r: r.get("__meta__") is not True).remove_columns("__meta__")
    if "row_index" in dataset.column_names:
        dataset = dataset.remove_columns("row_index")
    split = dataset.train_test_split(test_size=HELDOUT_N, seed=SEED)
    heldout = split["test"]
    print(f"  heldout size: {len(heldout)}", flush=True)

    results_by_size: dict[int, list[dict]] = defaultdict(list)

    print(f"\ngenerating + scoring {len(heldout)} rows...", flush=True)
    for i, row in enumerate(heldout):
        n_trigs = len(row["ground_truth_triggers"])
        prompt = row["prompt"]

        inputs = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

        scores = score_completion(
            gen_text,
            row["ground_truth_triggers"],
            row["input_json"],
        )
        scores["n_triggers"] = n_trigs
        results_by_size[n_trigs].append(scores)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(heldout)}] sizes seen so far: "
                  f"{ {k: len(v) for k, v in sorted(results_by_size.items())} }",
                  flush=True)

    print("\n" + "═" * 72, flush=True)
    print("PER-TRIGGER-COUNT BREAKDOWN", flush=True)
    print("═" * 72, flush=True)
    print(f"{'size':<5} {'n':>5} {'f1':>7} {'no_halluc_mean':>16} {'no_halluc_pass%':>16}",
          flush=True)
    print("─" * 72, flush=True)

    summary = {}
    for sz in sorted(results_by_size):
        rs = results_by_size[sz]
        f1m = mean(r["f1_triggers"] for r in rs)
        nhm = mean(r["no_hallucinated_facts"] for r in rs)
        # pass = score >= 1.0 (the +1 branch); fail = -3.0; skip = 0.0
        nh_pass = sum(1 for r in rs if r["no_hallucinated_facts"] >= 1.0)
        nh_pass_pct = nh_pass / len(rs) * 100 if rs else 0
        wfm = mean(r["well_formed"] for r in rs)
        pam = mean(r["prev_amount_correct"] for r in rs)
        summary[str(sz)] = {
            "n": len(rs),
            "f1_mean": f1m,
            "no_halluc_mean": nhm,
            "no_halluc_pass_pct": nh_pass_pct,
            "well_formed_mean": wfm,
            "prev_amount_mean": pam,
        }
        print(f"{sz:<5} {len(rs):>5} {f1m:>7.3f} {nhm:>16.3f} {nh_pass_pct:>15.1f}%",
              flush=True)

    OUT_PATH.write_text(json.dumps({"summary": summary, "config": {
        "adapter": ADAPTER_DIR,
        "heldout_n": HELDOUT_N,
        "seed": SEED,
    }}, indent=2))
    print(f"\nfull results → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
