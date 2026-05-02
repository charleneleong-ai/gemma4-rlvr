"""Cross-experiment progression plot for the dd_explainer task.

Twin-axis chart: ``mean_total`` (left) and ``no_hallucinated_facts``
(right) across every checkpoint that has been A/B'd on the n=1000
heldout split. Hand-curated data table — small enough that
hand-maintaining is cleaner than parsing scattered eval JSONs.

Rendering is delegated to ``autoresearch.compare.plot_milestone_progression``
(autoresearch ≥ 0.6.0 / PR #14). This file stays the source-of-truth
for the *milestones*, the upstream package owns the *chart*.

Usage:

    uv run python experiments/_dd_explainer_progression.py
    # writes experiments/progress/dd_explainer_two_stage/v2_progression.png
"""

from __future__ import annotations

import sys
from pathlib import Path

# Drop ``experiments/`` from sys.path so the installed ``autoresearch``
# package wins over our local ``experiments/autoresearch.py`` orchestrator
# (Python adds the script's dir to sys.path[0] when invoked as a file).
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p != _HERE]

from autoresearch import Milestone, plot_milestone_progression  # noqa: E402


MILESTONES: list[Milestone] = [
    Milestone(
        label="vanilla",
        description="Gemma 4 4B base, no fine-tune (vanilla baseline)",
        metrics={"mean_total": 8.354, "no_halluc": -0.876, "pass_all_pct": 11.9},
    ),
    Milestone(
        label="E18",
        description="v2 GRPO champion (single-stage, no two-stage architecture)",
        metrics={"mean_total": 9.324, "no_halluc": -0.840, "pass_all_pct": 18.5},
    ),
    Milestone(
        label="v0_pipeline",
        description="Two-stage v3 (bge-small + extra features) + E18 LoRA — first config to clear 10",
        metrics={"mean_total": 10.293, "no_halluc": -0.876, "pass_all_pct": 14.4},
    ),
    Milestone(
        label="E25",
        description="v3 + GRPO retrain on Stage-1-injected prompts (90 min sweep)",
        metrics={"mean_total": 10.804, "no_halluc": -0.844, "pass_all_pct": 12.9},
    ),
    Milestone(
        label="v4_mlp+E18",
        description="MLP head Stage 1 + E18 LoRA (no Stage 2 retrain) — ties E25 cheaper",
        metrics={"mean_total": 10.816, "no_halluc": -0.848, "pass_all_pct": 22.4},
    ),
    Milestone(
        label="v1_constrained",
        description="v4_mlp + E18 + prompt-time GROUNDING CONSTRAINT (PR #12)",
        metrics={"mean_total": 10.961, "no_halluc": -0.732, "pass_all_pct": 23.8},
    ),
    Milestone(
        label="v2_slot",
        description="v4_mlp + E18 + LMFE slot mask (PR-B) — flat vs PR #12",
        metrics={"mean_total": 10.966, "no_halluc": -0.724, "pass_all_pct": 23.2},
    ),
    Milestone(
        label="E26 (×3.0)",
        description="PR-C iter 1 — slot reward ×3.0 + 80 steps (over-weighted, regressed prose)",
        metrics={"mean_total": 10.801, "no_halluc": -0.924, "pass_all_pct": 12.1},
    ),
    Milestone(
        label="E27 (×2.0)",
        description="PR-C iter 2 — slot reward ×2.0 + 160 steps (new SOTA mean_total, prose recovered)",
        metrics={"mean_total": 11.192, "no_halluc": -0.744, "pass_all_pct": 17.0},
    ),
]


TITLE = (
    "dd_explainer two_stage — cross-experiment progression on n=1000 heldout\n"
    "PR-C E27 (×2.0) is new SOTA on mean_total but prose plateau didn't break "
    "(-0.74 ≈ PR-B -0.72)"
)


if __name__ == "__main__":
    out = Path("experiments/progress/dd_explainer_two_stage/v2_progression.png")
    plot_milestone_progression(
        MILESTONES,
        primary_metric="mean_total",
        secondary_metric="no_halluc",
        primary_color="#2563eb",
        secondary_color="#dc2626",
        primary_ylim=(7.5, 11.7),
        secondary_ylim=(-1.0, 0.0),
        threshold=-0.5,
        threshold_label="-0.5 ship-as-champion",
        title=TITLE,
        out_path=out,
    )
    print(f"wrote {out}")
