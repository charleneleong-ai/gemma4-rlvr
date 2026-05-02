"""Cross-experiment progression plot for the dd_explainer task.

Renders a single twin-axis chart showing mean_total (left) and
no_hallucinated_facts (right) across every checkpoint that has been A/B'd
on the n=1000 heldout split. Hand-authored data table — small enough that
hand-maintaining is cleaner than parsing scattered eval JSONs.

Usage:

    uv run python experiments/_dd_explainer_progression.py
    # writes experiments/progress/dd_explainer_two_stage/v2_progression.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Milestone:
    label: str  # short id for x-axis
    full: str   # tooltip / annotation
    mean_total: float
    no_halluc: float
    pass_all_pct: float


# Hand-curated history. Sources cited inline.
MILESTONES: list[Milestone] = [
    Milestone(
        label="vanilla",
        full="Gemma 4 4B base, no fine-tune (vanilla baseline)",
        mean_total=8.354,
        no_halluc=-0.876,
        pass_all_pct=11.9,
    ),
    Milestone(
        label="E18",
        full="v2 GRPO champion (single-stage, no two-stage architecture)",
        mean_total=9.324,
        no_halluc=-0.840,
        pass_all_pct=18.5,
    ),
    Milestone(
        label="v0_pipeline",
        full="Two-stage v3 (bge-small + extra features) + E18 LoRA — first config to clear 10",
        mean_total=10.293,
        no_halluc=-0.876,
        pass_all_pct=14.4,
    ),
    Milestone(
        label="E25",
        full="v3 + GRPO retrain on Stage-1-injected prompts (90 min sweep)",
        mean_total=10.804,
        no_halluc=-0.844,
        pass_all_pct=12.9,
    ),
    Milestone(
        label="v4_mlp+E18",
        full="MLP head Stage 1 + E18 LoRA (no Stage 2 retrain) — ties E25 cheaper",
        mean_total=10.816,
        no_halluc=-0.848,
        pass_all_pct=22.4,
    ),
    Milestone(
        label="v1_constrained",
        full="v4_mlp + E18 + prompt-time GROUNDING CONSTRAINT (PR #12)",
        mean_total=10.961,
        no_halluc=-0.732,
        pass_all_pct=23.8,
    ),
    Milestone(
        label="v2_slot",
        full="v4_mlp + E18 + LMFE slot mask (PR-B, this PR) — flat vs PR #12",
        mean_total=10.966,
        no_halluc=-0.724,
        pass_all_pct=23.2,
    ),
]


def render(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    xs = list(range(len(MILESTONES)))
    labels = [m.label for m in MILESTONES]
    mean_totals = [m.mean_total for m in MILESTONES]
    no_hallucs = [m.no_halluc for m in MILESTONES]

    fig, ax_total = plt.subplots(figsize=(11, 5.5))
    ax_halluc = ax_total.twinx()

    line_total = ax_total.plot(
        xs, mean_totals, "-o", color="#2563eb", linewidth=2.2, markersize=7,
        label="mean_total (left axis)",
    )[0]
    line_halluc = ax_halluc.plot(
        xs, no_hallucs, "--s", color="#dc2626", linewidth=2.0, markersize=6,
        label="no_halluc (right axis)",
    )[0]

    # Horizontal reference lines — falsification thresholds + "ship as champion" line
    ax_halluc.axhline(-0.5, color="#16a34a", linestyle=":", alpha=0.5, linewidth=1)
    ax_halluc.text(
        len(MILESTONES) - 0.5, -0.5, "  -0.5 ship-as-champion", color="#16a34a",
        fontsize=8, va="bottom", ha="right",
    )

    # Annotate each point with mean_total value
    for x, m in zip(xs, MILESTONES):
        ax_total.annotate(
            f"{m.mean_total:.2f}",
            (x, m.mean_total),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=8, color="#1e40af",
        )

    # Highlight the new PR-B point
    ax_total.scatter([xs[-1]], [mean_totals[-1]], s=160, facecolors="none",
                     edgecolors="#2563eb", linewidth=2, zorder=5)

    ax_total.set_xticks(xs)
    ax_total.set_xticklabels(labels, rotation=20, ha="right")
    ax_total.set_ylabel("mean_total", color="#2563eb")
    ax_halluc.set_ylabel("no_hallucinated_facts (rendered prose)", color="#dc2626")
    ax_total.tick_params(axis="y", labelcolor="#2563eb")
    ax_halluc.tick_params(axis="y", labelcolor="#dc2626")
    ax_total.set_ylim(7.5, 11.5)
    ax_halluc.set_ylim(-1.0, 0.0)
    ax_total.grid(True, alpha=0.25)

    ax_total.set_title(
        "dd_explainer two_stage — cross-experiment progression on n=1000 heldout\n"
        "PR-B (v2_slot) flat vs v1_constrained → slot enforcement is a no-op without retrain (→ PR-C)",
        fontsize=11, pad=12,
    )

    # Combined legend
    ax_total.legend(
        handles=[line_total, line_halluc],
        loc="lower right", framealpha=0.9, fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    render(Path("experiments/progress/dd_explainer_two_stage/v2_progression.png"))
