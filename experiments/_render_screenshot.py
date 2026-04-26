"""One-shot matplotlib renderer for the PR screenshot.

Reads the per-config `experiments/dd_explainer/<config_name>/results.jsonl`
the Plotly chart uses and renders a static PNG that mirrors the visual
encoding (status colour + best-run halo + kill_reason inline). Output
filename is also config-scoped: `docs/autoresearch_progress_<config>.png`.

Usage:
  python experiments/_render_screenshot.py <config_name>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from experiment_progress import _STATUS_STYLE, _kill_tag, load_results

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: _render_screenshot.py <config_name>")
    config_name = sys.argv[1]
    rows = sorted(load_results("dd_explainer", config_name=config_name),
                  key=lambda r: r["experiment"])
    if not rows:
        raise SystemExit(f"no results to plot for config {config_name!r}")
    out = DOCS / f"autoresearch_progress_{config_name}.png"

    best_exp = max(rows, key=lambda r: r["score"])["experiment"]

    fig, ax = plt.subplots(figsize=(14, 7), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Plot one marker per row
    for r in rows:
        cfg = _STATUS_STYLE.get(r["status"], _STATUS_STYLE["DISCARD"])
        is_best = (r["experiment"] == best_exp)
        ax.scatter(
            r["experiment"], r["score"],
            s=400 if is_best else 220,
            c=cfg["color"], edgecolors="#27ae60" if is_best else cfg["line_color"],
            linewidths=3 if is_best else 1.2,
            zorder=3,
        )

    # Running-best line over KEEP/BASELINE only
    kept = [r for r in rows if r["status"] in ("KEEP", "BASELINE")]
    if kept:
        xs = [r["experiment"] for r in kept]
        ys, best = [], float("-inf")
        for r in kept:
            best = max(best, r["score"])
            ys.append(best)
        ax.step(xs, ys, where="post", color="#27ae60", lw=2, alpha=0.6, zorder=2,
                label="running best (KEEP/BASELINE)")

    # Inline label per dot, alternating above/below
    for j, r in enumerate(rows):
        cfg = _STATUS_STYLE.get(r["status"], _STATUS_STYLE["DISCARD"])
        is_best = (r["experiment"] == best_exp)
        m = r.get("metrics") or {}
        if r["status"] == "EARLY_KILL":
            tag = _kill_tag(m.get("kill_reason", ""))
        elif r["status"] == "CRASH" and m.get("crash_reason"):
            cr = m["crash_reason"]
            tag = f"crashed: {cr[:30]}{'…' if len(cr) > 30 else ''}"
        else:
            tag = r["status"].lower()
        runtime = f"{int(r['runtime_min'])}min" if r.get("runtime_min") else ""
        head = f"E{r['experiment']} · {runtime} · {tag}".strip(" ·")
        bits = [f"score={r['score']:.2f}"]
        if isinstance(m.get("final_kl"), (int, float)):
            bits.append(f"kl={m['final_kl']:.2f}")
        if r.get("steps"):
            bits.append(f"{r['steps']}st")
        body = " · ".join(bits)
        text = f"{head}\n{body}"

        y_off = 1.6 if j % 2 == 0 else -1.8
        ax.annotate(
            text, xy=(r["experiment"], r["score"]),
            xytext=(0, y_off * 18), textcoords="offset points",
            ha="center", va="center",
            fontsize=9 if not is_best else 10,
            fontweight="bold" if is_best else "normal",
            color=("#1a7a3a" if is_best else cfg["text_color"]),
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=("#f0fff0" if is_best else "white"),
                edgecolor=("#27ae60" if is_best else cfg["color"]),
                linewidth=2 if is_best else 1,
            ),
        )

    n = len(rows)
    n_kept = sum(1 for r in rows if r["status"] in ("KEEP", "BASELINE"))
    n_killed = sum(1 for r in rows if r["status"] == "EARLY_KILL")
    n_crash = sum(1 for r in rows if r["status"] == "CRASH")
    runtime = sum(r.get("runtime_min", 0) for r in rows)

    ax.set_title(
        f"GRPO Autoresearch — dd_explainer · {config_name}\n"
        f"{n} experiments · {n_kept} kept · {n_killed} killed early · {n_crash} crashed · {runtime:.0f}min total",
        fontsize=14, color="#222", pad=20,
    )
    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_ylabel("Train reward (higher is better)", fontsize=11)
    ax.grid(True, color="#eee", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xticks(range(0, n))
    ax.set_xlim(-0.5, n - 0.5)
    ymin = min(r["score"] for r in rows) - 2
    ymax = max(r["score"] for r in rows) + 3
    ax.set_ylim(ymin, ymax)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
