"""Experiment progress tracker (autoresearch-style) for gemma4 GRPO runs.

Adapted from charleneleong-ai/orak-2025-starter-kit for the single-task
direct_debit_explainer GRPO setup. Each completed `train` invocation appends
one row to `experiments/<task>/results.jsonl`; `plot` renders a Plotly
progress chart with KEEP/DISCARD/BASELINE markers and a running-best line.

CLI:
  python experiments/experiment_progress.py log \\
      --score 11.5 --steps 300 --status KEEP \\
      --description "lower beta, +max_completion=1024" \\
      --wandb-url https://wandb.ai/chaleong/gemma4-rlvr/runs/abc123

  python experiments/experiment_progress.py plot          # all tasks
  python experiments/experiment_progress.py plot --task dd_explainer
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
import typer
from plotly.subplots import make_subplots

# Sibling import that works both as direct script and as package member.
try:
    from experiments._chart_widgets import plotly_label_toggle
except ImportError:
    from _chart_widgets import plotly_label_toggle

import re

EXPERIMENTS_DIR = Path(__file__).parent
DEFAULT_TASK = "dd_explainer"

_REWARD_RE = re.compile(r"'reward': '([\-0-9.eE+]+)'")
_STEP_RE = re.compile(r"'step_time': '([\-0-9.eE+]+)'")
_WANDB_URL_RE = re.compile(r"https://wandb\.ai/[\w\-./]+/runs/[\w\-]+")


def _task_dir(task: str) -> Path:
    d = EXPERIMENTS_DIR / task.lower().replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_results(task: str = DEFAULT_TASK, include_superseded: bool = False) -> list[dict]:
    """Load result rows for a task. By default, rows with `superseded_by` set
    are filtered out so charts and aggregations show only canonical entries
    (the rerun supersedes the original). Pass `include_superseded=True` for
    bookkeeping that needs the full history (e.g. assigning the next exp #).
    """
    f = _task_dir(task) / "results.jsonl"
    if not f.exists():
        return []
    rows = [json.loads(line) for line in f.read_text().strip().split("\n") if line]
    if include_superseded:
        return rows
    return [r for r in rows if not r.get("superseded_by")]


def _scrape_in_flight_run(task: str) -> dict | None:
    """Read `<task>/current_run.json` (written by autoresearch at iter start)
    and scrape the live training log for max-reward / step-count so far.
    Returns a synthetic RUNNING row, or None if no iter is in flight.
    """
    cur_f = _task_dir(task) / "current_run.json"
    if not cur_f.exists():
        return None
    try:
        cur = json.loads(cur_f.read_text())
    except json.JSONDecodeError:
        return None
    finished = load_results(task)
    if finished and finished[-1]["experiment"] >= cur["experiment"]:
        return None
    log_path = Path(cur["log_path"])
    iter_marker = cur.get("iter_marker", "")
    if not log_path.exists():
        return None

    text = log_path.read_text(errors="replace")
    if iter_marker:
        idx = text.rfind(iter_marker)
        if idx >= 0:
            text = text[idx:]
    rewards = [float(m) for m in _REWARD_RE.findall(text)]
    steps = len(_STEP_RE.findall(text))
    best = max(rewards) if rewards else 0.0
    urls = _WANDB_URL_RE.findall(text)
    wandb_url = cur.get("wandb_url") or (urls[-1] if urls else "")

    started = datetime.fromisoformat(cur["started_at"].replace("Z", "+00:00"))
    elapsed_min = (datetime.now(timezone.utc) - started).total_seconds() / 60.0

    return {
        "experiment": cur["experiment"],
        "task": task,
        "config_name": cur.get("config_name", ""),
        "score": best,
        "metrics": {"best_reward": best, "steps_so_far": steps},
        "steps": steps,
        "runtime_min": elapsed_min,
        "status": "RUNNING",
        "description": cur.get("description", "(in flight)"),
        "notes": cur.get("notes", ""),
        "wandb_url": wandb_url,
        "wandb_run_id": "", "wandb_run_name": "",
        "timestamp": cur["started_at"],
    }


def promotion_score(row_or_metrics: dict, fallback_score: float | None = None) -> float | None:
    """The score used for KEEP/DISCARD comparisons.

    Prefers `metrics.heldout.mean_total` (true generalization signal). Falls
    back to the train reward (`score`) when eval hasn't run — e.g. EARLY_KILL
    rows that died before the post-train eval block, or older rows from
    before the eval gate shipped.

    Accepts either a full row dict (reads its `metrics` + `score`) or a raw
    metrics dict + `fallback_score`. Returns None if neither is available.
    """
    if "metrics" in row_or_metrics or "score" in row_or_metrics:
        metrics = row_or_metrics.get("metrics") or {}
        fallback_score = row_or_metrics.get("score")
    else:
        metrics = row_or_metrics or {}
    ho = (metrics.get("heldout") or {})
    if ho.get("n"):
        return ho.get("mean_total")
    return fallback_score


def _decide_status(prior: list[dict], score: float, metrics: dict | None = None) -> str:
    """BASELINE for first run; KEEP if better than prior best (KEEP|BASELINE).

    Promote on **held-out generalization** (`metrics.heldout.mean_total`) when
    both the new row and at least one prior KEEP/BASELINE row have heldout —
    that's the apples-to-apples comparison. Otherwise fall back to train
    reward, never mixing the two scales (mixing would let stale train-scored
    baselines block heldout-scored runs from ever winning).
    """
    kept = [r for r in prior if r["status"] in ("KEEP", "BASELINE")]
    if not kept:
        return "BASELINE"

    new_heldout = ((metrics or {}).get("heldout") or {}).get("mean_total")
    prior_heldout = [
        ((r.get("metrics") or {}).get("heldout") or {}).get("mean_total")
        for r in kept
    ]
    prior_heldout = [s for s in prior_heldout if s is not None]

    if new_heldout is not None and prior_heldout:
        return "KEEP" if new_heldout > max(prior_heldout) else "DISCARD"
    # Fallback: pure train-reward comparison (legacy rows / EARLY_KILL with no eval)
    return "KEEP" if score > max(r["score"] for r in kept) else "DISCARD"


def log_experiment(
    *,
    score: float,
    description: str,
    task: str = DEFAULT_TASK,
    config_name: str = "",
    steps: int = 0,
    runtime_min: float = 0.0,
    status: Optional[str] = None,
    metrics: Optional[dict] = None,
    notes: str = "",
    wandb_url: str = "",
    wandb_run_id: str = "",
    wandb_run_name: str = "",
) -> dict:
    """Append a result row. If `status` is None, decide BASELINE/KEEP/DISCARD."""
    prior = load_results(task, include_superseded=True)
    if status is None:
        status = _decide_status(prior, score, metrics=metrics)

    entry = {
        "experiment": len(prior),
        "task": task,
        "config_name": config_name,
        "score": score,
        "metrics": metrics or {},
        "steps": steps,
        "runtime_min": runtime_min,
        "status": status.upper(),
        "description": description,
        "notes": notes,
        "wandb_url": wandb_url,
        "wandb_run_id": wandb_run_id,
        "wandb_run_name": wandb_run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    f = _task_dir(task) / "results.jsonl"
    with f.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"Logged to {f}: #{entry['experiment']} score={score} [{status}] {description}")
    return entry


def _green_gradient(t: float) -> str:
    """Linear interp #b8e6c8 (low heldout) → #1a7a3a (high heldout). t in [0,1]."""
    t = max(0.0, min(1.0, t))
    lo = (0xb8, 0xe6, 0xc8)
    hi = (0x1a, 0x7a, 0x3a)
    rgb = tuple(int(lo[i] + (hi[i] - lo[i]) * t) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


_STATUS_STYLE = {
    "DISCARD":    {"color": "#cccccc", "size": 18, "opacity": 0.7, "line_color": "#999",    "symbol": "circle", "text_color": "#777"},
    "KEEP":       {"color": "#2ecc71", "size": 20, "opacity": 1.0, "line_color": "black",   "symbol": "circle", "text_color": "#1a7a3a"},
    "BASELINE":   {"color": "#2ecc71", "size": 20, "opacity": 1.0, "line_color": "black",   "symbol": "circle", "text_color": "#1a7a3a"},
    "RUNNING":    {"color": "#f1c40f", "size": 20, "opacity": 1.0, "line_color": "#9a7d0a", "symbol": "circle", "text_color": "#7d6608"},
    "EARLY_KILL": {"color": "#7f8c8d", "size": 20, "opacity": 0.9, "line_color": "#34495e", "symbol": "circle", "text_color": "#34495e"},
    "CRASH":      {"color": "#e74c3c", "size": 20, "opacity": 1.0, "line_color": "#922b21", "symbol": "circle", "text_color": "#922b21"},
}
_LEGEND_NAMES = {
    "DISCARD": "Discarded", "KEEP": "Kept", "BASELINE": "Baseline", "RUNNING": "Running",
    "EARLY_KILL": "Killed early (triage)", "CRASH": "Crashed",
}
_WANDB_PROJECT_URL = "https://wandb.ai/chaleong/gemma4-rlvr"

_REWARD_SUMMARIES = {
    "dd_explainer": (
        "reward = schema(±1) + triggers_in_enum(±1) + triggers_match_gt(±10) + "
        "prev_dd_amount(±2) + no_hallucinated_facts(±1) + underpayment_lang(±0.5) + "
        "well_formed(±0.5) — max ≈ 16, untrained baseline ≈ 7 (higher is better)"
    ),
}


def _hover(r: dict) -> str:
    h = f"<b>E{r['experiment']} — {_LEGEND_NAMES.get(r['status'], r['status'])}</b>"
    h += f"<br>{r['description']}"
    if r.get("config_name"):
        h += f"<br>config: {r['config_name']}"
    m = r.get("metrics") or {}
    bits = []
    for k in ("best_reward", "final_kl", "final_loss", "min_completion_length"):
        if k in m:
            v = m[k]
            bits.append(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}")
    if bits:
        h += "<br>" + " | ".join(bits)
    if r.get("steps"):
        h += f"<br>steps={r['steps']}"
    if r.get("runtime_min"):
        h += f" | runtime={r['runtime_min']:.0f}min"
    if m.get("kill_reason"):
        h += f"<br><b>kill_reason</b>: {m['kill_reason']}"
    if m.get("crash_reason"):
        h += f"<br><b>crash_reason</b>: {m['crash_reason']}"
    if m.get("train_status") and m["train_status"] != r["status"]:
        h += f"<br>train_status={m['train_status']}"
    # Held-out + regression eval blocks (when present)
    for eval_key, label in (("heldout", "held-out"), ("regression", "regression")):
        ev = m.get(eval_key) or {}
        if ev.get("n"):
            h += (f"<br><b>{label}</b>: pass {ev.get('pass_all', 0)}/{ev['n']} "
                  f"({ev.get('pass_all_pct', 0)}%) · mean={ev.get('mean_total', 0):.2f}")
    notes = r.get("notes") or ""
    if notes and notes != r["description"]:
        h += f"<br><i>{notes}</i>"
    if r.get("wandb_url"):
        h += f"<br><a href='{r['wandb_url']}' target='_blank'>W&B Run</a>"
    return h


def _short_summary(r: dict) -> str:
    """Headline: strip `[crash]`/`[early-kill]`/`[autoresearch N/M]`/`config=…;`."""
    s = r.get("notes") or r.get("description") or ""
    s = re.sub(r"^\[(crash|early-kill|early_stopped)\]\s*", "", s)
    s = re.sub(r"^config=\w+;\s*", "", s)
    s = re.sub(r"^\[autoresearch \d+/\d+\]\s*", "", s)
    return s.strip()


def _kill_tag(kill_reason: str) -> str:
    """Map a long triage reason to a short category for the inline label."""
    kr = (kill_reason or "").lower()
    if "kl" in kr and "divergence" in kr:
        m = re.search(r"\|kl\|=([\d.]+)", kr)
        return f"killed: kl={m.group(1)} (policy)" if m else "killed: policy divergence"
    if "loss" in kr and ("divergence" in kr or "blow" in kr):
        m = re.search(r"\|loss\|=([\d.]+)", kr)
        return f"killed: |loss|={m.group(1)}" if m else "killed: loss blow-up"
    if "step_time spike" in kr:
        m = re.search(r"spike ([\d.]+)s", kr)
        return f"killed: {m.group(1)}s GPU spike" if m else "killed: GPU spike"
    if "mean step_time" in kr or "step_time" in kr:
        m = re.search(r"= ?([\d.]+)s", kr)
        return f"killed: {m.group(1)}s/step (slow)" if m else "killed: GPU slow"
    if "no reward" in kr or "baseline" in kr:
        return "killed: no learning"
    return f"killed: {kill_reason[:30]}" if kill_reason else "killed early"


def _label(r: dict, is_best: bool = False) -> str:
    """Multi-line inline callout. Full detail lives in hover."""
    head = f"<b>E{r['experiment']}</b>"
    if r.get("runtime_min"):
        head += f" · {int(r['runtime_min'])}min"
    m = r.get("metrics") or {}
    if r["status"] == "EARLY_KILL":
        head += f" · {_kill_tag(m.get('kill_reason', ''))}"
    elif r["status"] == "CRASH" and m.get("crash_reason"):
        cr = m["crash_reason"]
        head += f" · crashed: {cr[:30]}{'…' if len(cr) > 30 else ''}"
    else:
        head += f" · {_LEGEND_NAMES.get(r['status'], r['status']).lower()}"

    summary = _short_summary(r)
    if summary and len(summary) > 64:
        summary = summary[:61] + "…"

    bits = [f"train={r['score']:.2f}"]
    if isinstance(m.get("final_kl"), (int, float)):
        bits.append(f"kl={m['final_kl']:.2f}")
    if r.get("steps"):
        bits.append(f"{r['steps']}st")

    lines = [head]
    if summary:
        lines.append(summary)
    lines.append(" · ".join(bits))

    # Eval scores get their own line so the promotion signal stands out.
    ho = (m.get("heldout") or {})
    rg = (m.get("regression") or {})
    eval_bits = []
    if ho.get("n") and ho.get("mean_total") is not None:
        eval_bits.append(f"<b>ho={ho['mean_total']:.2f}</b>/16")
    if rg.get("n") and rg.get("mean_total") is not None:
        eval_bits.append(f"reg={rg['mean_total']:.2f}/16")
    if eval_bits:
        lines.append("eval mean_total · " + " · ".join(eval_bits))
    return "<br>".join(lines)


def plot_progress(task: Optional[str] = None) -> Path:
    """Plot a per-task progress chart. With no `task`, plots every task with
    a results.jsonl in `experiments/`. Returns the HTML output path.
    """
    tasks = [task] if task else sorted(
        d.name for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and (d / "results.jsonl").exists()
    )
    tasks = [t for t in tasks if load_results(t)]
    if not tasks:
        print("No results yet. Use `log` to add experiments.")
        return Path()

    subtitles = []
    for t in tasks:
        rs = load_results(t)
        n_kept = sum(1 for r in rs if r["status"] in ("KEEP", "BASELINE"))
        rt = sum(r.get("runtime_min", 0) for r in rs)
        subtitles.append(f"{t} — {len(rs)} experiments, {n_kept} kept{f', {rt:.0f}min total' if rt else ''}")

    fig = make_subplots(rows=len(tasks), cols=1, subplot_titles=subtitles, vertical_spacing=0.22)
    # Capture the count *before* we add per-row labels — these initial entries
    # are subplot-title annotations and must NOT be toggled by the labels
    # show/hide button below.
    n_static_annotations = len(fig.layout.annotations)

    # Centered horizontally on the dot (xshift=0); alternate y across 4
    # levels so adjacent boxes don't overlap even though they share x.
    POSITIONS = [
        ("center",  0, -110),  # above
        ("center",  0,  110),  # below
        ("center",  0, -200),  # higher above
        ("center",  0,  200),  # lower below
    ]

    for i, t in enumerate(tasks, 1):
        results = sorted(load_results(t), key=lambda r: r["experiment"])
        in_flight = _scrape_in_flight_run(t)
        if in_flight and not any(r["experiment"] == in_flight["experiment"] for r in results):
            results.append(in_flight)
        legend_seen: set[str] = set()
        # "Best" follows the promotion rule (heldout mean_total when available,
        # else train reward). Matches what `_decide_status` uses to pick KEEP.
        if results:
            scored = [(r, promotion_score(r)) for r in results]
            scored = [(r, s) for r, s in scored if s is not None]
            best_exp = max(scored, key=lambda rs: rs[1])[0]["experiment"] if scored else None
        else:
            best_exp = None
        # Heldout-score range across KEEP/BASELINE rows for marker tinting.
        keep_heldouts = [
            ((r.get("metrics") or {}).get("heldout") or {}).get("mean_total")
            for r in results if r["status"] in ("KEEP", "BASELINE")
        ]
        keep_heldouts = [s for s in keep_heldouts if s is not None]
        ho_min = min(keep_heldouts) if keep_heldouts else None
        ho_max = max(keep_heldouts) if keep_heldouts else None

        for j, r in enumerate(results):
            cfg = _STATUS_STYLE.get(r["status"], _STATUS_STYLE["DISCARD"])
            is_best = (r["experiment"] == best_exp)
            show_legend = (i == 1) and r["status"] not in legend_seen
            legend_seen.add(r["status"])

            # KEEP/BASELINE markers tinted by heldout mean_total: deeper green
            # for higher heldout, light grey-green for the lowest heldout in
            # range, mid-green when heldout is unavailable. Other statuses
            # keep their flat status color.
            marker_color = cfg["color"]
            if r["status"] in ("KEEP", "BASELINE") and ho_max is not None:
                ho_val = ((r.get("metrics") or {}).get("heldout") or {}).get("mean_total")
                if ho_val is not None and ho_max > ho_min:
                    # Linear interp from #b8e6c8 (light) to #1a7a3a (deep green)
                    t_norm = (ho_val - ho_min) / (ho_max - ho_min)
                    marker_color = _green_gradient(t_norm)
                elif ho_val is None:
                    marker_color = "#7fbc8c"  # mid-green = "no heldout data"

            marker_kwargs = dict(color=marker_color, size=cfg["size"], opacity=cfg["opacity"],
                                 line=dict(width=1, color=cfg["line_color"]), symbol=cfg["symbol"])
            if is_best:
                marker_kwargs.update(size=cfg["size"] + 6, line=dict(width=3, color="#27ae60"))

            # Marker carries the same hover content as its label box, but
            # `hoverinfo="skip"` keeps it dormant while labels are visible.
            # The toggle button flips hoverinfo→"text" when labels go off so
            # you can still hover the dots.
            fig.add_trace(go.Scatter(
                x=[r["experiment"]], y=[r["score"]], mode="markers",
                marker=marker_kwargs,
                name=_LEGEND_NAMES[r["status"]], legendgroup=r["status"], showlegend=show_legend,
                hovertext=[_hover(r)],
                hovertemplate="%{hovertext}<extra></extra>",
                hoverinfo="skip",
                hoverlabel=dict(bgcolor="white", bordercolor=cfg["color"]),
            ), row=i, col=1)

            xanchor, ax, ay = POSITIONS[j % len(POSITIONS)]
            xref, yref = (f"x{i}", f"y{i}") if i > 1 else ("x", "y")

            ann_text = _label(r, is_best=is_best)
            if r.get("wandb_url"):
                link_color = "#1a7a3a" if is_best else cfg["text_color"]
                ann_text += (
                    f"<br><a href='{r['wandb_url']}' target='_blank' "
                    f"style='color:{link_color}'>↗ W&amp;B</a>"
                )
            elif r["status"] == "RUNNING":
                ann_text += "<br><i>(W&amp;B link pending)</i>"

            hover_html = _hover(r)
            hoverlabel = dict(bgcolor="white", bordercolor=cfg["color"],
                              font=dict(size=11))
            if is_best:
                fig.add_annotation(
                    x=r["experiment"], y=r["score"], xref=xref, yref=yref,
                    text=ann_text, hovertext=hover_html, captureevents=True,
                    hoverlabel=hoverlabel,
                    showarrow=False, xshift=ax, yshift=ay,
                    xanchor=xanchor, yanchor="middle",
                    font=dict(size=10, color="#1a7a3a", family="Arial"),
                    align="left", bgcolor="rgba(248,255,248,0.98)",
                    bordercolor="#27ae60", borderwidth=2, borderpad=6,
                    width=240,
                )
            else:
                fig.add_annotation(
                    x=r["experiment"], y=r["score"], xref=xref, yref=yref,
                    text=ann_text, hovertext=hover_html, captureevents=True,
                    hoverlabel=hoverlabel,
                    showarrow=False, xshift=ax, yshift=ay,
                    xanchor=xanchor, yanchor="middle",
                    font=dict(size=9, color=cfg["text_color"]),
                    align="left", bgcolor="rgba(255,255,255,0.98)",
                    bordercolor=cfg["color"], borderwidth=1, borderpad=5,
                    width=220,
                )

        kept = [(r["experiment"], r["score"]) for r in results if r["status"] in ("KEEP", "BASELINE")]
        if kept:
            best = float("-inf")
            xs, ys = [], []
            for x, y in kept:
                best = max(best, y)
                xs.append(x); ys.append(best)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="#27ae60", width=2, shape="hv"),
                name="Running best", legendgroup="best", showlegend=(i == 1),
                hoverinfo="skip",
            ), row=i, col=1)

        # (best-row callout is rendered inline above with green styling +
        # ↗ W&B link; no separate ★ annotation needed.)

        # Dotted horizontal line at the BASELINE score (the first/only
        # successful clean run). Anchors the eye to the bar future runs
        # need to clear.
        baseline_row = next(
            (r for r in results if r["status"] == "BASELINE"), None,
        )
        if baseline_row:
            fig.add_hline(
                y=baseline_row["score"],
                line=dict(color="#27ae60", width=1.5, dash="dot"),
                annotation_text=f"baseline E{baseline_row['experiment']} = {baseline_row['score']:.2f}",
                annotation_position="top right",
                annotation_font=dict(color="#1a7a3a", size=10),
                row=i, col=1,
            )

        fig.update_yaxes(title_text="Train reward (higher is better)",
                         rangemode="tozero",
                         gridcolor="#eee", zerolinecolor="#ddd", row=i, col=1)
        fig.update_xaxes(title_text="Experiment #", dtick=1,
                         gridcolor="#f4f4f4", row=i, col=1)

    # Append reward summary + W&B link to each subplot title (plotly stores
    # subplot titles as layout annotations).
    for t in tasks:
        formula = _REWARD_SUMMARIES.get(t)
        if not formula:
            continue
        for ann in fig.layout.annotations:
            if ann.text and ann.text.startswith(f"{t} —"):
                ann.text = (
                    f"{ann.text}<br>"
                    f"<span style='font-size:11px;color:#666'>{formula} "
                    f"<a href='{_WANDB_PROJECT_URL}' target='_blank'>[W&B]</a></span>"
                )

    # Bump subplot title font + nudge titles up so they sit above the plot,
    # not crammed against it. Plotly stores subplot titles as annotations.
    for ann in fig.layout.annotations:
        if ann.text and any(ann.text.startswith(f"{t} —") or f"<br>" in (ann.text or "") for t in tasks):
            if ann.font is None or ann.font.size is None:
                ann.font = dict(size=14, color="#333")
            ann.yshift = 24

    # Real HTML switch (injected via post_script) — sits top-left of the
    # chart, calls Plotly.relayout + Plotly.restyle directly. Toggling off
    # hides every per-row annotation AND flips marker hoverinfo skip→text
    # so the dots become hoverable with the same tooltip.
    label_indices = list(range(n_static_annotations, len(fig.layout.annotations)))
    n_traces = len(fig.data)

    fig.update_layout(
        title=dict(text="GRPO Experiment Progress",
                   font=dict(size=22, color="#222"),
                   x=0.02, xanchor="left", y=0.985, yanchor="top"),
        height=720 * len(tasks),
        margin=dict(t=130, b=80, l=80, r=80),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#222"),
        showlegend=False,
        hovermode="closest",
        hoverlabel=dict(align="left", bgcolor="white", bordercolor="#bbb",
                        font=dict(size=11, color="#222"),
                        namelength=-1),
    )

    out_dir = _task_dir(tasks[0]) if len(tasks) == 1 else EXPERIMENTS_DIR
    html = out_dir / "progress.html"
    fig.write_html(
        str(html),
        post_script=plotly_label_toggle(
            label_indices=label_indices,
            n_traces=n_traces,
            label="labels",
            position="top-left",
        ),
    )
    print(f"Saved {html}")
    try:
        png = out_dir / "progress.png"
        fig.write_image(str(png), width=1800, height=650 * len(tasks))
        print(f"Saved {png}")
    except Exception as e:
        print(f"(skipped PNG export: {e})")

    # Eval-metrics chart appended to the same progress.html so train + eval
    # live in one page. plotly.js is already loaded by the main fig above, so
    # the second fig embeds with include_plotlyjs=False. A short description
    # block explains what mean_total is + cites the heldout technique.
    eval_fig, _ = _build_eval_fig(tasks)
    if eval_fig is not None:
        with open(html, "a") as fh:
            fh.write(eval_fig.to_html(include_plotlyjs=False, full_html=False))
            fh.write(_eval_description_html())
        print(f"Appended eval chart to {html}")
    return html


def _eval_description_html() -> str:
    """Static explainer block appended below the eval chart."""
    return """
<style>
  .eval-explainer { max-width: 1200px; margin: 24px auto 48px; padding: 20px 28px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #222; background: #fafbfc; border: 1px solid #e1e4e8; border-radius: 6px;
    line-height: 1.55; }
  .eval-explainer h3 { margin-top: 0; color: #1a7a3a; font-size: 15px; }
  .eval-explainer h4 { margin: 18px 0 6px; font-size: 13px; color: #444; }
  .eval-explainer code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px;
    font-family: 'SF Mono', Monaco, monospace; font-size: 12px; }
  .eval-explainer ul { margin: 6px 0; padding-left: 22px; font-size: 13px; }
  .eval-explainer a { color: #1a7a3a; }
  .eval-explainer .formula { background: #fff; padding: 8px 12px; border-left: 3px solid #27ae60;
    font-family: 'SF Mono', Monaco, monospace; font-size: 12px; margin: 8px 0; }
</style>
<div class="eval-explainer">
  <h3>What you're looking at: held-out generalization for an RLVR policy</h3>
  <p>Each iter trains a GRPO LoRA on synthetic <code>direct_debit_explainer</code>
  prompts. After training, the frozen policy is evaluated at <code>temperature=0</code>
  on two distinct splits the model never saw during training. The KEEP / DISCARD
  promotion decision uses <strong><code>heldout.mean_total</code></strong> rather
  than train reward, so we promote on generalization, not on what the policy
  was directly optimized against.</p>

  <h4>mean_total — the per-rubric sum</h4>
  <p>Each of the 7 verifiable rewards from <code>dd_explainer_rewards.py</code> scores
  one rubric. <code>mean_total</code> is the row-mean of their sum:</p>
  <div class="formula">mean_total = mean<sub>row</sub>(
    schema_valid + in_enum + f1_triggers + prev_amount_correct
    + no_hallucinated_facts + underpayment_ok + well_formed )</div>
  <p>Range is <code>[-9.5, 16.0]</code> in principle (worst-case all rubrics return
  their negative penalty; best-case all hit their max). On the current dataset,
  random-init tends to sit near <code>~6</code>, trained policies climb past
  <code>~8.5</code>. Higher = better. Solid green diamond line on the chart.</p>

  <h4>pass_all_pct — the joint-AND pass rate (option-A semantics)</h4>
  <p>Fraction of rows where <em>every</em> rubric clears its pass threshold. A
  defensive guardrail (<code>prev_amount_correct &ge; 0</code>) counts as pass — only
  active hallucinations (<code>-3</code>) fail it — because the prompt never
  asks the model to cite previous £ amounts, so requiring <code>+2</code> would
  be unreachable. Dotted blue (held-out) and purple (regression) lines.</p>

  <h4>The two splits</h4>
  <ul>
    <li><strong>held-out (val):</strong> 1000 IID rows from the synthetic dataset,
    excluded from training (seed=42). Catches prompt overfitting + reward hacking
    that's invisible on the train distribution. ~18% of the 5500-row dataset
    &rarr; ±1.6% std-err on pass-rate.</li>
    <li><strong>regression (test):</strong> 100 of 187 known-failed real-prod traces
    from <code>.error_analysis_cache/</code>. The "did we actually fix the failure
    modes that motivated the project" signal — never touched during HP search.</li>
  </ul>

  <h4>Why heldout for promotion (not train reward)</h4>
  <p>RL with verifiable rewards is uniquely vulnerable to <em>reward hacking</em>
  — the policy finds outputs that maximise the verifier without satisfying the
  spirit of the rubric. Train reward is computed on the same prompts the policy
  was just gradient-stepped against, so a wrapper that maximises train reward
  alone preferentially promotes hacks. Held-out evaluation breaks this loop:
  same rubrics, prompts the model has never been pushed against. Standard
  practice in RLHF/RLVR — see references below.</p>

  <h4>Why <code>mean_total</code> for promotion, <code>pass_all_pct</code> for reporting</h4>
  <p>The two metrics answer different questions. <strong><code>mean_total</code></strong>
  uses gradient signal from every rubric on every row (1000 rows × 7 rubrics =
  7000 numeric values), so it cleanly separates similar policies, catches partial
  regressions, and gives the autoresearch loop a continuous comparator. It
  answers <em>"is this policy better"</em>. <strong><code>pass_all_pct</code></strong>
  collapses each row to a single pass/fail bit (~85% information loss) and is
  bottlenecked by the worst rubric — currently <code>well_formed</code> caps it
  at ~40%. It answers <em>"would I ship this"</em> — the right metric for product
  reporting, the wrong metric for promotion ranking.</p>

  <h4>References</h4>
  <ul>
    <li>Ouyang et al. 2022, <em>Training Language Models to Follow Instructions
    with Human Feedback</em> (InstructGPT) —
    <a href="https://arxiv.org/abs/2203.02155">arXiv:2203.02155</a>.
    Establishes the held-out prompt-set evaluation pattern for RLHF policies.</li>
    <li>Stiennon et al. 2020, <em>Learning to summarize from human feedback</em>
    — <a href="https://arxiv.org/abs/2009.01325">arXiv:2009.01325</a>.
    Earlier formal use of held-out prompt eval to detect reward over-optimization.</li>
    <li>Lambert et al. 2024, <em>T&uuml;lu 3: Pushing Frontiers in Open Language
    Model Post-Training</em> —
    <a href="https://arxiv.org/abs/2411.15124">arXiv:2411.15124</a>.
    Coins "RLVR" (Reinforcement Learning with Verifiable Rewards) and
    formalises the held-out + regression split for verifier-based RL.</li>
    <li>Shao et al. 2024, <em>DeepSeekMath: Pushing the Limits of Mathematical
    Reasoning in Open Language Models</em> —
    <a href="https://arxiv.org/abs/2402.03300">arXiv:2402.03300</a>.
    Introduces GRPO and reports held-out math-bench accuracy as the policy's
    promotion criterion.</li>
    <li>Gao, Schulman, Hilton 2023, <em>Scaling Laws for Reward Model
    Overoptimization</em> — <a href="https://arxiv.org/abs/2210.10760">arXiv:2210.10760</a>.
    The canonical paper on RL reward hacking; quantifies why train-reward
    can't be trusted and held-out is needed.</li>
  </ul>
</div>
"""


def _build_eval_fig(tasks: list[str]):
    """Build the held-out + regression pass-rate figure. Returns (fig, any_data).

    Returns (None, False) if no task has any eval data yet.
    """
    fig = make_subplots(
        rows=len(tasks), cols=1, vertical_spacing=0.18,
        subplot_titles=[f"{t} — eval mean_total + pass-rates" for t in tasks],
    )
    any_data = False
    for i, t in enumerate(tasks, 1):
        rows = sorted(load_results(t), key=lambda r: r["experiment"])
        for series_key, label, color, axis in (
            ("heldout_mean", "held-out mean_total", "#27ae60", "y2"),
            ("heldout", "held-out pass_all (%)", "#3498db", "y"),
            ("regression", "regression pass_all (%)", "#9b59b6", "y"),
        ):
            xs, ys, hover = [], [], []
            for r in rows:
                ev_key = "heldout" if series_key == "heldout_mean" else series_key
                ev = (r.get("metrics") or {}).get(ev_key) or {}
                if not ev.get("n"):
                    continue
                if series_key == "heldout_mean":
                    val = ev.get("mean_total", 0)
                    hover.append(
                        f"<b>E{r['experiment']} — held-out</b><br>"
                        f"mean_total = {val:.2f}<br>"
                        f"pass_all = {ev.get('pass_all', 0)}/{ev['n']}"
                    )
                else:
                    val = ev.get("pass_all_pct", 0)
                    hover.append(
                        f"<b>E{r['experiment']} — {label}</b><br>"
                        f"pass_all = {ev.get('pass_all', 0)}/{ev['n']} ({val:.1f}%)<br>"
                        f"mean_total = {ev.get('mean_total', 0):.2f}"
                    )
                xs.append(r["experiment"])
                ys.append(val)
            if xs:
                any_data = True
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers", name=label,
                    line=dict(color=color, width=2,
                              dash="solid" if series_key == "heldout_mean" else "dot"),
                    marker=dict(size=10, color=color, line=dict(width=1, color="white"),
                                symbol="diamond" if series_key == "heldout_mean" else "circle"),
                    hovertext=hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                    yaxis=axis,
                    legendgroup=series_key, showlegend=(i == 1),
                ), row=i, col=1)
        fig.update_yaxes(title_text="pass_all (%)", range=[0, 100],
                         gridcolor="#eee", row=i, col=1)
        fig.update_xaxes(title_text="Experiment #", dtick=1,
                         gridcolor="#f4f4f4", row=i, col=1)

    if not any_data:
        return None, False

    # Secondary y-axis on the right for mean_total (rubric scale, not %)
    fig.update_layout(
        yaxis2=dict(title="mean_total (Σ 7 rubric scores, max 16.0)",
                    overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
        title=dict(
            text=(
                "<b>GRPO Eval Progress</b> · promotion uses held-out "
                "<code>mean_total</code> (= mean<sub>row</sub>(Σ 7 rubrics))"
                "<br><span style='font-size:11px;color:#666'>"
                "held-out (1000 IID rows excluded from training) · "
                "regression (100 known-failed prod traces) · "
                "see explainer + references below"
                "</span>"
            ),
            font=dict(size=18, color="#222"),
            x=0.02, xanchor="left", y=0.98, yanchor="top",
        ),
        height=520 * len(tasks),
        margin=dict(t=130, b=60, l=80, r=80),
        template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#bbb", borderwidth=1),
        hovermode="closest",
        hoverlabel=dict(align="left", bgcolor="white", bordercolor="#bbb",
                        font=dict(size=11), namelength=-1),
    )
    return fig, True


# ── CLI ─────────────────────────────────────────────────────────────


class Status(str, Enum):
    KEEP = "KEEP"
    DISCARD = "DISCARD"
    BASELINE = "BASELINE"
    RUNNING = "RUNNING"
    EARLY_KILL = "EARLY_KILL"
    CRASH = "CRASH"


app = typer.Typer(help="Experiment progress tracker for gemma4 GRPO runs.", add_completion=False)


@app.command()
def log(
    score: float = typer.Option(..., help="Primary metric — typically best train reward."),
    description: str = typer.Option(..., "-d", help="Short summary of what this run changed."),
    task: str = typer.Option(DEFAULT_TASK, help="Task identifier (one subplot per task)."),
    config_name: str = typer.Option("", help="Hydra config name used (e.g. train, train_mem)."),
    steps: int = typer.Option(0, help="Steps actually completed."),
    runtime_min: float = typer.Option(0.0, help="Wall-clock runtime in minutes."),
    status: Optional[Status] = typer.Option(None, help="Override auto status (BASELINE/KEEP/DISCARD/RUNNING/CRASH)."),
    notes: str = typer.Option("", help="Free-text notes mirroring wandb run notes."),
    wandb_url: str = typer.Option("", help="W&B run URL."),
    wandb_run_id: str = typer.Option(""),
    wandb_run_name: str = typer.Option(""),
    metric_kv: list[str] = typer.Option(
        [], "--metric", "-m",
        help="Extra metric as key=value (repeatable), e.g. -m best_reward=11.5 -m final_kl=0.012",
    ),
):
    """Append one experiment row + refresh the plot."""
    metrics: dict = {}
    for kv in metric_kv:
        if "=" not in kv:
            raise typer.BadParameter(f"--metric expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        try:
            metrics[k] = float(v)
        except ValueError:
            metrics[k] = v
    log_experiment(
        score=score, description=description, task=task, config_name=config_name,
        steps=steps, runtime_min=runtime_min,
        status=status.value if status else None,
        metrics=metrics, notes=notes,
        wandb_url=wandb_url, wandb_run_id=wandb_run_id, wandb_run_name=wandb_run_name,
    )
    plot_progress(task=task)


@app.command()
def plot(task: Optional[str] = typer.Option(None, help="Plot just this task (default: all).")):
    """Regenerate the progress chart from current results.jsonl."""
    plot_progress(task=task)


if __name__ == "__main__":
    app()
