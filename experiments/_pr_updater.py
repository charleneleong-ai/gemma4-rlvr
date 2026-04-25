"""Periodic PR refresher for the autoresearch sweep.

Polls every N seconds and:
1. Re-renders `docs/autoresearch_progress.png` from `results.jsonl`.
2. If the PNG changed: git add + commit + push (so the embedded image
   in the PR body refreshes — GitHub serves it via `?raw=true`).
3. Re-builds the sweep-narrative table from results.jsonl and PATCHes
   PR #4's body between the markers
   `<!-- SWEEP_NARRATIVE_START -->` … `<!-- SWEEP_NARRATIVE_END -->`.

Detach with `setsid nohup .venv/bin/python -u experiments/_pr_updater.py …`
so it survives Claude / SSH disconnect.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "dd_explainer" / "results.jsonl"
PNG = ROOT / "docs" / "autoresearch_progress.png"
RENDER = ROOT / "experiments" / "_render_screenshot.py"
POLL_S = 300
PR_NUMBER = 4
REPO = "charleneleong-ai/gemma4-rlvr"
BRANCH = "feat/auto-research-loop"
MARKER_START = "<!-- SWEEP_NARRATIVE_START -->"
MARKER_END = "<!-- SWEEP_NARRATIVE_END -->"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _kill_short(kill_reason: str) -> str:
    kr = (kill_reason or "").lower()
    if "kl" in kr and "policy" in kr or "kl" in kr and "divergence" in kr:
        import re
        m = re.search(r"\|kl\|=([\d.]+)", kr)
        return f"`kl={m.group(1)}` (policy)" if m else "policy divergence"
    if "step_time" in kr and "spike" in kr:
        return "GPU spike"
    if "step_time" in kr or "slow" in kr:
        return "GPU slow"
    if "loss" in kr:
        return "loss blow-up"
    if "no reward" in kr or "baseline" in kr:
        return "no learning"
    return kill_reason[:40] if kill_reason else "killed"


def _build_narrative() -> str:
    if not RESULTS.exists():
        return "_(no results yet)_"
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    if not rows:
        return "_(no results yet)_"
    n_kept = sum(1 for r in rows if r["status"] in ("KEEP", "BASELINE"))
    n_killed = sum(1 for r in rows if r["status"] == "EARLY_KILL")
    n_crash = sum(1 for r in rows if r["status"] == "CRASH")
    n_run = sum(1 for r in rows if r["status"] == "RUNNING")
    runtime = sum(r.get("runtime_min", 0) for r in rows)
    best = max(rows, key=lambda r: r["score"])

    lines = [
        f"_Last refresh: {_ts()}._ "
        f"**{len(rows)}** experiments · {n_kept} kept · {n_killed} killed · {n_crash} crashed"
        + (f" · {n_run} running" if n_run else "")
        + f" · {runtime:.0f}min total · best so far: **{best['score']:.2f}** (E{best['experiment']})\n",
        "| E | status | score | runtime | notes |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        m = r.get("metrics") or {}
        if r["status"] == "EARLY_KILL":
            tag = f"killed: {_kill_short(m.get('kill_reason', ''))}"
        elif r["status"] == "CRASH":
            cr = m.get("crash_reason") or ""
            tag = f"crashed: {cr[:40]}" if cr else "crashed"
        elif r["status"] == "RUNNING":
            tag = "running"
        else:
            tag = r["status"].lower()
        notes = (r.get("notes") or "").replace("|", "\\|")[:80]
        rt = f"{r.get('runtime_min', 0):.0f}min"
        score = f"{r['score']:.2f}"
        link = f" [↗](https://wandb.ai/chaleong/gemma4-rlvr/runs/{r.get('wandb_run_id', '')})" if r.get("wandb_run_id") else ""
        lines.append(f"| E{r['experiment']} | {tag} | {score} | {rt} | {notes}{link} |")
    return "\n".join(lines)


def _refresh_png() -> bool:
    before_mtime = PNG.stat().st_mtime if PNG.exists() else -1
    venv_py = ROOT / ".venv" / "bin" / "python"
    subprocess.run(
        [str(venv_py), str(RENDER)],
        cwd=str(ROOT / "experiments"), check=False,
        capture_output=True,
    )
    if not PNG.exists():
        return False
    return PNG.stat().st_mtime > before_mtime  # was rewritten


def _git_push_png_if_changed() -> bool:
    """Stage png, commit + push only if working-tree differs from HEAD."""
    subprocess.run(["git", "add", str(PNG)], cwd=str(ROOT), check=True)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=str(ROOT),
    )
    if diff.returncode == 0:
        return False  # no actual content change after staging
    subprocess.run(
        ["git", "commit", "-m", f"docs: refresh autoresearch screenshot ({_ts()})"],
        cwd=str(ROOT), check=True,
    )
    push = subprocess.run(
        ["git", "push", "origin", BRANCH], cwd=str(ROOT),
        capture_output=True, text=True,
    )
    if push.returncode != 0:
        print(f"[pr_updater] push failed: {push.stderr.strip()[:200]}")
    return push.returncode == 0


def _patch_pr_body(narrative: str) -> bool:
    body_proc = subprocess.run(
        ["gh", "api", f"repos/{REPO}/pulls/{PR_NUMBER}", "--jq", ".body"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if body_proc.returncode != 0:
        print(f"[pr_updater] gh api failed: {body_proc.stderr.strip()[:200]}")
        return False
    body = body_proc.stdout
    if MARKER_START not in body or MARKER_END not in body:
        print("[pr_updater] markers missing in PR body — manual setup needed")
        return False
    pre, _, rest = body.partition(MARKER_START)
    _, _, post = rest.partition(MARKER_END)
    new = pre + MARKER_START + "\n" + narrative + "\n" + MARKER_END + post
    if new == body:
        return False
    payload = json.dumps({"body": new})
    proc = subprocess.run(
        ["gh", "api", f"repos/{REPO}/pulls/{PR_NUMBER}",
         "--method", "PATCH", "--input", "-"],
        input=payload, text=True, capture_output=True, cwd=str(ROOT),
    )
    if proc.returncode != 0:
        print(f"[pr_updater] PATCH failed: {proc.stderr.strip()[:200]}")
        return False
    return True


def main() -> None:
    print(f"[pr_updater] starting — poll every {POLL_S}s, PR #{PR_NUMBER} on {REPO}")
    while True:
        try:
            png_changed = _refresh_png()
            pushed = _git_push_png_if_changed() if png_changed else False
            narrative = _build_narrative()
            patched = _patch_pr_body(narrative)
            print(f"[pr_updater] {_ts()} — png_changed={png_changed} pushed={pushed} pr_patched={patched}")
        except Exception as e:
            print(f"[pr_updater] tick error: {e}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
