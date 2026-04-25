"""Keep `experiments/dd_explainer/current_run.json` in sync with whatever
iter the autoresearch loop is on.

Runs as a detached daemon (setsid + nohup) so it survives Claude / SSH
disconnects. Polls the latest `logs/autoresearch_*.log` every N seconds:

  * On the most recent `Iter N/M: ...` line with no matching
    `Iter N/M finished ...` below it → write current_run.json.
  * On `Iter N/M finished` with no fresher `Iter K/M:` after → delete
    current_run.json.

Useful as a fallback when the autoresearch process predates the inline
sidecar writes (Python doesn't hot-reload an edited module).
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
SIDECAR = ROOT / "experiments" / "dd_explainer" / "current_run.json"
RESULTS = ROOT / "experiments" / "dd_explainer" / "results.jsonl"
POLL_S = 15

ITER_START_RE = re.compile(r"\[(?P<ts>[\d\-T:Z]+)\] Iter (?P<n>\d+)/(?P<m>\d+): (?P<rest>.*)")
ITER_END_RE = re.compile(r"\[[\d\-T:Z]+\] Iter (?P<n>\d+)/(?P<m>\d+) finished")
DESC_RE = re.compile(r"-d (\[autoresearch [^\]]+\][^-]+?)(?= --|$)")
WANDB_RE = re.compile(r"https://wandb\.ai/[\w\-./]+/runs/[\w\-]+")


def _latest_log() -> Path | None:
    # Match training stdout logs only (timestamped name); skip the sidecar
    # `autoresearch_events.log` which has no step_time / `$ ...` lines.
    logs = sorted(LOG_DIR.glob("autoresearch_*T*Z.log"))
    return logs[-1] if logs else None


def _experiment_count() -> int:
    if not RESULTS.exists():
        return 0
    return sum(1 for line in RESULTS.read_text().splitlines() if line.strip())


def _tick() -> None:
    log = _latest_log()
    if log is None:
        return
    text = log.read_text(errors="replace")

    # Find every Iter N/M start, and whether each has a matching finish.
    starts = list(ITER_START_RE.finditer(text))
    ends = {int(m.group("n")) for m in ITER_END_RE.finditer(text)}
    if not starts:
        return

    last = starts[-1]
    iter_n = int(last.group("n"))
    iter_m = int(last.group("m"))

    if iter_n in ends:
        # Latest started iter has finished — no in-flight run. Drop sidecar.
        if SIDECAR.exists():
            SIDECAR.unlink()
            print(f"[updater] iter {iter_n}/{iter_m} finished — sidecar removed")
        return

    # Pull description from the `$ ... -d <desc> --max-steps ...` line that
    # follows the Iter line.
    after_iter = text[last.end():]
    cmd_line = next((ln for ln in after_iter.splitlines() if ln.startswith("$ ")), "")
    m_desc = DESC_RE.search(cmd_line)
    desc = m_desc.group(1).strip() if m_desc else last.group("rest").strip()

    # Wandb URL inside this iter's chunk
    chunk = text[last.start():]
    urls = WANDB_RE.findall(chunk)
    wandb_url = urls[-1] if urls else ""

    # Iter timestamp from the log line
    started_at = last.group("ts")
    if not started_at.endswith("Z") and "+" not in started_at:
        started_at += "Z"

    payload = {
        "experiment": _experiment_count(),
        "config_name": "train_fast",  # all current schedule entries
        "description": desc,
        "notes": desc,
        "started_at": started_at,
        "log_path": str(log),
        "iter_marker": f"Iter {iter_n}/{iter_m}",
        "wandb_url": wandb_url,
    }

    # Only write if changed (avoids touching mtime every poll).
    if SIDECAR.exists():
        try:
            cur = json.loads(SIDECAR.read_text())
            if cur == payload:
                return
        except json.JSONDecodeError:
            pass
    SIDECAR.write_text(json.dumps(payload, indent=2))
    print(f"[updater] sidecar → iter {iter_n}/{iter_m} (E{payload['experiment']}, "
          f"wandb={wandb_url or 'pending'})")


def main() -> None:
    print(f"[updater] starting — poll every {POLL_S}s, log dir={LOG_DIR}")
    while True:
        try:
            _tick()
        except Exception as e:
            print(f"[updater] tick error: {e}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
