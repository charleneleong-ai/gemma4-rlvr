"""Autonomous overnight research loop for gemma4 GRPO with active triage.

Sequentially launches a small sweep of `train.py` invocations. Each child
auto-logs to `experiments/dd_explainer/results.jsonl` and refreshes the
progress plot via train.py's try/finally hook (always — early-stop, SIGINT,
or unexpected exception).

The loop monitors child stdout in real time and SIGINTs the child when the
run is clearly non-optimal — so a bad config does not burn the overnight
budget. SIGINT lets the child's finally hook log a CRASH row instead of
vanishing silently. Triage is conservative: act only on signals that mean
the run cannot recover.

Triage thresholds (per-step dict logs from TRL):
  * mean step_time over last 5 steps > 90s   → too slow vs ~60s target
  * single step_time > 200s                  → memory/runtime spike
  * |kl| > 1.0                               → policy divergence
  * |loss| > 10                              → numerical divergence
  * no reward beat baseline-1 in last 25 steps → not learning

Time bounding: every iteration also runs with --max-steps 80 + patience=8,
so even with no triage trigger a run caps at ~80 min.

Usage:
  python experiments/autoresearch.py                  # full schedule
  python experiments/autoresearch.py --max-iters 3
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python"
TRAIN = ROOT / "train.py"
TASK = "dd_explainer"
LOG_PATH_ENV = "AUTORESEARCH_LOG_PATH"


def _results_path(config_name: str) -> Path:
    p = ROOT / "experiments" / TASK / config_name
    p.mkdir(parents=True, exist_ok=True)
    return p / "results.jsonl"


def _current_path(config_name: str) -> Path:
    p = ROOT / "experiments" / TASK / config_name
    p.mkdir(parents=True, exist_ok=True)
    return p / "current_run.json"

SCHEDULES_DIR = ROOT / "configs" / "schedules"


def _load_schedule(name: str) -> tuple[list[tuple[str, list[str], str]], list[str]]:
    """Load a schedule yaml from `configs/schedules/<name>.yaml`.

    Returns (iters, common_overrides) where iters is the same shape as the
    historical hardcoded SCHEDULE: list of (config_name, cli_overrides, description).
    """
    import yaml  # local import — only needed for the autoresearch CLI
    path = SCHEDULES_DIR / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in SCHEDULES_DIR.glob("*.yaml")))
        raise FileNotFoundError(
            f"Schedule {name!r} not found at {path}. Available: {available or '(none)'}"
        )
    data = yaml.safe_load(path.read_text())
    common = list(data.get("common_overrides") or [])
    iters_raw = data.get("iters") or []
    iters: list[tuple[str, list[str], str]] = []
    for entry in iters_raw:
        iters.append((
            entry["config"],
            list(entry.get("overrides") or []),
            entry["description"],
        ))
    return iters, common

# ── Triage thresholds ──────────────────────────────────────────────
SLOW_WINDOW = 5            # rolling window for step_time check
SLOW_MEAN_S = 90.0         # mean s/step above this = abandon
SLOW_SPIKE_S = 200.0       # any single step over this = abandon
KL_DIVERGE = 1.0           # |kl| above this = policy collapse
LOSS_DIVERGE = 10.0        # |loss| above this = numerical blow-up
NO_LEARN_WINDOW = 25       # steps without beating baseline-1 reward = abandon

GPU_LOW_UTIL_PCT  = 8      # % below which GPU counts as idle
GPU_LOW_UTIL_S    = 300    # 5 min consecutive idle → hang kill
GPU_GRACE_S       = 180    # 3 min startup grace (model loading)
GPU_POLL_S        = 30     # nvidia-smi poll interval (s)
GPU_MEM_HEADROOM  = 0.65   # if peak_mem/total < this → suggest bigger batch
GPU_UTIL_LOW_WARN = 50     # mean util% below this → suggest tuning



def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _baseline_score(config_name: str) -> float:
    """Best train-reward score among prior KEEP/BASELINE rows for this config.

    Used by the mid-training triage gate (the `no reward > baseline-1` rule
    in `_run_with_triage`). Triage compares against in-flight train rewards,
    so this must stay train-reward-scaled — *not* heldout. The KEEP/DISCARD
    decision uses heldout via `experiment_progress.promotion_score` instead.

    Scoped per-config so triage thresholds don't bleed across presets.
    """
    p = _results_path(config_name)
    if not p.exists():
        return float("-inf")
    rows = [json.loads(l) for l in p.read_text().strip().splitlines() if l]
    kept = [r["score"] for r in rows if r.get("status") in ("KEEP", "BASELINE")]
    return max(kept) if kept else float("-inf")


_CRASH_PATTERNS = [
    (re.compile(r"torch\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError"), "CUDA OOM"),
    (re.compile(r"Killed\s*$", re.MULTILINE), "killed by host (likely cgroup OOM)"),
    (re.compile(r"AssertionError:?\s*(.*)"), lambda m: f"AssertionError: {m.group(1).strip()[:80]}"),
    (re.compile(r"RuntimeError:?\s*(.*)"), lambda m: f"RuntimeError: {m.group(1).strip()[:80]}"),
    (re.compile(r"ValueError:?\s*(.*)"), lambda m: f"ValueError: {m.group(1).strip()[:80]}"),
    (re.compile(r"FileNotFoundError:?\s*(.*)"), lambda m: f"FileNotFoundError: {m.group(1).strip()[:80]}"),
    (re.compile(r"^([A-Z][A-Za-z]+Error):?\s*(.*)", re.MULTILINE),
     lambda m: f"{m.group(1)}: {m.group(2).strip()[:80]}"),
]


def _crash_reason_from_lines(lines: list[str]) -> str:
    """Best-effort: scan recent child stdout for a crash cause.

    Looks for OOM markers first (most common on this hardware), then any
    `*Error:` line. Falls back to the last non-empty line if no pattern hits.
    """
    text = "".join(lines[-200:])  # cap memory; tail is where the trace lives
    for pat, mapper in _CRASH_PATTERNS:
        m = pat.search(text)
        if m:
            return mapper(m) if callable(mapper) else mapper
    last = next((ln.strip() for ln in reversed(lines) if ln.strip()), "")
    return f"unknown: {last[:120]}" if last else "unknown crash"


def _patch_last_with_crash_reason(config_name: str, crash_reason: str) -> None:
    """Add `crash_reason` to the latest results.jsonl row's metrics.

    Called when autoresearch detects a non-zero child exit with no triage
    kill_reason — the row was already written by train.py's finally hook
    with status=CRASH but no detected cause.
    """
    p = _results_path(config_name)
    if not p.exists():
        return
    lines = p.read_text().splitlines()
    if not lines:
        return
    try:
        last = json.loads(lines[-1])
    except json.JSONDecodeError:
        return
    if last.get("status") != "CRASH":
        return
    last.setdefault("metrics", {})["crash_reason"] = crash_reason
    lines[-1] = json.dumps(last)
    p.write_text("\n".join(lines) + "\n")
    print(f"[{_ts()}] [autoresearch] tagged last CRASH row with crash_reason={crash_reason!r}")


def _relabel_last_as_early_kill(config_name: str, kill_reason: str) -> None:
    """Rewrite the last results.jsonl row from CRASH→EARLY_KILL.

    train.py's finally hook can't distinguish a triage SIGINT from a real
    crash — it always logs CRASH. Only autoresearch knows the kill was
    deliberate, so we patch the row here so the chart can colour triaged
    kills (grey) separately from true crashes (red).
    """
    p = _results_path(config_name)
    if not p.exists():
        return
    lines = p.read_text().splitlines()
    if not lines:
        return
    try:
        last = json.loads(lines[-1])
    except json.JSONDecodeError:
        return
    if last.get("status") != "CRASH":
        return
    last["status"] = "EARLY_KILL"
    last.setdefault("metrics", {})["kill_reason"] = kill_reason
    desc = last.get("description", "")
    if desc.startswith("[crash] "):
        desc = "[early-kill] " + desc[len("[crash] "):]
    elif not desc.startswith("[early-kill]"):
        desc = "[early-kill] " + desc
    last["description"] = desc
    lines[-1] = json.dumps(last)
    p.write_text("\n".join(lines) + "\n")
    print(f"[{_ts()}] [autoresearch] relabelled last row CRASH→EARLY_KILL ({kill_reason})")


def _wait_for_train_to_clear(poll_s: int = 30, max_wait_s: int = 600) -> None:
    start = time.monotonic()
    while time.monotonic() - start < max_wait_s:
        out = subprocess.run(["pgrep", "-f", "train.py train"],
                             capture_output=True, text=True)
        if not out.stdout.strip():
            return
        print(f"[{_ts()}] waiting for prior train.py to exit (pids={out.stdout.split()})")
        time.sleep(poll_s)
    print(f"[{_ts()}] WARN: prior train.py still running after {max_wait_s}s; proceeding")


# ── Step-line parsing ──────────────────────────────────────────────

_FIELD_RE_CACHE: dict[str, re.Pattern] = {}


def _extract(line: str, key: str) -> float | None:
    """Pull `'<key>': '<float>'` from a TRL step dict line. Returns None if missing."""
    pat = _FIELD_RE_CACHE.get(key)
    if pat is None:
        pat = re.compile(rf"'{re.escape(key)}': '([^']+)'")
        _FIELD_RE_CACHE[key] = pat
    m = pat.search(line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _is_step_line(line: str) -> bool:
    return "'step_time'" in line and "'reward'" in line



def _gpu_stats() -> dict | None:
    """Return {util_pct, mem_used_gb, mem_total_gb} or None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=10, text=True,
        ).strip().split("\n")[0]
        util, used, total = [x.strip() for x in out.split(",")]
        return {
            "util_pct": int(util),
            "mem_used_gb": round(int(used) / 1024, 1),
            "mem_total_gb": round(int(total) / 1024, 1),
        }
    except Exception:
        return None


def _gpu_advisor(samples: list[dict], cmd: list[str]) -> None:
    """Log GPU utilisation summary + optimisation hints after an iter."""
    if not samples:
        return
    utils = [s["util_pct"] for s in samples]
    mems  = [s["mem_used_gb"] for s in samples]
    total = samples[0]["mem_total_gb"]
    mean_util = sum(utils) / len(utils)
    peak_mem  = max(mems)

    print(f"\n[{_ts()}] [gpu_advisor] mean_util={mean_util:.0f}%  "
          f"peak_mem={peak_mem:.1f}/{total:.0f}GB  "
          f"({len(samples)} samples over run)")

    hints = []
    if peak_mem / total < GPU_MEM_HEADROOM:
        headroom_gb = total - peak_mem
        hints.append(
            f"  • Memory underused (peak {peak_mem:.1f}/{total:.0f}GB, "
            f"{headroom_gb:.0f}GB free) — consider:"
        )
        # Extract current values from cmd for concrete suggestions
        def _get(flag):
            try:
                return cmd[cmd.index(flag) + 1]
            except (ValueError, IndexError):
                return None
        bs   = _get("--batch-size")   or _get("-b")
        ng   = _get("--num-generations")
        msl  = _get("--max-seq-length")
        if bs:
            hints.append(f"    batch_size {bs} → {int(bs)*2} (double)")
        if ng:
            hints.append(f"    num_generations {ng} → {min(int(ng)*2, 16)}"
                         f" (keep divisible by batch*grad_accum)")
        if msl:
            hints.append(f"    max_seq_length {msl} → {int(msl)*2} (longer context)")
        if not any([bs, ng, msl]):
            hints.append("    increase batch_size / num_generations / max_seq_length")

    if mean_util < GPU_UTIL_LOW_WARN:
        hints.append(
            f"  • Compute underused (mean {mean_util:.0f}%) — possible causes:"
        )
        hints.append("    dataloader CPU bottleneck · eval/preview overhead · small batch")
    elif mean_util >= 85:
        hints.append(f"  • GPU well-utilised (mean {mean_util:.0f}%) ✓")

    if hints:
        print("\n".join(hints))
    print()

# ── Per-iteration execution with triage ────────────────────────────

def _run_with_triage(cmd: list[str], baseline_score: float, config_name: str = "") -> tuple[int, str | None]:
    """Run one training subprocess, monitor stdout, SIGINT on triage trigger.

    Returns (exit_code, kill_reason). kill_reason is None if no triage fired.
    """
    proc = subprocess.Popen(cmd, cwd=str(ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, text=True)
    assert proc.stdout is not None


    # ── GPU watcher thread ──────────────────────────────────────────
    gpu_kill_reason: list[str] = []   # mutable, written by thread
    gpu_samples:     list[dict] = []  # util/mem snapshots for advisor
    _gpu_stop = threading.Event()

    def _gpu_watcher():
        _gpu_stop.wait(GPU_GRACE_S)  # wait out model loading
        low_since: float | None = None
        while not _gpu_stop.is_set() and proc.poll() is None:
            s = _gpu_stats()
            if s:
                gpu_samples.append(s)
                if s["util_pct"] < GPU_LOW_UTIL_PCT:
                    if low_since is None:
                        low_since = time.monotonic()
                    elif time.monotonic() - low_since >= GPU_LOW_UTIL_S:
                        gpu_kill_reason.append(
                            f"GPU util {s['util_pct']}% < {GPU_LOW_UTIL_PCT}% "
                            f"for {GPU_LOW_UTIL_S // 60}min+ — likely hang"
                        )
                        _gpu_stop.set()
                        break
                else:
                    low_since = None
            _gpu_stop.wait(GPU_POLL_S)

    _gpu_thread = threading.Thread(target=_gpu_watcher, daemon=True, name="gpu-watcher")
    _gpu_thread.start()
    # ────────────────────────────────────────────────────────────────
    step_times: deque[float] = deque(maxlen=SLOW_WINDOW)
    recent_rewards: deque[float] = deque(maxlen=NO_LEARN_WINDOW)
    recent_lines: deque[str] = deque(maxlen=200)
    n_steps = 0
    kill_reason: str | None = None

    try:
        for line in iter(proc.stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
            recent_lines.append(line)

            if not _is_step_line(line):
                continue

            n_steps += 1
            st = _extract(line, "step_time")
            rw = _extract(line, "reward")
            kl = _extract(line, "kl")
            loss = _extract(line, "loss")

            if st is not None:
                step_times.append(st)
                if st > SLOW_SPIKE_S:
                    kill_reason = f"step_time spike {st:.1f}s > {SLOW_SPIKE_S}s on step {n_steps}"
                    break
                if len(step_times) == SLOW_WINDOW:
                    mean_st = sum(step_times) / SLOW_WINDOW
                    if mean_st > SLOW_MEAN_S:
                        kill_reason = (f"mean step_time over last {SLOW_WINDOW} = "
                                       f"{mean_st:.1f}s > {SLOW_MEAN_S}s")
                        break

            if kl is not None and abs(kl) > KL_DIVERGE:
                kill_reason = f"|kl|={abs(kl):.3f} > {KL_DIVERGE} suggests policy divergence"
                break

            if loss is not None and abs(loss) > LOSS_DIVERGE:
                kill_reason = f"|loss|={abs(loss):.3f} > {LOSS_DIVERGE} suggests divergence"
                break

            # GPU hang check (set by watcher thread)
            if gpu_kill_reason:
                kill_reason = gpu_kill_reason[0]
                break

            if rw is not None:
                recent_rewards.append(rw)
                if (baseline_score != float("-inf")
                        and len(recent_rewards) == NO_LEARN_WINDOW
                        and max(recent_rewards) < baseline_score - 1.0):
                    kill_reason = (f"no reward > baseline-1 ({baseline_score - 1:.2f}) "
                                   f"in last {NO_LEARN_WINDOW} steps; max={max(recent_rewards):.2f}")
                    break

        if kill_reason:
            _gpu_stop.set()
            _gpu_thread.join(timeout=5)
            _gpu_advisor(gpu_samples, cmd)
            print(f"\n[{_ts()}] [triage] {kill_reason}")
            print(f"[{_ts()}] [triage] sending SIGINT — child's finally hook will log a CRASH row")
            try:
                proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                print(f"[{_ts()}] [triage] child not exiting after SIGINT — escalating to SIGTERM")
                proc.terminate()
                proc.wait(timeout=30)
        else:
            proc.wait()
        _gpu_stop.set()
        _gpu_thread.join(timeout=5)
        _gpu_advisor(gpu_samples, cmd)
    except KeyboardInterrupt:
        _gpu_stop.set()
        print(f"\n[{_ts()}] [autoresearch] interrupted; forwarding SIGINT to child")
        proc.send_signal(signal.SIGINT)
        proc.wait()
        raise

    crash_reason: str | None = None
    if kill_reason is None and proc.returncode not in (0, None):
        crash_reason = _crash_reason_from_lines(list(recent_lines))
        print(f"[{_ts()}] [autoresearch] child exited {proc.returncode} — crash_reason: {crash_reason}")
        _patch_last_with_crash_reason(config_name, crash_reason)

    return proc.returncode, kill_reason, crash_reason


# ── Main loop ──────────────────────────────────────────────────────

app = typer.Typer(help="Autonomous overnight GRPO research loop with triage.",
                  add_completion=False)


@app.command()
def main(
    schedule: str = typer.Option(
        "v1_explore", "--schedule",
        help=f"Schedule name in configs/schedules/<name>.yaml. "
             f"Available: {sorted(p.stem for p in SCHEDULES_DIR.glob('*.yaml'))}",
    ),
    max_iters: int = typer.Option(0, "--max-iters",
                                  help="Cap the schedule (0 = run all iters)."),
    skip_baseline: bool = typer.Option(False, "--skip-baseline"),
    start_iter: int = typer.Option(1, "--start-iter", help="1-based iter index to resume from (skip earlier iters)."),
    pause_s: int = typer.Option(15, "--pause-s",
                                help="Seconds to wait between runs (lets GPU mem fully free)."),
):
    iters_full, common_overrides = _load_schedule(schedule)
    cap = max_iters or len(iters_full)
    print(f"[{_ts()}] Autoresearch starting — schedule={schedule!r}, "
          f"{min(cap, len(iters_full))}/{len(iters_full)} iterations.")
    _wait_for_train_to_clear()

    started = time.monotonic()
    iters = iters_full[:cap]
    if skip_baseline:
        iters = iters[1:]
    if start_iter > 1:
        skip_n = start_iter - 1
        print(f"[{_ts()}] Skipping first {skip_n} iter(s) as requested (--start-iter {start_iter}).")
        iters = iters[skip_n:]
    total = len(iters)

    for i, (config, extras, notes) in enumerate(iters, start=1):
        baseline = _baseline_score(config)
        results_p = _results_path(config)
        current_p = _current_path(config)
        desc = f"[autoresearch {i}/{total}] {notes}"
        cmd = [
            str(PYTHON), str(TRAIN), "train",
            "-c", config, "-d", desc,
            *common_overrides, *extras,
        ]
        print(f"\n{'='*70}\n[{_ts()}] Iter {i}/{total}: {config} {' '.join(extras)}")
        print(f"  baseline_score so far = {baseline if baseline != float('-inf') else 'none'}")
        print(f"$ {' '.join(cmd)}\n{'='*70}")

        # Sidecar lets plot_progress render the in-flight iter.
        log_path = os.environ.get(LOG_PATH_ENV, "")
        if not log_path:
            logs = sorted((ROOT / "logs").glob("autoresearch_*.log"))
            log_path = str(logs[-1]) if logs else ""
        prior_n = sum(1 for _ in results_p.read_text().splitlines() if _.strip()) if results_p.exists() else 0
        current_p.write_text(json.dumps({
            "experiment": prior_n,
            "config_name": config,
            "description": desc,
            "notes": desc,
            "started_at": _ts(),
            "log_path": log_path,
            "iter_marker": f"Iter {i}/{total}",
        }))

        iter_start = time.monotonic()
        try:
            ret, kill_reason, crash_reason = _run_with_triage(cmd, baseline, config)
        except KeyboardInterrupt:
            print(f"[{_ts()}] [autoresearch] aborted by SIGINT at iter {i}/{total}.")
            current_p.unlink(missing_ok=True)
            return
        if kill_reason:
            _relabel_last_as_early_kill(config, kill_reason)
        elif crash_reason:
            _patch_last_with_crash_reason(config, crash_reason)
        current_p.unlink(missing_ok=True)
        mins = (time.monotonic() - iter_start) / 60.0
        if kill_reason:
            outcome = f"triaged: {kill_reason}"
        elif crash_reason:
            outcome = f"crashed ({ret}): {crash_reason}"
        else:
            outcome = f"natural exit ({ret})"
        print(f"\n[{_ts()}] Iter {i}/{total} finished in {mins:.1f}min — {outcome}")

        time.sleep(pause_s)

    total_h = (time.monotonic() - started) / 3600.0
    print(f"\n[{_ts()}] Autoresearch finished — {total} iterations in {total_h:.2f}h.")
    # Per-config artifacts: list every config touched in this schedule.
    seen_configs = sorted({cfg for cfg, _, _ in iters})
    for cfg in seen_configs:
        rp = _results_path(cfg)
        print(f"[{_ts()}] Final results [{cfg}]: {rp}")
        print(f"[{_ts()}] Plot           [{cfg}]: {rp.parent / 'progress.html'}")


if __name__ == "__main__":
    app()
