"""Autonomous overnight research loop for gemma4 GRPO via shared SweepRunner.

The project-specific pieces stay local:
- schedule loading from `configs/schedules/<name>.yaml`
- stdout/GPU triage signals
- child-process row patching for `kill_reason` / `crash_reason`

The generic orchestration now lives in `autoresearch.SweepRunner`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

from autoresearch import IterPlan, SweepRunner  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

try:
    from experiments.experiment_progress import DEFAULT_TASK, load_results, plot_progress
except ImportError:
    from experiment_progress import DEFAULT_TASK, load_results, plot_progress

PYTHON = ROOT / ".venv" / "bin" / "python"
TRAIN = ROOT / "train.py"
LOG_PATH_ENV = "AUTORESEARCH_LOG_PATH"
SCHEDULES_DIR = ROOT / "configs" / "schedules"

# ── schedule loading ──────────────────────────────────────────────────


def _load_schedule(name: str) -> tuple[list[tuple[str, list[str], str]], list[str]]:
    """Load `configs/schedules/<name>.yaml`.

    Returns `(iters, common_overrides)` where each iter is `(config, overrides,
    description)`.
    """
    import yaml  # local import — CLI-only path

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
        iters.append(
            (
                entry["config"],
                list(entry.get("overrides") or []),
                entry["description"],
            )
        )
    return iters, common


# ── thresholds / parsing ──────────────────────────────────────────────

SLOW_WINDOW = 5
SLOW_MEAN_S = 130.0
SLOW_SPIKE_S = 200.0
KL_DIVERGE = 1.0
LOSS_DIVERGE = 10.0
NO_LEARN_WINDOW = 25

GPU_LOW_UTIL_PCT = 8
GPU_LOW_UTIL_S = 300
GPU_UNDERUTIL_PCT = 35
GPU_UNDERUTIL_S = 900
GPU_LOW_MEM_PCT = 35
GPU_LOW_MEM_S = 1800
GPU_GRACE_S = 180
GPU_POLL_S = 30
GPU_MEM_HEADROOM = 0.65
GPU_UTIL_LOW_WARN = 50

_FIELD_RE_CACHE: dict[str, re.Pattern[str]] = {}
WANDB_RE = re.compile(r"https://wandb\.ai/[\w\-./]+/runs/[\w\-]+")
_CRASH_PATTERNS: list[tuple[re.Pattern[str], str | Any]] = [
    (
        re.compile(r"torch\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError"),
        "CUDA OOM",
    ),
    (
        re.compile(r"Killed\s*$", re.MULTILINE),
        "killed by host (likely cgroup OOM)",
    ),
    (
        re.compile(r"AssertionError:?\s*(.*)"),
        lambda m: f"AssertionError: {m.group(1).strip()[:80]}",
    ),
    (
        re.compile(r"RuntimeError:?\s*(.*)"),
        lambda m: f"RuntimeError: {m.group(1).strip()[:80]}",
    ),
    (
        re.compile(r"ValueError:?\s*(.*)"),
        lambda m: f"ValueError: {m.group(1).strip()[:80]}",
    ),
    (
        re.compile(r"FileNotFoundError:?\s*(.*)"),
        lambda m: f"FileNotFoundError: {m.group(1).strip()[:80]}",
    ),
    (
        re.compile(r"^([A-Z][A-Za-z]+Error):?\s*(.*)", re.MULTILINE),
        lambda m: f"{m.group(1)}: {m.group(2).strip()[:80]}",
    ),
]


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _results_path(config_name: str) -> Path:
    p = ROOT / "experiments" / DEFAULT_TASK / config_name
    p.mkdir(parents=True, exist_ok=True)
    return p / "results.jsonl"


def _autoresearch_log_path() -> str:
    log_path = os.environ.get(LOG_PATH_ENV, "")
    if log_path:
        return log_path
    logs = sorted((ROOT / "logs").glob("autoresearch_*.log"))
    return str(logs[-1]) if logs else ""


def _baseline_score(config_name: str) -> float:
    rows = load_results(DEFAULT_TASK, config_name=config_name, include_superseded=True)
    kept = [r["score"] for r in rows if r.get("status") in ("KEEP", "BASELINE")]
    return max(kept) if kept else float("-inf")


def _extract(line: str, key: str) -> float | None:
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


def _gpu_stats() -> dict[str, float] | None:
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                timeout=10,
                text=True,
            )
            .strip()
            .split("\n")[0]
        )
        util, used, total = [x.strip() for x in out.split(",")]
        return {
            "util_pct": int(util),
            "mem_used_gb": round(int(used) / 1024, 1),
            "mem_total_gb": round(int(total) / 1024, 1),
        }
    except Exception:
        return None


def _gpu_advisor(samples: list[dict[str, float]], cmd: list[str]) -> None:
    if not samples:
        return
    utils = [s["util_pct"] for s in samples]
    mems = [s["mem_used_gb"] for s in samples]
    total = samples[0]["mem_total_gb"]
    mean_util = sum(utils) / len(utils)
    peak_mem = max(mems)

    print(
        f"\n[{_ts()}] [gpu_advisor] mean_util={mean_util:.0f}%  "
        f"peak_mem={peak_mem:.1f}/{total:.0f}GB  "
        f"({len(samples)} samples over run)",
        flush=True,
    )

    hints: list[str] = []
    if peak_mem / total < GPU_MEM_HEADROOM:
        headroom_gb = total - peak_mem
        hints.append(
            "  • Memory underused (peak "
            f"{peak_mem:.1f}/{total:.0f}GB, {headroom_gb:.0f}GB free) "
            "— consider:"
        )

        def _get(flag: str) -> str | None:
            try:
                return cmd[cmd.index(flag) + 1]
            except (ValueError, IndexError):
                return None

        bs = _get("--batch-size") or _get("-b")
        ng = _get("--num-generations")
        msl = _get("--max-seq-length")
        if bs:
            hints.append(f"    batch_size {bs} → {int(bs) * 2}")
        if ng:
            hints.append(
                f"    num_generations {ng} → {min(int(ng) * 2, 16)} "
                "(keep divisible by batch*grad_accum)"
            )
        if msl:
            hints.append(f"    max_seq_length {msl} → {int(msl) * 2}")
        if not any([bs, ng, msl]):
            hints.append("    increase batch_size / num_generations / max_seq_length")

    if mean_util < GPU_UTIL_LOW_WARN:
        hints.append(f"  • Compute underused (mean {mean_util:.0f}%) — possible causes:")
        hints.append("    dataloader CPU bottleneck · eval/preview overhead · small batch")
    elif mean_util >= 85:
        hints.append(f"  • GPU well-utilised (mean {mean_util:.0f}%) ✓")

    if hints:
        print("\n".join(hints), flush=True)
    print(flush=True)


def _crash_reason_from_lines(lines: list[str]) -> str:
    text = "".join(lines[-200:])
    for pat, mapper in _CRASH_PATTERNS:
        m = pat.search(text)
        if m:
            return mapper(m) if callable(mapper) else mapper
    last = next((ln.strip() for ln in reversed(lines) if ln.strip()), "")
    return f"unknown: {last[:120]}" if last else "unknown crash"


def _patch_logged_row(
    config_name: str,
    experiment_num: int,
    *,
    metrics_update: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = _results_path(config_name)
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return None

    patched_row: dict[str, Any] | None = None
    for idx in range(len(lines) - 1, -1, -1):
        row = json.loads(lines[idx])
        if row.get("experiment") != experiment_num:
            continue
        if metrics_update:
            row.setdefault("metrics", {}).update(metrics_update)
        lines[idx] = json.dumps(row)
        patched_row = row
        break

    if patched_row is None:
        return None
    path.write_text("\n".join(lines) + "\n")
    return patched_row


# ── adapters ──────────────────────────────────────────────────────────


class ScheduleIterPlanner:
    def __init__(
        self,
        *,
        schedule: str,
        max_iters: int,
        skip_baseline: bool,
        start_iter: int,
    ) -> None:
        self.schedule = schedule
        iters_full, self.common_overrides = _load_schedule(schedule)
        cap = max_iters or len(iters_full)
        iters = iters_full[:cap]
        if skip_baseline:
            iters = iters[1:]
        if start_iter > 1:
            iters = iters[start_iter - 1 :]
        self.iters = iters
        self.total = len(iters)
        self.seen_configs = sorted({cfg for cfg, _, _ in iters})

    def plan_iters(self, history: list[dict[str, Any]]):
        for i, (config, extras, notes) in enumerate(self.iters, start=1):
            desc = f"[autoresearch {i}/{self.total}] {notes}"
            cmd = [
                str(PYTHON),
                str(TRAIN),
                "train",
                "-c",
                config,
                "-d",
                desc,
                *self.common_overrides,
                *extras,
            ]
            print(f"\n{'=' * 70}\n[{_ts()}] Iter {i}/{self.total}: {config} {' '.join(extras)}")
            print(
                f"  baseline_score so far = "
                f"{_baseline_score(config) if _baseline_score(config) != float('-inf') else 'none'}"
            )
            print(f"$ {' '.join(cmd)}\n{'=' * 70}")
            yield IterPlan(
                cmd=cmd,
                description=desc,
                notes=desc,
                config_name=config,
                timeout_min=24 * 60,
                popen_kwargs={
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "bufsize": 1,
                    "text": True,
                },
                sidecar_payload={
                    "log_path": _autoresearch_log_path(),
                    "iter_marker": f"Iter {i}/{self.total}",
                },
                cwd=ROOT,
            )


class GemmaTriageMonitor:
    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.config_name = ""
        self.expected_experiment = 0
        self.baseline_score = float("-inf")
        self.kill_reason: str | None = None
        self.wandb_url = ""
        self.step_times: deque[float] = deque(maxlen=SLOW_WINDOW)
        self.recent_rewards: deque[float] = deque(maxlen=NO_LEARN_WINDOW)
        self.recent_lines: deque[str] = deque(maxlen=200)
        self.n_steps = 0
        self.gpu_samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._gpu_thread: threading.Thread | None = None
        self._proc: subprocess.Popen[str] | None = None
        self.started_at = 0.0
        self.last_exit_code = 0

    def _set_kill_reason(self, reason: str) -> None:
        if self.kill_reason is None:
            self.kill_reason = reason

    def setup(self, plan: IterPlan, proc: subprocess.Popen[bytes], baseline: float) -> str | None:
        del baseline
        self._reset()
        self.config_name = plan.config_name or ""
        self.expected_experiment = len(
            load_results(DEFAULT_TASK, config_name=self.config_name, include_superseded=True)
        )
        self.baseline_score = _baseline_score(self.config_name)
        self._proc = proc  # type: ignore[assignment]
        self.started_at = time.monotonic()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="stdout-reader",
        )
        self._gpu_thread = threading.Thread(
            target=self._gpu_loop,
            daemon=True,
            name="gpu-watcher",
        )
        self._reader_thread.start()
        self._gpu_thread.start()
        return None

    def _reader_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for line in iter(self._proc.stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
            self.recent_lines.append(line)

            urls = WANDB_RE.findall(line)
            if urls:
                self.wandb_url = urls[-1]

            if not _is_step_line(line):
                continue

            self.n_steps += 1
            st = _extract(line, "step_time")
            rw = _extract(line, "reward")
            kl = _extract(line, "kl")
            loss = _extract(line, "loss")

            if st is not None:
                self.step_times.append(st)
                if st > SLOW_SPIKE_S:
                    self._set_kill_reason(
                        f"step_time spike {st:.1f}s > {SLOW_SPIKE_S}s on step {self.n_steps}"
                    )
                elif len(self.step_times) == SLOW_WINDOW:
                    mean_st = sum(self.step_times) / SLOW_WINDOW
                    if mean_st > SLOW_MEAN_S:
                        self._set_kill_reason(
                            "mean step_time over last "
                            f"{SLOW_WINDOW} = {mean_st:.1f}s > {SLOW_MEAN_S}s"
                        )

            if kl is not None and abs(kl) > KL_DIVERGE:
                self._set_kill_reason(
                    f"|kl|={abs(kl):.3f} > {KL_DIVERGE} suggests policy divergence"
                )

            if loss is not None and abs(loss) > LOSS_DIVERGE:
                self._set_kill_reason(
                    f"|loss|={abs(loss):.3f} > {LOSS_DIVERGE} suggests divergence"
                )

            if rw is not None:
                self.recent_rewards.append(rw)
                if (
                    self.baseline_score != float("-inf")
                    and len(self.recent_rewards) == NO_LEARN_WINDOW
                    and max(self.recent_rewards) < self.baseline_score - 1.0
                ):
                    self._set_kill_reason(
                        f"no reward > baseline-1 ({self.baseline_score - 1:.2f}) "
                        f"in last {NO_LEARN_WINDOW} steps; max={max(self.recent_rewards):.2f}"
                    )

            if self._stop.is_set():
                break

    def _gpu_loop(self) -> None:
        assert self._proc is not None
        self._stop.wait(GPU_GRACE_S)
        low_since: float | None = None
        under_since: float | None = None
        mem_low_since: float | None = None
        peak_mem_pct = 0.0

        while not self._stop.is_set() and self._proc.poll() is None:
            sample = _gpu_stats()
            if sample:
                self.gpu_samples.append(sample)
                util = sample["util_pct"]
                mem_pct = sample["mem_used_gb"] / sample["mem_total_gb"] * 100
                peak_mem_pct = max(peak_mem_pct, mem_pct)

                if util < GPU_LOW_UTIL_PCT:
                    if low_since is None:
                        low_since = time.monotonic()
                    elif time.monotonic() - low_since >= GPU_LOW_UTIL_S:
                        self._set_kill_reason(
                            f"GPU util {util}% < {GPU_LOW_UTIL_PCT}% "
                            f"for {GPU_LOW_UTIL_S // 60}min+ — likely hang"
                        )
                        self._stop.set()
                        break
                    if under_since is None:
                        under_since = time.monotonic()
                elif util < GPU_UNDERUTIL_PCT:
                    low_since = None
                    if under_since is None:
                        under_since = time.monotonic()
                    elif time.monotonic() - under_since >= GPU_UNDERUTIL_S:
                        self._set_kill_reason(
                            f"GPU util sustained <{GPU_UNDERUTIL_PCT}% "
                            f"for {GPU_UNDERUTIL_S // 60}min+ — wasted compute "
                            f"(last sample {util}%, "
                            f"{sample['mem_used_gb']:.0f}/{sample['mem_total_gb']:.0f}GB)"
                        )
                        self._stop.set()
                        break
                else:
                    low_since = None
                    under_since = None

                if peak_mem_pct < GPU_LOW_MEM_PCT:
                    if mem_low_since is None:
                        mem_low_since = time.monotonic()
                    elif time.monotonic() - mem_low_since >= GPU_LOW_MEM_S:
                        self._set_kill_reason(
                            f"peak GPU mem {peak_mem_pct:.0f}% < {GPU_LOW_MEM_PCT}% "
                            f"for {GPU_LOW_MEM_S // 60}min+ — undersized config "
                            f"({sample['mem_used_gb']:.0f}/{sample['mem_total_gb']:.0f}GB; "
                            "try larger batch / num_generations / max_seq_length)"
                        )
                        self._stop.set()
                        break
                else:
                    mem_low_since = None
            self._stop.wait(GPU_POLL_S)

    def check(self, elapsed_s: float) -> str | None:
        del elapsed_s
        return self.kill_reason

    def teardown(self) -> None:
        self._stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5)
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=5)
        if self._proc is not None and self._proc.returncode is not None:
            self.last_exit_code = self._proc.returncode
        if self._proc is not None:
            _gpu_advisor(self.gpu_samples, [str(x) for x in self._proc.args])

    @property
    def runtime_min(self) -> float:
        if not self.started_at:
            return 0.0
        return (time.monotonic() - self.started_at) / 60.0

    @property
    def crash_reason(self) -> str | None:
        if self.kill_reason is not None:
            return None
        if self.last_exit_code in (0, None):
            return None
        return _crash_reason_from_lines(list(self.recent_lines))


class GemmaResultExtractor:
    def __init__(self, monitor: GemmaTriageMonitor) -> None:
        self.monitor = monitor

    def _logged_row(self, config_name: str) -> dict[str, Any] | None:
        rows = load_results(DEFAULT_TASK, config_name=config_name, include_superseded=True)
        for row in reversed(rows):
            if row.get("experiment") == self.monitor.expected_experiment:
                return row
        return rows[-1] if rows else None

    def extract(
        self,
        plan: IterPlan,
        run_id: str | None,
        exit_code: int,
    ) -> list[dict[str, Any]]:
        del run_id
        config_name = plan.config_name or ""
        row = self._logged_row(config_name)

        if row is not None:
            if self.monitor.kill_reason:
                patched = _patch_logged_row(
                    config_name,
                    row.get("experiment", self.monitor.expected_experiment),
                    metrics_update={"kill_reason": self.monitor.kill_reason},
                )
                row = patched or row
                row = dict(row)
                row["status"] = "EARLY_KILL"
            elif exit_code not in (0, None) and self.monitor.crash_reason:
                patched = _patch_logged_row(
                    config_name,
                    row.get("experiment", self.monitor.expected_experiment),
                    metrics_update={"crash_reason": self.monitor.crash_reason},
                )
                row = patched or row
                row = dict(row)
            else:
                row = dict(row)
            row["_prelogged"] = True
            return [row]

        metrics: dict[str, Any] = {}
        status = "DISCARD"
        if self.monitor.kill_reason:
            status = "EARLY_KILL"
            metrics["kill_reason"] = self.monitor.kill_reason
        elif exit_code not in (0, None):
            status = "CRASH"
            if self.monitor.crash_reason:
                metrics["crash_reason"] = self.monitor.crash_reason

        score = max(self.monitor.recent_rewards) if self.monitor.recent_rewards else 0.0
        return [
            {
                "task": DEFAULT_TASK,
                "config_name": config_name,
                "score": score,
                "metrics": metrics,
                "steps": self.monitor.n_steps,
                "runtime_min": self.monitor.runtime_min,
                "status": status,
                "description": plan.description,
                "notes": plan.notes,
                "wandb_url": self.monitor.wandb_url,
                "wandb_run_id": "",
                "wandb_run_name": "",
            }
        ]


# ── CLI ───────────────────────────────────────────────────────────────

app = typer.Typer(
    help="Autonomous overnight GRPO research loop with shared SweepRunner.",
    add_completion=False,
)


@app.command()
def main(
    schedule: str = typer.Option(
        "v1_explore",
        "--schedule",
        help=(
            "Schedule name in configs/schedules/<name>.yaml. Available: "
            f"{sorted(p.stem for p in SCHEDULES_DIR.glob('*.yaml'))}"
        ),
    ),
    max_iters: int = typer.Option(0, "--max-iters", help="Cap the schedule (0 = run all iters)."),
    skip_baseline: bool = typer.Option(False, "--skip-baseline"),
    start_iter: int = typer.Option(1, "--start-iter", help="1-based iter index to resume from."),
    pause_s: int = typer.Option(15, "--pause-s", help="Seconds to wait between runs."),
) -> None:
    planner = ScheduleIterPlanner(
        schedule=schedule,
        max_iters=max_iters,
        skip_baseline=skip_baseline,
        start_iter=start_iter,
    )
    print(f"[{_ts()}] Autoresearch starting — schedule={schedule!r}, {planner.total} iterations.")

    monitor = GemmaTriageMonitor()
    extractor = GemmaResultExtractor(monitor)
    started = time.monotonic()
    runner = SweepRunner(
        tag=DEFAULT_TASK,
        planner=planner,
        triage=monitor,
        extractor=extractor,
        experiments_dir=ROOT / "experiments",
        iter_timeout_min=24 * 60,
        triage_poll_s=1,
        pause_between_iters_s=pause_s,
    )
    result = runner.run()

    total_h = (time.monotonic() - started) / 3600.0
    print(f"\n[{_ts()}] Autoresearch finished — {result.iterations} iterations in {total_h:.2f}h.")
    for cfg in planner.seen_configs:
        rp = _results_path(cfg)
        print(f"[{_ts()}] Final results [{cfg}]: {rp}")
        print(f"[{_ts()}] Plot           [{cfg}]: {rp.parent / 'progress.html'}")
        plot_progress(config_name=cfg)


if __name__ == "__main__":
    app()
