"""Build the outlier-detection training set for the encoder gate.

Produces `data/outlier_set_v0.jsonl` with 200 rows:
  - 100 OOD examples — real account contexts mutated in obvious ways the
    LLM would fabricate against (drop contract_history, swap tariff names
    to gibberish, NaN/negative numerics, empty dd_change_history, broken
    dates, truncated payment_history).
  - 100 in-distribution examples — sampled from the same dataset, untouched.

Each output row carries:
  - `input_json`: the (possibly mutated) input
  - `is_outlier`: 0/1 label
  - `mutation`: name of the mutation applied (None for in-distribution)
  - `source_row_index`: original row index for traceability

The mutation distribution round-robins across all six mutators so the
encoder sees roughly equal counts per failure mode — useful for
per-mutation AUROC diagnosis once training lands.

Usage:

    uv run python scripts/build_outlier_set.py \\
      --dataset data/dd_dataset_20260427T201521Z_5500rows.jsonl \\
      --out data/outlier_set_v0.jsonl

Add `--n 200` to override the row count or `--seed 1337` for variants.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)

# A deterministic gibberish tariff name — short, lowercase, deliberately
# unlike anything in the real tariff vocabulary so the encoder gets a clean
# signal. Reused across all calls so the encoder doesn't learn "any string
# starting with z" but rather "this exact OOD marker".
_GIBBERISH_TARIFFS = ["zxq42_payg", "qrkk_var_v2", "blvf_fix99"]


def _drop_contract_history(inp: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Mutation 1: empty contract_history. LLM cites a tariff that isn't there."""
    inp["account_context"]["contract_history"] = []
    return inp


def _gibberish_tariff_names(inp: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Mutation 2: replace tariff_name with gibberish. LLM tends to ignore the
    gibberish and invent a plausible-looking real name in its citation."""
    for ch in inp["account_context"].get("contract_history", []) or []:
        ch["tariff_name"] = rng.choice(_GIBBERISH_TARIFFS)
    return inp


def _negative_numerics(inp: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Mutation 3: invert numeric DD amounts to a clearly-impossible negative.
    LLM tends to hallucinate a plausible positive amount."""
    ldd = inp["latest_dd_change"]
    for k in ("dd_amount", "dd_amount_change", "recommended_dd_amount", "yearly_predicted_energy_cost_gbp"):
        if isinstance(ldd.get(k), (int, float)):
            ldd[k] = -abs(ldd[k]) - 1.0
    for change in inp["account_context"].get("dd_change_history", []) or []:
        for k in ("dd_amount", "dd_amount_change", "recommended_dd_amount"):
            if isinstance(change.get(k), (int, float)):
                change[k] = -abs(change[k]) - 1.0
    return inp


def _empty_dd_change_history(inp: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Mutation 4: clear dd_change_history. LLM hallucinates a previous amount
    when prompted on contexts that need one."""
    inp["account_context"]["dd_change_history"] = []
    return inp


def _broken_dates(inp: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Mutation 5: end-before-start dates on the latest DD change + contracts.
    Reasoning failure — LLM often invents a remediation that doesn't match."""
    ldd = inp["latest_dd_change"]
    if ldd.get("datetime_from"):
        ldd["datetime_to"] = "1970-01-01T00:00:00"
    for ch in inp["account_context"].get("contract_history", []) or []:
        if ch.get("contract_start_date"):
            ch["contract_end_date"] = "1970-01-01"
    return inp


def _truncate_payment_history(inp: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Mutation 6: truncate payment_history to 1 row. LLM extrapolates a
    spurious pattern from a single data point."""
    ph = inp["account_context"].get("payment_history") or []
    if len(ph) > 1:
        inp["account_context"]["payment_history"] = ph[:1]
    return inp


def _large_debt(inp: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Mutation 7: very large DD amount (~£500-£1700 from a typical £130 base).
    The synthetic distribution has p99=£211 / max=£219, so this is well outside
    what Gemma saw at train time. Production routinely sees these — high-usage
    customers, commercial accounts, accumulated arrears — and the model tends
    to fabricate plausible-looking explanations for unfamiliar magnitudes.
    Domain-realistic OOD, unlike `negative_numerics`."""
    multiplier = rng.uniform(4.0, 8.0)
    ldd = inp["latest_dd_change"]
    for k in ("dd_amount", "recommended_dd_amount", "yearly_predicted_energy_cost_gbp"):
        if isinstance(ldd.get(k), (int, float)):
            ldd[k] = round(ldd[k] * multiplier, 2)
    return inp


def _significant_increase(inp: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Mutation 8: dd_amount_change £150-£300 (synthetic max is £40). Big
    monthly jumps are real production signal — tariff resets, regulatory
    price-cap moves, customer-induced rebanding — but never in the train
    set. LLM tends to invent a cause that doesn't match the actual context."""
    increase = round(rng.uniform(150, 300), 2)
    ldd = inp["latest_dd_change"]
    if isinstance(ldd.get("dd_amount_change"), (int, float)):
        ldd["dd_amount_change"] = increase
    if isinstance(ldd.get("dd_amount"), (int, float)):
        ldd["dd_amount"] = round(ldd["dd_amount"] + increase, 2)
    return inp


def _many_missed_payments(inp: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Mutation 9: chronic-misser — flip 6-10 of payment_history to
    is_payment_successful=False. Synthetic distribution has max 3 missed
    out of ~12. Production sees chronic financial-difficulty customers
    with way more — the LLM tends to hallucinate explanations (e.g.
    inventing arrears amounts) for unfamiliar miss densities."""
    ph = inp.get("account_context", {}).get("payment_history") or []
    if not ph:
        return inp
    n_to_miss = min(rng.randint(6, 10), len(ph))
    indices = rng.sample(range(len(ph)), n_to_miss)
    for idx in indices:
        ph[idx]["is_payment_successful"] = False
    return inp


def _significant_decrease(inp: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Mutation 10: dd_amount_change -£150 to -£300 (synthetic min is -£40).
    Mirror of significant_increase — large drops do happen in production
    (energy-efficiency upgrades, summer seasonal recalibration, leaving a
    fixed-rate tariff onto cheaper SVR) but are well outside the synthetic
    range. LLM has never seen jumps this big DOWNWARD and hallucinates."""
    decrease = round(rng.uniform(-300, -150), 2)
    ldd = inp["latest_dd_change"]
    if isinstance(ldd.get("dd_amount_change"), (int, float)):
        ldd["dd_amount_change"] = decrease
    if isinstance(ldd.get("dd_amount"), (int, float)):
        # Reflect the drop in the new dd_amount; clamp to a minimum of £5 so
        # we don't overlap with `negative_numerics`.
        ldd["dd_amount"] = round(max(5.0, ldd["dd_amount"] + decrease), 2)
    return inp


def _customer_in_credit(inp: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Mutation 11: customer in credit — paying 3-10× the recommended DD.
    Synthetic distribution has dd_amount/recommended ratio ∈ [0.96, 1.05]
    (DDs are well-calibrated to recommendations). Production sees
    accumulated credit from chronic overpayment after meter-read corrections,
    seasonal usage drops, etc. LLM tends to invent reasons for the gap."""
    ratio = rng.uniform(3.0, 10.0)
    ldd = inp["latest_dd_change"]
    if isinstance(ldd.get("dd_amount"), (int, float)):
        ldd["recommended_dd_amount"] = round(ldd["dd_amount"] / ratio, 2)
    return inp


MUTATIONS: dict[str, Callable[[dict[str, Any], random.Random], dict[str, Any]]] = {
    "drop_contract_history": _drop_contract_history,
    "gibberish_tariff_names": _gibberish_tariff_names,
    "negative_numerics": _negative_numerics,
    "empty_dd_change_history": _empty_dd_change_history,
    "broken_dates": _broken_dates,
    "truncate_payment_history": _truncate_payment_history,
    "large_debt": _large_debt,
    "significant_increase": _significant_increase,
    "many_missed_payments": _many_missed_payments,
    "significant_decrease": _significant_decrease,
    "customer_in_credit": _customer_in_credit,
}


def _load_dataset(dataset: Path) -> list[dict[str, Any]]:
    """Load the synthetic dataset, skipping the first row (which is metadata
    about the generator, not a training example)."""
    rows: list[dict[str, Any]] = []
    with dataset.open() as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tail_flags(
    input_json: dict[str, Any],
    dd_p90: float,
    change_p90: float,
    missed_threshold: int,
    ratio_p95: float,
) -> list[str]:
    """Return zero or more tail-flag tags for in-distribution rows that sit
    in the natural distribution tail. Tags are independent — a row can be
    `high_debt` AND `many_missed` if the customer is genuinely a chronic
    high-usage misser.

    Tags:
      - `high_debt`     — dd_amount in top 10% of natural distribution
      - `high_change`   — |dd_amount_change| in top 10% (covers both increase + decrease)
      - `many_missed`   — n_missed at the natural maximum (synthetic max=3)
      - `in_credit`     — dd_amount/recommended_dd_amount in top 5% (slightly overpaying)

    Used by the encoder eval to measure false-positive rate on legitimate
    tail customers — we don't want the gate to confuse natural extremes with
    the synthetic mutations of the same shape.
    """
    ldd = input_json.get("latest_dd_change", {}) or {}
    ph = input_json.get("account_context", {}).get("payment_history") or []
    dd = ldd.get("dd_amount")
    rec = ldd.get("recommended_dd_amount")
    change = ldd.get("dd_amount_change")
    n_missed = sum(1 for p in ph if p.get("is_payment_successful") is False)

    flags: list[str] = []
    if dd is not None and dd >= dd_p90:
        flags.append("high_debt")
    if change is not None and abs(change) >= change_p90:
        flags.append("high_change")
    if n_missed >= missed_threshold:
        flags.append("many_missed")
    if dd is not None and rec and rec > 0 and (dd / rec) >= ratio_p95:
        flags.append("in_credit")
    return flags


def _compute_thresholds(rows: list[dict[str, Any]]) -> tuple[float, float, int, float]:
    """Returns (dd_amount_p90, |dd_amount_change|_p90, max natural n_missed,
    dd/recommended ratio p95) — thresholds above which a row counts as
    'natural tail'. Computed on the full dataset so thresholds are stable
    across reruns."""
    dds: list[float] = []
    changes: list[float] = []
    miss_counts: list[int] = []
    ratios: list[float] = []
    for r in rows:
        inp = r.get("input_json", {})
        ldd = inp.get("latest_dd_change", {}) or {}
        ph = inp.get("account_context", {}).get("payment_history") or []
        if ldd.get("dd_amount") is not None:
            dds.append(abs(float(ldd["dd_amount"])))
        if ldd.get("dd_amount_change") is not None:
            changes.append(abs(float(ldd["dd_amount_change"])))
        if ldd.get("dd_amount") and ldd.get("recommended_dd_amount"):
            ratios.append(float(ldd["dd_amount"]) / float(ldd["recommended_dd_amount"]))
        miss_counts.append(sum(1 for p in ph if p.get("is_payment_successful") is False))
    dds.sort()
    changes.sort()
    ratios.sort()
    return (
        dds[int(len(dds) * 0.9)],
        changes[int(len(changes) * 0.9)],
        max(miss_counts) if miss_counts else 0,
        ratios[int(len(ratios) * 0.95)] if ratios else 1.0,
    )


@app.command()
def main(
    dataset: Path = typer.Option(
        Path("data/dd_dataset_20260427T201521Z_5500rows.jsonl"),
        help="Source dataset (multi-trigger v2.0.0 by default).",
    ),
    out: Path = typer.Option(
        Path("data/outlier_set_v0.jsonl"),
        help="Output JSONL path.",
    ),
    n: int = typer.Option(200, help="Total rows to emit (50/50 OOD/in-distribution)."),
    seed: int = typer.Option(42, help="RNG seed for reproducibility."),
) -> None:
    """Emit a 200-row outlier-detection training set."""
    if n % 2 != 0:
        raise typer.BadParameter(f"--n must be even (50/50 split). Got {n}.")
    if not dataset.exists():
        raise typer.BadParameter(f"dataset not found: {dataset}")

    rng = random.Random(seed)
    rows = _load_dataset(dataset)
    dd_p90, change_p90, missed_max, ratio_p95 = _compute_thresholds(rows)
    typer.echo(
        f"loaded {len(rows)} rows from {dataset.name}  "
        f"(p90 dd_amount=£{dd_p90:.0f}, p90 |dd_amount_change|=£{change_p90:.1f}, "
        f"max n_missed={missed_max}, p95 dd/rec ratio={ratio_p95:.2f})"
    )

    half = n // 2
    sampled = rng.sample(rows, n)
    ood_source, indist_source = sampled[:half], sampled[half:]

    out_rows: list[dict[str, Any]] = []

    mutation_names = list(MUTATIONS.keys())
    for i, src in enumerate(ood_source):
        mutation = mutation_names[i % len(mutation_names)]
        inp = json.loads(json.dumps(src["input_json"]))  # deep copy
        MUTATIONS[mutation](inp, rng)
        out_rows.append(
            {
                "input_json": inp,
                "is_outlier": 1,
                "mutation": mutation,
                "tail_flags": [],  # OOD rows: tail-flag is not meaningful
                "source_row_index": src.get("row_index", -1),
            }
        )

    for src in indist_source:
        out_rows.append(
            {
                "input_json": src["input_json"],
                "is_outlier": 0,
                "mutation": None,
                # Naturally-tail in-dist rows — used by the encoder eval to
                # check that the gate doesn't over-flag legitimate high-debt /
                # high-change / chronic-misser customers.
                "tail_flags": _tail_flags(
                    src["input_json"], dd_p90, change_p90, missed_max, ratio_p95
                ),
                "source_row_index": src.get("row_index", -1),
            }
        )

    rng.shuffle(out_rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    counts = {m: 0 for m in mutation_names}
    counts["__indist__"] = 0
    tail_counts: dict[str, int] = {
        "high_debt": 0,
        "high_change": 0,
        "many_missed": 0,
        "in_credit": 0,
    }
    for row in out_rows:
        counts[row["mutation"] or "__indist__"] += 1
        for tag in row.get("tail_flags") or []:
            tail_counts[tag] = tail_counts.get(tag, 0) + 1

    typer.echo(f"wrote {out}  ({len(out_rows)} rows)")
    for k, v in counts.items():
        typer.echo(f"  {k:30s} {v:4d}")
    typer.echo("  tail-flagged in-dist rows (natural distribution tail; rows can carry multiple tags):")
    for k, v in tail_counts.items():
        typer.echo(f"    tail.{k:24s} {v:4d}")


if __name__ == "__main__":
    app()
