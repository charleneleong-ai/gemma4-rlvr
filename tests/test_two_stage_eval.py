"""Unit tests for scripts/two_stage_eval.py refactor (EvalArm + rescore-from)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def test_eval_arm_description_combinations() -> None:
    from scripts.two_stage_eval import EvalArm

    assert EvalArm("vanilla", False, False, False).description() == \
        "vanilla (no Stage-1 injection, no constraints)"
    assert EvalArm("two_stage", True, False, False).description() == "Stage-1-injected"
    assert EvalArm("two_stage_constrain", True, True, False).description() == \
        "Stage-1-injected + VALID FACTS prompt list"
    assert EvalArm("two_stage_lmfe", True, False, True).description() == \
        "Stage-1-injected + LMFE slot mask"
    assert EvalArm("two_stage_full", True, True, True).description() == \
        "Stage-1-injected + VALID FACTS prompt list + LMFE slot mask"


def test_rescore_from_per_row(tmp_path: Path) -> None:
    """A new per-row dump (with completions) is rescored without touching the model."""
    from scripts.two_stage_eval import _rescore_from_per_row

    valid_completion = json.dumps({
        "explanations": [{
            "trigger": "Change in unit rates",
            "explanation": "Your unit rate changed.",
            "tariff_cited": "tariffA",
            "rate_change_pct_cited": 5.0,
        }],
    })
    invalid_completion = json.dumps({
        "explanations": [{
            "trigger": "Change in unit rates",
            "explanation": "Your tariff XYZ went up 99%.",
        }],
    })

    inp = {
        "account_context": {
            "contract_history": [{
                "tariff_name": "tariffA",
                "contract_rates_history": [{
                    "rates": [{"change_since_previous_rate_percent": 5.0}],
                }],
            }],
        },
        "latest_dd_change": {"dd_amount": 100.0, "dd_amount_change": 0.0},
    }
    rows = [{
        "i": 0,
        "ground_truth_triggers": ["Change in unit rates"],
        "stage1_triggers": ["Change in unit rates"],
        "input_json": inp,
        "completions": {"vanilla": invalid_completion, "two_stage": valid_completion},
        "scores": {},
    }]
    per_row_path = tmp_path / "in.per_row.jsonl"
    per_row_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "out.json"

    _rescore_from_per_row(per_row_path, out)

    payload = json.loads(out.read_text())
    assert "vanilla" in payload and "two_stage" in payload
    assert payload["rescored_from"] == str(per_row_path)
    assert payload["n_heldout"] == 1
    # Aggregate keys present (score values are rubric-determined, not under test here).
    for arm in ("vanilla", "two_stage"):
        for k in ("mean_total", "no_hallucinated_facts_mean", "pass_all_pct"):
            assert k in payload[arm], f"{arm} missing {k}"


def test_rescore_from_legacy_format_errors_clearly(tmp_path: Path) -> None:
    """Old per-row dumps (no `completions` field) raise typer.BadParameter, not silently."""
    import typer

    from scripts.two_stage_eval import _rescore_from_per_row

    legacy_row = {
        "i": 0,
        "ground_truth_triggers": ["Change in unit rates"],
        "stage1_triggers": ["Change in unit rates"],
        "vanilla": {"no_hallucinated_facts": 1.0},
        "two_stage": {"no_hallucinated_facts": 1.0},
    }
    p = tmp_path / "legacy.per_row.jsonl"
    p.write_text(json.dumps(legacy_row) + "\n")

    try:
        _rescore_from_per_row(p, tmp_path / "out.json")
    except typer.BadParameter as e:
        assert "completions" in str(e)
        return
    raise AssertionError("expected typer.BadParameter for legacy format")
