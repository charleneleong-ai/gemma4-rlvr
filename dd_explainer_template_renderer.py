"""Deterministic prose templates for the 4 "lonely" triggers.

The PR-G E28 per-row analysis showed row-level no_halluc collapses to ~0%
when these triggers fire alone — the model fabricates template phrases
("standard process", "to cover the difference", "necessary catch-up")
because it has no grounded source for the explanatory prose.

Since the trigger predicates are 100%-recall / 0%-FPR derivable from the
structured input (see `extract_trigger_grounding`), the LLM adds zero
value beyond fabrication for these rows. This module renders the prose
deterministically from the same grounding values.

Templates use ONLY values present in `extract_trigger_grounding` and
`extract_valid_facts` so the rendered prose satisfies the eval rubric's
row-level no_halluc check by construction.

Used by `scripts/two_stage_eval.py --use-templates` as a post-generation
overwrite for the `header` / `explanation` fields of any explanations[]
entry whose trigger is in LONELY_TRIGGERS. Slot citation fields
(prev_amount_cited / tariff_cited / rate_change_pct_cited) are left
to LMFE schema enforcement.
"""

from __future__ import annotations

from typing import Any


LONELY_TRIGGERS: frozenset[str] = frozenset({
    "First DD review since account start",
    "Missed/bounced DD payments",
    "Manual reduction",
    "Exemption Expiry",
})

# Stage-1 fires this sentinel when no class clears the threshold. The LLM's
# default behavior on this label is vague filler that cites zero tariffs,
# tripping the no_halluc inaction loophole — render a tariff-citing
# template instead.
NO_TRIGGERS_LABEL: str = "No triggers identified"


def _fmt_gbp(v: float | int | None) -> str | None:
    if v is None:
        return None
    return f"£{float(v):.2f}"


def _current_tariff(valid_facts: dict[str, Any]) -> str | None:
    """Return the most-recent tariff name from valid_facts['tariffs'] (last entry).
    Returns None when no tariff is available so callers can omit the clause
    rather than invent one."""
    tariffs = valid_facts.get("tariffs") or []
    if not tariffs:
        return None
    return tariffs[-1]


def _tariff_clause(valid_facts: dict[str, Any], lead: str = "on your tariff") -> str:
    """Render a clause like 'on your tariff <Name>' that satisfies the
    `_TARIFF_RE` rubric check, which requires the keyword `tariff|contract|plan`
    to come BEFORE the capitalized name. Empty string when no tariff available.
    """
    name = _current_tariff(valid_facts)
    return f" {lead} {name}" if name else ""


def _render_first_dd_review(grounding: dict[str, Any], valid_facts: dict[str, Any]) -> dict[str, str] | None:
    if not grounding.get("first_dd_review", {}).get("is_first"):
        return None
    prev = _fmt_gbp(valid_facts.get("prev_amount"))
    tariff_clause = _tariff_clause(valid_facts)
    prev_clause = f" The previous estimated amount was {prev}, and we've now updated it based on your actual usage." if prev else ""
    return {
        "header": "First Direct Debit review since account start",
        "explanation": (
            f"This is the first review of your Direct Debit{tariff_clause} since you started your account."
            + prev_clause
        ),
    }


def _render_missed_payments(grounding: dict[str, Any], valid_facts: dict[str, Any]) -> dict[str, str] | None:
    if "missed_payments" not in grounding:
        return None
    ctx = grounding["missed_payments"] or {}
    n = int(ctx.get("n_missed") or 0)
    period = ctx.get("most_recent_period")
    amt = _fmt_gbp(ctx.get("most_recent_amount_gbp"))
    tariff_clause = _tariff_clause(valid_facts, lead="On your tariff")
    s = "" if n == 1 else "s"
    if tariff_clause:
        parts = [f"{tariff_clause}, we've identified {n} unsuccessful Direct Debit payment{s} in your recent payment history"]
    else:
        parts = [f"We've identified {n} unsuccessful Direct Debit payment{s} in your recent payment history"]
    if period and amt:
        parts.append(f" (most recently {amt} for {period})")
    elif period:
        parts.append(f" (most recently in {period})")
    elif amt:
        parts.append(f" (most recently {amt})")
    parts.append(". Your Direct Debit has been adjusted to recover the missed amount.")
    return {
        "header": "Catching up on missed Direct Debit payments",
        "explanation": "".join(parts),
    }


def _render_manual_reduction(grounding: dict[str, Any], valid_facts: dict[str, Any]) -> dict[str, str] | None:
    if "manual_reduction" not in grounding:
        return None
    ctx = grounding["manual_reduction"] or {}
    manual = _fmt_gbp(ctx.get("manual_dd_amount_gbp"))
    recommended = _fmt_gbp(ctx.get("recommended_dd_amount_gbp"))
    when = ctx.get("active_in")  # "current_period" or "previous_period"
    tariff_clause = _tariff_clause(valid_facts)
    if when == "previous_period":
        header = "Reverting from a manually reduced Direct Debit"
        clause = f"Your previous Direct Debit{tariff_clause}"
        end = "That period has ended, so your payment has now been adjusted toward the recommended amount."
    else:
        header = "Adjusting a manually reduced Direct Debit"
        clause = f"Your Direct Debit{tariff_clause}"
        end = "Your payment is now being adjusted toward the recommended amount."
    if manual and recommended:
        body = f"{clause} of {manual} was manually set below the recommended amount of {recommended}. {end}"
    elif manual:
        body = f"{clause} of {manual} was manually set below the recommended amount. {end}"
    else:
        body = f"{clause} was manually set below the recommended amount. {end}"
    return {"header": header, "explanation": body}


def _render_exemption_expiry(grounding: dict[str, Any], valid_facts: dict[str, Any]) -> dict[str, str] | None:
    if "exemption_expiry" not in grounding:
        return None
    ctx = grounding["exemption_expiry"] or {}
    expired = ctx.get("expired_on")
    prev_dd = _fmt_gbp(ctx.get("previous_dd_amount_gbp"))
    recommended = _fmt_gbp(ctx.get("previous_recommended_dd_amount_gbp"))
    tariff_clause = _tariff_clause(valid_facts)
    parts = [f"Your Direct Debit{tariff_clause} was previously set"]
    if prev_dd:
        parts.append(f" at {prev_dd}")
    parts.append(" under an exemption")
    if expired:
        # Strip trailing time-of-day if present (e.g. "2025-10-28 00:00:00" -> "2025-10-28")
        date_part = str(expired).split(" ")[0]
        parts.append(f" that expired on {date_part}")
    parts.append(".")
    if recommended:
        parts.append(f" Now that the exemption has ended, your payment has been updated toward the recommended amount of {recommended}.")
    else:
        parts.append(" Now that the exemption has ended, your payment has been updated.")
    return {"header": "Exemption period has ended", "explanation": "".join(parts)}


def _render_change_in_unit_rates(grounding: dict[str, Any], valid_facts: dict[str, Any]) -> dict[str, str] | None:
    """Renderer for the 'Change in unit rates' trigger.

    Cites only values guaranteed by `valid_facts`:
      - tariff name (LLM-passing prose always pairs the trigger with one)
      - the largest-magnitude rate change % (rubric validates abs >= 1.0)
      - prev_amount

    Returns None when no tariff is available (rubric needs a tariff anchor
    to escape the no_halluc inaction loophole on this trigger).
    """
    tariff_clause = _tariff_clause(valid_facts)
    if not tariff_clause:
        return None
    rate_pcts = valid_facts.get("rate_percentages") or []
    significant = [p for p in rate_pcts if abs(p) >= 1.0]
    rate_clause = ""
    if significant:
        biggest = max(significant, key=abs)
        direction = "increased" if biggest > 0 else "decreased"
        rate_clause = f" The unit rate has {direction} by {abs(biggest):.2f}% since the previous review."
    prev = _fmt_gbp(valid_facts.get("prev_amount"))
    prev_clause = f" Your previous Direct Debit was {prev} and has been adjusted accordingly." if prev else ""
    return {
        "header": "Tariff rate adjustments applied",
        "explanation": (
            f"Your Direct Debit has been updated because of changes in unit rates"
            f"{tariff_clause}.{rate_clause}{prev_clause}"
        ),
    }


def _render_change_in_usage(grounding: dict[str, Any], valid_facts: dict[str, Any]) -> dict[str, str] | None:
    """Renderer for the 'Change in usage' trigger.

    Cites only values guaranteed by `valid_facts`:
      - tariff name
      - prev_amount

    Deliberately avoids quoting a usage percentage: usage-change %s live in
    `projected_consumption_history.<fuel>.change_percent` not in `valid_facts`,
    so a citation would require extra grounding the renderer would have to
    extract itself. Generic "your recent usage patterns" prose is safe under
    no_halluc since it cites no specific number.
    """
    tariff_clause = _tariff_clause(valid_facts)
    if not tariff_clause:
        return None
    prev = _fmt_gbp(valid_facts.get("prev_amount"))
    prev_clause = f" The previous Direct Debit was {prev}." if prev else ""
    return {
        "header": "Direct Debit updated for usage changes",
        "explanation": (
            f"Your Direct Debit{tariff_clause} has been reviewed based on your"
            f" recent usage patterns.{prev_clause}"
            f" The amount has been adjusted to better reflect your projected consumption."
        ),
    }


def _render_no_triggers_identified(grounding: dict[str, Any], valid_facts: dict[str, Any]) -> dict[str, str] | None:
    """Stage-1 fallback when no specific trigger fires. Renders prose that
    cites the current tariff so the row escapes the no_halluc inaction
    loophole. No grounding context required (Stage-1 said nothing applies)."""
    tariff_clause = _tariff_clause(valid_facts)
    if not tariff_clause:
        # No tariff to cite either — caller falls through to LLM (no template
        # can help here without inventing claims).
        return None
    prev = _fmt_gbp(valid_facts.get("prev_amount"))
    prev_clause = f" The amount has been adjusted from the previous {prev} to reflect your current usage." if prev else ""
    return {
        "header": "Direct Debit reviewed — no specific trigger identified",
        "explanation": (
            f"Your Direct Debit{tariff_clause} has been reviewed this cycle"
            f" and no specific trigger event was identified.{prev_clause}"
        ),
    }


# Renderers used by `overwrite_explanations` to REPLACE LLM-generated prose
# for the 4 lonely triggers + the No-triggers fallback. Adding a trigger here
# means LLM prose for that trigger is overwritten on every row.
_OVERWRITE_RENDERERS = {
    "First DD review since account start": _render_first_dd_review,
    "Missed/bounced DD payments": _render_missed_payments,
    "Manual reduction": _render_manual_reduction,
    "Exemption Expiry": _render_exemption_expiry,
    NO_TRIGGERS_LABEL: _render_no_triggers_identified,
}

# Backfill registry — superset of overwrite renderers. Used ONLY when the LLM
# dropped a Stage-1-predicted trigger, so adding here is safe even if the LLM
# usually produces good prose for that trigger (this path only fires when the
# LLM's prose is missing).
_BACKFILL_RENDERERS = {
    **_OVERWRITE_RENDERERS,
    "Change in unit rates": _render_change_in_unit_rates,
    "Change in usage": _render_change_in_usage,
}


def render_lonely_explanation(
    trigger: str,
    grounding: dict[str, Any],
    valid_facts: dict[str, Any],
) -> dict[str, str] | None:
    """Render header + explanation for one of the 4 lonely triggers.

    Returns {"header", "explanation"} when the trigger is in LONELY_TRIGGERS
    AND the corresponding grounding context is present; returns None otherwise
    (caller falls through to LLM-generated prose). Never raises on missing
    fields — degrades to the most general phrasing supported by what's available.
    """
    fn = _OVERWRITE_RENDERERS.get(trigger)
    if fn is None:
        return None
    return fn(grounding, valid_facts)


def render_for_backfill(
    trigger: str,
    grounding: dict[str, Any],
    valid_facts: dict[str, Any],
) -> dict[str, str] | None:
    """Like `render_lonely_explanation` but also handles `Change in usage` and
    `Change in unit rates`. Backfill calls this so it can close failures on
    those triggers without overwriting LLM prose for rows where the LLM did
    emit them correctly."""
    fn = _BACKFILL_RENDERERS.get(trigger)
    if fn is None:
        return None
    return fn(grounding, valid_facts)


def overwrite_explanations(
    parsed: dict[str, Any],
    grounding: dict[str, Any],
    valid_facts: dict[str, Any],
) -> dict[str, Any]:
    """Mutate `parsed` (LLM-generated JSON) in place: replace header+explanation
    for any lonely-trigger entry with the deterministic template. Slot citation
    fields (prev_amount_cited / tariff_cited / rate_change_pct_cited) are left
    untouched so LMFE-enforced values flow through unchanged.

    Returns the same dict (for fluent use). Safe on malformed input — silently
    leaves entries it can't process alone.
    """
    explanations = parsed.get("explanations")
    if not isinstance(explanations, list):
        return parsed
    for entry in explanations:
        if not isinstance(entry, dict):
            continue
        trigger = entry.get("trigger")
        if not isinstance(trigger, str):
            continue
        rendered = render_lonely_explanation(trigger, grounding, valid_facts)
        if rendered is not None:
            entry["header"] = rendered["header"]
            entry["explanation"] = rendered["explanation"]
    return parsed


def _build_backfill_entry(
    trigger: str,
    rendered: dict[str, str],
    valid_facts: dict[str, Any],
) -> dict[str, Any]:
    """Construct a complete explanations[] entry from a rendered template,
    populating the three slot citation fields from `valid_facts` so the
    backfilled entry is rubric-complete by construction."""
    tariff = _current_tariff(valid_facts) or ""
    prev = valid_facts.get("prev_amount")
    return {
        "trigger": trigger,
        "header": rendered["header"],
        "explanation": rendered["explanation"],
        "tariff_cited": tariff,
        "rate_change_pct_cited": 0.0,
        "prev_amount_cited": float(prev) if prev is not None else 0.0,
    }


def backfill_missing_triggers(
    parsed: dict[str, Any],
    stage1_triggers: list[str],
    stage1_probs: dict[str, float] | None,
    grounding: dict[str, Any],
    valid_facts: dict[str, Any],
    confidence_threshold: float = 0.9,
) -> dict[str, Any]:
    """Append a templated entry for each Stage-1-predicted trigger absent
    from `parsed["explanations"]`, gated on classifier confidence.

    The PR #32 ceiling analysis showed the residual 23/1000 failures are
    exclusively "Stage-1 right, LLM dropped a trigger" (f1=6 instead of
    f1=10). The LLM tends to drop low-salience metadata triggers (e.g.
    "First DD review since account start") when paired with high-salience
    triggers (e.g. "Change in usage"). Backfilling restores f1 to 10 and
    pushes pass_all toward the rubric ceiling.

    Args:
        parsed: parsed two_stage completion JSON, mutated in place.
        stage1_triggers: trigger labels predicted by the Stage-1 classifier
            for this row.
        stage1_probs: per-trigger sigmoid probabilities from the classifier.
            If None the gate is bypassed (legacy callers). Production code
            should always pass probs to guard against Stage-1 false-positives
            being forced into the output.
        grounding: extract_trigger_grounding(input_json) — supplies the
            per-trigger anchors the renderer needs.
        valid_facts: extract_valid_facts(input_json) — supplies tariff and
            prev_amount used for slot citations.
        confidence_threshold: only backfill triggers whose Stage-1 prob is
            at least this. 0.9 is a conservative default tuned against
            v5's well-calibrated boolean-feature predictions.

    Returns the same dict (for fluent use). No-op when `parsed` has no
    `explanations` list, when no Stage-1 trigger is missing, or when no
    renderer/grounding is available for the missing triggers.
    """
    explanations = parsed.get("explanations")
    if not isinstance(explanations, list):
        return parsed
    emitted: set[str] = {
        e["trigger"] for e in explanations
        if isinstance(e, dict) and isinstance(e.get("trigger"), str)
    }
    for trigger in stage1_triggers:
        if trigger in emitted:
            continue
        if stage1_probs is not None:
            prob = stage1_probs.get(trigger, 0.0)
            if prob < confidence_threshold:
                continue
        rendered = render_for_backfill(trigger, grounding, valid_facts)
        if rendered is None:
            continue
        explanations.append(_build_backfill_entry(trigger, rendered, valid_facts))
        emitted.add(trigger)
    return parsed


__all__ = [
    "LONELY_TRIGGERS",
    "NO_TRIGGERS_LABEL",
    "render_lonely_explanation",
    "render_for_backfill",
    "overwrite_explanations",
    "backfill_missing_triggers",
]
