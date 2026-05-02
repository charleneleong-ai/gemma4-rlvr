"""Synthetic dataset + reward oracle for the direct_debit_explainer API mock.

Input schema  : DDExplainerPromptInput  (account_context + latest_dd_change)
Output schema : DirectDebitExplainerResponse  (list[TriggerExplanation])
Trigger enum  : 7 values matching the gemini-2.5-pro response_schema.

The generator is scenario-first: sample a target set of `Trigger`s, then
synthesise an account context that satisfies the domain rules for exactly
that set. `detect_triggers` is the companion oracle. Both use the same
private predicates (`_has_manual_reduction_history` etc.) so the self-check
in `__main__` is meaningful.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import typer
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)


# =============================================================================
# Input schemas (copied verbatim from the user's spec)
# =============================================================================


class DDChange(BaseModel):
    datetime_from: Optional[datetime] = Field(
        default=None,
        validation_alias=AliasChoices("datetime_from", "subscriptionCreatedTimestamp"),
    )
    datetime_to: Optional[datetime] = Field(
        default=None,
        validation_alias=AliasChoices("datetime_to", "subscriptionCancelledTimestamp"),
    )
    is_currently_active_DD: bool = Field(
        default=False,
        validation_alias=AliasChoices("is_currently_active_DD", "activeSubscriptionFlag"),
    )
    reason_for_DD_change: str = Field(
        validation_alias=AliasChoices("reason_for_DD_change", "friendlyOrigin"),
    )
    dd_amount: float = Field(
        validation_alias=AliasChoices("dd_amount", "newDdAmountPounds"),
    )
    dd_amount_change: Optional[float] = None
    recommended_dd_amount: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices(
            "recommended_dd_amount", "recommendedDirectDebitAmountPounds"
        ),
    )
    yearly_predicted_energy_cost_gbp: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices(
            "yearly_predicted_energy_cost_gbp", "yearlyPredictedEnergyCost"
        ),
    )
    description: str = Field(
        validation_alias=AliasChoices("description", "reason"),
    )
    collectionDay: int
    is_exemption: bool = Field(
        default=False, validation_alias=AliasChoices("is_exemption", "exemptionFlag")
    )
    exemption_expiry_date: Optional[date] = Field(
        default=None,
        validation_alias=AliasChoices(
            "exemption_expiry_date", "exemptionExpirationDate"
        ),
    )
    is_amount_manually_reduced_lower_than_recommended_amount: bool = False

    @field_validator("datetime_from", "datetime_to", mode="before")
    @classmethod
    def strip_timezone_dt(cls, v):
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return dt.replace(tzinfo=None)
        if isinstance(v, datetime) and v.tzinfo is not None:
            return v.replace(tzinfo=None)
        return v


class PaymentAttempt(BaseModel):
    transaction_timestamp: datetime = Field(
        validation_alias=AliasChoices("transaction_timestamp", "transactionTimeStamp")
    )
    transaction_amount_in_pounds: float
    is_payment_successful: bool = True

    @computed_field
    @property
    def payment_period(self) -> str:
        return self.transaction_timestamp.strftime("%b %Y")

    @field_validator("transaction_timestamp", mode="before")
    @classmethod
    def strip_timezone(cls, v):
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return dt.replace(tzinfo=None)
        if isinstance(v, datetime) and v.tzinfo is not None:
            return v.replace(tzinfo=None)
        return v


class ProjectedEnergyCost(BaseModel):
    date_from: date
    date_to: date
    yearlyPredictedEnergyCost: float


class Rate(BaseModel):
    fuel: str
    rate_type: str
    amount_GBP: float
    change_since_previous_rate: Optional[float] = None
    change_since_previous_rate_percent: Optional[float] = None


class ContractRatesHistory(BaseModel):
    rate_effective_from: date
    rate_effective_to: Optional[date] = None
    rates: List[Rate]


class Contract(BaseModel):
    tariff_name: str
    is_current_contract: bool
    contract_start_date: date
    contract_end_date: Optional[date] = None
    contract_rates_history: List[ContractRatesHistory]

    @field_validator("contract_end_date", mode="before")
    @classmethod
    def parse_null_string(cls, v):
        if isinstance(v, str) and v.lower() == "null":
            return None
        return v


class FuelProjectedConsumption(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    change_kwh: Optional[float] = Field(default=None, alias="change_kWh")
    change_percent: Optional[float] = None
    latest_projected_annual_consumption_kwh: float = Field(
        alias="latest_projected_annual_consumption_kWh",
    )
    latest_projection_date: date
    previous_projected_annual_consumption_kwh: Optional[float] = Field(
        default=None, alias="previous_projected_annual_consumption_kWh",
    )
    previous_projection_date: Optional[date] = None


class ProjectedConsumptionHistory(BaseModel):
    electricity: Optional[FuelProjectedConsumption] = None
    gas: Optional[FuelProjectedConsumption] = None


class AccountContext(BaseModel):
    dd_change_history: List[DDChange] = Field(default_factory=list)
    payment_history: List[PaymentAttempt] = Field(default_factory=list)
    contract_history: List[Contract] = Field(default_factory=list)
    projected_consumption_history: Optional[ProjectedConsumptionHistory] = None


class DDExplainerPromptInput(BaseModel):
    account_context: AccountContext
    latest_dd_change: DDChange


# =============================================================================
# Output schemas (from the api_schema.yaml codegen block)
# =============================================================================


class DirectDebitExplainerRequest(BaseModel):
    account_id: str
    client_id: str
    conversation_id: Optional[str] = None
    include_context: Optional[bool] = False


class Trigger(str, Enum):
    Manual_reduction = "Manual reduction"
    Exemption_Expiry = "Exemption Expiry"
    Change_in_usage = "Change in usage"
    Change_in_unit_rates = "Change in unit rates"
    Missed_bounced_DD_payments = "Missed/bounced DD payments"
    First_DD_review_since_account_start = "First DD review since account start"
    No_triggers_identified = "No triggers identified"


class TriggerExplanation(BaseModel):
    trigger: Trigger
    header: str
    explanation: str
    # Optional structured slots — when populated, the rubric validates these
    # directly against the allowed-list from input_json instead of regex-scanning
    # prose. Old-format completions (slots absent) fall back to legacy prose
    # regex so PR #12 baselines remain comparable.
    tariff_cited: Optional[str] = None
    rate_change_pct_cited: Optional[float] = None


class DirectDebitExplainerResponse(BaseModel):
    explanations: List[TriggerExplanation]
    context: Optional[Dict[str, Any]] = None


# =============================================================================
# Prompt + domain knowledge
# =============================================================================


DOMAIN_KNOWLEDGE = """<background_information_on_DD>
DDs are a common payment method for energy customers.
One benefit of a DD is consistency: customers pay the same amount for their energy each month.
This helps smooth and balance out seasonal variation in energy consumption, as most customers use more energy in winter compared to summer.
The idea is to build up a surplus credit balance by overpaying in summer. This creates a reserve that can then be used to underpay in winter.
Energy companies factor this seasonality into their calculations when forecasting expected usage to set the recommended DD amount for each customer.
As customers send meter reads throughout the year, the energy supplier is then able to compare the customer's actual usage against expected.
If actual and expected usage differ significantly then the energy supplier will periodically review and change the DD amount accordingly.
</background_information_on_DD>

<common_triggers_for_DD_changes>
1) Manual reduction:
Customer service advisors can manually reduce the DD amount at the request of a customer.
When the dd_amount is reduced to be lower than the recommended_dd_amount, then the customer will be underpaying for their usage.
This usually results in a subsequent increase to the dd_amount at the next DD review.
IMPORTANT: Manual reductions are evident only when a previous DD change shows "is_amount_manually_reduced_lower_than_recommended_amount"=True.

2) Exemption Expiry:
Customer service advisors can place an exemption to the normal DD amount.
Exemptions are different to manual reductions, and will be clearly marked within the customer data.
Typically the reason for an exemption is due to the customer wanting to pay less than the recommended value.
Exemption expiry (after 90 days) results in a DD becoming eligible for a review (DDR).
During the exemption period the account may have underpaid for their usage, and so requires an increase to the DD review but has been exempt.
After expiry, the automated DDR process will increase the customers DD to cover this underpayment.

3) Change in usage:
When customers send meter reads energy suppliers use these to calculate their actual usage AND generate a new projected usage calculation.
Where actual usage is significantly higher than previous expected usage, the new projected usage will also increase.
This can then trigger a DDR, and the DD amount will increase to capture the higher expected usage.
To identify this trigger, check `projected_consumption_history` which shows the change in projected annual consumption (kWh) between the latest and previous DD reviews for each fuel.
A `change_percent` of >=5% (positive or negative) for either fuel is a meaningful change. Below 5% treat as insignificant.
Only flag usage change as a trigger if the direction of change is consistent with the DD change direction (e.g. increased usage explains a DD increase, but decreased usage does not).
IMPORTANT: Do not use `yearly_predicted_energy_cost_gbp` alone to conclude usage has changed -- predicted cost is affected by both usage AND rates. Use `projected_consumption_history` to assess usage changes.
If `projected_consumption_history` is missing or shows no data for a fuel, you cannot confirm a usage change for that fuel.

4) Change in unit rates:
Customers on a standard variable tariff (SVT) pay a variable unit cost and standing charge rates for their energy, whereas fixed term contracts lock in a static price for an agreed period of time.
If unit costs and stading charges have increased (e.g. due to price cap events) since the previous DDR or start of contract, then the DD amount will also increase.
This effect can also occur when customers change to a new contariff with different pricing, either;
- Fixed to SVT: when customer reach the end of their fixed term contract, and by default "roll on" to a SVT.
- SVT to Fixed: alternatively, transferring from a SVT to a new fixed term contract with different rates.
- Fixed to Fixed: renewing a fixed term contract with different rates.
The `change_since_previous_rate` and `change_since_previous_rate_percent` fields indicate whether rates have increased or decreased.
When rate changes are mixed (some up, some down), consider the net impact: if the dominant changes by magnitude point in the direction of the DD change, treat this as a valid trigger.
To confirm this trigger, check `contract_rates_history` for rate entries with `change_since_previous_rate_percent` values.
Rate changes below 2% are typically insignificant and unlikely to be a meaningful trigger on their own.
If `contract_rates_history` is empty or shows no `change_since_previous_rate` values, you cannot confirm a rate change -- even if `yearly_predicted_energy_cost_gbp` has changed (the cost change may instead be due to usage).
When a contract transition has occurred (e.g. Fixed to SVT), compare rates from the new contract against the previous contract to assess the direction and scale of rate changes.

5) Missed/bounced DD payments:
Customers are expected to pay their DD on the agreed collection date each month.
Sometimes payments may be unsuccessful if the customer doesn't have sufficient funds available in their account, or if the bank recalls the funds.
If there is an unsuccessful payment the energy supplier may attempt to collect the payment again on a subsequent day in that month.
Any previous missed or failed payment means there was a period where energy wasnt paid for.
As a result the account balance changes and future DD payments need to increase to 'recoup' that loss.

6) First DD review since account start:
When a customer first starts supply, there is often limited information available about their property's projected energy consumption.
Therefore the DD amount set for new customers may be subject to change via a DD review some time after starting supply.
As more information becomes available about the customer's actual consumption (i.e. when they submit meter reads), this can trigger an update to their projected usage.
Updated projections can in turn trigger a first DD review for the customer.
If the initial DD amount was set too low for the customer's actual consumption, the DDR may increase the DD amount.
</common_triggers_for_DD_changes>
"""


SYSTEM_PROMPT = """<task>
Your mission is to provide an explanation to a household energy customer about why their direct debit (DD) has changed (this could be an increase or decrease in their DD amount in £ GBP).
Below you are provided with a list of common triggers for DD changes, and raw data from the customer's account.
You should analyse the triggers and account data together to determine which triggers are relevant to the customer.
For each relevant trigger you should generate a concise explanation (no more than 3 sentences).
If multiple triggers are relevant then return a 3 sentence explanation for each.
</task>

<domain_knowledge>
{domain_knowledge}
</domain_knowledge>

<instructions>
Determine whether each of the above common triggers are relevant for the latest DD change on the customer's account (multiple triggers can cause a DD change).
You must only identify triggers for the latest DD change event, not historical DD changes.
For each relevant trigger generate a 3 sentence explanation for the customer about why their DD has changed.
Include all relevant account data points to justify your answer - this is particularly important when referencing rate changes and missed payments.
When assessing "Change in usage" and "Change in unit rates" triggers, cross-reference `yearly_predicted_energy_cost_gbp` with both `projected_consumption_history` and `contract_rates_history` to correctly attribute the cause of any cost change.
You must relate this data to the relevant information for each trigger.
Use user-friendly language to explain any technical terms.
Be sympathetic, especially in the case of missed payments. When the amount they have to pay has gone up, customers may experience distress.
</instructions>

Return JSON matching:
{{"explanations": [{{"trigger": "<one of: Manual reduction | Exemption Expiry | Change in usage | Change in unit rates | Missed/bounced DD payments | First DD review since account start | No triggers identified>", "header": "<5-word header>", "tariff_cited": "<verbatim tariff_name from contract_history, or null if you don't reference a tariff>", "rate_change_pct_cited": "<verbatim change_since_previous_rate_percent number from contract_rates_history, or null if you don't reference a rate change>", "explanation": "<3 sentences; reference the cited tariff and rate as {{tariff_cited}} and {{rate_change_pct_cited}} placeholders so they substitute the slot values exactly>"}}]}}

The `tariff_cited` and `rate_change_pct_cited` fields are optional but strongly preferred when your explanation references a specific tariff name or rate change percentage. They MUST come verbatim from the input account_context — pulling them into structured slots prevents fabrication. Use the `{{tariff_cited}}` / `{{rate_change_pct_cited}}` placeholders inside the explanation prose so the rendered output is grounded in the slot values.
"""


HUMAN_PROMPT_TEMPLATE = """<latest_DD_change>
**Latest DD change:**
{latest_dd_change}
</latest_DD_change>

<customers_account_data>
{account_context}
</customers_account_data>
"""


def build_chat_messages(pin: DDExplainerPromptInput) -> List[Dict[str, object]]:
    """Render the full direct_debit_explainer prompt into chat messages.

    Content is wrapped as a list of typed blocks (`[{"type": "text", "text": ...}]`)
    rather than a bare string. Both forms render identically through Gemma 4's
    chat template, but transformers >=5.5.0's `apply_chat_template(tokenize=True)`
    iterates string content as if it were a list of multimodal blocks and crashes
    on `content_block["type"]`. List-of-blocks bypasses that path cleanly.
    """
    system = SYSTEM_PROMPT.format(domain_knowledge=DOMAIN_KNOWLEDGE)
    human = HUMAN_PROMPT_TEMPLATE.format(
        latest_dd_change=json.dumps(pin.latest_dd_change.model_dump(mode="json"), indent=2),
        account_context=json.dumps(pin.account_context.model_dump(mode="json"), indent=2),
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": human}]},
    ]


# =============================================================================
# Shared rule predicates (used by both generator and oracle)
# =============================================================================


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _has_manual_reduction_history(history: List[DDChange]) -> bool:
    return any(c.is_amount_manually_reduced_lower_than_recommended_amount for c in history)


def _has_recent_exemption_expiry(
    history: List[DDChange], latest_dt: Optional[datetime], days: int = 120
) -> bool:
    if latest_dt is None:
        return False
    cutoff_start = latest_dt.date() - timedelta(days=days)
    cutoff_end = latest_dt.date() + timedelta(days=30)
    for c in history:
        if not c.is_exemption or c.exemption_expiry_date is None:
            continue
        if cutoff_start <= c.exemption_expiry_date <= cutoff_end:
            return True
    return False


def _usage_change_direction(
    pch: Optional[ProjectedConsumptionHistory], dd_sign: int, threshold_pct: float = 5.0
) -> bool:
    if pch is None or dd_sign == 0:
        return False
    for fuel in (pch.electricity, pch.gas):
        if fuel is None or fuel.change_percent is None:
            continue
        if abs(fuel.change_percent) >= threshold_pct and _sign(fuel.change_percent) == dd_sign:
            return True
    return False


def _rate_change_direction(
    contract_history: List[Contract], dd_sign: int, threshold_pct: float = 2.0
) -> bool:
    if dd_sign == 0 or not contract_history:
        return False
    current = next((c for c in contract_history if c.is_current_contract), contract_history[-1])
    if not current.contract_rates_history:
        return False
    latest_rates = current.contract_rates_history[-1].rates
    changes = [
        r.change_since_previous_rate_percent
        for r in latest_rates
        if r.change_since_previous_rate_percent is not None
    ]
    if not changes:
        return False
    dominant = max(changes, key=abs)
    return abs(dominant) >= threshold_pct and _sign(dominant) == dd_sign


def _had_failed_payment(
    payments: List[PaymentAttempt], latest_dt: Optional[datetime], months: int = 6
) -> bool:
    if latest_dt is None:
        return any(not p.is_payment_successful for p in payments)
    cutoff = latest_dt - timedelta(days=months * 30)
    return any(
        (not p.is_payment_successful) and p.transaction_timestamp >= cutoff
        for p in payments
    )


def _is_first_review(
    dd_history: List[DDChange],
    contract_history: List[Contract],
    latest_dt: Optional[datetime],
    months: int = 18,
) -> bool:
    if len(dd_history) > 1 or latest_dt is None or not contract_history:
        return False
    earliest_start = min(c.contract_start_date for c in contract_history)
    months_since_start = (latest_dt.date() - earliest_start).days / 30.0
    return months_since_start <= months


# =============================================================================
# Oracle: detect triggers from a given input
# =============================================================================


def detect_triggers(pin: DDExplainerPromptInput) -> Set[Trigger]:
    """Return the set of Triggers implied by the input, per the domain rules."""
    latest = pin.latest_dd_change
    ctx = pin.account_context
    dd_sign = _sign(latest.dd_amount_change or 0.0)
    # Exclude the `latest` from the history set so Manual_reduction /
    # Exemption_Expiry are driven only by prior DD changes.
    prior_dd = [c for c in ctx.dd_change_history if c is not latest]
    latest_dt = latest.datetime_from

    found: Set[Trigger] = set()
    if _has_manual_reduction_history(prior_dd) and dd_sign > 0:
        found.add(Trigger.Manual_reduction)
    if _has_recent_exemption_expiry(prior_dd, latest_dt) and dd_sign > 0:
        found.add(Trigger.Exemption_Expiry)
    if _usage_change_direction(ctx.projected_consumption_history, dd_sign):
        found.add(Trigger.Change_in_usage)
    if _rate_change_direction(ctx.contract_history, dd_sign):
        found.add(Trigger.Change_in_unit_rates)
    if _had_failed_payment(ctx.payment_history, latest_dt):
        found.add(Trigger.Missed_bounced_DD_payments)
    if _is_first_review(ctx.dd_change_history, ctx.contract_history, latest_dt):
        found.add(Trigger.First_DD_review_since_account_start)

    if not found:
        found.add(Trigger.No_triggers_identified)
    return found


# =============================================================================
# Scenario-first generator
# =============================================================================


_TARIFFS = ["Simpler Energy", "Better Energy Fixed", "2-Year Fixed", "Standard Variable", "Green Fixed"]
_REASONS = ["automatic direct debit review", "customer request", "annual DDR", "price cap update"]


def _baseline_input(rng: random.Random, dd_direction: int) -> DDExplainerPromptInput:
    """Construct a minimal DDExplainerPromptInput with no triggers firing."""
    latest_dt = datetime(2026, rng.randint(1, 4), rng.randint(1, 28), rng.randint(8, 17), 0)
    start_months_ago = rng.randint(36, 72)  # ≥ 18 so First_DDR is off by default
    contract_start = latest_dt.date() - timedelta(days=start_months_ago * 30)

    prev_amount = round(rng.uniform(60.0, 180.0), 2)
    amount_change = round(rng.uniform(8.0, 40.0), 2) * (1 if dd_direction >= 0 else -1)
    new_amount = round(prev_amount + amount_change, 2)

    rate_entry = ContractRatesHistory(
        rate_effective_from=contract_start,
        rate_effective_to=None,
        rates=[
            Rate(fuel="electricity", rate_type="unit_rate", amount_GBP=0.27,
                 change_since_previous_rate=0.0, change_since_previous_rate_percent=0.0),
            Rate(fuel="electricity", rate_type="standing_charge", amount_GBP=0.53,
                 change_since_previous_rate=0.0, change_since_previous_rate_percent=0.0),
            Rate(fuel="gas", rate_type="unit_rate", amount_GBP=0.07,
                 change_since_previous_rate=0.0, change_since_previous_rate_percent=0.0),
        ],
    )
    contract = Contract(
        tariff_name=rng.choice(_TARIFFS),
        is_current_contract=True,
        contract_start_date=contract_start,
        contract_end_date=None,
        contract_rates_history=[rate_entry],
    )

    older_dt = latest_dt - timedelta(days=rng.randint(180, 365))
    older = DDChange(
        datetime_from=older_dt,
        datetime_to=latest_dt,
        is_currently_active_DD=False,
        reason_for_DD_change=rng.choice(_REASONS),
        dd_amount=prev_amount,
        dd_amount_change=0.0,
        recommended_dd_amount=round(prev_amount * 1.02, 2),
        yearly_predicted_energy_cost_gbp=round(prev_amount * 12, 2),
        description="Previous DD review",
        collectionDay=rng.randint(1, 28),
        is_exemption=False,
        exemption_expiry_date=None,
        is_amount_manually_reduced_lower_than_recommended_amount=False,
    )
    latest = DDChange(
        datetime_from=latest_dt,
        datetime_to=None,
        is_currently_active_DD=True,
        reason_for_DD_change="automatic direct debit review",
        dd_amount=new_amount,
        dd_amount_change=amount_change,
        recommended_dd_amount=round(new_amount * rng.uniform(0.95, 1.05), 2),
        yearly_predicted_energy_cost_gbp=round(new_amount * 12, 2),
        description="Your DD has been updated.",
        collectionDay=older.collectionDay,
        is_exemption=False,
        exemption_expiry_date=None,
        is_amount_manually_reduced_lower_than_recommended_amount=False,
    )

    projection = ProjectedConsumptionHistory(
        electricity=FuelProjectedConsumption(
            change_kwh=0.0,
            change_percent=0.0,
            latest_projected_annual_consumption_kwh=3000.0,
            latest_projection_date=latest_dt.date(),
            previous_projected_annual_consumption_kwh=3000.0,
            previous_projection_date=older_dt.date(),
        ),
        gas=FuelProjectedConsumption(
            change_kwh=0.0,
            change_percent=0.0,
            latest_projected_annual_consumption_kwh=10000.0,
            latest_projection_date=latest_dt.date(),
            previous_projected_annual_consumption_kwh=10000.0,
            previous_projection_date=older_dt.date(),
        ),
    )

    payments: List[PaymentAttempt] = []
    for i in range(6, 0, -1):
        ts = latest_dt - timedelta(days=i * 28)
        payments.append(
            PaymentAttempt(
                transaction_timestamp=ts,
                transaction_amount_in_pounds=prev_amount,
                is_payment_successful=True,
            )
        )

    return DDExplainerPromptInput(
        account_context=AccountContext(
            dd_change_history=[older, latest],
            payment_history=payments,
            contract_history=[contract],
            projected_consumption_history=projection,
        ),
        latest_dd_change=latest,
    )


def _apply_change_in_usage(pin: DDExplainerPromptInput, rng: random.Random) -> None:
    sign = _sign(pin.latest_dd_change.dd_amount_change or 0.0) or 1
    pct = rng.uniform(6.0, 25.0) * sign
    fuel_key = rng.choice(["electricity", "gas"])
    pch = pin.account_context.projected_consumption_history
    target = pch.electricity if fuel_key == "electricity" else pch.gas
    baseline = target.previous_projected_annual_consumption_kwh or target.latest_projected_annual_consumption_kwh
    new_kwh = baseline * (1.0 + pct / 100.0)
    target.change_percent = round(pct, 2)
    target.change_kwh = round(new_kwh - baseline, 1)
    target.latest_projected_annual_consumption_kwh = round(new_kwh, 1)


def _apply_change_in_unit_rates(pin: DDExplainerPromptInput, rng: random.Random) -> None:
    sign = _sign(pin.latest_dd_change.dd_amount_change or 0.0) or 1
    pct = rng.uniform(3.0, 15.0) * sign
    current = next(
        (c for c in pin.account_context.contract_history if c.is_current_contract),
        pin.account_context.contract_history[-1],
    )
    rates = current.contract_rates_history[-1].rates
    target_rate = next((r for r in rates if r.rate_type == "unit_rate"), rates[0])
    target_rate.change_since_previous_rate_percent = round(pct, 2)
    target_rate.change_since_previous_rate = round(target_rate.amount_GBP * pct / 100.0, 4)


def _apply_missed_payments(pin: DDExplainerPromptInput, rng: random.Random) -> None:
    payments = pin.account_context.payment_history
    n_to_flip = rng.randint(1, min(3, len(payments)))
    payments_sorted = sorted(payments, key=lambda p: p.transaction_timestamp, reverse=True)
    for p in payments_sorted[:n_to_flip]:
        p.is_payment_successful = False


def _apply_manual_reduction(pin: DDExplainerPromptInput, rng: random.Random) -> None:
    older = pin.account_context.dd_change_history[0]
    older.is_amount_manually_reduced_lower_than_recommended_amount = True
    older.reason_for_DD_change = "customer request"
    # Manual_reduction trigger requires dd_sign > 0 — force increase.
    if (pin.latest_dd_change.dd_amount_change or 0.0) <= 0:
        pin.latest_dd_change.dd_amount_change = abs(pin.latest_dd_change.dd_amount_change or 1.0) + 5.0
        pin.latest_dd_change.dd_amount = round(older.dd_amount + pin.latest_dd_change.dd_amount_change, 2)


def _apply_exemption_expiry(pin: DDExplainerPromptInput, rng: random.Random) -> None:
    older = pin.account_context.dd_change_history[0]
    older.is_exemption = True
    older.exemption_expiry_date = (
        pin.latest_dd_change.datetime_from.date() - timedelta(days=rng.randint(10, 100))
    )
    if (pin.latest_dd_change.dd_amount_change or 0.0) <= 0:
        pin.latest_dd_change.dd_amount_change = abs(pin.latest_dd_change.dd_amount_change or 1.0) + 5.0
        pin.latest_dd_change.dd_amount = round(older.dd_amount + pin.latest_dd_change.dd_amount_change, 2)


def _apply_first_review(pin: DDExplainerPromptInput, rng: random.Random) -> None:
    latest_dt = pin.latest_dd_change.datetime_from
    new_start = latest_dt.date() - timedelta(days=rng.randint(90, 365))
    for c in pin.account_context.contract_history:
        c.contract_start_date = new_start
        c.contract_rates_history[0].rate_effective_from = new_start
    pin.account_context.dd_change_history = [pin.latest_dd_change]
    pin.account_context.payment_history = [
        p for p in pin.account_context.payment_history
        if p.transaction_timestamp.date() >= new_start
    ]


_APPLIERS = {
    Trigger.Change_in_usage: _apply_change_in_usage,
    Trigger.Change_in_unit_rates: _apply_change_in_unit_rates,
    Trigger.Missed_bounced_DD_payments: _apply_missed_payments,
    Trigger.Manual_reduction: _apply_manual_reduction,
    Trigger.Exemption_Expiry: _apply_exemption_expiry,
    Trigger.First_DD_review_since_account_start: _apply_first_review,
}


def generate_dd_example(
    target_triggers: Set[Trigger], rng: random.Random
) -> DDExplainerPromptInput:
    """Build a DDExplainerPromptInput whose data implies exactly `target_triggers`.

    Raises AssertionError if the oracle disagrees with the target after mutation.
    """
    if not target_triggers or target_triggers == {Trigger.No_triggers_identified}:
        pin = _baseline_input(rng, dd_direction=rng.choice([-1, 1]))
        triggers = detect_triggers(pin)
        assert triggers == {Trigger.No_triggers_identified}, (
            f"baseline unexpectedly triggered {triggers}"
        )
        return pin

    needs_increase = bool(
        target_triggers & {
            Trigger.Manual_reduction,
            Trigger.Exemption_Expiry,
            Trigger.First_DD_review_since_account_start,
        }
    )
    dd_direction = 1 if needs_increase else rng.choice([-1, 1])

    pin = _baseline_input(rng, dd_direction=dd_direction)
    # Apply mutations. Manual_reduction / Exemption_Expiry may bump the DD
    # amount, which must happen BEFORE usage/rate appliers re-read dd_sign.
    apply_order = [
        Trigger.Manual_reduction,
        Trigger.Exemption_Expiry,
        Trigger.First_DD_review_since_account_start,
        Trigger.Change_in_usage,
        Trigger.Change_in_unit_rates,
        Trigger.Missed_bounced_DD_payments,
    ]
    for trig in apply_order:
        if trig in target_triggers:
            _APPLIERS[trig](pin, rng)

    detected = detect_triggers(pin)
    assert detected == target_triggers, (
        f"generator/oracle drift: requested {target_triggers}, got {detected}"
    )
    return pin


# =============================================================================
# Dataset builder
# =============================================================================


_NON_TERMINAL_TRIGGERS = [t for t in Trigger if t != Trigger.No_triggers_identified]

_SINGLE_WEIGHTS = {
    Trigger.Change_in_usage: 30,
    Trigger.Change_in_unit_rates: 25,
    Trigger.Missed_bounced_DD_payments: 15,
    Trigger.First_DD_review_since_account_start: 15,
    Trigger.Manual_reduction: 8,
    Trigger.Exemption_Expiry: 7,
}

# Distribution of target-set sizes — realistic skew for v2 regeneration.
# Most rows are 1-3 triggers; 4+ kept as a thin robustness tail.
# Was: hardcoded 5% no-triggers + 80% single + 15% pair (zero 3+).
_SIZE_WEIGHTS = {
    1: 0.45,
    2: 0.30,
    3: 0.20,
    4: 0.04,
    5: 0.01,
}

# First_DDR requires ≤1 prior DD; Manual_reduction and Exemption_Expiry both
# require prior DDs. So First_DDR cannot coexist with either. Used by the
# multi-trigger sampler to reject structurally-incompatible combos.
_FIRST_DDR_INCOMPAT = {Trigger.Manual_reduction, Trigger.Exemption_Expiry}


def _sample_target_set(rng: random.Random) -> Set[Trigger]:
    size = rng.choices(
        list(_SIZE_WEIGHTS.keys()),
        weights=list(_SIZE_WEIGHTS.values()),
        k=1,
    )[0]

    if size == 1:
        # Within the size=1 bucket, ~5% are the "No triggers identified"
        # sanity baseline; the rest sample from _SINGLE_WEIGHTS.
        if rng.random() < 0.05:
            return {Trigger.No_triggers_identified}
        triggers = list(_SINGLE_WEIGHTS.keys())
        weights = [_SINGLE_WEIGHTS[t] for t in triggers]
        return {rng.choices(triggers, weights=weights, k=1)[0]}

    # Size >= 2: rejection-sample to respect First_DDR's structural rule.
    # Combinatorial counts: 13 valid 2-combos, 13 valid 3-combos,
    # 6 valid 4-combos, exactly 1 valid 5-combo.
    for _ in range(20):
        candidate = set(rng.sample(_NON_TERMINAL_TRIGGERS, size))
        if (Trigger.First_DD_review_since_account_start in candidate
                and candidate & _FIRST_DDR_INCOMPAT):
            continue
        return candidate
    # Fallback: exclude First_DDR from the pool. Guarantees a valid combo
    # for any size up to 5 (the 5 remaining triggers compose freely).
    pool = [t for t in _NON_TERMINAL_TRIGGERS
            if t != Trigger.First_DD_review_since_account_start]
    return set(rng.sample(pool, size))


def build_dataset(n: int = 1000, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = n * 5
    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        target = _sample_target_set(rng)
        try:
            pin = generate_dd_example(target, rng)
        except AssertionError:
            # A few 2-trigger combos are structurally incompatible (e.g.
            # First_DDR + Manual_reduction needs both ≤1 prior DDs and a prior
            # DD with the manual-reduction flag). Re-sample instead of crash.
            continue
        rows.append(
            {
                "prompt": build_chat_messages(pin),
                "ground_truth_triggers": sorted(t.value for t in target),
                "input_json": pin.model_dump(mode="json"),
            }
        )
    if len(rows) < n:
        raise RuntimeError(
            f"Only produced {len(rows)}/{n} rows after {max_attempts} attempts"
        )
    return rows


# =============================================================================
# Self-check
# =============================================================================


def _self_check() -> None:
    rng = random.Random(0)
    tested = 0
    skipped = 0
    for size in (1, 2, 3, 4, 5):
        for combo in itertools.combinations(_NON_TERMINAL_TRIGGERS, size):
            target = set(combo)
            # Skip structurally-incompatible combos (First_DDR clashes with
            # Manual_reduction or Exemption_Expiry — see _FIRST_DDR_INCOMPAT).
            if (Trigger.First_DD_review_since_account_start in target
                    and target & _FIRST_DDR_INCOMPAT):
                skipped += 1
                continue
            try:
                generate_dd_example(target, rng)
                tested += 1
            except AssertionError as e:
                raise RuntimeError(f"Drift for target={target}: {e}")
    generate_dd_example({Trigger.No_triggers_identified}, rng)
    tested += 1
    print(f"Self-check: {tested} combinations OK, {skipped} structurally skipped.")


_GENERATOR_VERSION = "2.0.0"


def _summarise_triggers(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter(tuple(r["ground_truth_triggers"]) for r in rows)
    return {" + ".join(k) if k else "(none)": v for k, v in c.most_common()}


def write_dataset_jsonl(n: int, seed: int, out_dir: Path) -> Path:
    """Generate `n` rows and persist as UTC-datetime + rowcount stamped JSONL.

    JSONL layout:
      line 0      : metadata record with `{"__meta__": true, ...provenance}`
      lines 1..n  : data records `{"row_index", "prompt", "ground_truth_triggers", "input_json"}`

    A sidecar `<stem>.meta.json` duplicates the metadata as a standalone JSON
    file for convenience (no JSONL parsing needed to inspect provenance).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc)
    rows = build_dataset(n=n, seed=seed)
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"dd_dataset_{stamp}_{len(rows)}rows.jsonl"

    meta = {
        "__meta__": True,
        "generator": "dd_explainer_data_generator.py",
        "generator_version": _GENERATOR_VERSION,
        "generated_at_utc": generated_at.isoformat(),
        "seed": seed,
        "n_rows_requested": n,
        "n_rows_written": len(rows),
        "jsonl_file": path.name,
        "row_schema": ["row_index", "prompt", "ground_truth_triggers", "input_json"],
        "trigger_enum": [t.value for t in Trigger],
        "trigger_distribution": _summarise_triggers(rows),
        "sampling_weights_single_trigger": {t.value: w for t, w in _SINGLE_WEIGHTS.items()},
        "rule_thresholds": {
            "change_in_usage_min_change_percent": 5.0,
            "change_in_unit_rates_min_change_percent": 2.0,
            "exemption_expiry_window_days": 120,
            "missed_payment_lookback_months": 6,
            "first_ddr_max_months_since_start": 18,
        },
    }

    with path.open("w") as f:
        f.write(json.dumps(meta, default=str) + "\n")
        for i, r in enumerate(rows):
            f.write(json.dumps({"row_index": i, **r}, default=str) + "\n")

    # Sidecar: same metadata, plus size of the JSONL (known only after writing).
    meta_with_size = {**meta, "jsonl_size_bytes": path.stat().st_size}
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta_with_size, indent=2))
    return path


def main(
    n_rows: int = typer.Option(5500, "--n-rows", "-n", help="Number of rows to generate."),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
    out_dir: Path = typer.Option(
        Path("/workspace/gemma4_rl/data"), "--out-dir",
        help="Directory to write the JSONL into.",
    ),
    skip_self_check: bool = typer.Option(
        False, "--skip-self-check", help="Skip the generator/oracle drift self-check.",
    ),
) -> None:
    """Generate a synthetic DD explainer dataset."""
    if not skip_self_check:
        _self_check()
    path = write_dataset_jsonl(n=n_rows, seed=seed, out_dir=out_dir)
    size_mb = path.stat().st_size / 1024 / 1024
    typer.echo(f"Wrote {n_rows} rows to {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    typer.run(main)
