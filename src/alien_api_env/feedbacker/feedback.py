"""Deterministic teaching feedback: corrections that name the preference-class.

The feedback contract (the certification pillar binds to it):

- **teaches the class, never the answer**: a correction states the violated convention in
  general terms — "I use twelve times MRR", "round aggregate totals to the nearest
  thousand" — and never contains the instance's accepted answer or any defensible value.
- **generalizable**: each sentence transfers to every future instance invoking the same
  dimension. Where applying a convention needs a class-level parameter (the fiscal-year
  start month), the parameter is part of the taught rule, rendered as a word, never as a
  digit string that could collide with a value.
- **deterministic**: a pure function of the violated (dimension, chosen-option) pairs.
  The Phase 3 persona feedbacker only rephrases these sentences in a character's voice
  (cached to data, audited against this template); it never adds information.

The v2 outcome-only feedback ("the correct answer was X") is gone by design: revealing
the answer collapses a sequence to n=1. Teaching the convention is what makes instances
progressively easier while leaving the agent to apply it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from alien_api_env.world.preferences import dimension

# Literal answer tokens that are instance-independent (the status vocabulary is a closed
# set, mirrored from the stored world's status_code_table).
_STATUS_WORDS = ("received", "processing", "shipped", "delivered", "cancelled", "refunded")


@dataclass(frozen=True)
class Violation:
    """One violated preference: the dimension and the option the feedbacker holds."""

    dimension: str
    chosen: str


# Per (dimension, chosen option): the generalizable correction, in the feedbacker's
# first-person voice. ``{month}`` renders the fiscal-year start month by name (a
# class-level parameter of the convention, not an instance value).
_CORRECTIONS: dict[str, dict[str, str]] = {
    "annual_revenue_basis": {
        "arr_field": (
            "For an account's annual revenue I take the stored ARR field as-is; "
            "do not recompute it from monthly figures."
        ),
        "mrr_derived": (
            "For an account's annual revenue I use twelve times the monthly recurring "
            "revenue; the stored annual field lags and is not to be trusted."
        ),
    },
    "country_basis": {
        "billing": "When I ask where an account is based, I mean its billing country.",
        "shipping": "When I ask where an account is based, I mean its shipping country.",
    },
    "count_authority": {
        "search": (
            "For counts I take the search listing as authoritative, even when an "
            "aggregate disagrees with it."
        ),
        "aggregate": (
            "For counts I use the server-side aggregate; listings can be incomplete "
            "and are never the number I want."
        ),
    },
    "process_page_authority": {
        "v1": (
            "On governed process pages I follow the original page; a migration "
            "write-up does not move my sign-off."
        ),
        "v2": (
            "On governed process pages the migrated page governs; the original is "
            "history and I do not work from history."
        ),
    },
    "quarter_calendar": {
        "calendar": "Quarters are standard calendar quarters; do not apply any fiscal offset.",
        "fiscal": "Quarters follow our fiscal calendar; the fiscal year starts in {month}.",
    },
    "revenue_stage_scope": {
        "won_only": "Booked revenue means won opportunities only; pipeline does not count.",
        "won_plus_open": (
            "Booked revenue includes the open pipeline alongside won deals; only lost "
            "opportunities are excluded."
        ),
    },
    "overdue_boundary": {
        "strict": (
            "Overdue past a day threshold means strictly past it; an invoice exactly on "
            "the boundary day is out."
        ),
        "inclusive": (
            "I count an invoice sitting exactly on the day threshold as overdue; the "
            "boundary day is in."
        ),
    },
    "inventory_scope": {
        "on_hand": "Units available means gross on-hand stock, before any reservations.",
        "net_of_reserved": (
            "Units available means on-hand net of reservations; reserved stock is not "
            "available."
        ),
    },
    "money_unit": {
        "cents": "Monetary figures are reported as integer cents.",
        "dollars": "Monetary figures are reported in whole dollars, never cents.",
    },
    "money_rounding": {
        "exact": "Aggregate totals are reported exactly; never round them for me.",
        "thousands": "Aggregate totals are rounded to the nearest thousand.",
    },
    "status_style": {
        "word": "Order status is reported as the translated status word, never the raw code.",
        "code": "Order status is reported as the raw stored code, not a translation.",
    },
    "date_style": {
        "iso": "Dates are reported as ISO calendar dates, UTC.",
        "unix": "Timestamps are reported as raw Unix seconds, not calendar dates.",
    },
    "page_naming": {
        "id": "Cite knowledge-base pages by page id, not by title.",
        "title": "Cite knowledge-base pages by their title, not the raw id.",
    },
    "account_naming": {
        "id": "Refer to accounts by account id, not the company name.",
        "name": "Refer to accounts by company name, not the raw id.",
    },
    "missing_data_policy": {
        "report_zero": "An empty result is still a result; report the zero.",
        "escalate": (
            "I never take a bare empty result as an answer; raise it to me instead of "
            "reporting it."
        ),
    },
}

ACCEPTED_FEEDBACK = "Accepted."

NOT_DEFENSIBLE_FEEDBACK = (
    "Rejected. That does not follow from the records at all; recheck the data before "
    "you worry about my conventions."
)


def correction_sentence(violation: Violation, hints: Mapping[str, str] | None = None) -> str:
    """The teaching sentence for one violated (dimension, chosen-option) pair."""
    dim = dimension(violation.dimension)  # KeyError on an unknown dimension (fail fast)
    if violation.chosen not in dim.options:
        raise ValueError(f"{violation.chosen!r} is not an option of {dim.id!r}")
    sentence = _CORRECTIONS[dim.id][violation.chosen]
    if "{month}" in sentence:
        month = (hints or {}).get(dim.id)
        if not month:
            raise ValueError(f"dimension {dim.id!r} needs a feedback hint to render")
        sentence = sentence.format(month=month)
    return sentence


def templated_feedback(
    violations: list[Violation],
    *,
    value_defensible: bool = True,
    hints: Mapping[str, str] | None = None,
) -> str:
    """The full feedback string for one reviewed answer.

    Accepted work gets a bare acceptance; a non-defensible value gets the data-recheck
    rejection (no convention is taught for an answer that is wrong under every reading);
    otherwise every violated preference-class is corrected, in violation order.
    """
    if not value_defensible:
        return NOT_DEFENSIBLE_FEEDBACK
    if not violations:
        return ACCEPTED_FEEDBACK
    corrections = " ".join(correction_sentence(v, hints) for v in violations)
    return f"Rejected. {corrections}"


# --------------------------------------------------------------------------- #
# Persona voice (Phase 3): cached LLM rephrasings of the template sentences.    #
# The cache is frozen data; a cache miss falls back to the template, so the     #
# rollout path never calls a model and never blocks on the cache's coverage.    #
# --------------------------------------------------------------------------- #

CACHE_DIR = Path(__file__).resolve().parent / "feedback_cache"

# Content rule every cached rephrasing must satisfy (the same rule the templates obey):
# no digits (numeric answers can never leak through a digit-free sentence) and none of
# the instance-independent literal answer tokens.
_FORBIDDEN_CACHE_TOKENS = ("escalate", "sop-refunds", *_STATUS_WORDS)


def cache_key(violation: Violation, hints: Mapping[str, str] | None = None) -> str:
    """The frozen-cache key for one violated (dimension, chosen-option) pair."""
    month = (hints or {}).get(violation.dimension, "")
    return f"{violation.dimension}={violation.chosen}#hint={month}"


def _load_cache(persona_id: str) -> dict[str, str]:
    path = CACHE_DIR / f"{persona_id}.json"
    if not path.exists():
        return {}
    return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def _cache_entry_ok(sentence: str) -> bool:
    if not sentence.strip():
        return False
    if re.search(r"\d", sentence):
        return False
    lowered = sentence.lower()
    return not any(tok in lowered for tok in _FORBIDDEN_CACHE_TOKENS)


def audit_feedback_cache(persona_id: str) -> list[str]:
    """Cache entries violating the feedback-content rule (empty list == clean).

    The deterministic template is the ground truth; a cached rephrasing may change the
    voice, never the information. Digits and literal answer tokens are how information
    leaks, so both are rejected outright.
    """
    return [
        f"{key}: {sentence!r}"
        for key, sentence in _load_cache(persona_id).items()
        if not _cache_entry_ok(sentence)
    ]


def persona_feedback(
    violations: list[Violation],
    *,
    persona_id: str,
    value_defensible: bool = True,
    hints: Mapping[str, str] | None = None,
) -> str:
    """Feedback in the persona's voice, from the frozen cache; template on any miss.

    Offline and deterministic: reads committed cache files only. Cached sentences that
    fail the content audit are ignored (the template is always safe). The acceptance and
    data-recheck strings stay canonical — only corrections carry voice.
    """
    if not value_defensible:
        return NOT_DEFENSIBLE_FEEDBACK
    if not violations:
        return ACCEPTED_FEEDBACK
    cache = _load_cache(persona_id)
    sentences = []
    for v in violations:
        cached = cache.get(cache_key(v, hints))
        if cached is not None and _cache_entry_ok(cached):
            sentences.append(cached)
        else:
            sentences.append(correction_sentence(v, hints))
    return "Rejected. " + " ".join(sentences)


def generate_feedback_cache(persona_id: str, brief: str | None = None) -> dict[str, str] | None:
    """Endpoint-gated: rephrase every correction sentence in the persona's voice and
    freeze the cache to ``feedback_cache/<persona_id>.json``.

    Returns the cache dict, or ``None`` (SKIPPED) when no endpoint is configured.
    Rephrasings that fail the content audit fall back to the template sentence, so a
    frozen cache is always clean. Hint-bearing sentences are generated per month name.
    """
    from alien_api_env.feedbacker.persona import PERSONA_BRIEFS, endpoint_configured

    if not endpoint_configured():
        return None
    brief = brief or PERSONA_BRIEFS[persona_id]

    import os

    from openai import OpenAI  # lazy: build-time dependency only

    client = OpenAI(
        base_url=os.environ["VF_PROBE_BASE_URL"], api_key=os.environ["VF_PROBE_API_KEY"]
    )
    model = os.environ["VF_PROBE_MODEL"]

    months = ("February", "April", "July", "October")  # the sampled fiscal starts
    cache: dict[str, str] = {}
    for dim_id, per_option in _CORRECTIONS.items():
        for option in per_option:
            hint_values = months if "{month}" in per_option[option] else ("",)
            for month in hint_values:
                hints = {dim_id: month} if month else None
                violation = Violation(dimension=dim_id, chosen=option)
                template = correction_sentence(violation, hints)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"You are {brief}\n\nRephrase this review correction in "
                                "your own voice, one or two short sentences, keeping "
                                "EXACTLY the same rule and nothing more. Never use "
                                "digits and never state a specific answer value.\n\n"
                                f"Correction: {template}"
                            ),
                        }
                    ],
                )
                text = (response.choices[0].message.content or "").strip()
                cache[cache_key(violation, hints)] = text if _cache_entry_ok(text) else template
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{persona_id}.json"
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache
