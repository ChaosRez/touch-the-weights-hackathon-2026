"""Feedback (Phase 2): teaches the preference-class, never the answer.

- accepted work -> bare acceptance; violations -> one correction per violated dimension
  naming the class (the chosen convention), in the feedbacker's voice;
- a non-defensible value -> the data-recheck rejection (no convention taught);
- **no answer leak**: across seeds and every wrong-presentation reply, the feedback
  string never contains the accepted answer or any defensible value;
- positive control: a deliberately answer-leaking feedback string is caught by the same
  leak predicate the sweep uses (the predicate has teeth).
"""

from __future__ import annotations

import re

import pytest

from alien_api_env.feedbacker.feedback import (
    ACCEPTED_FEEDBACK,
    NOT_DEFENSIBLE_FEEDBACK,
    Violation,
    correction_sentence,
    persona_feedback,
    templated_feedback,
)
from alien_api_env.vf import AlienApiTask, AlienApiTaskset, AlienApiTasksetConfig

from ._trace import build_trace


def _tasks(offset: int = 0, n: int = 12) -> list[AlienApiTask]:
    return AlienApiTaskset(AlienApiTasksetConfig(id="alien-api", split="")).select(offset + n)[offset:]


def _leaks(feedback: str, values: set[str]) -> list[str]:
    """Defensible values leaked into a feedback string (word-boundary for short values,
    substring for longer ones)."""
    out = []
    for v in values:
        if len(v) <= 3:
            if re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", feedback):
                out.append(v)
        elif v in feedback:
            out.append(v)
    return out


async def test_accepted_answer_gets_bare_acceptance() -> None:
    for task in _tasks():
        trace = build_trace(task, task.data.accepted, tool_returns=1)
        await task.finalize(trace, None)
        assert trace.info["feedback"] == ACCEPTED_FEEDBACK
        assert trace.info["violated"] == []
        assert trace.info["value_defensible"] is True
        assert trace.info["accepted"] == task.data.accepted
        assert trace.info["invoked"] == list(task.data.invoked)
        assert trace.info["world_traps"] == list(task.data.world_traps)


async def test_violation_feedback_teaches_the_class() -> None:
    task = next(t for t in _tasks() if len(t.data.defensible) > 1)
    wrong = next(v for v in task.data.defensible.values() if v != task.data.accepted)
    trace = build_trace(task, wrong, tool_returns=1)
    await task.finalize(trace, None)
    assert trace.info["feedback"].startswith("Rejected.")
    assert trace.info["violated"], "a wrong defensible reading must violate something"
    for dim in trace.info["violated"]:
        assert dim in task.data.invoked
    # the feedback is exactly the persona-voiced (cache-or-template) corrections for the
    # violated dimensions' chosen options
    violations = [
        Violation(dimension=d, chosen=task.data.choices[d]) for d in trace.info["violated"]
    ]
    assert trace.info["feedback"] == persona_feedback(
        violations, persona_id=task.data.persona_id, hints=task.data.feedback_hints
    )


async def test_non_defensible_value_gets_data_recheck() -> None:
    task = _tasks()[0]
    trace = build_trace(task, "certainly-not-a-defensible-value-9999999999", tool_returns=1)
    await task.finalize(trace, None)
    assert trace.info["feedback"] == NOT_DEFENSIBLE_FEEDBACK
    assert trace.info["violated"] == list(task.data.invoked)
    assert trace.info["value_defensible"] is False  # the disambiguating flag


@pytest.mark.parametrize("offset", (0, 12, 24))
async def test_feedback_never_leaks_answers(offset) -> None:
    """Every wrong-presentation reply, every instance: the feedback names classes only."""
    for task in _tasks(offset):
        values = set(task.data.defensible.values())
        for reply in values | {"__wrong__"}:
            trace = build_trace(task, reply, tool_returns=0)
            await task.finalize(trace, None)
            fb = trace.info["feedback"]
            if reply == task.data.accepted:
                continue  # acceptance carries no content to leak
            leaked = _leaks(fb, values)
            assert not leaked, f"feedback leaks {leaked} for {task.data.prompt!r}: {fb!r}"


def test_leak_predicate_has_teeth() -> None:
    """Positive control: an answer-leaking feedback string is caught."""
    assert _leaks("Rejected. The correct answer was 103000.", {"103000"}) == ["103000"]
    assert _leaks("Report it as escalate next time.", {"escalate"}) == ["escalate"]


def test_correction_sentences_cover_every_dimension_option() -> None:
    from alien_api_env.world.preferences import SCHEMA

    hints = {"quarter_calendar": "April"}
    for d in SCHEMA:
        for option in d.options:
            sentence = correction_sentence(Violation(dimension=d.id, chosen=option), hints=hints)
            assert sentence and sentence[0].isupper()
            assert not re.search(r"\d", sentence), (
                f"correction for {d.id}={option} contains digits (leak risk): {sentence!r}"
            )


def test_fiscal_correction_requires_and_renders_the_month_hint() -> None:
    v = Violation(dimension="quarter_calendar", chosen="fiscal")
    assert "April" in correction_sentence(v, hints={"quarter_calendar": "April"})
    with pytest.raises(ValueError, match="hint"):
        correction_sentence(v, hints={})


def test_templated_feedback_composes_multiple_corrections() -> None:
    fb = templated_feedback(
        [
            Violation(dimension="money_unit", chosen="dollars"),
            Violation(dimension="money_rounding", chosen="thousands"),
        ]
    )
    assert fb.startswith("Rejected.")
    assert "whole dollars" in fb and "nearest thousand" in fb


def test_unknown_dimension_or_option_fails_fast() -> None:
    with pytest.raises(KeyError):
        correction_sentence(Violation(dimension="nope", chosen="x"))
    with pytest.raises(ValueError, match="not an option"):
        correction_sentence(Violation(dimension="money_unit", chosen="doubloons"))
