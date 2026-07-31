"""v4 reward: binary acceptance, plus the curve metrics.

- exact accepted answer -> ``1.0`` regardless of tool calls (efficiency is weight-0
  observability only, 2026-07-28 change);
- right value, wrong presentation -> ``0.0`` with ``value_correct==1`` and
  ``preference_accepted==0`` (the two layers separate);
- wrong value -> ``0.0`` with ``value_correct==0`` and every invoked dimension violated.
"""

from __future__ import annotations

import verifiers.v1 as vf

from alien_api_env.vf import AlienApiData, AlienApiTask

from ._trace import build_trace

# A synthetic 2-dimension instance: annual revenue under (basis x unit). The profile
# holds mrr_derived + dollars, so accepted == "103000".
_DEFENSIBLE = {
    "annual_revenue_basis=arr_field|money_unit=cents": "9700000",
    "annual_revenue_basis=arr_field|money_unit=dollars": "97000",
    "annual_revenue_basis=mrr_derived|money_unit=cents": "10300000",
    "annual_revenue_basis=mrr_derived|money_unit=dollars": "103000",
}
_CHOICES = {"annual_revenue_basis": "mrr_derived", "money_unit": "dollars"}


def _task(budget: int = 4) -> AlienApiTask:
    return AlienApiTask(
        AlienApiData(
            idx=0,
            name="t",
            prompt="p",
            invoked=("annual_revenue_basis", "money_unit"),
            defensible=dict(_DEFENSIBLE),
            accepted="103000",
            choices=dict(_CHOICES),
            budget=budget,
            kind="money",
        )
    )


async def _score(task: AlienApiTask, reply: str, tool_returns) -> vf.Trace:
    trace = build_trace(task, reply, tool_returns=tool_returns)
    await task.score(trace)
    return trace


async def test_accepted_scores_full_regardless_of_calls() -> None:
    trace = await _score(_task(budget=4), "103000", 1)
    assert trace.reward == 1.0
    assert trace.metrics["preference_accepted"] == 1.0
    assert trace.metrics["value_correct"] == 1.0
    assert trace.metrics["preferences_violated"] == 0.0
    assert trace.metrics["tool_calls"] == 1.0
    assert trace.metrics["over_budget"] == 0.0


async def test_zero_tool_calls_accepted_scores_full() -> None:
    trace = await _score(_task(budget=4), "103000", 0)
    assert trace.reward == 1.0


async def test_right_value_wrong_presentation_scores_zero_but_is_value_correct() -> None:
    # cents instead of dollars: defensible reading, wrong convention -> layer 2 rejects.
    trace = await _score(_task(budget=4), "10300000", 1)
    assert trace.reward == 0.0
    assert trace.metrics["value_correct"] == 1.0
    assert trace.metrics["preference_accepted"] == 0.0
    assert trace.metrics["preferences_violated"] == 1.0  # money_unit missed, basis matched


async def test_wrong_source_and_unit_counts_both_violations() -> None:
    trace = await _score(_task(budget=4), "9700000", 0)  # arr_field + cents
    assert trace.reward == 0.0
    assert trace.metrics["value_correct"] == 1.0
    assert trace.metrics["preferences_violated"] == 2.0


async def test_wrong_value_scores_zero_for_any_k() -> None:
    task = _task(budget=4)
    for k in (0, 2, 10):
        trace = await _score(task, "42", k)
        assert trace.reward == 0.0
        assert trace.metrics["value_correct"] == 0.0
        assert trace.metrics["preference_accepted"] == 0.0
        # nothing matched: every invoked dimension counts as violated
        assert trace.metrics["preferences_violated"] == 2.0


async def test_over_budget_accepted_still_scores_full_but_is_flagged() -> None:
    trace = await _score(_task(budget=3), "103000", 5)
    assert trace.reward == 1.0  # acceptance is the whole reward
    assert trace.metrics["over_budget"] == 1.0  # efficiency stays observable
    assert trace.metrics["preference_accepted"] == 1.0


async def test_at_budget_accepted_scores_full() -> None:
    trace = await _score(_task(budget=4), "103000", 4)
    assert trace.reward == 1.0
    assert trace.metrics["over_budget"] == 0.0


async def test_prose_wrapped_value_is_diagnosed_but_not_accepted() -> None:
    """Acceptance is exact — presentation is the learned preference, so prose around the
    right value is rejected; the extractor still credits value_correct."""
    trace = await _score(_task(budget=4), "The answer is **103,000** dollars.", 0)
    assert trace.reward == 0.0
    assert trace.metrics["value_correct"] == 1.0
    assert trace.metrics["preference_accepted"] == 0.0


async def test_artifact_tokens_metric_sums_tool_return_tokens() -> None:
    trace = await _score(_task(budget=4), "103000", ["hello world", "more tokens here"])
    assert trace.metrics["artifact_tokens"] > 0.0
    assert trace.metrics["tool_calls"] == 2.0
