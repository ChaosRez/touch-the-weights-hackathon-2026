"""Deterministic scoring smoke test (no model, no network).

Drives the framework's real scoring path — ``Task.score`` over a hand-built tool-using
``Trace`` from the *real* taskset — to prove the v3 acceptance reward wiring end to end.
"""

from __future__ import annotations

import verifiers.v1 as vf

from alien_api_env.vf import AlienApiTaskset, AlienApiTasksetConfig

from ._trace import answer_trace, build_trace


def _one_task() -> vf.Task:
    return AlienApiTaskset(AlienApiTasksetConfig(id="alien-api")).select(1)[0]


async def test_acceptance_rewards_accepted_answer_under_budget() -> None:
    task = _one_task()
    trace = build_trace(task, task.data.accepted, tool_returns=2)
    await task.score(trace)
    assert trace.reward == 1.0
    assert trace.rewards["acceptance"].score == 1.0
    assert trace.metrics["tool_calls"] == 2.0
    assert trace.metrics["preference_accepted"] == 1.0


async def test_acceptance_zero_on_wrong_answer() -> None:
    task = _one_task()
    trace = build_trace(task, "not-the-answer", tool_returns=1)
    await task.score(trace)
    assert trace.reward == 0.0
    assert trace.metrics["preference_accepted"] == 0.0
    assert trace.metrics["value_correct"] == 0.0


async def test_typed_channel_end_to_end_score_then_finalize() -> None:
    task = _one_task()
    trace = answer_trace(task, task.data.accepted, tool_returns=1)
    await task.score(trace)
    assert trace.reward == 1.0
    await task.finalize(trace, None)
    assert trace.info["accepted"] == task.data.accepted
    assert trace.info["feedback"] == "Accepted."
