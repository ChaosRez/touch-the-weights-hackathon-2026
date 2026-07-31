"""The typed answer channel under the v3 reward: ``submit_answer`` records the value,
the reward reads it back through the trace, and the call stays exempt from
``calls_used``. Exactness matters in v3 — the channel carries the presentation."""

from __future__ import annotations

import verifiers.v1 as vf

from alien_api_env.vf import AlienApiData, AlienApiTask, AnswerToolset
from alien_api_env.vf.tools.answer import submitted_value

from ._trace import answer_trace, build_trace

_DEFENSIBLE = {
    "missing_data_policy=report_zero": "0",
    "missing_data_policy=escalate": "escalate",
}


def _task(accepted: str = "escalate", budget: int = 2) -> AlienApiTask:
    return AlienApiTask(
        AlienApiData(
            idx=0,
            name="t",
            prompt="p",
            invoked=("missing_data_policy",),
            defensible=dict(_DEFENSIBLE),
            accepted=accepted,
            choices={"missing_data_policy": "escalate" if accepted == "escalate" else "report_zero"},
            budget=budget,
            kind="policy",
        )
    )


async def test_answer_toolset_wired_into_task() -> None:
    assert AnswerToolset in AlienApiTask.tools


async def test_submit_answer_tool_records_value() -> None:
    ts = AnswerToolset(vf.ToolsetConfig())
    env = await ts.submit_answer(value="escalate")
    assert env["ok"] is True
    assert env["data"]["submitted"] == "escalate"


async def test_reward_reads_submitted_value_over_last_reply() -> None:
    task = _task(budget=2)
    trace = answer_trace(task, "escalate", reply="Let me know if you need more detail.", tool_returns=1)
    assert submitted_value(trace) == "escalate"
    await task.score(trace)
    assert trace.reward == 1.0
    assert trace.metrics["preference_accepted"] == 1.0
    assert trace.metrics["tool_calls"] == 1.0  # the one world read; submit_answer exempt


async def test_submit_answer_exempt_from_calls_used() -> None:
    task = _task("escalate", budget=4)
    with_submit = answer_trace(task, "escalate", tool_returns=2)
    without = build_trace(task, "escalate", tool_returns=2)
    await task.score(with_submit)
    await task.score(without)
    assert with_submit.reward == without.reward == 1.0
    # the exemption still matters for the observability metric:
    assert with_submit.metrics["tool_calls"] == 2.0


async def test_submitted_value_is_whitespace_stripped_only() -> None:
    task = _task("escalate", budget=2)
    trace = answer_trace(task, "  escalate ", tool_returns=0)
    await task.score(trace)
    assert trace.reward == 1.0  # stripped
    wrapped = answer_trace(task, "I would **escalate** this.", tool_returns=0)
    await task.score(wrapped)
    assert wrapped.reward == 0.0  # prose is not the value; presentation is learned
    assert wrapped.metrics["value_correct"] == 1.0


async def test_free_text_fallback_when_no_submit_answer() -> None:
    task = _task("escalate", budget=2)
    trace = build_trace(task, "escalate", tool_returns=1)
    assert submitted_value(trace) is None
    await task.score(trace)
    assert trace.reward == 1.0


async def test_wrong_submitted_value_scores_zero() -> None:
    task = _task("escalate", budget=2)
    trace = answer_trace(task, "0", tool_returns=0)
    await task.score(trace)
    assert trace.reward == 0.0
    assert trace.metrics["value_correct"] == 1.0  # "0" is the other defensible reading
    assert trace.metrics["preferences_violated"] == 1.0
