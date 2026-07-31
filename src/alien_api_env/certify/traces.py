"""Hand-built `Trace`s for in-process certification (no model, no network).

A v1 rollout always proxies model calls through the interception server to a live
endpoint, so the deterministic certify pillars drive `Task.score` / `finalize` over a
`Trace` built by hand: a linear chain of tool-result nodes followed by the final sampled
assistant reply. `trace.tool_messages` then returns the tool results (the reward's
`calls_used`) and `trace.last_reply` the reply.

Promoted out of `tests/_trace.py` (Plan 04-01) so library code — the certify subpackage —
can build synthetic traces without importing `tests/`. The test shim re-exports from here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import ToolMessage


def build_trace(task: vf.Task, reply: str, tool_returns: list[str] | int = 0) -> vf.Trace:
    """A finished trace: `tool_returns` tool messages, then `reply` as the sampled answer.

    `tool_returns` may be a list of tool-result payload strings (their tokens feed the
    `artifact_tokens` metric) or an int count of empty tool results. Nodes are chained via
    `parent` so the trace has a single branch (leaf = the assistant reply).
    """
    if isinstance(tool_returns, int):
        tool_returns = ["" for _ in range(tool_returns)]

    trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data))
    parent: int | None = None
    for i, payload in enumerate(tool_returns):
        trace.nodes.append(
            MessageNode(
                parent=parent,
                message=ToolMessage(tool_call_id=f"call-{i}", content=payload, name="tool"),
                sampled=False,
            )
        )
        parent = len(trace.nodes) - 1
    trace.nodes.append(
        MessageNode(parent=parent, message=vf.AssistantMessage(content=reply), sampled=True)
    )
    return trace


def witness_trace(task: vf.Task, tool_call_names: Sequence[str], reply: str) -> vf.Trace:
    """A witness trace: one `ToolMessage` per named call, then `reply` as the answer.

    The call *names* become the tool-result payloads so `len(trace.tool_messages)` is the
    realistic call-count `k` for the efficiency reward — the witness's minimal quirk-aware
    path made concrete, not a bare integer.
    """
    return build_trace(task, reply, tool_returns=list(tool_call_names))


def answer_trace(
    task: vf.Task,
    value: str,
    *,
    reply: str = "",
    tool_returns: int = 0,
) -> vf.Trace:
    """A trace whose answer comes through the typed ``submit_answer`` channel (Rework 1).

    Emits ``tool_returns`` world-read tool results, then a ``submit_answer`` tool result carrying
    ``value`` (named so the reward's ``submitted_value`` recovers it and ``calls_used`` exempts it),
    then ``reply`` as the sampled assistant message. Mirrors how a real rollout that calls
    ``submit_answer`` lands in the trace, so certify can score the typed channel in-process.
    """
    from alien_api_env.vf.tools.answer import SUBMIT_ANSWER_TOOL, encode_submission

    trace = build_trace(task, reply, tool_returns=tool_returns)
    # Splice the submit_answer tool result in just before the sampled assistant leaf.
    leaf = trace.nodes.pop()
    parent = leaf.parent
    trace.nodes.append(
        MessageNode(
            parent=parent,
            message=ToolMessage(
                tool_call_id="call-submit",
                content=json.dumps(encode_submission(value)),
                name=SUBMIT_ANSWER_TOOL,
            ),
            sampled=False,
        )
    )
    leaf.parent = len(trace.nodes) - 1
    trace.nodes.append(leaf)
    return trace
