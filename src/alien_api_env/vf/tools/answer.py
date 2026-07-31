"""The typed answer channel (Rework 1, v1.1): a ``submit_answer`` tool.

v1.0 read the answer off the last assistant message, which made *format compliance* dominate the
signal (a reasoning model that answered "**190,265 cents**" scored zero on a right value). The
rework adds a typed channel: the model calls ``submit_answer(value=...)`` with the bare value, so
the answer is a structured field that never mixes with prose. Free text (the last assistant
message) remains a fallback for models that do not call the tool.

The toolset and the reward run in different processes (the toolset is served over MCP), so the
submitted value cannot be read off a toolset instance attribute — it must travel through the
trace. ``submit_answer`` therefore *echoes* the value into its result envelope, and the reward
recovers it with :func:`submitted_value` from the last ``submit_answer`` tool result. The call is
**exempt from ``calls_used``** (it is the answer, not a world read), so adding the channel does not
shift the budget economics; the reward filters it out by tool name.
"""

from __future__ import annotations

import json
import re
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.types import content_text

from alien_api_env.vf.tools.envelopes import ok

# The tool name the reward keys on (to read the submitted value and to exempt it from the tool
# budget). Matched as a substring of ``ToolMessage.name`` so an MCP server-name prefix is tolerated.
SUBMIT_ANSWER_TOOL = "submit_answer"

_SUBMITTED_RE = re.compile(r'"submitted"\s*:\s*"((?:[^"\\]|\\.)*)"')


def encode_submission(value: str) -> dict[str, Any]:
    """The ``submit_answer`` result envelope carrying the submitted value under ``data.submitted``
    (the shape :func:`submitted_value` recovers). Kept tiny — this is the answer, not an artifact."""
    return ok({"submitted": value})


def _extract(text: str) -> str | None:
    """Recover the submitted value from a serialized ``submit_answer`` result. JSON-parse first
    (the normal path), with a regex fallback robust to whatever wrapping MCP applies to the text."""
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict) and "submitted" in data:
            return str(data["submitted"])
        if isinstance(data, str):
            return data
    m = _SUBMITTED_RE.search(text or "")
    if m:
        try:
            return json.loads(f'"{m.group(1)}"')
        except ValueError:
            return m.group(1)
    return None


def submitted_value(trace: vf.Trace) -> str | None:
    """The value from the last ``submit_answer`` result in the trace, or ``None`` if the model
    never used the typed channel (the reward then falls back to ``trace.last_reply``)."""
    found: str | None = None
    for m in trace.tool_messages:
        if SUBMIT_ANSWER_TOOL in (m.name or ""):
            recovered = _extract(content_text(m.content) or "")
            if recovered is not None:
                found = recovered
    return found


def is_submit_answer(name: str | None) -> bool:
    """Whether a tool-result message name is the answer channel (exempt from ``calls_used``)."""
    return SUBMIT_ANSWER_TOOL in (name or "")


class AnswerToolset(vf.Toolset[vf.ToolsetConfig]):
    """The typed answer channel, served over MCP. Stateless and world-agnostic: it only records
    the submitted value into its result so the reward can read it back through the trace."""

    @vf.tool
    async def submit_answer(self, value: str) -> dict[str, Any]:
        """Submit your final answer. Call this once, with ``value`` set to exactly the answer and
        nothing else — no explanation, label, units, thousands separators, or markdown. This is the
        answer channel; it does not count against your tool-call budget.

        Args:
            value: the answer value, verbatim (e.g. ``13923977``, ``shipped``, ``DE``, ``inventory_v2``).
        """
        return encode_submission(value)


if __name__ == "__main__":  # MCP server entrypoint: `python -m alien_api_env.vf.tools.answer`
    AnswerToolset.run()
