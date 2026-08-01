"""Strict legal-memory boundary for the Phase 4 Alien API experiment."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from alien_api_env.feedbacker.feedback import ACCEPTED_FEEDBACK
from alien_api_env.vf.tools import is_submit_answer
from cartridge_memory.models import ToolExecution

_NOT_DEFENSIBLE_PREFIX = "Rejected. That does not follow from the records"
_REJECTED_PREFIX = "Rejected. "


def _tool_fields(execution: ToolExecution | dict[str, Any]) -> tuple[str, dict, dict]:
    if isinstance(execution, ToolExecution):
        return execution.call.name, execution.call.arguments, execution.result
    if not isinstance(execution, dict):
        raise TypeError("tool executions must be ToolExecution instances or explicit dictionaries")
    name = execution.get("name")
    arguments = execution.get("arguments")
    result = execution.get("result")
    if not isinstance(name, str) or not isinstance(arguments, dict) or not isinstance(result, dict):
        raise TypeError("tool execution dictionaries require name, arguments, and result fields")
    return name, arguments, result


def legal_events(
    *,
    feedback: str,
    tool_executions: Sequence[ToolExecution | dict[str, Any]],
) -> tuple[str, ...]:
    """Serialize only legal reviewer corrections and observed non-answer tool outcomes.

    The keyword-only API deliberately cannot accept a rollout/result record. No reward,
    accepted label, answer, metric, fleet metadata, or offline reporting tag is in scope.
    """

    if not isinstance(feedback, str):
        raise TypeError("feedback must be the explicit trace.info['feedback'] string")
    if isinstance(tool_executions, (str, bytes, dict)):
        raise TypeError("tool_executions must be an explicit sequence, not a result record")

    events: list[str] = []
    stripped = feedback.strip()
    if (
        stripped
        and stripped != ACCEPTED_FEEDBACK
        and not stripped.startswith(_NOT_DEFENSIBLE_PREFIX)
    ):
        correction = stripped.removeprefix(_REJECTED_PREFIX)
        events.append(f"REVIEWER: {correction}")

    for execution in tool_executions:
        name, arguments, result = _tool_fields(execution)
        if is_submit_answer(name):
            continue
        canonical_arguments = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if result.get("ok"):
            outcome = "ok"
        else:
            error = result.get("error")
            if isinstance(error, dict):
                code = str(error.get("code", "?"))
                message = str(error.get("message", ""))
            else:
                code, message = "?", ""
            outcome = f"ERROR:{code}:{message}"
        events.append(f"TOOL: {name}({canonical_arguments}) -> {outcome}")
    return tuple(events)
