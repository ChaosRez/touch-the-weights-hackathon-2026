from __future__ import annotations

import pytest

from cartridge_memory.legal_memory import legal_events


def _tool(name="get_account", arguments=None, result=None):
    return {
        "name": name,
        "arguments": {"b": 2, "a": 1} if arguments is None else arguments,
        "result": (
            {"ok": True, "data": {"secret": "not serialized"}}
            if result is None
            else result
        ),
    }


def test_legal_serializer_is_keyword_only_and_rejects_whole_records():
    record = {"feedback": "Rejected. Keep this.", "tool_executions": []}

    with pytest.raises(TypeError):
        legal_events(record)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        legal_events(feedback=record, tool_executions=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        legal_events(feedback="Rejected. Keep this.", tool_executions=record)  # type: ignore[arg-type]


def test_legal_serializer_is_canonical_verbatim_and_excludes_answer_channel():
    events = legal_events(
        feedback="Rejected. Preserve this correction exactly.",
        tool_executions=[
            _tool(),
            _tool(name="submit_answer", arguments={"value": "forbidden-answer"}),
            _tool(
                name="missing",
                arguments={},
                result={"ok": False, "error": {"code": "unknown", "message": "no such tool"}},
            ),
        ],
    )

    assert events == (
        "REVIEWER: Preserve this correction exactly.",
        'TOOL: get_account({"a":1,"b":2}) -> ok',
        "TOOL: missing({}) -> ERROR:unknown:no such tool",
    )
    assert "forbidden-answer" not in "\n".join(events)
    assert "secret" not in "\n".join(events)


@pytest.mark.parametrize(
    "feedback",
    [
        "Accepted.",
        "Rejected. That does not follow from the records at all; recheck the data before answering.",
    ],
)
def test_legal_serializer_ignores_contentless_feedback(feedback):
    assert legal_events(feedback=feedback, tool_executions=[]) == ()
