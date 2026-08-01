from __future__ import annotations

import pytest

from cartridge_memory.qwen_agent import ToolCallParseError, parse_qwen_turn


def test_parses_reasoning_text_and_tool_call() -> None:
    parsed = parse_qwen_turn(
        '<think>Need the exact aggregate.</think>\n'
        '<tool_call>{"name":"count_accounts","arguments":{"region":2}}</tool_call>',
        turn_index=3,
    )

    assert parsed.reasoning == "Need the exact aggregate."
    assert parsed.content == ""
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].id == "call-3-0"
    assert parsed.tool_calls[0].name == "count_accounts"
    assert parsed.tool_calls[0].arguments == {"region": 2}


def test_parses_multiple_calls_and_string_arguments() -> None:
    parsed = parse_qwen_turn(
        "I'll inspect both.\n"
        '<tool_call>{"name":"list_spaces","arguments":{}}</tool_call>\n'
        '<tool_call>{"name":"read_page","arguments":"{\\"page_id\\":\\"p-1\\"}"}</tool_call>',
        turn_index=0,
    )

    assert parsed.content == "I'll inspect both."
    assert [call.name for call in parsed.tool_calls] == ["list_spaces", "read_page"]
    assert parsed.tool_calls[1].arguments == {"page_id": "p-1"}


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name":"list_spaces"',
        "<tool_call>not-json</tool_call>",
        '<tool_call>{"arguments":{}}</tool_call>',
        '<tool_call>{"name":"x","arguments":[]}</tool_call>',
    ],
)
def test_rejects_malformed_tool_calls(text: str) -> None:
    with pytest.raises(ToolCallParseError):
        parse_qwen_turn(text, turn_index=0)


def test_plain_final_text_is_preserved() -> None:
    parsed = parse_qwen_turn("<think>done</think>\n13923977", turn_index=0)
    assert parsed.reasoning == "done"
    assert parsed.content == "13923977"
    assert parsed.tool_calls == ()


def test_parses_qwen_bare_followup_tool_call() -> None:
    parsed = parse_qwen_turn(
        '<think>I have the value.</think>\nsubmit_answer\n{"value": "4117925"}',
        turn_index=1,
    )

    assert parsed.content == ""
    assert parsed.reasoning == "I have the value."
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "submit_answer"
    assert parsed.tool_calls[0].arguments == {"value": "4117925"}
