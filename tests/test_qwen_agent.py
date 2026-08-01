from __future__ import annotations

import json

from alien_api_env.vf import AlienApiTaskset, AlienApiTasksetConfig
from cartridge_memory.models import GeneratedTurn
from cartridge_memory.qwen_agent import QwenToolAgent


class ScriptedBackend:
    def __init__(self, turns: list[str]) -> None:
        self.turns = iter(turns)
        self.requests: list[tuple[list[dict], list[dict], object, int]] = []

    async def generate(self, messages, tools, attachment, seed) -> GeneratedTurn:
        self.requests.append((messages, tools, attachment, seed))
        text = next(self.turns)
        return GeneratedTurn(text=text, token_ids=(1, 2), input_tokens=10, output_tokens=2)


async def test_agent_calls_tool_submits_answer_and_scores_real_trace() -> None:
    task = next(iter(AlienApiTaskset(AlienApiTasksetConfig(split="", artifact_verbosity=0)).load()))
    backend = ScriptedBackend(
        [
            '<tool_call>{"name":"list_spaces","arguments":{}}</tool_call>',
            '<tool_call>{"name":"submit_answer","arguments":{"value":'
            + json.dumps(task.data.accepted)
            + "}}</tool_call>",
        ]
    )
    agent = QwenToolAgent(backend)

    record = await agent.run(task, attachment=None, seed=7)

    assert record.submitted is True
    assert record.stop_reason == "submitted"
    assert record.answer == task.data.accepted
    assert record.answered is True
    assert record.to_dict()["answered"] is True
    assert record.reward == 1.0
    assert record.feedback == "Accepted."
    assert len(record.assistant_turns) == 2
    assert [execution.call.name for execution in record.tool_executions] == [
        "list_spaces",
        "submit_answer",
    ]
    assert len(record.trace.tool_messages) == 2
    assert record.metrics["tool_calls"] == 1.0
    assert backend.requests[1][0][-1]["role"] == "tool"


async def test_agent_uses_plain_final_text_fallback() -> None:
    task = next(iter(AlienApiTaskset(AlienApiTasksetConfig(split="", artifact_verbosity=0)).load()))
    backend = ScriptedBackend([task.data.accepted])
    record = await QwenToolAgent(backend).run(task, attachment=None, seed=1)

    assert record.submitted is False
    assert record.stop_reason == "final_text"
    assert record.final_text == task.data.accepted
    assert record.answer == task.data.accepted
    assert record.answered is True
    assert record.reward == 1.0
