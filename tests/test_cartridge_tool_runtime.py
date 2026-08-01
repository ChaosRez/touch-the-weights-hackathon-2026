from __future__ import annotations

from alien_api_env.vf import AlienApiTaskset, AlienApiTasksetConfig
from cartridge_memory.models import ParsedToolCall
from cartridge_memory.tool_runtime import ToolRuntime


async def test_runtime_builds_qwen_schemas_and_dispatches_in_process() -> None:
    task = next(iter(AlienApiTaskset(AlienApiTasksetConfig(split="", artifact_verbosity=0)).load()))
    runtime = await ToolRuntime.create(task)

    names = [definition["function"]["name"] for definition in runtime.definitions]
    assert "count_accounts" in names
    assert "search_wiki" in names
    assert "submit_answer" in names
    assert all(definition["type"] == "function" for definition in runtime.definitions)

    execution = await runtime.execute(
        ParsedToolCall(id="call-0-0", name="list_spaces", arguments={}),
        turn_index=0,
    )
    assert execution.result["ok"] is True
    assert execution.outcome == "ok"


async def test_runtime_surfaces_unknown_tool_as_a_result() -> None:
    task = next(iter(AlienApiTaskset(AlienApiTasksetConfig(split="", artifact_verbosity=0)).load()))
    runtime = await ToolRuntime.create(task)
    execution = await runtime.execute(
        ParsedToolCall(id="call-0-0", name="does_not_exist", arguments={}),
        turn_index=0,
    )
    assert execution.result["ok"] is False
    assert execution.result["error"]["code"] == "unknown_tool"
