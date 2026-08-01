"""Qwen tool schemas and in-process alien-api dispatch."""

from __future__ import annotations

import inspect
import json
import types
import typing
from dataclasses import dataclass
from typing import Any, get_args, get_origin, get_type_hints

import verifiers.v1 as vf
from verifiers.v1.decorators import discover_decorated

from alien_api_env.vf.tools import AnswerToolset, CrmToolset, WikiToolset
from cartridge_memory.models import ParsedToolCall, ToolExecution

_TYPE_MAP = {int: "integer", str: "string", float: "number", bool: "boolean"}


def _json_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin in (types.UnionType, typing.Union):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        annotation = args[0] if args else str
    return _TYPE_MAP.get(annotation, "string")


def build_tool_definitions(toolsets: tuple[object, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build Qwen chat-template schemas and a name-to-bound-method dispatch table."""

    definitions: list[dict[str, Any]] = []
    dispatch: dict[str, Any] = {}
    for toolset in toolsets:
        for function in discover_decorated(toolset, "tool"):
            name = getattr(function, "tool_name", None) or function.__name__
            signature = inspect.signature(function)
            hints = get_type_hints(function)
            properties: dict[str, Any] = {}
            required: list[str] = []
            for parameter_name, parameter in signature.parameters.items():
                if parameter_name == "self":
                    continue
                annotation = hints.get(parameter_name, parameter.annotation)
                properties[parameter_name] = {"type": _json_type(annotation)}
                if parameter.default is inspect.Parameter.empty:
                    required.append(parameter_name)
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": inspect.cleandoc(function.__doc__ or name),
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False,
                        },
                    },
                }
            )
            dispatch[name] = getattr(toolset, function.__name__)
    return definitions, dispatch


@dataclass(slots=True)
class ToolRuntime:
    """Per-episode tool instances, schemas, and safe in-process dispatch."""

    toolsets: tuple[object, ...]
    definitions: list[dict[str, Any]]
    dispatch: dict[str, Any]

    @classmethod
    async def create(cls, task: Any) -> ToolRuntime:
        crm = CrmToolset(vf.ToolsetConfig())
        wiki = WikiToolset(vf.ToolsetConfig())
        answer = AnswerToolset(vf.ToolsetConfig())
        await crm.setup_task(task.data)
        await wiki.setup_task(task.data)
        toolsets = (crm, wiki, answer)
        definitions, dispatch = build_tool_definitions(toolsets)
        return cls(toolsets=toolsets, definitions=definitions, dispatch=dispatch)

    async def execute(self, call: ParsedToolCall, turn_index: int) -> ToolExecution:
        function = self.dispatch.get(call.name)
        if function is None:
            result: dict[str, Any] = {
                "ok": False,
                "error": {
                    "code": "unknown_tool",
                    "message": f"unknown tool {call.name!r}",
                },
            }
        else:
            try:
                value = await function(**call.arguments)
                result = value if isinstance(value, dict) else {"ok": True, "data": value}
            except Exception as error:
                result = {
                    "ok": False,
                    "error": {
                        "code": "tool_error",
                        "message": str(error),
                    },
                }
        serialized = json.dumps(result, default=str, separators=(",", ":"))
        return ToolExecution(
            turn_index=turn_index,
            call=call,
            result=result,
            serialized_result=serialized,
        )
