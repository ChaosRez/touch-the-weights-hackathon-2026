"""Shared attachment and rollout records for every cartridge arm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    import verifiers.v1 as vf
    from still.model.attention import CompactCache


@dataclass(frozen=True, slots=True)
class TextAttachment:
    """Retrieved memory represented as ordinary prompt text."""

    text: str


@dataclass(frozen=True, slots=True)
class RawKVAttachment:
    """Retrieved memory represented as an ordinary, lossless model KV cache."""

    cache: object
    source_tokens: int


@dataclass(frozen=True, slots=True)
class CompactKVAttachment:
    """Retrieved memory represented as a learned fixed-size Still cache."""

    cache: CompactCache
    source_tokens: int
    latent_count: int


MemoryAttachment: TypeAlias = TextAttachment | RawKVAttachment | CompactKVAttachment


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    """One normalized Qwen tool call."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GeneratedTurn:
    """Raw output returned by a generation backend."""

    text: str
    token_ids: tuple[int, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """A complete assistant action retained for later compactor training."""

    turn_index: int
    raw_text: str
    content: str
    reasoning: str
    token_ids: tuple[int, ...]
    tool_calls: tuple[ParsedToolCall, ...]
    input_tokens: int
    output_tokens: int
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """One in-process dispatch and the exact result returned to Qwen."""

    turn_index: int
    call: ParsedToolCall
    result: dict[str, Any]
    serialized_result: str

    @property
    def outcome(self) -> str:
        if self.result.get("ok"):
            return "ok"
        error = self.result.get("error")
        code = error.get("code", "?") if isinstance(error, dict) else "?"
        return f"ERROR:{code}"


@dataclass(slots=True)
class RolloutRecord:
    """One scored episode, including every assistant action and tool result."""

    episode_id: str
    prompt: str
    assistant_turns: list[AssistantTurn]
    tool_executions: list[ToolExecution]
    trace: vf.Trace
    final_text: str
    submitted: bool
    stop_reason: str
    answer: str
    reward: float
    metrics: dict[str, float] = field(default_factory=dict)
    feedback: str = ""

    @property
    def usage_in(self) -> int:
        return sum(turn.input_tokens for turn in self.assistant_turns)

    @property
    def usage_out(self) -> int:
        return sum(turn.output_tokens for turn in self.assistant_turns)

    @property
    def answered(self) -> bool:
        """Whether the verifier received a non-empty answer through either channel."""

        return bool(self.answer.strip())

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe evaluation record; the live vf.Trace stays in memory."""

        return {
            "episodeId": self.episode_id,
            "prompt": self.prompt,
            "assistant_turns": [
                {
                    "turn_index": turn.turn_index,
                    "raw_text": turn.raw_text,
                    "content": turn.content,
                    "reasoning": turn.reasoning,
                    "token_ids": list(turn.token_ids),
                    "tool_calls": [
                        {
                            "id": call.id,
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                        for call in turn.tool_calls
                    ],
                    "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                    "parse_error": turn.parse_error,
                }
                for turn in self.assistant_turns
            ],
            "tool_executions": [
                {
                    "turn_index": execution.turn_index,
                    "call_id": execution.call.id,
                    "name": execution.call.name,
                    "arguments": execution.call.arguments,
                    "result": execution.result,
                    "outcome": execution.outcome,
                }
                for execution in self.tool_executions
            ],
            "final_text": self.final_text,
            "answered": self.answered,
            "submitted": self.submitted,
            "stop_reason": self.stop_reason,
            "answer": self.answer,
            "reward": self.reward,
            "metrics": self.metrics,
            "feedback": self.feedback,
            "usage_in": self.usage_in,
            "usage_out": self.usage_out,
        }
