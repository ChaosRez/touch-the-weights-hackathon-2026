"""In-process Qwen agent with local alien-api tool dispatch."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import ToolCall, ToolMessage

from alien_api_env.vf.tools import is_submit_answer
from cartridge_memory.models import (
    AssistantTurn,
    CompactKVAttachment,
    GeneratedTurn,
    MemoryAttachment,
    ParsedToolCall,
    RawKVAttachment,
    RolloutRecord,
    TextAttachment,
    ToolExecution,
)
from cartridge_memory.tool_runtime import ToolRuntime

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_BARE_TOOL_CALL_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\n\s*(\{.*\})\s*$",
    re.DOTALL,
)


class ToolCallParseError(ValueError):
    """Qwen emitted tool-call markup that cannot be dispatched."""


class UnsupportedAttachmentError(NotImplementedError):
    """The selected generation backend cannot consume this attachment type yet."""


@dataclass(frozen=True, slots=True)
class ParsedAssistantTurn:
    content: str
    reasoning: str
    tool_calls: tuple[ParsedToolCall, ...]


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        return stripped[7:-3].strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return stripped[3:-3].strip()
    return stripped


def parse_qwen_turn(text: str, turn_index: int) -> ParsedAssistantTurn:
    """Parse all native Qwen ``<tool_call>`` blocks from one assistant action."""

    blocks = _TOOL_CALL_RE.findall(text)
    if "<tool_call>" in text and not blocks:
        raise ToolCallParseError("unclosed <tool_call> block")

    reasoning_parts = _THINK_RE.findall(text)
    reasoning = "\n".join(part.strip() for part in reasoning_parts if part.strip())
    visible = _THINK_RE.sub("", text).strip()

    call_payloads: list[str] = list(blocks)
    if not call_payloads:
        # Qwen3 sometimes emits a subsequent call as ``tool_name\n{...}``
        # after receiving a tool result, even though its first call used tags.
        bare = _BARE_TOOL_CALL_RE.fullmatch(visible)
        if bare:
            try:
                bare_arguments = json.loads(bare.group(2))
            except json.JSONDecodeError as error:
                raise ToolCallParseError(f"invalid bare tool-call JSON: {error.msg}") from error
            call_payloads.append(
                json.dumps({"name": bare.group(1), "arguments": bare_arguments})
            )

    calls: list[ParsedToolCall] = []
    for call_index, block in enumerate(call_payloads):
        try:
            payload = json.loads(_strip_json_fence(block))
        except json.JSONDecodeError as error:
            raise ToolCallParseError(f"invalid tool-call JSON: {error.msg}") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            raise ToolCallParseError("tool call must be an object with a string name")
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as error:
                raise ToolCallParseError("tool-call arguments string is not JSON") from error
        if not isinstance(arguments, dict):
            raise ToolCallParseError("tool-call arguments must be an object")
        calls.append(
            ParsedToolCall(
                id=f"call-{turn_index}-{call_index}",
                name=payload["name"],
                arguments=arguments,
            )
        )

    content = _TOOL_CALL_RE.sub("", text)
    content = _THINK_RE.sub("", content).strip()
    if calls and not blocks:
        content = ""
    return ParsedAssistantTurn(content=content, reasoning=reasoning, tool_calls=tuple(calls))


class QwenGenerationBackend(Protocol):
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        attachment: MemoryAttachment | None,
        seed: int,
    ) -> GeneratedTurn: ...


@dataclass(frozen=True, slots=True)
class QwenGenerationConfig:
    model_name: str = "Qwen/Qwen3-8B"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    max_new_tokens: int = 2048
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    enable_thinking: bool = True


def render_qwen_chat(
    tokenizer: Any,
    config: QwenGenerationConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> Any:
    """Render the one shared live chat request for cold, text, and Still backends."""

    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=config.enable_thinking,
    )


def qwen_generation_kwargs(tokenizer: Any, config: QwenGenerationConfig) -> dict[str, Any]:
    """Return identical generation controls for every Qwen Phase 4 arm."""

    generation: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if config.do_sample:
        generation.update(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
    return generation


def _input_ids(encoded: Any, device: str) -> tuple[dict[str, Any], Any]:
    if hasattr(encoded, "input_ids"):
        model_inputs = {
            name: tensor.to(device)
            for name, tensor in encoded.items()
            if hasattr(tensor, "to")
        }
        return model_inputs, model_inputs["input_ids"]
    input_ids = encoded.to(device)
    return {"input_ids": input_ids}, input_ids


def _validate_context(model_config: Any, input_tokens: int, memory_tokens: int, maximum: int) -> None:
    context_limit = getattr(model_config, "max_position_embeddings", None)
    if context_limit and input_tokens + memory_tokens + maximum > context_limit:
        raise ValueError(
            f"live prompt ({input_tokens}) + memory ({memory_tokens}) + generation ({maximum}) "
            f"exceeds context window {context_limit}"
        )


class HuggingFaceQwenBackend:
    """Single-device Hugging Face generation, loaded lazily to keep local tests light."""

    def __init__(self, config: QwenGenerationConfig | None = None) -> None:
        self.config = config or QwenGenerationConfig()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - exercised on the GPU image
            raise RuntimeError("HuggingFaceQwenBackend requires torch and transformers") from error

        dtype = getattr(torch, self.config.dtype)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.config.device)
        self.model.eval()

    @staticmethod
    def _attach_text(
        messages: list[dict[str, Any]], attachment: MemoryAttachment | None
    ) -> list[dict[str, Any]]:
        if attachment is None:
            return [dict(message) for message in messages]
        if isinstance(attachment, (RawKVAttachment, CompactKVAttachment)):
            raise UnsupportedAttachmentError(
                f"{type(attachment).__name__} generation is implemented in a later phase"
            )
        if not isinstance(attachment, TextAttachment):
            raise UnsupportedAttachmentError(type(attachment).__name__)
        attached = [dict(message) for message in messages]
        for message in attached:
            if message.get("role") == "user":
                message["content"] = (
                    "Retrieved memory for this task:\n"
                    f"{attachment.text.strip()}\n\n"
                    "Current task:\n"
                    f"{message.get('content', '')}"
                )
                break
        return attached

    def _generate_sync(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        attachment: MemoryAttachment | None,
        seed: int,
    ) -> GeneratedTurn:
        torch = self._torch
        rendered_messages = self._attach_text(messages, attachment)
        encoded = render_qwen_chat(self.tokenizer, self.config, rendered_messages, tools)
        model_inputs, input_ids = _input_ids(encoded, self.config.device)
        _validate_context(
            self.model.config,
            int(input_ids.shape[-1]),
            0,
            self.config.max_new_tokens,
        )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generation = qwen_generation_kwargs(self.tokenizer, self.config)
        with torch.inference_mode():
            output = self.model.generate(**model_inputs, **generation)
        new_ids = output[0, input_ids.shape[-1] :].tolist()
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GeneratedTurn(
            text=text,
            token_ids=tuple(int(token) for token in new_ids),
            input_tokens=int(input_ids.shape[-1]),
            output_tokens=len(new_ids),
        )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        attachment: MemoryAttachment | None,
        seed: int,
    ) -> GeneratedTurn:
        return await asyncio.to_thread(self._generate_sync, messages, tools, attachment, seed)


class StillQwenBackend:
    """In-process Qwen generation against an optional trained fixed-size Still cache."""

    def __init__(
        self,
        config: QwenGenerationConfig,
        checkpoint_path: str,
        still_config: Any | None = None,
    ) -> None:
        try:
            import torch
            from still.config import STILLConfig
            from still.model.wrapper import STILLModel
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - exercised on the GPU image
            raise RuntimeError("StillQwenBackend requires torch, transformers, and still") from error

        self.config = config
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        cfg = still_config or STILLConfig(
            model_name=config.model_name,
            num_latents=64,
            latent_dim=256,
            num_blocks=2,
            device=config.device,
        )
        if cfg.model_name != config.model_name:
            raise ValueError("Still and Qwen model names must match")
        self.still_model = STILLModel(
            config.model_name,
            cfg=cfg,
            device=config.device,
            dtype=getattr(torch, config.dtype),
            attn_implementation="sdpa",
        )
        payload = torch.load(
            checkpoint_path,
            map_location=self.still_model.perceiver_device,
            weights_only=True,
        )
        if not isinstance(payload, dict) or "perceiver" not in payload:
            raise ValueError("Phase 4 requires a nested Phase 2 checkpoint")
        recorded_model = payload.get("model_name") or payload.get("config", {}).get("model_name")
        if recorded_model != config.model_name:
            raise ValueError(
                f"checkpoint model {recorded_model!r} does not match {config.model_name!r}"
            )
        recorded = payload.get("config", {})
        expected = {
            "num_latents": cfg.num_latents,
            "latent_dim": cfg.latent_dim,
            "num_blocks": cfg.num_blocks,
        }
        mismatches = {
            key: (recorded.get(key), value)
            for key, value in expected.items()
            if recorded.get(key) != value
        }
        if mismatches:
            raise ValueError(f"checkpoint Still configuration mismatch: {mismatches}")
        self.still_model.perceiver.load_state_dict(payload["perceiver"])
        self.still_model.perceiver.eval()
        self.checkpoint_metadata = {
            "path": checkpoint_path,
            "format": payload.get("format"),
            "stage": payload.get("stage"),
            "completed_steps": payload.get("completed_steps"),
        }

    def _generate_sync(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        attachment: MemoryAttachment | None,
        seed: int,
    ) -> GeneratedTurn:
        if attachment is not None and not isinstance(attachment, CompactKVAttachment):
            raise UnsupportedAttachmentError(type(attachment).__name__)
        cache = attachment.cache if attachment is not None else None
        memory_positions = attachment.latent_count if attachment is not None else 0
        if attachment is not None:
            if memory_positions != self.still_model.cfg.num_latents:
                raise ValueError("compact attachment latent count does not match Still config")
            if int(cache.num_latents) != memory_positions:
                raise ValueError("compact attachment metadata does not match its cache")

        encoded = render_qwen_chat(self.tokenizer, self.config, messages, tools)
        model_inputs, input_ids = _input_ids(encoded, self.config.device)
        _validate_context(
            self.still_model.base.config,
            int(input_ids.shape[-1]),
            memory_positions,
            self.config.max_new_tokens,
        )
        torch = self._torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generation = qwen_generation_kwargs(self.tokenizer, self.config)
        if "attention_mask" in model_inputs:
            generation["attention_mask"] = model_inputs["attention_mask"]
        with torch.inference_mode():
            new_ids = self.still_model.decode_generate(
                input_ids,
                cache,
                **generation,
            )
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GeneratedTurn(
            text=text,
            token_ids=tuple(int(token) for token in new_ids),
            input_tokens=int(input_ids.shape[-1]),
            output_tokens=len(new_ids),
        )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        attachment: MemoryAttachment | None,
        seed: int,
    ) -> GeneratedTurn:
        return await asyncio.to_thread(self._generate_sync, messages, tools, attachment, seed)


@dataclass(frozen=True, slots=True)
class QwenAgentConfig:
    max_tool_turns: int = 12
    malformed_call_retries: int = 2


class QwenToolAgent:
    """One common in-process tool agent used by every evaluation arm."""

    def __init__(
        self,
        backend: QwenGenerationBackend,
        config: QwenAgentConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or QwenAgentConfig()

    async def run(
        self,
        task: Any,
        attachment: MemoryAttachment | None,
        seed: int,
    ) -> RolloutRecord:
        runtime = await ToolRuntime.create(task)
        trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": task.data.system_prompt or ""},
            {"role": "user", "content": task.data.prompt},
        ]
        parent: int | None = None
        assistant_turns: list[AssistantTurn] = []
        tool_executions: list[ToolExecution] = []
        submitted = False
        final_text = ""
        stop_reason = "turn_limit"
        parse_failures = 0

        for turn_index in range(self.config.max_tool_turns + 1):
            generated = await self.backend.generate(
                messages,
                runtime.definitions,
                attachment,
                seed + turn_index,
            )
            parse_error: str | None = None
            try:
                parsed = parse_qwen_turn(generated.text, turn_index)
            except ToolCallParseError as error:
                parse_error = str(error)
                parsed = ParsedAssistantTurn(content="", reasoning="", tool_calls=())

            assistant_turns.append(
                AssistantTurn(
                    turn_index=turn_index,
                    raw_text=generated.text,
                    content=parsed.content,
                    reasoning=parsed.reasoning,
                    token_ids=generated.token_ids,
                    tool_calls=parsed.tool_calls,
                    input_tokens=generated.input_tokens,
                    output_tokens=generated.output_tokens,
                    parse_error=parse_error,
                )
            )
            vf_calls = [
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=json.dumps(call.arguments, separators=(",", ":")),
                )
                for call in parsed.tool_calls
            ]
            trace.nodes.append(
                MessageNode(
                    parent=parent,
                    message=vf.AssistantMessage(
                        content=parsed.content or None,
                        reasoning_content=parsed.reasoning or None,
                        tool_calls=vf_calls or None,
                    ),
                    sampled=True,
                    token_ids=list(generated.token_ids),
                )
            )
            parent = len(trace.nodes) - 1
            messages.append({"role": "assistant", "content": generated.text})

            if parse_error is not None:
                parse_failures += 1
                if parse_failures > self.config.malformed_call_retries:
                    stop_reason = "malformed_tool_call"
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your tool call could not be parsed: {parse_error}. "
                            "Emit valid JSON inside <tool_call>...</tool_call>."
                        ),
                    }
                )
                continue

            if not parsed.tool_calls:
                final_text = parsed.content
                stop_reason = "final_text"
                break

            for call in parsed.tool_calls:
                execution = await runtime.execute(call, turn_index)
                tool_executions.append(execution)
                trace.nodes.append(
                    MessageNode(
                        parent=parent,
                        message=ToolMessage(
                            tool_call_id=call.id,
                            content=execution.serialized_result,
                            name=call.name,
                        ),
                        sampled=False,
                    )
                )
                parent = len(trace.nodes) - 1
                messages.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "content": execution.serialized_result,
                    }
                )
                if is_submit_answer(call.name):
                    submitted = True

            if submitted:
                stop_reason = "submitted"
                break

        await task.score(trace)
        await task.finalize(trace, None)
        return RolloutRecord(
            episode_id=task.data.name,
            prompt=task.data.prompt,
            assistant_turns=assistant_turns,
            tool_executions=tool_executions,
            trace=trace,
            final_text=final_text,
            submitted=submitted,
            stop_reason=stop_reason,
            answer=task._answer(task.data, trace),
            reward=float(trace.reward),
            metrics={key: float(value) for key, value in trace.metrics.items()},
            feedback=str(trace.info.get("feedback", "")),
        )
