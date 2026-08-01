"""Minimal global text ledger used by Workstream A2."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from cartridge_memory.legal_memory import legal_events
from cartridge_memory.models import TextAttachment


class GlobalTextLedger:
    """Deduplicated reviewer corrections and mechanical tool outcomes."""

    def __init__(self, tokenizer: Any, max_tokens: int) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.entries: list[str] = []
        self._seen: set[str] = set()

    def _append(self, entry: str) -> None:
        normalized = entry.strip()
        if normalized and normalized not in self._seen:
            self._seen.add(normalized)
            self.entries.append(normalized)

    def update(self, record: dict[str, Any]) -> None:
        """Append one episode's tool outcomes, then its verbatim correction."""

        for execution in record.get("tool_executions", []):
            arguments = json.dumps(
                execution.get("arguments", {}),
                sort_keys=True,
                separators=(",", ":"),
            )
            self._append(
                f"{execution.get('name', 'tool')}({arguments}) -> "
                f"{execution.get('outcome', 'unknown')}"
            )
        feedback = str(record.get("feedback", "")).strip()
        if feedback and feedback != "Accepted.":
            self._append(feedback)

    def render(self) -> str:
        """Render only the most recent tokenizer positions."""

        text = "\n".join(self.entries)
        if not text:
            return ""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.decode(token_ids[-self.max_tokens :], skip_special_tokens=True)

    @property
    def rendered_tokens(self) -> int:
        rendered = self.render()
        if not rendered:
            return 0
        return len(self.tokenizer.encode(rendered, add_special_tokens=False))

    def attachment(self) -> TextAttachment | None:
        rendered = self.render()
        return TextAttachment(rendered) if rendered else None


class StreamingTextLedger:
    """A deduplicated legal-event stream retaining only its last tokenizer positions."""

    def __init__(self, tokenizer: Any, max_tokens: int = 64) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.token_ids: list[int] = []
        self.event_hashes: set[str] = set()
        self.source_tokens = 0
        self.updates = 0

    def update(self, *, feedback: str, tool_executions: list[Any]) -> dict[str, int]:
        added_events = 0
        added_tokens = 0
        for event in legal_events(feedback=feedback, tool_executions=tool_executions):
            digest = sha256(event.encode()).hexdigest()
            if digest in self.event_hashes:
                continue
            event_tokens = [
                int(token) for token in self.tokenizer.encode(event, add_special_tokens=False)
            ]
            self.event_hashes.add(digest)
            self.token_ids.extend(event_tokens)
            self.token_ids = self.token_ids[-self.max_tokens :]
            self.source_tokens += len(event_tokens)
            self.updates += 1
            added_events += 1
            added_tokens += len(event_tokens)
        return {"added_events": added_events, "added_source_tokens": added_tokens}

    def render(self) -> str:
        if not self.token_ids:
            return ""
        return self.tokenizer.decode(self.token_ids, skip_special_tokens=True)

    @property
    def rendered_tokens(self) -> int:
        return len(self.token_ids)

    def attachment(self) -> TextAttachment | None:
        rendered = self.render()
        return TextAttachment(rendered) if rendered else None

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "streaming-text-ledger-v1",
            "max_tokens": self.max_tokens,
            "token_ids": list(self.token_ids),
            "event_hashes": sorted(self.event_hashes),
            "source_tokens": self.source_tokens,
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") != "streaming-text-ledger-v1":
            raise ValueError("unsupported streaming text ledger checkpoint")
        if int(state["max_tokens"]) != self.max_tokens:
            raise ValueError("streaming text ledger token budget mismatch")
        token_ids = [int(token) for token in state["token_ids"]]
        if len(token_ids) > self.max_tokens:
            raise ValueError("streaming text ledger checkpoint exceeds its token budget")
        self.token_ids = token_ids
        self.event_hashes = set(state["event_hashes"])
        self.source_tokens = int(state["source_tokens"])
        self.updates = int(state["updates"])
