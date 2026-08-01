"""Minimal global text ledger used by Workstream A2."""

from __future__ import annotations

import json
from typing import Any

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
