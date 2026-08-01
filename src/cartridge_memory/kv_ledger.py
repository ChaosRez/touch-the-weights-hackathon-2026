"""Fixed-budget recurrent KV ledger for legal Phase 4 memory events."""

from __future__ import annotations

import time
from hashlib import sha256
from typing import Any

from cartridge_memory.legal_memory import legal_events
from cartridge_memory.models import CompactKVAttachment


class RecurrentKVLedger:
    """Incrementally compact legal events without retaining their decoded history."""

    def __init__(self, model: Any, tokenizer: Any, *, chunk_tokens: int = 64) -> None:
        if chunk_tokens < 1:
            raise ValueError("chunk_tokens must be positive")
        if int(model.cfg.num_latents) != chunk_tokens:
            raise ValueError("the Phase 4 chunk and fixed KV budgets must match")
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens
        self.state: Any | None = None
        self.event_hashes: set[str] = set()
        self.source_tokens = 0
        self.recurrence_count = 0
        self.last_compaction_seconds = 0.0

    @property
    def memory_positions(self) -> int:
        return 0 if self.state is None else int(self.state.num_latents)

    def attachment(self) -> CompactKVAttachment | None:
        if self.state is None:
            return None
        return CompactKVAttachment(
            cache=self.state,
            source_tokens=self.source_tokens,
            latent_count=self.memory_positions,
        )

    def _synchronize(self) -> None:
        try:
            import torch

            if str(self.model.device_str).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
        except (AttributeError, ImportError):
            return

    def _validate_state(self) -> None:
        if self.state is None:
            return
        import torch

        if self.memory_positions != self.chunk_tokens:
            raise RuntimeError(
                f"recurrent cache has {self.memory_positions} positions, expected {self.chunk_tokens}"
            )
        tensors = [*self.state.compact_k, *self.state.compact_v, *self.state.bias]
        if not tensors or not all(torch.isfinite(tensor).all().item() for tensor in tensors):
            raise FloatingPointError("recurrent cache contains a non-finite tensor")

    def update(self, *, feedback: str, tool_executions: list[Any]) -> dict[str, Any]:
        events = legal_events(feedback=feedback, tool_executions=tool_executions)
        added_events = 0
        added_tokens = 0
        compactions = 0
        self._synchronize()
        started = time.perf_counter()
        for event in events:
            digest = sha256(event.encode()).hexdigest()
            if digest in self.event_hashes:
                continue
            token_ids = [
                int(token) for token in self.tokenizer.encode(event, add_special_tokens=False)
            ]
            self.event_hashes.add(digest)
            self.source_tokens += len(token_ids)
            added_events += 1
            added_tokens += len(token_ids)
            for offset in range(0, len(token_ids), self.chunk_tokens):
                chunk = token_ids[offset : offset + self.chunk_tokens]
                if not chunk:
                    continue
                self.state = (
                    self.model.compact_tokens(chunk)
                    if self.state is None
                    else self.model.recompact(self.state, chunk)
                )
                self.recurrence_count += 1
                compactions += 1
                self._validate_state()
        self._synchronize()
        self.last_compaction_seconds = time.perf_counter() - started
        return {
            "added_events": added_events,
            "added_source_tokens": added_tokens,
            "compactions": compactions,
            "compaction_seconds": self.last_compaction_seconds,
            "memory_positions": self.memory_positions,
            "source_tokens": self.source_tokens,
            "recurrence_count": self.recurrence_count,
        }

    def state_dict(self) -> dict[str, Any]:
        cache = None
        if self.state is not None:
            cache = {
                "compact_k": [tensor.detach() for tensor in self.state.compact_k],
                "compact_v": [tensor.detach() for tensor in self.state.compact_v],
                "bias": [tensor.detach() for tensor in self.state.bias],
            }
        return {
            "format": "recurrent-kv-ledger-v1",
            "chunk_tokens": self.chunk_tokens,
            "cache": cache,
            "event_hashes": sorted(self.event_hashes),
            "source_tokens": self.source_tokens,
            "recurrence_count": self.recurrence_count,
            "last_compaction_seconds": self.last_compaction_seconds,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("format") != "recurrent-kv-ledger-v1":
            raise ValueError("unsupported recurrent KV ledger checkpoint")
        if int(payload["chunk_tokens"]) != self.chunk_tokens:
            raise ValueError("recurrent KV ledger token budget mismatch")
        cache_payload = payload.get("cache")
        if cache_payload is None:
            self.state = None
        else:
            from still.model.attention import CompactCache

            cache = CompactCache()
            device = self.model.perceiver_device
            for key, value, bias in zip(
                cache_payload["compact_k"],
                cache_payload["compact_v"],
                cache_payload["bias"],
                strict=True,
            ):
                cache.add(key.to(device), value.to(device), bias.to(device))
            self.state = cache
        self.event_hashes = set(payload["event_hashes"])
        self.source_tokens = int(payload["source_tokens"])
        self.recurrence_count = int(payload["recurrence_count"])
        self.last_compaction_seconds = float(payload["last_compaction_seconds"])
        self._validate_state()
