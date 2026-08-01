from __future__ import annotations

from types import SimpleNamespace

import torch
from still.model.attention import CompactCache

from cartridge_memory.kv_ledger import RecurrentKVLedger


class IntTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 31 for character in text]


class FakeStillModel:
    def __init__(self):
        self.cfg = SimpleNamespace(num_latents=64)
        self.device_str = "cpu"
        self.perceiver_device = torch.device("cpu")

    def _cache(self, value):
        cache = CompactCache()
        tensor = torch.full((2, 64, 4), float(value))
        cache.add(tensor, tensor + 1, torch.full((2, 64), float(value)))
        return cache

    def compact_tokens(self, token_ids):
        return self._cache(sum(token_ids) / max(1, len(token_ids)))

    def recompact(self, cache, token_ids):
        return self._cache(cache.compact_k[0].mean().item() + sum(token_ids) / len(token_ids))


def _feedback(index):
    return f"Rejected. Event correction number {index} with enough text to compact."


def test_one_hundred_updates_stay_fixed_finite_and_do_not_retain_text():
    ledger = RecurrentKVLedger(FakeStillModel(), IntTokenizer())
    for index in range(100):
        ledger.update(feedback=_feedback(index), tool_executions=[])
        assert ledger.memory_positions == 64
        assert all(torch.isfinite(tensor).all() for tensor in ledger.state.compact_k)
        assert all(torch.isfinite(tensor).all() for tensor in ledger.state.compact_v)
        assert all(torch.isfinite(tensor).all() for tensor in ledger.state.bias)

    assert len(ledger.event_hashes) == 100
    assert ledger.recurrence_count >= 100
    assert not hasattr(ledger, "events")


def test_checkpoint_resume_matches_uninterrupted_cache_and_counters():
    tokenizer = IntTokenizer()
    uninterrupted = RecurrentKVLedger(FakeStillModel(), tokenizer)
    resumed_source = RecurrentKVLedger(FakeStillModel(), tokenizer)
    for index in range(40):
        uninterrupted.update(feedback=_feedback(index), tool_executions=[])
        if index < 20:
            resumed_source.update(feedback=_feedback(index), tool_executions=[])

    resumed = RecurrentKVLedger(FakeStillModel(), tokenizer)
    resumed.load_state_dict(resumed_source.state_dict())
    for index in range(20, 40):
        resumed.update(feedback=_feedback(index), tool_executions=[])

    assert resumed.source_tokens == uninterrupted.source_tokens
    assert resumed.recurrence_count == uninterrupted.recurrence_count
    assert resumed.event_hashes == uninterrupted.event_hashes
    for expected, actual in zip(
        uninterrupted.state.compact_k + uninterrupted.state.compact_v + uninterrupted.state.bias,
        resumed.state.compact_k + resumed.state.compact_v + resumed.state.bias,
        strict=True,
    ):
        assert torch.equal(actual, expected)
