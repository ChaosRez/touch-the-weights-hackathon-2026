from __future__ import annotations

from cartridge_memory.models import TextAttachment
from cartridge_memory.text_ledger import GlobalTextLedger, StreamingTextLedger


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token) for token in token_ids)


def _record(feedback="Rejected. Use computed annual revenue."):
    return {
        "feedback": feedback,
        "tool_executions": [
            {
                "name": "get_account",
                "arguments": {"account_id": "acct-1"},
                "outcome": "ok",
            },
            {
                "name": "missing_tool",
                "arguments": {},
                "outcome": "ERROR:unknown_tool",
            },
        ],
    }


def test_ledger_deduplicates_verbatim_feedback_and_tool_outcomes():
    ledger = GlobalTextLedger(CharacterTokenizer(), max_tokens=1000)

    ledger.update(_record())
    ledger.update(_record())

    assert len(ledger.entries) == 3
    assert ledger.entries[-1] == "Rejected. Use computed annual revenue."
    assert "get_account({\"account_id\":\"acct-1\"}) -> ok" in ledger.entries
    assert "missing_tool({}) -> ERROR:unknown_tool" in ledger.entries


def test_ledger_omits_acceptance_and_caps_recent_token_positions():
    ledger = GlobalTextLedger(CharacterTokenizer(), max_tokens=32)
    ledger.update(_record(feedback="Accepted."))
    ledger.update(_record(feedback="Rejected. Prefer the newest value."))

    rendered = ledger.render()

    assert len(rendered) == 32
    assert rendered.endswith("Prefer the newest value.")
    assert ledger.rendered_tokens == 32
    assert isinstance(ledger.attachment(), TextAttachment)


def test_empty_ledger_has_no_attachment():
    ledger = GlobalTextLedger(CharacterTokenizer(), max_tokens=64)

    assert ledger.render() == ""
    assert ledger.rendered_tokens == 0
    assert ledger.attachment() is None


def test_streaming_ledger_retains_only_recent_tokens_and_resumes_exactly():
    ledger = StreamingTextLedger(CharacterTokenizer(), max_tokens=16)
    tool = {
        "name": "get_account",
        "arguments": {"id": "a"},
        "result": {"ok": True, "data": {"ignored": True}},
    }
    ledger.update(feedback="Rejected. First correction.", tool_executions=[tool])
    ledger.update(feedback="Rejected. Second correction.", tool_executions=[tool])

    assert ledger.rendered_tokens == 16
    assert not hasattr(ledger, "entries")
    assert ledger.source_tokens > ledger.rendered_tokens
    state = ledger.state_dict()
    resumed = StreamingTextLedger(CharacterTokenizer(), max_tokens=16)
    resumed.load_state_dict(state)
    assert resumed.state_dict() == state
    assert resumed.attachment() == ledger.attachment()
