"""Shared JSON envelope shapes + deterministic token sizing for the tool surface.

Every CRM/ERP tool returns one of two envelope shapes:

- success : ``{"ok": True,  "data": <records/value>, "meta": {...}}``
- error   : ``{"ok": False, "error": {"code": ..., "message": ...}, "meta": {...}}``

The behavioral quirks ride these: a deprecated endpoint returns a ``not_found``/404
error, an unknown id returns ``not_found``. The stale-report and truncation quirks return
a *success* envelope with a silently-wrong ``data`` (never a self-announcing ``stale`` or
``truncated`` flag — the divergence must not advertise itself; see the plan's manual
verification).

Sizing: list returns are padded toward a per-call share of the instance's
``artifact_verbosity`` budget so the model faces realistic context pressure (research
§F.2). ``count_tokens`` prefers ``tiktoken cl100k_base`` (the tokenizer Phase 2 used for
SOP pages) and falls back to a deterministic char estimate when tiktoken is absent —
``tiktoken`` is a **dev-only** extra (payload-sizing tests), never a runtime dependency,
so the tools must keep working when it is not installed. Padding and measurement both go
through ``count_tokens``, so within any one environment they stay consistent and the pad
is deterministic; the size tests run with tiktoken installed and measure exactly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

TOKENIZER = "cl100k_base"

# Default per-instance artifact-token budget (design default; provisional until Plan
# 01-03 calibration lands, per the plan's Open Risks). Threaded onto AlienApiData and
# read by the toolset in setup_task.
ARTIFACT_VERBOSITY = 22_000

# A representative instance makes a handful of list-returning calls; each list return is
# padded to this share so a typical sequence sums near ARTIFACT_VERBOSITY.
LIST_CALLS_PER_INSTANCE = 4


@lru_cache(maxsize=1)
def _encoding():
    """The cl100k_base encoder, or None when tiktoken (a dev extra) is not installed."""
    try:
        import tiktoken
    except Exception:  # pragma: no cover - runtime-without-tiktoken path
        return None
    return tiktoken.get_encoding(TOKENIZER)


def count_tokens(text: str) -> int:
    """Token count under cl100k_base, with a deterministic char-based fallback.

    ~4 chars/token is the cl100k rule of thumb; good enough for padding and the weight-0
    ``artifact_tokens`` metric when tiktoken is unavailable at runtime. The size tests run
    with tiktoken installed, so they measure exactly.
    """
    enc = _encoding()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def ok(data: Any, **meta: Any) -> dict[str, Any]:
    """A success envelope."""
    return {"ok": True, "data": data, "meta": dict(meta)}


def err(code: str, message: str, **meta: Any) -> dict[str, Any]:
    """An error envelope (``not_found``, ``404``, ``unknown_metric``, ...)."""
    env: dict[str, Any] = {"ok": False, "error": {"code": code, "message": message}}
    if meta:
        env["meta"] = dict(meta)
    return env


# On-topic operational filler, cycled + numbered so an arbitrary amount of *distinct*
# deterministic padding can ride in ``meta.notes`` without repeating a block verbatim
# (mirrors the Phase 2 SOP ``_fit`` appendix, but for tool envelopes). No secrets, no
# network, no record data.
_NOTE_TEMPLATES = (
    "Records are AND-combined on the supplied filters; unknown filter keys are ignored.",
    "Monetary fields are integer cents and timestamps are Unix seconds at UTC.",
    "Identifiers are lowercase, zero-padded, and prefixed by record type.",
    "Denormalized region and segment reflect the record's creation-time assignment.",
    "This response was served from the operational read replica.",
    "Downstream consumers should reconcile aggregates against the finance ledger at close.",
    "List ordering is stable within a store but is not a lifecycle ordering.",
    "Retry-safe reads back off on rate limits; confirm caps before trusting a count.",
    "Segment is set at account creation and reviewed at each renewal, not recomputed.",
    "A region filter scopes a list to one region on the endpoints that support it.",
    "Partial payments do not clear an invoice's paid flag; ageing continues.",
    "Credit notes are recorded against the originating invoice, not as negatives.",
)


def _payload_tokens(envelope: dict[str, Any]) -> int:
    return count_tokens(json.dumps(envelope, sort_keys=True, default=str))


def pad_to_tokens(envelope: dict[str, Any], target: int) -> dict[str, Any]:
    """Inflate ``envelope`` to ~``target`` tokens by appending operational notes to
    ``meta.notes``. ``data`` is never touched, so record semantics stay intact; padding
    is deterministic given the (deterministic) payload. A no-op when ``target`` <= 0 or
    the payload already exceeds it.
    """
    if target <= 0:
        return envelope
    base = _payload_tokens(envelope)
    if base >= target:
        return envelope
    meta = envelope.setdefault("meta", {})
    notes: list[str] = list(meta.get("notes", []))
    total = base
    i = 0
    # Bound is a safety backstop; the target is reached in ~target/note_tokens steps.
    while total < target and i < 20_000:
        note = f"Operational note {i + 1}: {_NOTE_TEMPLATES[i % len(_NOTE_TEMPLATES)]}"
        notes.append(note)
        total += count_tokens(note) + 4  # + JSON quoting/comma overhead per entry
        i += 1
    meta["notes"] = notes
    return envelope
