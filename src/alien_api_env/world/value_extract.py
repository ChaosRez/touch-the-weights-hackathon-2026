"""Value extraction for the ``value_correct`` diagnostic (repurposes v2's answer_norm).

Acceptance in v3 is an **exact match** against the profile-compiled label — presentation
is a learned preference, so the reward never normalizes it away. This module serves the
weight-0 diagnostics instead: did the model reach *some* defensible reading of the
records (``value_correct``), and which defensible rendering did its reply carry (for
violation attribution and the curve's learned-the-world-vs-learned-the-feedbacker split)?

``match(reply, candidates)`` is pure string handling — not an LLM judge. Numeric
candidates match token-wise (thousands separators stripped, so ``"**190,265 cents**"``
matches ``190265`` but ``"0"`` never matches inside ``103000``); textual candidates
(status words, ids, titles, names, ISO dates, ``escalate``) match case-insensitively on
word boundaries. Longer candidates are tried first so a more specific rendering wins.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_MARKDOWN = re.compile(r"[*`~_]{1,3}")
_INT_TOKEN = re.compile(r"\d[\d,]*")


def _clean(reply: str) -> str:
    return _MARKDOWN.sub("", (reply or "")).strip().strip("'\"").strip()


def match(reply: str, candidates: Iterable[str]) -> str | None:
    """The defensible value ``reply`` carries, or None if it carries none of them.

    Deterministic and order-stable: candidates are tried longest-first, numeric ones by
    token equality, textual ones by whole-token (word-boundary) containment.
    """
    cleaned = _clean(reply)
    if not cleaned:
        return None
    tokens = [t.replace(",", "") for t in _INT_TOKEN.findall(cleaned)]
    for value in sorted(set(candidates), key=len, reverse=True):
        if value.isdigit():
            if value in tokens:
                return value
        else:
            if cleaned.lower() == value.lower():
                return value
            if re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", cleaned, re.IGNORECASE):
                return value
    return None
