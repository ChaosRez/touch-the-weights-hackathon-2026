"""The feedbacker profile: one committed choice per preference dimension.

There is ONE feedbacker — Margot — and her profile is frozen data
(``feedbacker/profiles/margot.json``). The environment loads it for feedback rendering
and validation; the *labels* were compiled against it at build time in ``alien-api-synth``
and ride in the fleet, so scoring never resolves a profile at all.

No drift, no procedural sampling, no seeds: a different feedbacker (or a drifted Margot)
would be a new committed file and a fleet rebuild, not a runtime knob.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from alien_api_env.world.preferences import SCHEMA, dimension_ids, validate_schema

# Committed persona profiles live next to the persona sampler that writes them.
PROFILES_DIR = Path(__file__).resolve().parent.parent / "feedbacker" / "profiles"


@dataclass(frozen=True)
class Profile:
    """A frozen commitment: one chosen option per schema dimension."""

    choices: Mapping[str, str]
    persona_id: str | None = None

    def __getitem__(self, dim_id: str) -> str:
        return self.choices[dim_id]


def validate_profile(choices: Mapping[str, str]) -> None:
    """Full coverage, no unknown dimensions, every choice a valid option. Raises ValueError."""
    validate_schema()
    known = set(dimension_ids())
    got = set(choices)
    if got != known:
        missing, extra = sorted(known - got), sorted(got - known)
        raise ValueError(f"profile coverage mismatch: missing={missing} extra={extra}")
    for d in SCHEMA:
        if choices[d.id] not in d.options:
            raise ValueError(
                f"profile chooses {choices[d.id]!r} for {d.id!r}; valid options are {d.options}"
            )


def load_persona_profile(persona_id: str) -> Profile:
    """Load a committed persona profile and validate it against the schema.

    The file is frozen data written by the persona sampler; runtime never calls a model.
    """
    path = PROFILES_DIR / f"{persona_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    choices = dict(payload["choices"])
    validate_profile(choices)
    return Profile(choices=MappingProxyType(choices), persona_id=persona_id)
