"""alien-api v4: a preference-learning continual-learning environment over stored data.

ONE world (kestrel, a fictional CRM/ERP back office with a drifting SOP wiki) and ONE
feedbacker (Margot, whose 15 frozen conventions decide which defensible answer is
accepted). Every episode is ambiguous by construction; feedback teaches the violated
preference-class in her voice, never the answer; and sequences get progressively easier
as her conventions and the world's latent structure (the silent search cap, the
prefix-gated lookup, the deprecated inventory route, the stale report retry) are learned.

The environment is **pure hydration**: the world file and the certified episode fleet
are committed data built by `alien-api-synth` (the milestone-synth pattern); scoring is
a deterministic exact match against the stored label — no model in the reward path, no
seeds at runtime, no key needed to score.
"""

from alien_api_env.vf import (
    AlienApiData,
    AlienApiTask,
    AlienApiTaskset,
    AlienApiTasksetConfig,
    AnswerToolset,
    CrmToolset,
    WikiToolset,
)

__all__ = [
    "AlienApiData",
    "AlienApiTask",
    "AlienApiTaskset",
    "AlienApiTasksetConfig",
    "CrmToolset",
    "WikiToolset",
    "AnswerToolset",
]
