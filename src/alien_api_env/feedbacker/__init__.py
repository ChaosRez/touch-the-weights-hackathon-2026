"""The feedbacker layer: teaching feedback and the build-time LLM persona.

The feedbacker has two roles that never mix: it *teaches* (corrective feedback naming the
violated preference-class, written to ``Trace.info``) and, at build time, it *generates*
(a persona commits to a coherent frozen profile, and optionally a persona-voice feedback
cache). It never scores — the reward is a deterministic exact match against the compiled
label and has no model dependency. Frozen artifacts live here as committed data:
``profiles/<persona_id>.json`` and ``feedback_cache/<persona_id>.json``.
"""

from alien_api_env.feedbacker.feedback import (
    Violation,
    audit_feedback_cache,
    persona_feedback,
    templated_feedback,
)
from alien_api_env.feedbacker.persona import PERSONA_BRIEFS, generate_persona_profile

__all__ = [
    "Violation",
    "templated_feedback",
    "persona_feedback",
    "audit_feedback_cache",
    "generate_persona_profile",
    "PERSONA_BRIEFS",
]
