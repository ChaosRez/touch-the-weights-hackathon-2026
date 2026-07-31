"""Synthetic-trace builders (the in-process rollout stand-in for tests).

v4: certification lives in ``alien-api-synth`` (`scripts/audit_fleet.py`) — the
environment ships pre-certified data and keeps no certification of its own.
"""

from __future__ import annotations

from alien_api_env.certify.traces import answer_trace, build_trace, witness_trace

__all__ = ["answer_trace", "build_trace", "witness_trace"]
