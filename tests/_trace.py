"""Test shim: `build_trace` now lives in the library so certify (library code) can build
synthetic traces without importing `tests/`. Re-exported here so existing test imports
(`from ._trace import build_trace`) keep working after the Plan 04-01 promotion.
"""

from __future__ import annotations

from alien_api_env.certify.traces import answer_trace, build_trace, witness_trace

__all__ = ["build_trace", "witness_trace", "answer_trace"]
