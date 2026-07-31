"""Persona layer (Phase 3), fully offline: committed profiles validate against the
schema, the loader is deterministic, the sampler/feedbacker SKIP without an endpoint,
and the persona-voice cache is used only when it passes the content audit. CI never
calls a model — every test here runs with the endpoint env vars cleared."""

from __future__ import annotations

import json

import pytest

from alien_api_env.feedbacker import feedback as fb
from alien_api_env.feedbacker.feedback import (
    Violation,
    audit_feedback_cache,
    cache_key,
    correction_sentence,
    persona_feedback,
)
from alien_api_env.feedbacker.persona import (
    PERSONA_BRIEFS,
    endpoint_configured,
    generate_persona_profile,
    persona_prompt,
)
from alien_api_env.vf import AlienApiTaskset, AlienApiTasksetConfig
from alien_api_env.world import profile as profile_mod
from alien_api_env.world.preferences import SCHEMA, dimension_ids
from alien_api_env.world.profile import PROFILES_DIR, load_persona_profile, validate_profile

from ._trace import build_trace

_CHOICES = {d.id: d.options[0] for d in SCHEMA}


@pytest.fixture(autouse=True)
def _no_endpoint(monkeypatch):
    for var in ("VF_PROBE_BASE_URL", "VF_PROBE_API_KEY", "VF_PROBE_MODEL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def persona_dir(tmp_path, monkeypatch):
    d = tmp_path / "profiles"
    d.mkdir()
    (d / "test-persona.json").write_text(
        json.dumps({"persona_id": "test-persona", "brief": "a test reviewer", "choices": _CHOICES})
    )
    monkeypatch.setattr(profile_mod, "PROFILES_DIR", d)
    return d


def test_committed_persona_profiles_validate_offline():
    """Every profile checked into the repo validates against the schema."""
    for path in PROFILES_DIR.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_profile(payload["choices"])
        assert payload["persona_id"] == path.stem


def test_persona_loader_is_deterministic_and_validated(persona_dir):
    p1 = load_persona_profile("test-persona")
    p2 = load_persona_profile("test-persona")
    assert dict(p1.choices) == dict(p2.choices) == _CHOICES
    assert p1.persona_id == "test-persona"


def test_persona_loader_rejects_invalid_choices(persona_dir):
    (persona_dir / "broken.json").write_text(
        json.dumps({"persona_id": "broken", "choices": {**_CHOICES, "money_unit": "doubloons"}})
    )
    with pytest.raises(ValueError, match="valid options"):
        load_persona_profile("broken")


def test_margot_is_the_committed_feedbacker():
    """The one feedbacker: her committed profile validates and labels the fleet."""
    profile = load_persona_profile("margot")
    assert profile.persona_id == "margot"
    rows = [t.data for t in AlienApiTaskset(AlienApiTasksetConfig(id="alien-api")).select(24)]
    for r in rows:
        assert r.persona_id == "margot"
        assert r.choices == {d: profile[d] for d in r.invoked}


# ------------------------------------------------------------- endpoint gating


def test_sampler_skips_cleanly_without_endpoint():
    assert not endpoint_configured()
    assert generate_persona_profile("auditor") is None


def test_feedback_cache_generation_skips_cleanly_without_endpoint():
    assert fb.generate_feedback_cache("auditor") is None


def test_persona_cli_errors_clearly_without_endpoint():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "alien_api_env.feedbacker.persona", "--persona-id", "auditor"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 2
    assert "VF_PROBE_BASE_URL" in proc.stdout


def test_persona_prompt_lists_every_dimension():
    prompt = persona_prompt(PERSONA_BRIEFS["margot"])
    for did in dimension_ids():
        assert did in prompt
    assert "JSON" in prompt


# ------------------------------------------------------- persona-voice feedback


def test_persona_feedback_falls_back_to_template_without_cache():
    v = Violation(dimension="money_unit", chosen="dollars")
    assert persona_feedback([v], persona_id="no-such-persona") == (
        "Rejected. " + correction_sentence(v)
    )


def test_persona_feedback_uses_clean_cache_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "CACHE_DIR", tmp_path)
    v = Violation(dimension="money_unit", chosen="dollars")
    voiced = "Whole dollars on my desk, always. Cents are for the ledger, not for me."
    (tmp_path / "auditor.json").write_text(json.dumps({cache_key(v): voiced}))
    assert persona_feedback([v], persona_id="auditor") == f"Rejected. {voiced}"


def test_persona_feedback_rejects_leaky_cache_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "CACHE_DIR", tmp_path)
    v = Violation(dimension="money_unit", chosen="dollars")
    (tmp_path / "auditor.json").write_text(
        json.dumps({cache_key(v): "The answer was 103000 dollars."})  # digits: leak risk
    )
    assert audit_feedback_cache("auditor")  # audit flags it
    # and the runtime path ignores it, falling back to the safe template
    assert persona_feedback([v], persona_id="auditor") == "Rejected. " + correction_sentence(v)


def test_committed_feedback_caches_pass_the_audit():
    for path in fb.CACHE_DIR.glob("*.json"):
        assert audit_feedback_cache(path.stem) == [], path


async def test_finalize_uses_persona_voice_when_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "CACHE_DIR", tmp_path)
    cfg = AlienApiTasksetConfig(id="alien-api")
    task = AlienApiTaskset(cfg).select(12)[0]
    wrong = next(v for v in task.data.defensible.values() if v != task.data.accepted)
    # freeze a voiced entry for whichever dimension this violation hits
    trace = build_trace(task, wrong, tool_returns=0)
    await task.finalize(trace, None)
    violated = trace.info["violated"]
    assert violated
    voiced = "That is not how I read it; follow my convention here."
    key = cache_key(
        Violation(dimension=violated[0], chosen=task.data.choices[violated[0]]),
        task.data.feedback_hints,
    )
    (tmp_path / "margot.json").write_text(json.dumps({key: voiced}))
    trace2 = build_trace(task, wrong, tool_returns=0)
    await task.finalize(trace2, None)
    assert voiced in trace2.info["feedback"]
