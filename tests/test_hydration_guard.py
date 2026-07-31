"""The v4 invariant, enforced: the environment package contains NO generation code and
ships its data. One world, one feedbacker, zero seeds at runtime."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "alien_api_env"

# Tokens that only generator code uses. Any hit in the env module tree is a regression
# against the hydration-only cutover.
_FORBIDDEN = re.compile(
    r"rng_stream|generate_records|sample_quirks|render_sop|sample_profile|"
    r"world_version|profile_version|import random\b"
)


def test_no_generator_code_in_the_env_package():
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in _FORBIDDEN.finditer(text):
            offenders.append(f"{path.relative_to(SRC)}: {match.group(0)}")
    assert not offenders, f"generation/drift code in the env package: {offenders}"


def test_generator_modules_are_gone():
    world_dir = SRC / "world"
    present = {p.name for p in world_dir.glob("*.py")}
    assert present == {
        "__init__.py",
        "conventions.py",
        "models.py",
        "preferences.py",
        "profile.py",
        "store.py",
        "value_extract.py",
    }


def test_data_ships_with_this_repo():
    """INVERTED from the internal repo on purpose.

    Internally the fleet and world are never in git (they live in the artifact store,
    mirrored to HF, rebuilt by alien-api-synth). This distribution has no synth package
    and no credentials, so the certified data is COMMITTED here: a clone must hydrate
    offline, with no key, no GCS, no HF token. If this fails, the clone is broken for
    every participant, not just for you.
    """
    data = SRC / "data"
    assert (data / "episodes" / "alien_api_v4.jsonl").exists(), "fleet missing from the clone"
    assert (data / "worlds" / "kestrel" / "world.json").exists(), "world missing from the clone"
    assert (SRC / "feedbacker" / "profiles" / "margot.json").exists(), "Margot's profile missing"
    assert (SRC / "feedbacker" / "feedback_cache" / "margot.json").exists(), "feedback cache missing"


def test_the_fleet_is_the_full_240():
    """Sequential single pass over the whole fleet: no held-out split in this setup."""
    fleet = SRC / "data" / "episodes" / "alien_api_v4.jsonl"
    rows = [ln for ln in fleet.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 240, f"expected the certified 240-episode fleet, got {len(rows)}"


def test_no_knowledge_injection_helpers_ship():
    """The two-axis harness's `world_block()` / `prefs_block()` and the `curve` demo
    render the answer key straight out of the shipped data. They are stripped from this
    build; scoring, feedback, and the metrics do not need them."""
    assert not (SRC / "curve").exists(), "the curve demo renders Margot's conventions"
    driver = (SRC.parent.parent / "examples" / "responses_rollouts.py").read_text(encoding="utf-8")
    assert "def world_block" not in driver
    assert "def prefs_block" not in driver


def test_package_data_patterns_still_bundle_when_present():
    """Wheels built after --install embed the data (the self-contained publish path)."""
    pyproject = (SRC.parent.parent / "pyproject.toml").read_text()
    for pattern in ("data/worlds/*/world.json", "data/episodes/*.jsonl"):
        assert pattern in pyproject, f"package-data misses {pattern}"
