"""World hydration (v4): the stored file -> identical frozen worlds, loudly or not at all."""

from __future__ import annotations

import json

import pytest

from alien_api_env.world.models import Behavior, World
from alien_api_env.world.store import (
    DEFAULT_WORLD,
    load_world,
    world_from_dict,
    world_path,
)


def test_hydrates_the_committed_world():
    world = load_world()
    assert isinstance(world, World)
    assert world.name == DEFAULT_WORLD == "kestrel"
    assert isinstance(world.behavior, Behavior)
    assert world.behavior.search_cap > 0
    assert len(world.accounts) > 100
    assert "sop-refunds-v2" in world.sop.pages and "sop-expenses-v2" in world.sop.pages
    assert len(world.behavior.governed_page_pairs) == 2


def test_hydration_is_cached_and_deterministic():
    a, b = load_world(), load_world()
    assert a is b  # lru_cache: one parse per process
    fresh = world_from_dict(json.loads(world_path().read_text()))
    assert fresh == a  # and value-identical when re-parsed


def test_world_is_immutable():
    world = load_world()
    with pytest.raises(TypeError):
        world.accounts["acc-00000"] = None  # type: ignore[index]
    with pytest.raises(AttributeError):
        world.behavior.search_cap = 1  # type: ignore[misc]


def test_missing_world_raises_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="no stored world"):
        load_world("atlantis", str(tmp_path))


def test_behavior_matches_the_stored_file():
    artifact = json.loads(world_path().read_text())
    world = load_world()
    assert world.behavior.search_cap == artifact["behavior"]["search_cap"]
    assert world.behavior.fiscal_year_start_month == artifact["behavior"]["fiscal_year_start_month"]
    assert list(world.behavior.status_code_table) == artifact["behavior"]["status_code_table"]
