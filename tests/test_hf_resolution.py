"""hf:// dataset references route through the huggingface_hub resolver (offline test:
the downloader is stubbed to the bundled files; no network in CI)."""

from __future__ import annotations

import huggingface_hub
import pytest

from alien_api_env.vf.taskset import DEFAULT_DATASET, load_episodes
from alien_api_env.world import store


@pytest.fixture
def stub_hub(monkeypatch):
    calls = []

    def fake_download(repo_id: str, filename: str, repo_type: str):
        calls.append((repo_id, filename, repo_type))
        if filename.startswith("episodes/"):
            return str(DEFAULT_DATASET)
        return str(store.world_path())

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return calls


def test_hf_dataset_path_resolves_episodes(stub_hub):
    rows = load_episodes("hf://ConstructLabs/alien-api", split="eval")
    assert len(rows) == 26
    assert stub_hub == [("ConstructLabs/alien-api", "episodes/alien_api_v4.jsonl", "dataset")]


def test_hf_worlds_root_resolves_world(stub_hub):
    path = store.world_path("kestrel", "hf://ConstructLabs/alien-api")
    assert path == store.world_path()  # stub returns the bundled copy
    assert stub_hub == [("ConstructLabs/alien-api", "worlds/kestrel/world.json", "dataset")]


def test_local_paths_never_touch_the_hub(stub_hub):
    load_episodes(split="train")
    store.world_path("kestrel", "")
    assert stub_hub == []
