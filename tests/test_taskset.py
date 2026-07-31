"""Taskset (v4): pure hydration of the committed fleet — rows map verbatim onto
``AlienApiData``, splits filter, nothing is derived at runtime."""

from __future__ import annotations

import json

import pytest

from alien_api_env.vf import AlienApiData, AlienApiTaskset, AlienApiTasksetConfig
from alien_api_env.vf.taskset import DEFAULT_DATASET, DEFAULT_PERSONA, load_episodes


def _rows(split: str = "train") -> list[dict]:
    return load_episodes(split=split)


def test_packaged_fleet_loads_and_filters():
    train, eval_rows = _rows("train"), _rows("eval")
    everything = load_episodes()
    assert len(train) + len(eval_rows) == len(everything) == 240
    assert {r["split"] for r in train} == {"train"}
    assert {r["split"] for r in eval_rows} == {"eval"}


def test_taskset_maps_rows_verbatim():
    cfg = AlienApiTasksetConfig(id="alien-api", split="train")
    tasks = AlienApiTaskset(cfg).select(20)
    rows = {r["episodeId"]: r for r in _rows("train")}
    for t in tasks:
        d = t.data
        assert isinstance(d, AlienApiData)
        row = rows[d.name]
        assert d.prompt == row["prompt"]
        assert d.world == row["world"] == "kestrel"
        assert list(d.invoked) == row["invoked"]
        assert dict(d.defensible) == row["defensible"]
        assert d.accepted == row["accepted"]
        assert d.choices == row["choices"]
        assert d.budget == row["budget"]
        assert d.persona_id == DEFAULT_PERSONA
        assert list(d.world_traps) == row["world_traps"]  # the quirk tags ride the wire
        # the row is fully self-contained: the accepted label is one of its defensible answers
        assert d.accepted in row["defensible"].values()


def test_load_is_deterministic():
    a = [t.data.name for t in AlienApiTaskset(AlienApiTasksetConfig(id="alien-api")).select(30)]
    b = [t.data.name for t in AlienApiTaskset(AlienApiTasksetConfig(id="alien-api")).select(30)]
    assert a == b


def test_split_and_dataset_path_knobs(tmp_path):
    rows = _rows("eval")[:3]
    path = tmp_path / "mini.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    cfg = AlienApiTasksetConfig(id="alien-api", dataset_path=str(path), split="eval")
    got = AlienApiTaskset(cfg).select(3)
    assert [t.data.name for t in got] == [r["episodeId"] for r in rows]


def test_empty_selection_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no episodes"):
        load_episodes(str(path))


def test_default_dataset_location():
    assert DEFAULT_DATASET.name == "alien_api_v4.jsonl"
    if not DEFAULT_DATASET.exists():
        import pytest

        pytest.skip("data not installed locally; defaults fall back to hf://ConstructLabs/alien-api")
