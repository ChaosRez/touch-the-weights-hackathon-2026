from __future__ import annotations

import json

import pytest
import torch

from cartridge_memory.phase4_state import Phase4StateStore


def _store(tmp_path, resume=False):
    return Phase4StateStore(
        str(tmp_path / "arm"),
        arm="still_recurrent",
        model="tiny",
        seed=17,
        resume=resume,
        run_metadata={"checkpoint": "phase2.pt"},
    )


def test_state_advances_records_and_resumes_exact_ledger_tensors(tmp_path):
    store = _store(tmp_path)
    state = store.initialize({"cache": {"k": torch.tensor([1.0])}})
    first = {"index": 0, "metrics": {"preference_accepted": 0.0}}
    state = store.commit(
        state,
        record=first,
        ledger_state={"cache": {"k": torch.tensor([2.0])}},
    )

    resumed = _store(tmp_path, resume=True)
    loaded = resumed.load()

    assert loaded["next_episode"] == 1
    assert torch.equal(loaded["ledger_state"]["cache"]["k"], torch.tensor([2.0]))
    assert resumed.records(loaded) == [first]


def test_state_recovers_last_record_if_checkpoint_won_atomic_race(tmp_path):
    store = _store(tmp_path)
    state = store.initialize(None)
    record = {"index": 0, "answer": "x"}
    state = store.commit(state, record=record, ledger_state=None)
    (tmp_path / "arm/episodes/000000.json").unlink()

    loaded = _store(tmp_path, resume=True).load()

    assert json.loads((tmp_path / "arm/episodes/000000.json").read_text()) == record
    assert loaded["next_episode"] == 1


def test_state_refuses_overwrite_and_resume_metadata_mismatch(tmp_path):
    store = _store(tmp_path)
    store.initialize(None)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _store(tmp_path)
    mismatched = Phase4StateStore(
        str(tmp_path / "arm"),
        arm="cold",
        model="tiny",
        seed=17,
        resume=True,
        run_metadata={"checkpoint": "phase2.pt"},
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        mismatched.load()

    wrong_checkpoint = Phase4StateStore(
        str(tmp_path / "arm"),
        arm="still_recurrent",
        model="tiny",
        seed=17,
        resume=True,
        run_metadata={"checkpoint": "wrong.pt"},
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        wrong_checkpoint.load()


def test_failure_trace_is_immutable_and_does_not_advance_state(tmp_path):
    store = _store(tmp_path)
    state = store.initialize(None)

    path = store.write_failure(0, {"index": 0, "parse_errors": ["bad JSON"]})

    assert json.loads(path.read_text())["parse_errors"] == ["bad JSON"]
    assert state["next_episode"] == 0
    with pytest.raises(FileExistsError, match="immutable Phase 4 failure"):
        store.write_failure(0, {"index": 0})
