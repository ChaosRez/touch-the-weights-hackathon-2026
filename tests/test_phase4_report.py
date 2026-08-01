from __future__ import annotations

import json

import pytest

from examples.phase4_report import aggregate, load_arms, paired_bootstrap


def _record(index: int, accepted: float, calls: float) -> dict:
    return {
        "index": index,
        "episodeId": f"episode-{index}",
        "submitted": True,
        "feedback": "Reviewer correction" if not accepted else "Accepted.",
        "metrics": {
            "preference_accepted": accepted,
            "value_correct": accepted,
            "tool_calls": calls,
        },
        "memory_positions_after": 64,
        "memory_source_tokens": 128,
        "memory_recurrence_count": 2,
        "memory_compaction_seconds": 0.5,
    }


def test_paired_bootstrap_preserves_episode_pairing():
    result = paired_bootstrap([1, 1, 1], [0, 0, 0], samples=100, seed=2)
    assert result["delta"] == 1.0
    assert result["ci95_low"] == 1.0
    assert result["ci95_high"] == 1.0


def test_aggregate_has_required_metrics_and_conservative_claim():
    data = {}
    for arm in ("cold", "text64", "still_single", "still_recurrent"):
        accepted = 1.0 if arm == "still_recurrent" else 0.0
        data[arm] = {
            "model": "tiny",
            "seed": 17,
            "records": [_record(index, accepted, 2.0) for index in range(4)],
        }
    metadata = {
        f"episode-{index}": {"invoked": ["money_unit"], "world_traps": ["stale"]}
        for index in range(4)
    }
    metrics = aggregate(data, metadata, bootstrap_samples=100, bootstrap_seed=3)

    assert metrics["summary"]["still_recurrent"]["memory_positions"] == 64
    assert metrics["summary"]["still_recurrent"]["compression_ratio"] == 2.0
    assert metrics["by_preference"]["money_unit"]["cold"]["n"] == 4
    assert metrics["by_world_trap"]["stale"]["still_recurrent"]["acceptance"] == 1.0
    assert metrics["claim_supported"] is True


def test_load_arms_refuses_unpaired_results(tmp_path):
    specifications = []
    for arm in ("cold", "text64", "still_single", "still_recurrent"):
        path = tmp_path / f"{arm}.json"
        records = [_record(0, 0.0, 1.0)]
        records[0]["feedback"] = "Accepted."
        records[0]["tool_executions"] = []
        if arm == "text64":
            records[0]["episodeId"] = "different"
        path.write_text(
            json.dumps(
                {
                    "format": "phase4-arm-result-v1",
                    "arm": arm,
                    "model": "tiny",
                    "seed": 17,
                    "event_hashes": [],
                    "records": records,
                }
            )
        )
        specifications.append(f"{arm}={path}")

    with pytest.raises(ValueError, match="not paired"):
        load_arms(specifications)
