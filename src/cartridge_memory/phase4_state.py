"""Transactional, per-episode Phase 4 state and immutable record storage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_record(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def record_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_record(record).encode()).hexdigest()


class Phase4StateStore:
    """Atomically advance an arm while keeping episode result files immutable."""

    FORMAT = "phase4-arm-state-v1"

    def __init__(
        self,
        root: str,
        *,
        arm: str,
        model: str,
        seed: int,
        resume: bool,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.arm = arm
        self.model = model
        self.seed = seed
        self.run_metadata = run_metadata or {}
        self.state_path = self.root / "state.pt"
        self.episodes_path = self.root / "episodes"
        if resume:
            if not self.state_path.exists() or not self.episodes_path.is_dir():
                raise FileNotFoundError(f"no resumable Phase 4 state under {self.root}")
        else:
            if self.root.exists():
                raise FileExistsError(f"refusing to overwrite Phase 4 arm directory: {self.root}")
            self.episodes_path.mkdir(parents=True)

    def _save(self, payload: dict[str, Any]) -> None:
        import torch

        temporary = self.state_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, self.state_path)

    def initialize(self, ledger_state: dict[str, Any] | None) -> dict[str, Any]:
        if self.state_path.exists():
            raise FileExistsError(f"Phase 4 state already exists: {self.state_path}")
        payload = {
            "format": self.FORMAT,
            "arm": self.arm,
            "model": self.model,
            "seed": self.seed,
            "run_metadata": self.run_metadata,
            "next_episode": 0,
            "record_hashes": [],
            "last_record": None,
            "ledger_state": ledger_state,
        }
        self._save(payload)
        return payload

    def load(self) -> dict[str, Any]:
        import torch

        payload = torch.load(self.state_path, map_location="cpu", weights_only=True)
        expected = {
            "format": self.FORMAT,
            "arm": self.arm,
            "model": self.model,
            "seed": self.seed,
            "run_metadata": self.run_metadata,
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Phase 4 resume metadata mismatch: {mismatches}")
        next_episode = int(payload["next_episode"])
        hashes = list(payload["record_hashes"])
        if len(hashes) != next_episode:
            raise ValueError("Phase 4 checkpoint record-hash count does not match next_episode")
        self._recover_and_validate_records(payload)
        return payload

    def _episode_path(self, index: int) -> Path:
        return self.episodes_path / f"{index:06d}.json"

    def _write_episode(self, index: int, record: dict[str, Any], digest: str) -> None:
        path = self._episode_path(index)
        if path.exists():
            existing = json.loads(path.read_text())
            if record_digest(existing) != digest:
                raise ValueError(f"immutable episode record differs at index {index}")
            return
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, path)

    def _recover_and_validate_records(self, payload: dict[str, Any]) -> None:
        next_episode = int(payload["next_episode"])
        hashes = payload["record_hashes"]
        for index, digest in enumerate(hashes):
            path = self._episode_path(index)
            if not path.exists():
                last = payload.get("last_record")
                if index != next_episode - 1 or not isinstance(last, dict):
                    raise FileNotFoundError(f"missing committed episode record {index}")
                if int(last.get("index", -1)) != index or record_digest(last) != digest:
                    raise ValueError("last-record recovery payload does not match checkpoint")
                self._write_episode(index, last, digest)
            existing = json.loads(path.read_text())
            if record_digest(existing) != digest:
                raise ValueError(f"episode record hash mismatch at index {index}")
        extras = sorted(self.episodes_path.glob("*.json"))[next_episode:]
        if extras:
            raise ValueError(f"uncommitted episode records exist after index {next_episode - 1}")

    def commit(
        self,
        payload: dict[str, Any],
        *,
        record: dict[str, Any],
        ledger_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        index = int(payload["next_episode"])
        if int(record.get("index", -1)) != index:
            raise ValueError(f"expected episode index {index}, got {record.get('index')}")
        digest = record_digest(record)
        advanced = {
            **payload,
            "next_episode": index + 1,
            "record_hashes": [*payload["record_hashes"], digest],
            "last_record": record,
            "ledger_state": ledger_state,
        }
        self._save(advanced)
        self._write_episode(index, record, digest)
        return advanced

    def records(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        self._recover_and_validate_records(payload)
        return [
            json.loads(self._episode_path(index).read_text())
            for index in range(int(payload["next_episode"]))
        ]

    def write_result(self, payload: dict[str, Any], result: dict[str, Any], target_n: int) -> Path:
        if int(payload["next_episode"]) != target_n:
            raise ValueError("cannot finalize an incomplete Phase 4 prefix")
        target = self.root / f"results_{target_n:03d}.json"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite immutable Phase 4 result: {target}")
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, target)
        return target
