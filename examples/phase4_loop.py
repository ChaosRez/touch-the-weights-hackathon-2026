"""Run one strictly sequential, resumable Phase 4 Alien API memory arm."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any

from alien_api_env.vf import AlienApiTaskset, AlienApiTasksetConfig
from cartridge_memory.kv_ledger import RecurrentKVLedger
from cartridge_memory.phase4_state import Phase4StateStore
from cartridge_memory.qwen_agent import (
    HuggingFaceQwenBackend,
    QwenAgentConfig,
    QwenGenerationConfig,
    QwenToolAgent,
    StillQwenBackend,
)
from cartridge_memory.text_ledger import StreamingTextLedger

ARMS = ("cold", "text64", "still_single", "still_recurrent")


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    final = records[-60:]

    def mean(rows: list[dict], key: str) -> float:
        return sum(float(row["metrics"].get(key, 0.0)) for row in rows) / max(1, len(rows))

    return {
        "n": n,
        "errors": sum("error" in record for record in records),
        "answered": sum(bool(record.get("answered")) for record in records),
        "typed_submissions": sum(bool(record.get("submitted")) for record in records),
        "typed_submission_rate": sum(bool(record.get("submitted")) for record in records)
        / max(1, n),
        "preference_accepted": mean(records, "preference_accepted"),
        "final_60_preference_accepted": mean(final, "preference_accepted"),
        "value_correct": mean(records, "value_correct"),
        "mean_tool_calls": mean(records, "tool_calls"),
        "final_memory_positions": int(records[-1].get("memory_positions_after", 0)) if records else 0,
        "final_memory_source_tokens": int(records[-1].get("memory_source_tokens", 0))
        if records
        else 0,
        "final_recurrence_count": int(records[-1].get("memory_recurrence_count", 0))
        if records
        else 0,
        "total_compaction_seconds": sum(
            float(record.get("memory_compaction_seconds", 0.0)) for record in records
        ),
    }


def _ledger_state(ledger: Any | None) -> dict[str, Any] | None:
    return None if ledger is None else ledger.state_dict()


def _load_ledger_state(ledger: Any | None, state: dict[str, Any] | None) -> None:
    if ledger is None:
        if state is not None:
            raise ValueError("cold arm checkpoint unexpectedly contains memory state")
        return
    if state is None:
        raise ValueError("memory arm checkpoint is missing ledger state")
    ledger.load_state_dict(state)


def _memory_observability(ledger: Any | None) -> tuple[int, int, int, int]:
    if ledger is None:
        return 0, 0, 0, 0
    positions = (
        ledger.memory_positions if isinstance(ledger, RecurrentKVLedger) else ledger.rendered_tokens
    )
    recurrence = ledger.recurrence_count if isinstance(ledger, RecurrentKVLedger) else 0
    return positions, int(ledger.source_tokens), int(recurrence), len(ledger.event_hashes)


def _build_backend(args):
    generation = QwenGenerationConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        enable_thinking=not args.no_thinking,
    )
    if args.arm in ("cold", "text64"):
        return HuggingFaceQwenBackend(generation)
    from still.config import STILLConfig

    checkpoint = (
        args.single_checkpoint if args.arm == "still_single" else args.recurrent_checkpoint
    )
    return StillQwenBackend(
        generation,
        checkpoint,
        still_config=STILLConfig(
            model_name=args.model,
            num_latents=64,
            latent_dim=256,
            num_blocks=2,
            device=args.device,
        ),
    )


def _build_ledger(args, backend):
    if args.arm == "cold":
        return None
    if args.arm == "text64":
        return StreamingTextLedger(backend.tokenizer, max_tokens=64)
    return RecurrentKVLedger(backend.still_model, backend.tokenizer, chunk_tokens=64)


def _smoke_gates(records: list[dict[str, Any]], target_n: int) -> dict[str, bool]:
    gates = {
        "no_record_errors": not any("error" in record for record in records),
        "all_scored": all("preference_accepted" in record.get("metrics", {}) for record in records),
        "at_least_one_answer": any(record.get("answered") for record in records),
        "at_least_one_world_tool_result": any(
            any(execution["name"] != "submit_answer" for execution in record["tool_executions"])
            for record in records
        ),
    }
    if target_n >= 30:
        gates["typed_submission_rate>=0.1"] = (
            sum(bool(record.get("submitted")) for record in records[:30]) / 30 >= 0.1
        )
    return gates


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--split", default="")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default=os.environ.get("CARTRIDGES_DEVICE", "cuda:0"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-tool-turns", type=int, default=12)
    parser.add_argument("--artifact-verbosity", type=int, default=22000)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-smoke-gates", action="store_true")
    parser.add_argument("--state-root", required=True)
    parser.add_argument(
        "--single-checkpoint",
        default="/persist/cartridges/checkpoints/qwen3_4b_single_step.pt",
    )
    parser.add_argument(
        "--recurrent-checkpoint",
        default="/persist/cartridges/checkpoints/qwen3_4b_recurrent.pt",
    )
    args = parser.parse_args()
    if not 1 <= args.n <= 240:
        raise ValueError("n must be between 1 and 240")

    tasks = list(
        AlienApiTaskset(
            AlienApiTasksetConfig(
                id="alien-api",
                split=args.split,
                artifact_verbosity=args.artifact_verbosity,
            )
        ).load()
    )
    indices = [int(task.data.idx) for task in tasks]
    if indices != list(range(240)):
        raise RuntimeError("Phase 4 requires all 240 episodes in exact seq_index order")

    backend = _build_backend(args)
    ledger = _build_ledger(args, backend)
    run_metadata = {
        "split": args.split,
        "max_new_tokens": args.max_new_tokens,
        "max_tool_turns": args.max_tool_turns,
        "do_sample": not args.greedy,
        "enable_thinking": not args.no_thinking,
        "checkpoint": getattr(backend, "checkpoint_metadata", None),
    }
    store = Phase4StateStore(
        args.state_root,
        arm=args.arm,
        model=args.model,
        seed=args.seed,
        resume=args.resume,
        run_metadata=run_metadata,
    )
    if args.resume:
        state = store.load()
        _load_ledger_state(ledger, state["ledger_state"])
    else:
        state = store.initialize(_ledger_state(ledger))
    start_index = int(state["next_episode"])
    if start_index > args.n:
        raise ValueError(f"checkpoint is already past requested prefix {args.n}")

    agent = QwenToolAgent(backend, QwenAgentConfig(max_tool_turns=args.max_tool_turns))
    for index in range(start_index, args.n):
        task = tasks[index]
        if int(task.data.idx) != index:
            raise RuntimeError(f"episode order changed at index {index}")
        positions_before, _, _, _ = _memory_observability(ledger)
        attachment = None if ledger is None else ledger.attachment()
        started = time.monotonic()
        rollout = await agent.run(task, attachment=attachment, seed=args.seed + index)
        parse_errors = [turn.parse_error for turn in rollout.assistant_turns if turn.parse_error]
        if parse_errors:
            raise RuntimeError(f"episode {index} produced parse errors: {parse_errors}")
        update = {
            "added_events": 0,
            "added_source_tokens": 0,
            "compactions": 0,
            "compaction_seconds": 0.0,
        }
        if ledger is not None:
            update = ledger.update(
                feedback=rollout.feedback,
                tool_executions=rollout.tool_executions,
            )
        positions_after, source_tokens, recurrence_count, hash_count = _memory_observability(ledger)
        if args.arm.startswith("still") and positions_after not in (0, 64):
            raise RuntimeError(f"neural memory position violation: {positions_after}")

        record = rollout.to_dict()
        record.update(
            index=index,
            seq_index=int(task.data.idx),
            seconds=round(time.monotonic() - started, 3),
            memory_positions_before=positions_before,
            memory_positions_after=positions_after,
            memory_source_tokens=source_tokens,
            memory_recurrence_count=recurrence_count,
            memory_event_hash_count=hash_count,
            memory_added_events=int(update["added_events"]),
            memory_added_source_tokens=int(update["added_source_tokens"]),
            memory_compactions=int(update.get("compactions", 0)),
            memory_compaction_seconds=float(update.get("compaction_seconds", 0.0)),
        )
        state = store.commit(state, record=record, ledger_state=_ledger_state(ledger))
        print(
            f"[{args.arm} {index:03d}] accepted={record['metrics'].get('preference_accepted')} "
            f"value={record['metrics'].get('value_correct')} typed={record['submitted']} "
            f"calls={record['metrics'].get('tool_calls')} memory={positions_after} "
            f"source={source_tokens} {record['seconds']}s",
            flush=True,
        )

    records = store.records(state)
    gates = _smoke_gates(records, args.n)
    result = {
        "format": "phase4-arm-result-v1",
        "arm": args.arm,
        "model": args.model,
        "seed": args.seed,
        "split": args.split,
        "target_n": args.n,
        "checkpoint_metadata": getattr(backend, "checkpoint_metadata", None),
        "summary": _summary(records),
        "smoke_gates": gates,
        "event_hashes": sorted(ledger.event_hashes) if ledger is not None else [],
        "records": records,
    }
    output = store.write_result(state, result, args.n)
    print(json.dumps({"summary": result["summary"], "gates": gates}, indent=2), flush=True)
    print(f"results -> {output}", flush=True)
    if args.require_smoke_gates and not all(gates.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
