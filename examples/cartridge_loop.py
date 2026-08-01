"""Run the Phase 1 in-process Qwen tool agent on cold alien-api episodes.

The same ``QwenToolAgent`` is used by every later memory arm; this command is
the cold rollout/gating harness. It records every assistant action and every
in-process tool result, then scores and finalizes the real verifier trace.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path

from alien_api_env.vf import AlienApiTaskset, AlienApiTasksetConfig
from cartridge_memory.qwen_agent import (
    HuggingFaceQwenBackend,
    QwenAgentConfig,
    QwenGenerationConfig,
    QwenToolAgent,
)
from cartridge_memory.text_ledger import GlobalTextLedger


def _answered(record: dict) -> bool:
    """Accept new records and pre-field records that already contain the effective answer."""

    if "answered" in record:
        return bool(record["answered"])
    return bool(str(record.get("answer", "")).strip())


def _summary(records: list[dict], n: int) -> dict:
    successful = [record for record in records if "error" not in record]
    first_ten = records[: min(10, n)]
    answered_first_ten = sum(_answered(record) for record in first_ten)
    submitted_first_ten = sum(bool(record.get("submitted")) for record in first_ten)
    value_correct = sum(
        float(record.get("metrics", {}).get("value_correct", 0.0)) for record in records
    ) / max(1, n)
    tool_answer_episode = any(
        _answered(record)
        and any(
            execution["name"] != "submit_answer"
            for execution in record.get("tool_executions", [])
        )
        for record in records
    )
    return {
        "n": n,
        "completed": len(successful),
        "errors": n - len(successful),
        "answered": sum(_answered(record) for record in records),
        "answered_first_ten": answered_first_ten,
        "submitted": sum(bool(record.get("submitted")) for record in records),
        "submitted_first_ten": submitted_first_ten,
        "first_ten_n": len(first_ten),
        "first_ten_answer_rate": round(answered_first_ten / max(1, len(first_ten)), 3),
        "first_ten_submission_rate": round(submitted_first_ten / max(1, len(first_ten)), 3),
        "mean_value_correct": round(value_correct, 3),
        "mean_preference_accepted": round(
            sum(
                float(record.get("metrics", {}).get("preference_accepted", 0.0))
                for record in records
            )
            / max(1, n),
            3,
        ),
        "mean_tool_calls": round(
            sum(
                float(record.get("metrics", {}).get("tool_calls", 0.0))
                for record in records
            )
            / max(1, n),
            3,
        ),
        "tool_result_and_answer_episode": tool_answer_episode,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--split", default="")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default=os.environ.get("CARTRIDGES_DEVICE", "cuda:0"))
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-tool-turns", type=int, default=12)
    parser.add_argument("--artifact-verbosity", type=int, default=22000)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--text-ledger-tokens",
        type=int,
        default=0,
        help="enable the minimal global text ledger with this tokenizer-position budget",
    )
    parser.add_argument("--require-gates", action="store_true")
    parser.add_argument("--out", default="/persist/cartridges/runs/phase_1.json")
    args = parser.parse_args()

    tasks = list(
        AlienApiTaskset(
            AlienApiTasksetConfig(
                id="alien-api",
                split=args.split,
                artifact_verbosity=args.artifact_verbosity,
            )
        ).select(args.n)
    )
    backend = HuggingFaceQwenBackend(
        QwenGenerationConfig(
            model_name=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            enable_thinking=not args.no_thinking,
        )
    )
    agent = QwenToolAgent(backend, QwenAgentConfig(max_tool_turns=args.max_tool_turns))
    ledger = (
        GlobalTextLedger(backend.tokenizer, args.text_ledger_tokens)
        if args.text_ledger_tokens > 0
        else None
    )

    records: list[dict] = []
    for index, task in enumerate(tasks):
        started = time.monotonic()
        try:
            attachment = ledger.attachment() if ledger is not None else None
            rollout = await agent.run(task, attachment=attachment, seed=args.seed + index)
            record = rollout.to_dict()
            record["index"] = index
            record["seconds"] = round(time.monotonic() - started, 2)
            record["memory_tokens"] = ledger.rendered_tokens if ledger is not None else 0
            if ledger is not None:
                ledger.update(record)
            print(
                f"[{index:02d}] answered={record['answered']} "
                f"typed={record['submitted']} "
                f"value={record['metrics'].get('value_correct')} "
                f"calls={record['metrics'].get('tool_calls')} "
                f"turns={len(record['assistant_turns'])} {record['seconds']}s",
                flush=True,
            )
        except Exception as error:
            if not any("error" in previous for previous in records):
                traceback.print_exc()
            record = {
                "episodeId": task.data.name,
                "index": index,
                "error": f"{type(error).__name__}: {error}",
                "seconds": round(time.monotonic() - started, 2),
            }
            print(f"[{index:02d}] ERROR {record['error']}", flush=True)
        records.append(record)

    summary = _summary(records, args.n)
    if ledger is not None:
        summary.update(
            memory_mode="global_text_ledger",
            memory_budget_tokens=ledger.max_tokens,
            final_ledger_entries=len(ledger.entries),
            final_rendered_tokens=ledger.rendered_tokens,
        )
    payload = {
        "model": args.model,
        "split": args.split,
        "seed": args.seed,
        "artifact_verbosity": args.artifact_verbosity,
        "thinking": not args.no_thinking,
        "text_ledger_tokens": args.text_ledger_tokens,
        "summary": summary,
        "records": records,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("results ->", output, flush=True)

    if args.require_gates:
        gates = {
            "tool_result_and_answer_episode": summary["tool_result_and_answer_episode"],
            "first_ten_answer_rate>=0.8": summary["first_ten_answer_rate"] >= 0.8,
            "mean_value_correct>=0.5": summary["mean_value_correct"] >= 0.5,
        }
        print("gates:", json.dumps(gates, indent=2), flush=True)
        if not all(gates.values()):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
