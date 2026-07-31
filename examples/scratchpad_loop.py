"""The simplest outer loop: sequential episodes with a two-part scratchpad, paired
against a stateless baseline on the same episodes.

The **memory arm** runs fleet episodes in sequence order carrying a scratchpad with two
sections:

- **Reviewer corrections (ledger)** — maintained *mechanically by the loop*: every
  correction sentence Margot has issued, appended verbatim and deduplicated. Corrections
  are class-level by design (certified answer-free), so replaying them is legitimate
  memory, and taught rules can never decay out of the pad (the v1-loop failure).
- **Operational notes** — maintained *by the model* (a separate no-tools call after each
  rollout) from a compact digest of its tool calls and their outcomes, so world facts
  (the 404 route, the silent cap, the prefix) are learnable. The instruction demands
  concrete facts and forbids process platitudes (the v1-loop crowding failure).

The **stateless arm** runs the same episodes cold. Learning shows up as the memory arm's
acceptance rising and tool calls falling across the sequence while the stateless arm
stays flat; the paired difference is the gain (the CL-Bench framing, on this substrate).
The loop never injects labels, and feedback never contains answers (certified).

Usage (from the repo root):
  OPENAI_API_KEY=... uv run python examples/scratchpad_loop.py \
      [--n 30] [--split train] [--out scratchpad_loop.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from responses_rollouts import (  # noqa: E402
    CONCURRENCY,
    EFFORT,
    MODEL,
    create_with_retries,
    run_rollout,
    summarize,
)

from alien_api_env.vf import AlienApiTaskset, AlienApiTasksetConfig  # noqa: E402

NOTES_WORD_BUDGET = 300
NOTES_CHAR_CAP = 3000  # hard backstop over the word-budget instruction

MEMORIZE_INSTRUCTIONS = (
    "You maintain the OPERATIONAL NOTES section of a scratchpad used between back-office "
    "tasks against the same company systems (the reviewer's corrections are kept for you "
    "separately — do not restate them). Rewrite the notes now, keeping them under "
    f"{NOTES_WORD_BUDGET} words. Record ONLY concrete, reusable facts observed from tool "
    "calls and results: which endpoints work or return errors, exact prefixes or key "
    "formats, silent caps or truncation, stale/retry behavior, where specific data lives, "
    "computation recipes that produced defensible values. FORBIDDEN: generic process "
    "advice (validate, verify, double-check, be careful) — concrete facts only. Reply "
    "with ONLY the new notes text."
)

NOT_DEFENSIBLE_MARKER = "Rejected. That does not follow from the records"


def _compose_pad(corrections: list[str], notes: str) -> str:
    parts = []
    if corrections:
        parts.append(
            "Reviewer corrections received so far (verbatim, cumulative):\n- "
            + "\n- ".join(corrections)
        )
    if notes.strip():
        parts.append(f"Your operational notes on these systems:\n{notes.strip()}")
    return "\n\n".join(parts)


def _ledger_update(corrections: list[str], feedback: str) -> None:
    """Mechanical append+dedupe of Margot's correction sentences (class-level, answer-free
    by certification). Acceptance and the contentless data-recheck rejection add nothing."""
    if feedback == "Accepted." or feedback.startswith(NOT_DEFENSIBLE_MARKER):
        return
    body = feedback.removeprefix("Rejected.").strip()
    for sentence in body.split(". "):
        sentence = sentence.strip()
        if sentence and not sentence.endswith("."):
            sentence += "."
        if sentence and sentence not in corrections:
            corrections.append(sentence)


def _memory_prefix(scratchpad: str) -> str:
    if not scratchpad.strip():
        return ""
    return f"Your notes from previous tasks with this reviewer and these systems:\n{scratchpad}\n\n"


async def _memorize(client: OpenAI, notes: str, record: dict) -> str:
    """One no-tools call: distill the episode's tool observations into the notes."""
    digest = "\n".join(record.get("tool_digest", [])) or "(no tool calls)"
    episode_report = (
        f"Task: {record['prompt']}\n"
        f"Tool calls and outcomes:\n{digest}\n"
        f"You submitted: {record['answer'] or '(nothing)'}\n"
        f"Reviewer verdict: "
        f"{'accepted' if record['feedback'] == 'Accepted.' else 'rejected'}\n\n"
        f"Current operational notes:\n{notes or '(empty)'}"
    )
    response = await create_with_retries(
        client,
        dict(
            model=MODEL,
            instructions=MEMORIZE_INSTRUCTIONS,
            input=episode_report,
            reasoning={"effort": "low"},
        ),
    )
    text = (response.output_text or "").strip()
    return text[:NOTES_CHAR_CAP] if text else notes


async def run_memory_arm(
    client: OpenAI, tasks, checkpoint, state: dict
) -> tuple[list[dict], list[str]]:
    """Strictly sequential: rollout -> ledger append -> notes update -> next episode.
    Checkpoints after every episode; resumes from a restored state dict."""
    corrections: list[str] = state.get("corrections", [])
    notes = state.get("notes", "")
    records: list[dict] = state.get("memory", [])
    snapshots: list[str] = state.get("scratchpads", [])
    tasks = tasks[len(records):]
    for i, task in enumerate(tasks, start=len(records)):
        t0 = time.time()
        pad = _compose_pad(corrections, notes)
        rec = await run_rollout(client, task, _memory_prefix(pad) + task.data.prompt)
        rec.update(condition="memory", idx=i, seconds=round(time.time() - t0, 1))
        rec["prompt"] = task.data.prompt
        records.append(rec)
        _ledger_update(corrections, rec["feedback"])
        notes = await _memorize(client, notes, rec)
        snapshots.append(_compose_pad(corrections, notes))
        checkpoint(memory=records, scratchpads=snapshots, corrections=corrections, notes=notes)
        print(
            f"[memory {i:03d}] accepted={rec['preference_accepted']} calls={rec['tool_calls']} "
            f"ledger={len(corrections)} pad={len(snapshots[-1])}ch {rec['seconds']}s",
            flush=True,
        )
    return records, snapshots


async def run_stateless_arm(client: OpenAI, tasks) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[dict | None] = [None] * len(tasks)

    async def one(i: int, task) -> None:
        async with sem:
            t0 = time.time()
            rec = await run_rollout(client, task, task.data.prompt)
            rec.update(condition="stateless", idx=i, seconds=round(time.time() - t0, 1))
            results[i] = rec
            print(
                f"[stateless {i:03d}] accepted={rec['preference_accepted']} "
                f"calls={rec['tool_calls']} {rec['seconds']}s",
                flush=True,
            )

    await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    return [r for r in results if r is not None]


def _window(records: list[dict], key: str, lo: int, hi: int) -> float:
    xs = [r[key] for r in records[lo:hi]]
    return round(sum(xs) / len(xs), 3) if xs else 0.0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--split", default="", help="fleet split; empty = ALL 240 episodes in sequence order")
    ap.add_argument("--out", default="scratchpad_loop.json")
    args = ap.parse_args()

    client = OpenAI()
    tasks = AlienApiTaskset(AlienApiTasksetConfig(id="alien-api", split=args.split)).select(args.n)

    partial_path = args.out + ".partial"
    state: dict = {}
    if os.path.exists(partial_path):
        state = json.load(open(partial_path))
        print(
            f"resuming from {partial_path}: stateless={len(state.get('stateless', []))} "
            f"memory={len(state.get('memory', []))}",
            flush=True,
        )

    def checkpoint(**updates) -> None:
        state.update(updates)
        with open(partial_path, "w") as f:
            json.dump(state, f)

    if len(state.get("stateless", [])) == args.n:
        stateless = state["stateless"]
        print(f"== stateless baseline: {args.n} episodes (restored) ==", flush=True)
    else:
        print(f"== stateless baseline: {args.n} episodes ==", flush=True)
        stateless = await run_stateless_arm(client, tasks)
        checkpoint(stateless=stateless)
    print(f"\n== memory arm (sequential, scratchpad-carried): {args.n} episodes ==", flush=True)
    memory, snapshots = await run_memory_arm(client, tasks, checkpoint, state)

    q = max(1, args.n // 4)
    windows = {}
    for arm_name, arm in (("memory", memory), ("stateless", stateless)):
        windows[arm_name] = [
            {
                "episodes": f"{lo}-{min(lo + q, args.n) - 1}",
                "accepted": _window(arm, "preference_accepted", lo, lo + q),
                "tool_calls": _window(arm, "tool_calls", lo, lo + q),
            }
            for lo in range(0, args.n, q)
        ]
    verdict = {
        "stateless": summarize(stateless),
        "memory": summarize(memory),
        "windows": windows,
        "gain_overall": round(
            summarize(memory)["mean_preference_accepted"]
            - summarize(stateless)["mean_preference_accepted"],
            3,
        ),
    }
    payload = {
        "model": MODEL, "split": args.split, "n": args.n, "effort": EFFORT,
        "verdict": verdict,
        "stateless": stateless, "memory": memory, "scratchpads": snapshots,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print("\n=== verdict ===")
    print(json.dumps(verdict, indent=2))
    print(f"\nfinal scratchpad:\n{snapshots[-1] if snapshots else '(none)'}")
    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
