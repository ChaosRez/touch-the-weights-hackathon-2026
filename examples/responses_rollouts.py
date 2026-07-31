"""Stateless rollout driver for alien-api over the OpenAI Responses API.

This is the baseline: every episode runs cold, with nothing carried between episodes and
nothing injected. It is also the machinery `scratchpad_loop.py` imports (`run_rollout`,
`create_with_retries`, `summarize`), so read this first if you are writing your own loop.

What a rollout does here: build the three toolsets, hand their schemas to the model, let
it call tools in-process until it submits an answer, assemble a real verifiers `Trace`,
and score it with the real `task.score()`. Scoring never calls a model.

**Use the Responses API, not /chat/completions.** gpt-5.x with reasoning plus tools over
/chat/completions returns zero tool calls, silently. That is the single most expensive
mistake available on this environment.

Model: `ALIEN_API_MODEL` (default gpt-5.6). Key: `OPENAI_API_KEY`.

Usage (from the repo root, so `uv run` resolves this project's virtualenv):
  OPENAI_API_KEY=... uv run python examples/responses_rollouts.py \
      [--n 20] [--split ""] [--out rollouts.json]
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import time

import verifiers.v1 as vf
from openai import OpenAI
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import ToolMessage

from alien_api_env.vf import AlienApiTaskset, AlienApiTasksetConfig
from alien_api_env.vf.tools import AnswerToolset, CrmToolset, WikiToolset

# Override with ALIEN_API_MODEL. Must be an OpenAI Responses-API model id that actually
# exists on your account: the 5.6 family ships as named variants (`-luna`, `-sol`,
# `-terra`), there is no bare `gpt-5.6`. `luna` is the model the reports/ trial used, so
# it is the one whose numbers are comparable.
MODEL = os.environ.get("ALIEN_API_MODEL", "gpt-5.6-luna")
EFFORT = "medium"
MAX_TOOL_TURNS = 12
CONCURRENCY = 4
RETRY_ATTEMPTS = 5

_TRANSIENT = ("APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError")


async def create_with_retries(client: OpenAI, request: dict):
    """responses.create with exponential backoff on transient failures — a multi-hour
    sequential run must not die to one Cloudflare 5xx (learned 2026-07-28)."""
    loop = asyncio.get_event_loop()
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return await loop.run_in_executor(None, lambda: client.responses.create(**request))
        except Exception as e:
            status = getattr(e, "status_code", None)
            transient = (
                type(e).__name__ in _TRANSIENT
                or (isinstance(status, int) and (status == 429 or status >= 500))
            )
            if not transient or attempt == RETRY_ATTEMPTS - 1:
                raise
            delay = 2 ** attempt * 3
            print(f"  transient API error ({type(e).__name__}, status={status}); retry in {delay}s", flush=True)
            await asyncio.sleep(delay)

_TYPE_MAP = {int: "integer", str: "string", float: "number", bool: "boolean"}


# REMOVED for the hackathon build: `world_block()` and `prefs_block()`, the two
# knowledge-injection helpers from the internal two-axis validation harness. They read
# the world behaviour block and Margot's profile off disk and render them as a prompt
# prefix, i.e. they hand the model the exact answers it is supposed to learn from
# corrections. Injecting either one is the thing this benchmark measures the absence of.
# The ceiling they establish (0.55 acceptance / 8.4 calls with both injected) is quoted
# in README.md as the target a real memory system should chase.


def _tool_defs(toolsets):
    defs, dispatch = [], {}
    for ts in toolsets:
        for fn in discover_decorated(ts, "tool"):
            name = getattr(fn, "tool_name", None) or fn.__name__
            sig = inspect.signature(fn)
            props, required = {}, []
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                ann, optional = param.annotation, param.default is not inspect.Parameter.empty
                base = ann
                if getattr(ann, "__args__", None):
                    args = [a for a in ann.__args__ if a is not type(None)]
                    base = args[0] if args else str
                if isinstance(base, str):
                    base = {"int": int, "str": str, "float": float, "bool": bool}.get(
                        base.split(" ")[0].replace("| None", "").strip(), str
                    )
                props[pname] = {"type": _TYPE_MAP.get(base, "string")}
                if not optional:
                    required.append(pname)
            defs.append(
                {
                    "type": "function",
                    "name": name,
                    # The FULL docstring, not just the first paragraph. Several tools
                    # state their constraints below the summary line (query_report names
                    # the one metric it supports), and truncating to the first paragraph
                    # hides them: an agent then brute-forces metric names against a closed
                    # vocabulary, burns its whole turn budget on `unknown_metric`, and
                    # submits nothing. Measured: 28 wasted calls on one episode.
                    "description": inspect.cleandoc(fn.__doc__ or name),
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                        "additionalProperties": False,
                    },
                }
            )
            dispatch[name] = getattr(ts, fn.__name__)
    return defs, dispatch


async def run_rollout(client: OpenAI, task, prompt: str) -> dict:
    crm = CrmToolset(vf.ToolsetConfig())
    wiki = WikiToolset(vf.ToolsetConfig())
    ans = AnswerToolset(vf.ToolsetConfig())
    await crm.setup_task(task.data)
    await wiki.setup_task(task.data)
    tools, dispatch = _tool_defs((crm, wiki, ans))

    trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data))
    parent = None
    request = dict(
        model=MODEL,
        instructions=task.data.system_prompt,
        input=prompt,
        tools=tools,
        reasoning={"effort": EFFORT},
    )
    usage_in = usage_out = 0
    final_text = ""
    tool_names: list[str] = []
    tool_digest: list[str] = []
    prev_id = None
    for _turn in range(MAX_TOOL_TURNS + 1):
        request_now = {**request, "previous_response_id": prev_id} if prev_id else request
        response = await create_with_retries(client, request_now)
        prev_id = response.id
        usage_in += response.usage.input_tokens
        usage_out += response.usage.output_tokens
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            final_text = response.output_text or ""
            break
        outputs = []
        for call in calls:
            tool_names.append(call.name)
            try:
                result = await dispatch[call.name](**json.loads(call.arguments or "{}"))
            except Exception as e:  # surface tool misuse to the model
                result = {"ok": False, "error": {"code": "tool_error", "message": str(e)}}
            outcome = "ok" if result.get("ok") else f"ERROR:{result.get('error', {}).get('code', '?')}"
            tool_digest.append(f"{call.name}({call.arguments or ''}) -> {outcome}")
            payload = json.dumps(result, default=str)
            trace.nodes.append(
                MessageNode(
                    parent=parent,
                    message=ToolMessage(tool_call_id=call.call_id, content=payload, name=call.name),
                    sampled=False,
                )
            )
            parent = len(trace.nodes) - 1
            outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": payload})
        request = dict(
            model=MODEL,
            instructions=task.data.system_prompt,
            input=outputs,
            tools=tools,
            reasoning={"effort": EFFORT},
        )

    trace.nodes.append(
        MessageNode(parent=parent, message=vf.AssistantMessage(content=final_text), sampled=True)
    )
    await task.score(trace)
    await task.finalize(trace, None)
    return {
        "episodeId": task.data.name,
        "prompt": task.data.prompt,
        "tool_names": tool_names,
        "tool_digest": tool_digest,
        "kind": task.data.kind,
        "invoked": list(task.data.invoked),
        "accepted": task.data.accepted,
        "answer": task._answer(task.data, trace),
        "reward": trace.reward,
        "preference_accepted": trace.metrics["preference_accepted"],
        "value_correct": trace.metrics["value_correct"],
        "preferences_violated": trace.metrics["preferences_violated"],
        "tool_calls": trace.metrics["tool_calls"],
        "budget": task.data.budget,
        "over_budget": trace.metrics["over_budget"],
        "violated": trace.info["violated"],
        "feedback": trace.info["feedback"],
        "usage_in": usage_in,
        "usage_out": usage_out,
    }


async def run_condition(client: OpenAI, tasks, label: str, prefix: str) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[dict | None] = [None] * len(tasks)

    async def one(i: int, task) -> None:
        async with sem:
            t0 = time.time()
            try:
                rec = await run_rollout(client, task, prefix + task.data.prompt)
            except Exception as e:
                rec = {"episodeId": task.data.name, "error": f"{type(e).__name__}: {e}"}
            rec.update(condition=label, idx=i, seconds=round(time.time() - t0, 1))
            results[i] = rec
            print(
                f"[{label} {i:02d}] accepted={rec.get('preference_accepted')} "
                f"calls={rec.get('tool_calls')} answer={rec.get('answer', rec.get('error'))!r:36.36} "
                f"{rec['seconds']}s",
                flush=True,
            )

    await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    return [r for r in results if r is not None]


def summarize(records: list[dict]) -> dict:
    good = [r for r in records if "error" not in r]
    mean = lambda k: round(sum(r[k] for r in good) / len(good), 3) if good else None  # noqa: E731
    return {
        "n": len(records),
        "errors": sum(1 for r in records if "error" in r),
        "mean_reward": mean("reward"),
        "mean_preference_accepted": mean("preference_accepted"),
        "mean_value_correct": mean("value_correct"),
        "mean_tool_calls": mean("tool_calls"),
        "over_budget_rate": mean("over_budget"),
        "tokens_in": sum(r.get("usage_in", 0) for r in good),
        "tokens_out": sum(r.get("usage_out", 0) for r in good),
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    # "" = all 240 episodes in seq_index order, which is the continual-learning setup.
    # "train" (214) / "eval" (26) exist in the fleet but there is no held-out set here:
    # the run is a single sequential pass, never multi-epoch.
    ap.add_argument("--split", default="")
    ap.add_argument("--out", default="rollouts.json")
    args = ap.parse_args()

    client = OpenAI()
    tasks = AlienApiTaskset(AlienApiTasksetConfig(id="alien-api", split=args.split)).select(args.n)
    payload = {"model": MODEL, "split": args.split, "n": args.n, "effort": EFFORT,
               "conditions": {}, "records": {}}
    # One condition: cold. Every episode starts with no injected knowledge, which is the
    # only honest baseline. Anything your agent knows, it has to have learned in-run.
    print(f"\n== cold: {args.n} episodes (split={args.split or 'ALL'}) ==", flush=True)
    records = await run_condition(client, tasks, "cold", "")
    payload["records"]["cold"] = records
    payload["conditions"]["cold"] = summarize(records)

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print("\n=== summary ===")
    print(json.dumps(payload["conditions"], indent=2))
    print(f"results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
