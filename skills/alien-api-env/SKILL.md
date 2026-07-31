---
name: alien-api-env
description: Write agents, memory systems, and outer loops against the alien-api continual-learning environment. Use when working in this repo, running rollouts, reading rewards or feedback, or debugging why acceptance is not improving.
---

# alien-api

A continual-learning benchmark. 240 questions about one fictional back office, answered in
a fixed order, with a correction after each wrong answer. Learning shows up as acceptance
rising and tool calls falling across the stream.

## Hard rule

The accepted label for every episode is on disk, in
`src/alien_api_env/data/episodes/alien_api_v4.jsonl`, and again in `trace.info["accepted"]`
after `finalize()`. It has to be, because the corrections are computed locally from it.

**Never put a label into the model's context.** Not from the JSONL, not from
`trace.info["accepted"]`, not from `record["accepted"]` in a results file. A loop that
banks all of `trace.info` is cheating by accident. Bank `trace.info["feedback"]` and the
agent's own tool observations, nothing else.

Also do not reconstruct the key from `alien_api_env.world.preferences`,
`world.profile`, or the world's `behavior` block. The internal knowledge-injection helpers
have been stripped from this build on purpose.

## The two axes

Both must be learned, and they are learned differently.

| axis | what it is | how it is learned | what it buys |
|---|---|---|---|
| world | undocumented API behaviours, a partly-wrong SOP wiki | discovery only, never taught | efficiency (fewer calls) |
| conventions | Margot's unwritten preferences about basis, scope, format, naming, escalation | taught by her corrections, one class at a time | acceptance |

Episodes each invoke 1 to 3 convention dimensions and may cross world traps. Both ride into
`trace.info` per episode (`invoked`, `world_traps`), so attribution is per-episode.

## Running one episode

```python
import asyncio
from alien_api_env.certify.traces import build_trace
from alien_api_env.vf import AlienApiTaskset, AlienApiTasksetConfig

ts = AlienApiTaskset(AlienApiTasksetConfig(split=""))   # "" = all 240, in order
tasks = list(ts.load())

trace = build_trace(tasks[0], "answer string", tool_returns=[])
await tasks[0].score(trace)         # reward + metrics
await tasks[0].finalize(trace, None)  # feedback + tags; score() does NOT call this
```

For real rollouts with tools, do not hand-roll it. `examples/responses_rollouts.py`
`run_rollout()` builds the toolsets, exposes their schemas, dispatches calls in-process,
assembles a real `Trace`, and scores it. Import it.

## Reading the result

`reward` is binary: 1.0 iff the answer exactly equals Margot's accepted label. Everything
else is weight-0.

The diagnostic split:

- `value_correct=1, reward=0` → found a defensible value, presented it wrong. Conventions
  problem. Feedback will name the violated dimension. This is the learnable case.
- `value_correct=0` → never got the value. World or arithmetic problem. The feedback here
  deliberately teaches nothing about conventions, so the teaching channel is throttled.
  Fix this before evaluating any memory design: raise reasoning effort or lower
  `artifact_verbosity`, and get `value_correct` above ~0.5.

`over_budget` and `tool_calls` are observability. Rising tool calls can be correct: an
agent that learned a workaround does real work where it used to fail fast.

## Building a memory system

What is known to work, from the measured 240-episode trial:

- **Mechanical corrections ledger.** Append Margot's correction sentences verbatim,
  dedupe, never paraphrase, never let the model rewrite the section. Taught rules must not
  be able to decay. A free-form rewrite evicted a taught rule before its retest.
- **Separate model-written notes** for world facts, fed from a compact digest of tool
  calls and outcomes (`name(args) -> ok/ERROR:code`). Instruct it to record concrete
  observed facts only. Process platitudes drive tool calls up and teach nothing.
- **Retries and per-episode checkpoint/resume.** A sequential 240-episode run is long
  enough that one 429 or 5xx will otherwise cost you the run.

Results: 0.421 acceptance vs 0.150 stateless, 8.3 vs 14.6 calls. The naive version of the
same idea scored *worse* than stateless. The design details are the whole difference.

Headroom: injecting both axes outright reaches 0.55 acceptance at 8.4 calls. That is the
ceiling to chase, and it is not 1.0. Some dimensions resist correction even when the rule
is sitting in the pad, which is real and expected.

## Config

`AlienApiTasksetConfig`: `split` (`""` for all 240, default `"train"`), `dataset_path`,
`worlds_root`, `artifact_verbosity` (default 22000). No seed, no persona knob. In prime-rl
these are `--env.taskset.*`.

## Traps

- `finalize(trace, runtime)` needs two args. Offline, pass `None`.
- `score()` does not call `finalize()`. No feedback without it.
- Default split is `"train"` (214), not the full 240.
- gpt-5.x plus reasoning plus tools over `/chat/completions` silently makes zero tool
  calls. Responses API only.
- `verifiers` is pinned to `0.2.2.dev36`. Newer prereleases break `vf.Trace`.
- Results JSON files embed prompts and labels. Never commit them.
