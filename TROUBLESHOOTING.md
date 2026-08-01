# Troubleshooting

Every entry here has actually happened. Read the first two before you start.

## The model makes zero tool calls and answers from nothing

You are on `/chat/completions` with a gpt-5.x model, reasoning, and tools. That
combination returns no tool calls and no error. Use the **Responses API**.
`examples/responses_rollouts.py` does this correctly; copy its `run_rollout`.

Related contract for gpt-5.x: set `reasoning={"effort": ...}` explicitly, and omit
`temperature` entirely.

## `ValidationError: agent Field required` when building a Trace

Your `verifiers` moved. This env is pinned to **`verifiers==0.2.2.dev36`** in
`pyproject.toml`, deliberately and not as a range. `vf.Trace` gained a required `agent`
field somewhere after dev36, and `>=0.2.1,<0.3` silently resolves to a much newer
prerelease. If you have relaxed the pin or installed verifiers yourself:

```bash
uv pip install --prerelease=allow \
  --extra-index-url https://hub.primeintellect.ai/primeintellect/simple/ \
  "verifiers==0.2.2.dev36"
```

Do not relax the pin during the event.

## `uv add` / `uv sync` cannot find verifiers

`verifiers` is not on PyPI. It comes from the prime-hub index, and it is a prerelease
build, so both of these are required and both are already in `pyproject.toml`:

```toml
[tool.uv]
prerelease = "allow"

[[tool.uv.index]]
name = "prime-hub"
url = "https://hub.primeintellect.ai/primeintellect/simple/"
```

If you started a fresh project of your own, copy that block into it.

## `TypeError: finalize() missing 1 required positional argument: 'runtime'`

`finalize` takes `(trace, runtime)`. Offline, pass `None`:

```python
await task.finalize(trace, None)
```

`Task.score()` does **not** call `finalize()`. If you only call `score()` you get the
reward and the metrics but no feedback, which means your memory loop has nothing to learn
from. Call both.

## The taskset only gives me 214 episodes

The default `split` is `"train"`. For the full sequential stream:

```python
AlienApiTasksetConfig(split="")   # all 240, seq_index order
```

There is no held-out set in this setup. It is a single sequential pass, never multi-epoch.

## Acceptance is 0 and `value_correct` is also near 0

Your model is not computing the answers, so it never reaches the point where Margot's
conventions matter. The feedback it gets in that state deliberately teaches nothing about
conventions, so your memory system is being starved, not failing.

Fix the value axis first: raise reasoning effort, or lower `artifact_verbosity` (default
22000) so tool returns are less padded and exploration is cheaper. Get `value_correct`
comfortably above 0.5 before judging any memory design.

## Tool calls went UP after adding memory

Expected in one specific case, and a bug in another.

Expected: your agent learned a workaround that requires real work where it previously hit
an error and gave up. Fewer calls is not intrinsically better, which is exactly why tool
calls are weight-0 and not part of the reward.

A bug: your notes are accumulating generic process advice ("always validate pagination",
"verify completeness") rather than concrete facts. That drives exploration up and taught
nothing. This is the documented failure of the naive free-rewrite scratchpad. Make the
corrections ledger mechanical, append and dedupe, and instruct the notes step to record
only concrete observed facts.

## A long run died partway through

Use `examples/scratchpad_loop.py` as the model: it has retry with backoff on transient API
errors and per-episode checkpoint/resume. A multi-hour sequential run will hit a 429 or a
5xx eventually. Losing episode 200 of 240 to one blip is avoidable.

## Rate limits with a shared key

240 episodes in a paired run is roughly 25M input tokens. Budget accordingly, and use
`--n 30` while iterating. `CONCURRENCY` in `examples/responses_rollouts.py` defaults to 4;
raising it is the fastest way to get yourself rate limited.

## Cluster problems

See [`skills/cluster/SKILL.md`](skills/cluster/SKILL.md) for the full table. The three that
cost the most time:

- **Job dies in setup with `sudo: the "no new privileges" flag is set`** — you are missing
  `runAsUser: 0` in the container `securityContext`. The image runs as a non-root user;
  SkyPilot bootstraps ssh/ray as root and needs this.
- **Pod rejected `... hostPath volumes`** — you used a `hostPath` (e.g. `/mnt/nvme`, which the
  namespace forbids). Use the `/persist` `emptyDir` instead; it's durable for the box's life.
- **`sky launch` sits Pending** — the shared 20-GPU pool is full right now. Your box starts
  when GPUs free; lower your GPU ask or wait. (It won't fail — it queues.)
- **Files gone after `sky down`** — the box disk doesn't survive teardown. `rsync
  team-N-box:/persist ./` or push to your HF/W&B **before** tearing down.
- **`ErrImagePull` or `... may only run the approved image`** — use the public, digest-pinned
  `ml-hackathon/prime-rl-base` image the organizers gave you, with no `imagePullSecrets`.
