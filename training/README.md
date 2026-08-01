# RL training on alien-api

Two files: `rl.toml` (what to train) and `run.yaml` (how to get it onto the cluster).
Cluster access itself is in [`../skills/cluster/SKILL.md`](../skills/cluster/SKILL.md).

## Read this before you launch

**Verified on the training image (2026-07-31)**, launched as a hackathon guest via
`sky jobs launch`, 4x H100 (2 trainer + 2 inference), Qwen3-8B + LoRA:

- the pod is admitted and provisions; `setup` installs the env and the preflight passes
  (240 episodes hydrate offline)
- vLLM comes up and the **LoRA adapter loads** — this requires attention-only
  `target_modules` in `[trainer.model.lora]` (the prime-rl default targets all-MLP and makes
  vLLM's `load_lora_adapter` 500; see `rl.toml`)
- the trainer starts and **rollouts begin**

Two settings in `rl.toml` are load-bearing and easy to omit:

```toml
[orchestrator.train.env.env.agent]
harness = { id = "null", runtime = { type = "subprocess" } }
```

Without it the harness resolves via `default_harness_id()`, which falls back to **`bash`**
for any taskset exporting no `Harness` subclass — which alien-api does not. `bash` is a
code-executing coding agent (`EXECUTES_CODE=True`); this env wants the plain tool-calling
driver `null` (`SUPPORTS_MCP=True`, `EXECUTES_CODE=False`), the same one the fork's own MCP
example uses (`configs/basic/wiki-search/rl.toml`).

```toml
[inference.model]
tool_call_parser = "hermes"
```

Without a parser vLLM hands Qwen's tool calls back as plain assistant text: no tool calls
reach the harness and every rollout scores 0 with `tool_calls=0`, which reads as a weak
model rather than a config error.

**Status: VERIFIED on 4x H100 (job 122, 2026-08-01).** Qwen3-8B + LoRA, two steps, zero
harness errors, orchestrator finished cleanly. Rollouts execute, the agent takes turns
against the MCP toolsets, episodes score, and gradients are computed.

Getting there required one workaround that `run.yaml` applies for you, for an upstream
verifiers bug. The two sides of MCP resolve different major versions:

| side | where | version |
|---|---|---|
| MCP server | the verifiers venv | 1.29.0 |
| MCP client | the harness's isolated uv script env | 2.0.0 |

The harness program is a PEP 723 script with an unpinned `"mcp"` in its header, so
`uv run --script` takes the newest, and a 2.x client hangs forever in `initialize()`
against a 1.x server. Every rollout dies as `HarnessError: harness 'null' exited 1`. This
became fatal the day mcp 2.0.0 shipped; nothing in prime-rl or this env changed.

`run.yaml` patches the header to `mcp<2` during setup and fails loudly if the header shape
changes. The reverse fix is impossible: verifiers imports `mcp.server.fastmcp`, which
2.0.0 removed.

### Reading the step line

```
SUCCESS Step 2 | 22.6s | Reward 0.5000 | Trainable 2/20 | Turns 2.0 | Error 20.0% | Truncation 0.0%
```

Two of those numbers mislead at small batch sizes:

- **`Reward 0.5000` is an artifact.** With `group_size = 2` and the enforced
  `zero_advantage` filter, only groups holding both a 1.0 and a 0.0 survive, so the mean is
  exactly 0.5 by construction. Not a performance measurement.
- **`Error 20.0%` is benign.** The error type is `Cancelled`: the orchestrator kills
  surplus in-flight rollouts when it drains the pipeline
  (`Draining pipeline (cancelled N in-flight train rollout(s))`). Step 1 always shows 0%,
  and a `step_N+1/` trace directory exists for a step that never ran. With `batch_size = 4`
  the cancelled-to-shipped ratio is high; at 128 it is proportionally small.

What *is* worth reacting to:

- **`Truncation`** — 1024 completion tokens truncated 75% of step-1 generations, cutting
  the agent off mid-tool-call. The smoke now uses 4096, `rl.toml` uses 8192.
- **`stop_condition/context_length`** — with `artifact_verbosity = 22000` and
  `seq_len = 32768` there is little headroom. If this climbs, lower `artifact_verbosity`
  before raising `seq_len`.
- **`solved_none`** sitting at 0.67-0.77 is expected cold. Only `solved_some` groups
  survive the advantage filter and produce gradient.

## The one config trap

alien-api is a **verifiers-v1** environment. Its env block is nested:

```toml
[[orchestrator.train.env]]
name = "alien-api"

[orchestrator.train.env.env.taskset]
id = "alien-api"
split = ""
```

Older configs in our monorepo use the flat form, `id = "milestone_env"` directly under
`[[train.env]]`. That is the **v0/legacy bridge**, for environments built on the old
`MultiTurnEnv` API. Copy that shape here and prime-rl will try to load alien-api as a
legacy env. The validator's own message spells out the distinction:

```
no env configured — set env = { taskset = { id = "<id>" } } (v1) or id = "<id>" (v0/legacy)
```

## Smoke, in this order

1. **Local, no cluster.** Confirms the env is installed such that prime-rl can find it:

   ```bash
   uv run python -c "
   from verifiers.v1.loaders import taskset_class
   print(taskset_class('alien-api'))"
   ```

   Should print `AlienApiTaskset`. If it raises, the hub wrapper in
   `environments/alien-api/` is not installed: `uv pip install --no-deps -e environments/alien-api`.

2. **A preflight job, before any training.** Not a dev box: `preflight.yaml` is one GPU for
   under a minute, in the same guest task shape as `run.yaml`. It prints the image's
   `verifiers` build, installs the env, resolves the id, hydrates all 240 episodes, and
   runs a full score/finalize.

   ```bash
   sky jobs launch -y -n alien-preflight preflight.yaml
   sky jobs logs alien-preflight
   ```

   Verified 2026-07-31: the image ships **`verifiers 0.2.2.dev17`**, not the
   `0.2.2.dev36` this repo pins for laptop use, and the env runs on it unchanged. That is
   why both installs use `--no-deps` — letting uv apply our pin would swap the image's
   verifiers out from under prime-rl mid-setup.

3. **A two-step smoke**, before committing to a long run. Override the step count on the
   command line rather than editing `rl.toml`:

   ```bash
   # in run.yaml's run block, append --max-steps 2 to the `rl` invocation
   sky jobs launch -y -n alien-smoke run.yaml
   ```

   Success looks like `SUCCESS Step 1 ... SUCCESS Step 2 ... Orchestrator finished.` with
   **zero** `HarnessError` lines. Budget ~6 minutes, most of it model download and vLLM
   startup.

   Caveat worth knowing: the run that verified this path end to end (2026-08-01) also
   lowered every concurrency knob to the floor (`batch_size 4`, `group_size 2`,
   `max_inflight_rollouts 4`, `pool.num_workers 2`). `rl.toml` at `batch_size = 128` has
   **not** been re-verified since the mcp fix landed. If a full-scale run misbehaves, drop
   those four values first — that configuration is known good.

4. **Then the real run**, `run.yaml` + `rl.toml`.

## Debugging a run that fails or stalls

Three places hold the answer and **none of them reach the job log**. If you need them, add
this to your `run:` block after the `rl` invocation:

```bash
timeout 900 .venv/bin/rl @ /tmp/cfg/rl.toml || true   # bound it: a failing run retries forever

find /scratch -name 'metrics.jsonl' -exec tail -5 {} \;        # every scalar, incl. error types
find /scratch -name 'traces.jsonl' | head                       # per-step rollout traces
find /scratch -path '*logs/envs*' -name '*.log' -exec tail -100 {} \;   # env server + MCP errors
```

- **`metrics.jsonl`** only exists if you enable the sink: add a bare `[file_monitor]` table
  to `rl.toml`. Without it these scalars are computed and thrown away, because W&B is
  offline by default.
- **`logs/envs/{train,eval}/<name>.log`** is where prime-rl redirects each env server's
  stdout and stderr. The MCP servers are launched from inside that process, so MCP startup
  failures land there and nowhere else — upstream you only see the client saying
  `initialize` was cancelled, which cannot tell you why.
- A **stalled** job is indistinguishable from a healthy one in `sky jobs queue`: status
  stays `RUNNING`, duration climbs, the log just stops. Poll the last log *timestamp*, not
  the status.

Do not reach for an interactive box for any of this. `sky launch -c` holds its GPUs until
someone tears it down, and with every team on one pool that is capacity taken out of the
room for as long as you leave it up. Iterate by resubmitting jobs.

## Sizing

`batch_size` counts **rollouts, not prompts**. With `group_size = 8`, `batch_size = 128`
gives 16 unique prompts per gradient step. Setting it to 32 gives four, which starves
gradient diversity and plateaus in a way that looks like an algorithm problem and is not
one. This cost us a full ablation series to diagnose once.

`pool.num_workers` is the other throttle, in the opposite direction. Rollouts on this env
are env-bound rather than generation-bound: every episode makes many tool calls against an
in-process world. The default of 4 leaves the GPUs idle at single-digit utilisation. Start
at 32.

## What to watch

There is no eval block and no held-out split, on purpose. This is a single sequential pass
over 240 episodes. Learning is the *trend*, so watch windowed:

- `preference_accepted` rising, the conventions axis
- `value_correct` rising, the world/computation axis
- `tool_calls` falling, exploration collapsing as the world gets learned

If `value_correct` sits below ~0.5, fix that before reading anything else: the feedback
channel that teaches conventions only opens once the answer carries a defensible value.
Lower `artifact_verbosity` or raise effort.

Watch `over_budget` too, but do not optimise it. It is weight-0 and rising tool calls can
be correct behaviour.
