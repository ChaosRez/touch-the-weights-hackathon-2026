# Cluster run log — bringing alien-api up under prime-rl

Every job run against the cluster while getting this environment training, with the
decision that produced it and what it actually proved. Newest last. Kept so nobody repeats
a failure that has already been paid for.

All runs so far are as the **CC Max** service account, not a `test-hackathon` guest, so
they exercise the config's mechanics but **not** the hardened-namespace restrictions
(pod-security admission, the 4-GPU quota, image allow-listing). That distinction is still
unverified.

| job | what | result |
|---|---|---|
| 116 | `preflight.yaml`, 1x H100 | **SUCCEEDED** |
| 117 | `run.yaml`, 4x H100 | FAILED, never provisioned |
| 118 | `run.yaml` + 0.6B smoke config, 4x H100 | FAILED before first rollout |
| 119 | `run.yaml` committed config, 4x H100, `--max-steps 2` | see below |

---

## Job 116 — preflight, SUCCEEDED

Purpose: settle the only question a laptop cannot answer, whether the env runs against the
`verifiers` build the **image** ships. That build is a git submodule pin inside the
prime-rl fork, not the version this repo pins in `pyproject.toml`.

Result, and it is the answer to the whole version-consolidation question:

```
verifiers 0.2.2.dev17          <- what the image ships
verifiers after install 0.2.2.dev17
env id 'alien-api' -> AlienApiTaskset
fleet hydrated: 240 episodes
score on accepted label: 1.0 | feedback: Accepted.
PREFLIGHT OK
```

**Decision: install with `--no-deps` and let the image's verifiers win.** This repo pins
`verifiers==0.2.2.dev36` for laptop use. Without `--no-deps`, uv honours that pin and
swaps the image's verifiers out from under prime-rl mid-setup. There is no conflict to
resolve: the env resolves, hydrates, scores, and finalizes on dev17 unchanged.

Known-bad range: `0.2.2.dev58` breaks `build_trace` (`vf.Trace` gained a required `agent`
field). Working range therefore spans at least dev17..dev36.

## Job 117 — FAILED, never provisioned

`run.yaml` shipped with the literal placeholders `<registry>` and `<digest>` in
`image_id`. Kubernetes rejects that as `InvalidImageName`.

**The failure mode is worse than the bug.** SkyPilot classifies it as a *provisioning*
failure, so it retried with growing backoff (77s, 80s, 168s, 251s...) forever rather than
failing fast. Burned 16 minutes looking busy.

Fix: digest-pinned image. Lesson recorded in the repo: never commit a task file with a
placeholder where a real resource identifier goes.

## Job 118 — FAILED before the first rollout

Purpose: prove the pipeline end to end with a cheap model (Qwen3-0.6B, 2 trainer + 2
inference). Everything up to the weight sync worked:

```
Env alien-api ready: num_tasks=240 group_scoring=False
Starting inference on GPU(s) 0 1  /  trainer on GPU(s) 2 3
Policy inference pool ready
Training from scratch
Updating policy in-flight to v0     <- last progress
```

Then ~11 minutes of silence, then:

```
orchestrator.py:456 start -> client.py:444 update_weights
  -> client.py:394 _resume_engines -> client.py:362 _admin_post -> httpx.ReadTimeout
Orchestrator failed with exit code 1
```

**No rollout ever ran**, so this job proves nothing about the harness or MCP.

Prime suspect: inference topology mismatch. That smoke config omitted
`[inference.parallel]`, and the log shows `NCCL broadcast: 1 servers,
inference_world_size=1, gpus_per_server=1` while `num_infer_gpus = 2`. Two GPUs allocated
to inference, broadcast convinced there was one.

**Two lessons worth more than the run:**

1. A stalled prime-rl job is **indistinguishable from a healthy one** in `sky jobs queue`:
   status stays `RUNNING`, duration climbs, the log just stops. Poll the last *log
   timestamp*, never the status.
2. Do not omit `[inference.parallel]`. Set `tp`/`dp` explicitly to match `num_infer_gpus`.

## Config decisions carried into the committed `rl.toml`

Taken from the fork's own MCP tool-calling example,
`configs/basic/wiki-search/rl.toml`, rather than invented.

- **`env.agent.harness = { id = "null", ... }`** — without it the harness resolves via
  `default_harness_id()`, which falls back to **`bash`** for any taskset exporting no
  `Harness` subclass, which alien-api does not. Verified locally:
  `default_harness_id('alien-api') -> 'bash'`, and that harness is a code-executing coding
  agent (`EXECUTES_CODE=True`). `null` is the plain tool-calling driver
  (`SUPPORTS_MCP=True`, `EXECUTES_CODE=False`). This is also the harness named in the
  previously documented `HarnessError: harness 'bash' exited 1`. **Still unverified end to
  end.**
- **`[inference.model] tool_call_parser = "hermes"`** — without a parser vLLM returns
  Qwen's tool calls as ordinary assistant text: no tool calls reach the harness, the agent
  never reads the world, and every rollout scores 0 with `tool_calls=0`. That reads as a
  weak model rather than a config error, which is why it is worth stating.
- **v1 env block shape** `env.taskset = { id = "alien-api" }`. The flat `id = "..."` form
  is the v0/legacy bridge and fails for this env.

## Job 119 — CANCELLED after every rollout failed. Two real results.

Committed `rl.toml` unmodified (Qwen3-8B + LoRA), `--max-steps 2`.

**Result 1: the weight-sync stall that killed 118 is gone, but not for the reason I
guessed.** I expected `[inference.parallel] tp=1/dp=2` to be the fix. The log says
otherwise:

```
118:  Initializing weight broadcast (type='nccl')  ... NCCL broadcast: 1 servers,
      inference_world_size=1, gpus_per_server=1     -> hung, ReadTimeout after 11 min
119:  Initializing weight broadcast (type='filesystem')  -> proceeded in seconds
```

The LoRA path selects **filesystem** broadcast, and filesystem does not hang. So the
lesson is "LoRA/filesystem broadcast works, the NCCL path in this image hangs on a 2-GPU
inference pool", not anything about `tp`/`dp`. **A full fine-tune run that selects NCCL
should be expected to hit 118's stall again** — unverified, but do not assume the
committed config protects you if you drop LoRA.

**Result 2: the harness change took effect and did NOT fix the MCP blocker.**

```
21:44:31 Starting orchestrator loop (max_steps=2)
21:44:41 Train batch 0/128 (0.0%); 128 inflight rollouts (train=128, eval=0)
21:45:05 Rollout failed ... HarnessError: harness 'null' exited 1
           File "/tmp/vf-scripts/<hash>.py", line 86, in list_tools
             async with mcp_session(spec) as session:
             await session.initialize()
           asyncio.exceptions.CancelledError: Cancelled via cancel scope
```

It says **`harness 'null'`**, so `env.agent.harness` is being applied — that config is
correct and confirmed. The MCP `session.initialize()` failure is **independent of harness
choice**, so the previously documented "bash harness" framing was a red herring: the
harness was wrong *and* MCP was broken, and fixing the first did not fix the second.

### What has been ruled out for the MCP failure

- **Missing MCP entrypoints.** All three toolset modules bind and report a port locally in
  ~1s (`crm` 1.2s, `wiki` 0.9s, `answer` 0.9s). Note the repo's `test_mcp_launch.py` only
  covers `crm` and `wiki`; `answer` was probed manually and is fine. Worth adding to the
  test.
- **A read timeout.** The harness uses `MCP_TIMEOUT = httpx.Timeout(600.0, connect=5.0)`.
  Six hundred seconds of read budget, and the failure came ~34s in as a
  `CancelledError` from an outer cancel scope. So the server accepted the connection and
  never answered `initialize`, and something upstream gave up on it.
- **A stdio/env mismatch.** The harness connects over **streamable HTTP** to `spec["url"]`
  (`verifiers/v1/harnesses/null/program.py`), so the servers are launched by the env
  server inside `/app/.venv`, where `alien_api_env` is installed. The harness's isolated
  uv environment never needs to import the env package.

### Leading hypothesis: thundering herd

`Train batch 0/128; 128 inflight rollouts` — the orchestrator launched **128 rollouts
simultaneously** on one pod, each spawning a harness subprocess and its own MCP sessions
against three servers. That is the next thing to eliminate.

## Job 120 — CANCELLED. Concurrency hypothesis REJECTED.

`smoke-rl.toml`: every concurrency knob at the floor (`batch_size 4`, `group_size 2`,
`max_inflight_rollouts 4`, `pool.num_workers 2`).

```
22:24:42 Train batch 0/4 (0.0%); 4 inflight rollouts (train=4, eval=0)
22:24:45 Rollout failed ... HarnessError: harness 'null' exited 1
22:24:48 Rollout failed ... HarnessError: harness 'null' exited 1
```

**Four** inflight rollouts fail exactly as **128** did. The MCP failure is absolute, not
load-induced. Cancelled after 35 minutes — note it never self-terminates, it retries
failing rollouts forever, so always bound a debug run.

## What the local reproduction eliminated

Rather than keep paying 4 GPUs per hypothesis, the MCP path was reproduced on a laptop:
launch `CrmToolset` through the real `verifiers.v1.mcp.launch.serve()` and connect exactly
as the null harness does (streamable HTTP, `ClientSession`, `initialize()`, `list_tools()`).

```
toolset runtime config: type='subprocess'
server url: http://127.0.0.1:61815/mcp
connected, calling initialize()...
INITIALIZE OK
LIST_TOOLS OK: 11 tools -> ['count_accounts', 'get_account', 'get_inventory', ...]
```

Then repeated in a venv built to mirror the cluster exactly — `verifiers==0.2.2.dev17`
(the image's build) with the env installed `--no-deps`:

```
verifiers: 0.2.2.dev17
INITIALIZE OK
LIST_TOOLS OK: 11 tools
```

**So the verifiers version is not the cause, and neither is the environment's MCP code.**
The same toolset, the same launcher, the same client protocol, on the same verifiers build
the image ships, works on a laptop. Something about the pod is the difference.

Eliminated so far: missing entrypoints, read timeouts, stdio/env mismatch, concurrency,
verifiers version, and the toolset code itself.

## Job 121 — debug, in progress

The blind spot: prime-rl spawns each env server as a child process and redirects its
stdout/stderr to `<output_dir>/logs/envs/{train,eval}/<name>.log`. The MCP servers are
launched from inside that process, so **every MCP startup error lands in that file and
never in the job log**. Everything seen so far is the client side saying "initialize was
cancelled", which cannot say why.

`debug.yaml` runs the smoke under `timeout 900` and then dumps every `*.log` under
`/scratch`, which is where the answer should be.
