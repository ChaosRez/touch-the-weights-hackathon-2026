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

## Job 119 — in progress

Purpose: run the committed `rl.toml` unmodified (Qwen3-8B + LoRA, `tp=1/dp=2`) with
`--max-steps 2` passed on the command line, to test in order: whether `[inference.parallel]`
avoids 118's weight-sync timeout, whether the `null` harness clears MCP
`session.initialize()`, and whether real tool calls appear.

Result: pending.
