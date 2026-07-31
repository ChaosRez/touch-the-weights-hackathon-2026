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

**Candidate fix for the MCP blocker (added 2026-07-31, not yet confirmed end to end).** The
`HarnessError: harness 'bash' exited 1` failure names the harness, and `bash` is not a
harness this environment should ever run. It is resolved by default: `default_harness_id()`
falls back to `"bash"` whenever a taskset exports no `Harness` subclass, which alien-api does
not. The bash harness is a code-executing coding agent (`EXECUTES_CODE=True`), while this env
wants the plain tool-calling driver, `null` (`SUPPORTS_MCP=True`, `EXECUTES_CODE=False`) —
the same one the fork's own MCP example uses (`configs/basic/wiki-search/rl.toml`).

`rl.toml` now sets it explicitly:

```toml
[orchestrator.train.env.env.agent]
harness = { id = "null", runtime = { type = "subprocess" } }
```

`rl.toml` also now sets `[inference.model] tool_call_parser = "hermes"`, without which vLLM
hands Qwen's tool calls back as plain assistant text and every rollout scores 0 with
`tool_calls=0`.

**Status: the harness change is UNTESTED.** A 4x H100 Qwen3-0.6B smoke (job 118) was meant
to test it and never got far enough to say anything. It failed *before the first rollout*,
so the harness code path was never exercised. Do not read the change as verified.

That smoke failed on something else entirely, worth knowing about because it is upstream of
every rollout:

```
Updating policy in-flight to v0          <- last progress, then ~11 min of silence
orchestrator.py:456 start
  -> client.py:444 update_weights
    -> client.py:394 _resume_engines
      -> client.py:362 _admin_post
        -> httpx.ReadTimeout             -> Orchestrator failed with exit code 1
```

The weight-update admin POST to vLLM timed out. Prime suspect is an inference topology
mismatch: that smoke config omitted `[inference.parallel]`, and the log shows
`Initializing NCCL broadcast: 1 servers, inference_world_size=1, gpus_per_server=1` while
`num_infer_gpus = 2`. Two GPUs were allocated to inference but the broadcast believed there
was one. **The `rl.toml` in this repo sets `[inference.parallel] tp = 1, dp = 2`, so it may
not hit this** — but that is untested too, and it is the first thing to check if a run hangs
at `Updating policy in-flight`.

Note also that a stalled job looks identical to a healthy one in `sky jobs queue`: status
stays `RUNNING` and the streamed log goes quiet. Read the log, not the status.

**Cluster requirements** (all in `run.yaml` already): the public `ml-hackathon` image
digest-pinned, `runAsUser: 0`, an `emptyDir` scratch (no `hostPath`/`/mnt/nvme`), and 4 GPUs
max. See [`../skills/cluster/SKILL.md`](../skills/cluster/SKILL.md).

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

3. **Two steps, then stop.** `max_steps = 2`, watch that rollouts complete and rewards are
   not all zero, then set it back.

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
