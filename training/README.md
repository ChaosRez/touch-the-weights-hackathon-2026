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

**Known blocker (env side, being fixed by the organizers):** the `crm`/`wiki`/`answer`
toolset MCP servers time out on `session.initialize()` under prime-rl's rollout harness, so
rollouts currently fail with `HarnessError: harness 'bash' exited 1`. Ask an organizer whether
this is resolved before scheduling a real run.

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

2. **A preflight job, before any training.** Not a dev box: submit `run.yaml` with the
   `run:` block replaced by `echo preflight only`. The `setup:` block already installs the
   env and asserts the id resolves, the fleet hydrates to 240 episodes, and the image's
   `verifiers` build matches the pin. It costs one node for under a minute and catches the
   two things most likely to be wrong.

   ```bash
   sky jobs launch -y -n alien-preflight run.yaml
   sky jobs logs alien-preflight
   ```

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
