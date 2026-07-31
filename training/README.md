# RL training on alien-api

Two files: `rl.toml` (what to train) and `run.yaml` (how to get it onto the cluster).
Cluster access itself is in [`../skills/cluster/SKILL.md`](../skills/cluster/SKILL.md).

## Read this before you launch

**Verified locally**, against `verifiers 0.2.2.dev36`:

- the env id `alien-api` resolves to module `alien_api` and class `AlienApiTaskset`
- with no `Env` subclass exported, it runs under the builtin `SingleAgentEnv`
- every knob in the `.taskset` block is a real field on `AlienApiTasksetConfig`
- the committed fleet hydrates offline, 240 episodes, no credentials

**Not verified**: none of this has been run against the training image. The trainer and
inference sections are the standard single-node shape, not a config proven on that stack.
Budget time for the smoke below and do not schedule a long run before it passes.

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

2. **On a dev box, before any training.** Launch one GPU with the training image, no
   `run:` block, ssh in, and run the preflight from `run.yaml`'s `setup:` by hand. This
   catches the two things most likely to be wrong: the image's bundled `verifiers` build
   differing from the pin here, and the env failing to import into `/app/.venv`.

3. **Two steps, then stop.** `max_steps = 2`, watch that rollouts complete and rewards are
   not all zero, then set it back.

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
