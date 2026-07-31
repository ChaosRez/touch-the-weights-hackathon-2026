# alien-api

A continual-learning environment. One fictional company, one reviewer, 240 questions in a
fixed order. Your agent answers them one at a time and gets a correction back when it is
wrong. Nothing else changes. If your agent is learning, its acceptance rate climbs across
the stream and its tool calls fall. If it is not, the curve is flat.

Everything runs on a laptop. Scoring is deterministic and needs no API key.

## The setup

**The world** is *kestrel*, a CRM/ERP back office: accounts, contacts, opportunities,
orders, invoices, products, inventory, tickets, plus an SOP wiki. You query it through
tools. The API has undocumented behaviours, and the wiki is plausible but partly wrong.
Working those out buys **efficiency**, and nobody will tell you what they are.

**The reviewer** is Margot Reinholt. Every question is ambiguous *by construction*: at
least two answers are defensible from the records, and she accepts exactly one of them.
Which one depends on conventions she holds and has never written down. Learning those buys
**acceptance**, and she teaches them, one correction at a time, only when you get one
wrong.

You need both. Neither axis alone gets you far: with world knowledge alone and reviewer
knowledge alone, the episodes each one solves are almost disjoint sets.

## The rule

**Do not read the labels into your agent's context.**

`src/alien_api_env/data/episodes/alien_api_v4.jsonl` contains, for every episode, the
accepted answer, every defensible alternative, and Margot's convention for each dimension
the episode invokes. It has to: the corrections you learn from are computed locally from
exactly those fields, which is what makes the whole thing run offline with no key.

So the answer key is on your disk, and reading it is both trivial and pointless. A run
that scores 1.0 by loading the JSONL demonstrates nothing and will be obvious in your
traces. Same for `trace.info["accepted"]`, which `finalize()` writes next to the feedback
your loop legitimately reads. Bank the corrections, not the labels.

What your agent may legitimately carry between episodes:
- Margot's correction sentences, verbatim. They are certified answer-free.
- Anything it observed itself: tool calls, results, errors, what worked.

## Quickstart

```bash
uv sync --extra dev --extra examples
uv run pytest tests/ -q          # 100 tests, offline, no key
```

Hydrate and score with no model in the loop:

```python
import asyncio
from alien_api_env.certify.traces import build_trace
from alien_api_env.vf import AlienApiTaskset, AlienApiTasksetConfig

async def main():
    # split="" is all 240 in seq_index order. The default is "train" (214).
    ts = AlienApiTaskset(AlienApiTasksetConfig(split=""))
    task = list(ts.load())[0]
    print(task.data.prompt)

    trace = build_trace(task, "some answer", tool_returns=[])
    await task.score(trace)
    await task.finalize(trace, None)          # note the second arg
    print(trace.reward, trace.info["feedback"])

asyncio.run(main())
```

Then a real rollout, which needs a key:

```bash
export OPENAI_API_KEY=<your key>
export ALIEN_API_MODEL=gpt-5.6-luna                # optional, this is the default
uv run python examples/responses_rollouts.py --n 10  # stateless baseline
uv run python examples/scratchpad_loop.py --n 30     # memory arm vs stateless, paired
```

**Use the Responses API.** gpt-5.x with reasoning plus tools over `/chat/completions`
returns zero tool calls and no error. The drivers here already do the right thing; if you
write your own, start from `examples/responses_rollouts.py`.

## Scoring

Binary. `reward` is 1.0 if the submitted answer exactly matches Margot's accepted label,
else 0.0. No model in the reward path, no judge, no partial credit.

Everything else is weight-0 observability, and it is where the interesting signal lives:

| metric | reads as |
|---|---|
| `preference_accepted` | same as reward: did Margot accept it |
| `value_correct` | did the answer carry *some* defensible value, i.e. did you work the data out |
| `preferences_violated` | how many invoked conventions were missed |
| `tool_calls` | exploration cost |
| `over_budget` | exceeded the episode's call budget (observability, not a penalty) |
| `artifact_tokens` | tokens pulled through tool returns |

The two-layer split is the diagnostic. `value_correct=1, reward=0` means you found a
defensible answer and presented it the wrong way, which is a conventions problem.
`value_correct=0` means you never got the number right, which is a world problem, and no
memory system will save you until you fix it.

## The outer-loop seam

After `finalize()`, `trace.info` holds:

| key | use |
|---|---|
| `feedback` | Margot's correction, in her voice. **This is your learning signal.** |
| `violated` | which preference dimensions were missed |
| `invoked` | which dimensions this episode tested |
| `world_traps` | which world behaviours this episode's solution path crossed |
| `value_defensible` | false means the answer carried no defensible value at all |
| `accepted` | the label. For offline analysis. Do not feed it to the model. |

When `value_defensible` is false, the feedback deliberately teaches nothing about
conventions, it just says the data is wrong. That is not a bug. It does mean a model that
cannot compute the aggregates learns Margot very slowly, because the teaching channel is
throttled behind getting the value right. If your `value_correct` is sitting below ~0.5,
raise reasoning effort or lower `artifact_verbosity` before concluding your memory design
is bad.

## What good looks like

Measured on all 240 episodes, gpt-5.6-luna, paired arms on the same episodes:

| | stateless | scratchpad memory |
|---|---|---|
| acceptance | 0.150 | **0.421** |
| mean tool calls | 14.6 | **8.3** |

2.8x acceptance with 43% fewer tool calls. The full write-up with learning curves and
per-dimension breakdowns is in [`reports/scratchpad_memory/report.md`](reports/scratchpad_memory/report.md),
and the loop that produced it is `examples/scratchpad_loop.py`.

Two reference points worth knowing. A **naive** scratchpad (free-form rewrite each
episode) produced *no gain at all*: it evicted taught rules before they were retested and
distilled generic process caution instead of conventions. What worked was mechanical, an
append-and-dedupe ledger of correction sentences that cannot decay, plus separate
model-written notes for world facts. And the ceiling: injecting both world knowledge and
Margot's conventions outright gets 0.55 acceptance at 8.4 calls. That is the target a real
learner should chase, and it is not 1.0.

## Config

`AlienApiTasksetConfig`, the whole surface:

| knob | default | note |
|---|---|---|
| `split` | `"train"` | `""` for all 240 in order, which is the CL setup |
| `dataset_path` | packaged fleet | a local path |
| `worlds_root` | packaged world | a local path |
| `artifact_verbosity` | `22000` | tool returns pad toward this; lower it to make exploration cheaper |

No seed, no version, no persona knob. The fleet is committed data.

## Layout

```
src/alien_api_env/     the environment: hydration, tools, scoring, feedback
  data/                the committed world + 240-episode fleet
  vf/                  verifiers-v1 taskset + MCP toolsets
  feedbacker/          Margot's profile and her voiced corrections
examples/              stateless driver, the paired memory loop, the report generator
training/              prime-rl config for RL on the cluster
skills/                agent-readable guides for this env and for the cluster
reports/               the measured outer-loop result
```

## If you use the GPU cluster

You run in an isolated `hackathon` workspace: you see only your own jobs, you have a hard
**4-GPU cap**, and your jobs are low-priority (preemptible on spare capacity). Three rules,
because every team shares one pool:

- **Submit jobs, do not create clusters.** `sky jobs launch`, never `sky launch -c <name>`.
  A named cluster holds its GPUs until someone runs `sky down`; a managed job gives them back
  when it finishes. Check with `sky status`, which should show nothing.
- **Single node, <= 4 GPUs.** Do not set `num_nodes`. A request over 4 GPUs is quota-rejected
  and sits Pending forever.
- **Use the required pod shape.** Every guest task needs the public `ml-hackathon` image
  (digest-pinned, no pull secret), `runAsUser: 0`, and an `emptyDir` scratch volume (no
  `hostPath` / `/mnt/nvme`). Miss any and the job fails at admission or in setup.

The training config here is single node and fits the 4-GPU cap. The exact shape, the failure
table, and a copy-paste task template are in [`skills/cluster/SKILL.md`](skills/cluster/SKILL.md).

## Pointers

- Writing an agent against this env: [`skills/alien-api-env/SKILL.md`](skills/alien-api-env/SKILL.md)
- Running on the GPU cluster: [`skills/cluster/SKILL.md`](skills/cluster/SKILL.md)
- RL training: [`training/README.md`](training/README.md)
- When something breaks: [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)

Licensed for use during this event. See [`LICENSE`](LICENSE).
