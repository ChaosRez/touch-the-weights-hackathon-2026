# Examples

Three scripts. Read `responses_rollouts.py` first: the other two build on it.

```bash
export OPENAI_API_KEY=<your key>
export ALIEN_API_MODEL=gpt-5.6      # optional, this is the default
```

## `responses_rollouts.py` — the stateless baseline

Runs episodes cold, nothing carried between them. This is also the machinery the memory
loop imports (`run_rollout`, `create_with_retries`, `summarize`), so it is the file to
copy from when you write your own agent.

```bash
uv run python examples/responses_rollouts.py --n 10 --out baseline.json
```

`run_rollout()` builds the three toolsets, hands their schemas to the model, dispatches
tool calls in-process, assembles a real verifiers `Trace`, and scores it with the real
`task.score()`. It uses the **Responses API** deliberately: gpt-5.x with reasoning plus
tools over `/chat/completions` returns zero tool calls, silently.

## `scratchpad_loop.py` — the memory arm, paired against stateless

The reference outer loop, and the one that produced the numbers in `../reports/`. Runs the
same episodes twice, once carrying a scratchpad and once cold, so the difference is the
memory system rather than the episode mix.

```bash
uv run python examples/scratchpad_loop.py --n 30 --out my_run.json
```

The scratchpad has two sections, and the split is the whole point:

- **Corrections ledger**, maintained mechanically by the loop. Margot's correction
  sentences, appended verbatim and deduplicated. The model never rewrites this. Taught
  rules cannot decay.
- **Operational notes**, maintained by the model from a digest of its own tool calls and
  outcomes. Concrete observed facts only.

The naive version of this (one free-form pad the model rewrites each episode) scored
*worse than stateless*: it evicted taught rules before they were retested and filled up
with generic process caution that drove tool calls up. The mechanical ledger is the fix.

Cost: 240 episodes across both arms is roughly 25M input tokens. Use `--n 30` while
iterating.

## `scratchpad_report.py` — plots and metrics

```bash
uv run --with matplotlib python examples/scratchpad_report.py \
    --results my_run.json --out reports/my_run
```

Learning curve, tool calls, teach-then-retest, and per-dimension and per-trap breakdowns,
plus a `metrics.json` with every number.

## Do not commit your results files

They embed fleet prompts and the accepted labels. `.gitignore` covers the obvious names;
do not work around it.
