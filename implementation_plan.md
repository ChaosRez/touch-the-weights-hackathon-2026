# Implementation plan — agenda-aligned recurrent KV memory

## Current state and objective

Phases 1--3 are complete. Phase 4 was implemented and exercised through its declared cluster
smoke gates, then stopped before the 240-episode promotion because both neural arms violated the
predeclared runtime/submission criteria. Phases 1--3 establish two useful prerequisites:

1. The local Qwen agent can use the real Alien API tools and verifier.
2. A recurrence-aware Still compactor can retain synthetic facts through 100 updates while
   holding memory at 64 KV positions.

Those results validate the mechanism, but they are not yet the KV-compaction track result. The
track evaluates continual learning on the real 240-episode Alien API stream. The new primary
objective is therefore:

> Recurrently compact only the legal memory produced by prior Alien API rollouts into a fixed
> 64-position KV state, attach that state to Qwen3-4B on every later episode, and measure whether
> acceptance rises and tool calls fall relative to paired stateless and text-window controls.

The synthetic depth curve remains the controlled explanation for why recurrence-aware training
is necessary. The all-240 Alien API learning curve becomes the headline result.

The repository's published `0.421` scratchpad acceptance and `8.3` mean tool calls use
`gpt-5.6-luna`. They are the official track reference, but not a causal comparison to a Qwen3-4B
arm. The fair ablation keeps the same Qwen model, tools, episode order, seeds, generation settings,
and legal memory events across all arms; only the memory representation changes.

## Decisions after the organizer discussion

| Decision | Choice |
|---|---|
| Final experimental model | `Qwen/Qwen3-4B` |
| Existing integration result | Keep the completed Qwen3-8B Phase 1 result |
| Primary memory design | One global fixed-size recurrent KV state |
| Retrieval/RAG | Deferred; not part of the main comparison |
| Inference implementation | Hugging Face/PyTorch in-process |
| vLLM/Rust cache injection | Deferred until after the algorithmic result |
| Fixed memory budget | 64 KV positions |
| Incoming chunk size | Approximately 64 tokens |
| Train recurrence depths | `{1, 2, 4, 8}` |
| Evaluation depths | `{1, 2, 4, 8, 16, 32}` plus a small depth-100 probe |
| OpenAI synthetic generation | Not needed for the first result |

The equal-budget comparison is in **attention/KV positions**: the text baseline keeps the most
recent 64 token positions, while the neural arm always keeps 64 synthetic KV positions.

## Verified status

### Phase 0 — completed

- Existing persistent `team-4-box` reused.
- 240 episodes hydrated and the score/finalize path passed.
- Python 3.12.3 and image `verifiers==0.2.2.dev17` verified.
- Qwen3-8B loaded on one H100.
- Result: `/persist/cartridges/runs/phase_0.json`.

### Phase 1 — completed with Qwen3-8B

The uncommitted implementation contains:

- A reusable in-process `QwenToolAgent`.
- Native Qwen tool-call parsing, including bare follow-up calls.
- Real CRM, Wiki, and Answer tool dispatch.
- Real `vf.Trace` construction, scoring, and finalization.
- Retention of every assistant action and tool result for later training.
- Attachment interfaces for text, raw KV, and compact KV.

Cluster result from job 32:

| Metric | Result |
|---|---:|
| Cold episodes | 20 |
| Execution/parse errors | 0 |
| First-10 answer rate | 1.00 |
| Mean `value_correct` | 0.55 |
| Typed `submit_answer` episodes | 15/20 |
| Assistant turns retained | 60 |
| Tool executions retained | 56 |

Artifacts: `/persist/cartridges/runs/phase_1.json`.

### Phase 1 pre-commit verification

- Focused new tests: **18 passed**.
- Full Hackathon suite: **118 passed**.
- `git diff --check`: clean.
- No commit-blocking issue found in the Phase 1 code.

Commit the Phase 1 agent separately before recurrent-training work. The Phase 1 files are:

```text
hackathon/pyproject.toml
hackathon/src/cartridge_memory/
hackathon/examples/cartridge_loop.py
hackathon/tests/test_attachment_parity.py
hackathon/tests/test_cartridge_loop.py
hackathon/tests/test_cartridge_tool_runtime.py
hackathon/tests/test_qwen_agent.py
hackathon/tests/test_qwen_tool_parser.py
hackathon/training/cartridges_phase1.yaml
```

`hackathon/uv.lock` and `hackathon/skills/cluster/loops.md` are also untracked. Include them in
that commit only if intentional; they are not required to describe the Phase 1 implementation.

## Historical Phase 1--3 schedule

These checkpoints produced the completed results recorded below. They are retained as execution
history rather than as current deadlines.

| Deadline | Deliverable |
|---|---|
| T−105 min | Qwen3-4B download/cache started; synthetic generator assigned |
| T−80 min | One recurrent row passes forward/backward on a tiny or 4B model |
| T−60 min | Depth-1 checkpoint training started |
| T−35 min | Mixed-depth recurrent fine-tuning started |
| T−15 min | Evaluation frozen; plots and table generated |
| T−10 min | No more model/code changes; presentation only |

## Workstream A — Qwen3-4B agent comparison

Owner: teammate 1. Run in parallel with recurrent-compactor implementation.

The existing Phase 1 command already accepts `--model`; do not fork the agent implementation.

### A1. Cold 4B baseline

Run 20 cold episodes using `Qwen/Qwen3-4B`, the same seed, generation settings, and episode
order as the completed 8B run.

Output:

```text
/persist/cartridges/runs/phase_1_qwen3_4b_cold.json
```

This provides:

- A model-size comparison against the completed 8B result.
- A same-model baseline for any later text-memory or KV-memory Alien API run.

### A2. Optional minimal fixed-text ledger

Only implement this if A1 starts promptly and another teammate is not blocked.

Use a single global ledger, not retrieval:

- Append reviewer correction sentences verbatim and deduplicate.
- Append mechanical tool outcomes such as `tool(args) -> ok/ERROR:code`.
- Cap the rendered memory to the most recent 64 or 128 tokenizer positions.
- Pass it using the existing `TextAttachment` path.
- Run 20 sequential episodes.

Output:

```text
/persist/cartridges/runs/qwen3_4b_text64.json
```

This is a useful control but is secondary to the recurrent-retention plot. Do not add embeddings,
RAG, LLM-written world summaries, or a generalized memory package today.

Status: **completed on `team-4-box` (2026-08-01, jobs 51 and 52).** Both Qwen3-4B
arms ran the same 20 ordered episodes with seed 17 and the Phase 1 generation settings. Both
completed without execution or parse errors. The bounded-text arm used the existing
`TextAttachment` path; before each new episode it rendered the most recent 64 tokenizer
positions from one global, deduplicated ledger of verbatim rejection feedback and mechanical
tool outcomes. Its rendered history was 0 tokens for episode 0, 61 for episode 1, and exactly
64 for episodes 2--19.

| Metric | Cold 8B (job 32) | Cold 4B (job 51) | Text64 4B (job 52) |
|---|---:|---:|---:|
| Completed episodes | 20/20 | 20/20 | 20/20 |
| Execution/parse errors | 0 | 0 | 0 |
| First-10 answer rate | 1.00 | 1.00 | 1.00 |
| Mean `value_correct` | 0.55 | 0.50 | 0.50 |
| Mean `preference_accepted` | 0.20 | 0.20 | 0.00 |
| Typed `submit_answer` episodes | 15/20 | 11/20 | 1/20 |
| Mean tool calls | 2.05 | 0.75 | 0.85 |

The minimal ledger is therefore a valid equal-budget control, but it did not improve value
accuracy in this 20-episode run and substantially reduced typed submissions and preference
acceptance. Treat this as a negative result, not evidence that fixed text memory helped. The
final ledger contained 21 deduplicated entries and rendered to the full 64-token budget.

Artifacts:

```text
/persist/cartridges/runs/phase_1_qwen3_4b_cold.json
/persist/cartridges/runs/qwen3_4b_text64.json
```

Local verification after adding the ledger: **121 passed**, repository-wide Ruff clean,
generated YAML shell syntax valid, and `git diff --check` clean. The persistent cluster was
reused and left running.

## Workstream B — synthetic repeated-compaction dataset

Owner: teammate 2.

Add:

```text
still/src/still/data/recurrent_synthetic.py
```

Each example contains multiple short chunks, a query, and a one-token MCQ answer:

```python
{
    "chunks_input_ids": [[...], [...], ...],
    "query_input_ids": [...],
    "answer_input_ids": [...],
    "depth": 8,
    "target_chunk": 0,
    "target_age": "oldest",
}
```

Example:

```text
Chunk 0: The access code for Project RAVEN is 7314.
Chunk 1: The access code for Project LYNX is 2849.
...
Query: What is the access code for Project RAVEN? A. 7314 B. 9921 C. 6103 D. 2849
Answer: A
```

Dataset requirements:

- Deterministic train and evaluation seeds.
- Unique project names and values within every example.
- Correct answer position randomized across A/B/C/D.
- 50% of questions target chunk 0.
- 25% target a middle chunk.
- 25% target the newest chunk.
- Training depths sampled from `{1, 2, 4, 8}`.
- Evaluation depths include `{1, 2, 4, 8, 16, 32}`.
- Depth 100 uses a small number of examples to bound runtime.
- No Alien API prompts, feedback, accepted labels, or metadata are used.

OpenAI is unnecessary for this first dataset. Template-generated facts make the retention
signal deterministic and auditable.

## Workstream C — differentiable recurrent compaction

Owner: teammate 3.

The current inference recurrence in
`still/src/still/model/wrapper.py::recompact` is under `@torch.no_grad()`. Preserve that method
and add a separate training method:

```python
def recompact_train(
    self,
    cache: CompactCache,
    new_token_ids,
    *,
    detach_prior: bool = True,
) -> CompactCache:
    ...
```

Implementation:

1. Capture raw K/V for the new chunk with the frozen base model under `no_grad`.
2. Concatenate previous compact K/V with the new raw K/V per layer.
3. Detach the previous state for the bounded-memory training mode.
4. Run the Perceiver with gradients.
5. Return exactly 64 compact positions.

For depths greater than one, use a last-step-gradient approximation:

```python
with torch.no_grad():
    state = model.compact_tokens(chunks[0])
    for chunk in chunks[1:-1]:
        state = model.recompact(state, chunk)

state = model.recompact_train(state, chunks[-1], detach_prior=True)
```

Depth-1 examples use the existing differentiable `compress()` path. Sampling recurrent depth
on-policy exposes the trainable final step to drifted states produced by the current compactor
without retaining a 100-step autograd graph.

Required test:

- Ten recurrence calls always return exactly 64 positions.
- Perceiver parameters receive nonzero gradients.
- Frozen base-model parameters receive no gradients.
- Outputs and loss stay finite.

Status: **completed and revalidated on 2026-08-01.** The inference-only `recompact`
method remains under `@torch.no_grad()`, while `recompact_train` captures new raw K/V with
the frozen base under `no_grad`, optionally detaches prior compact K/V, concatenates prior and
new states per layer, and runs only the Perceiver differentiably. Depth-1 training uses
`compress`; deeper rows construct on-policy recurrence states without a retained history graph
and differentiate only the final recompact step. Focused tests cover ten fixed-64 recurrences,
finite K/V/bias/logits/loss, gradients on every Perceiver parameter, no base-model gradients,
and explicit prior-state detachment. This path produced both Phase 2 checkpoints in job 49.

## Phase 2 — train the two comparison checkpoints

Add:

```text
still/src/still/train_recurrent.py
```

Use `Qwen/Qwen3-4B`, BF16, and one H100.

### Checkpoint 1: single-step

Train only depth-1 examples:

```text
/persist/cartridges/checkpoints/qwen3_4b_single_step.pt
```

Initial target: 50–100 steps. Stop earlier if loss clearly falls and time is tight.

### Checkpoint 2: recurrence-aware

Resume the single-step checkpoint and train on mixed depths:

```text
depth probabilities:
1: 10%
2: 20%
4: 30%
8: 40%
```

Output:

```text
/persist/cartridges/checkpoints/qwen3_4b_recurrent.pt
```

Initial target: 75–150 steps, bounded by T−15 minutes.

Shared settings:

```text
num_latents: 64
latent_dim: 256
num_blocks: 2
dtype: bfloat16
optimizer: AdamW
learning_rate: 4e-5
answer length: one MCQ token where possible
base model: frozen
```

Log every step:

- Depth and target age.
- KL loss.
- Teacher and student answer CE.
- Perceiver gradient norm.
- Step time.
- GPU memory.

Do not implement the paper's full RoPE/identity-initialization fixes unless the depth-1 model
cannot learn at all. Record their absence as a limitation.

Status: **completed on `team-4-box` (2026-08-01, job 49).** Qwen3-4B trained on one H100
with a fixed budget of 64 KV positions. The deterministic single-step stage ran 50 depth-1
steps; endpoint KL moved from `0.01768` to `0.00219`, and mean KL moved from `0.01292`
over its first 10 steps to `0.00115` over its last 10. The recurrence-aware stage resumed
that checkpoint for 75 steps with exact depth counts `{1: 8, 2: 15, 4: 22, 8: 30}`;
endpoint KL moved from `0.00869` to `0.00505`, and first-10/last-10 mean KL moved from
`0.19546` to `0.01274`. Every step had finite loss and a nonzero Perceiver gradient, the
frozen base received no gradients, and peak allocated GPU memory was `9.98 GB`.

Artifacts:

```text
/persist/cartridges/checkpoints/qwen3_4b_single_step.pt  # 492 MB
/persist/cartridges/checkpoints/qwen3_4b_recurrent.pt    # 492 MB
/persist/cartridges/runs/phase_2_training.jsonl
```

The implementation uses deterministic synthetic facts, an exact 50/25/25 target-age schedule,
seeded model initialization, and a last-step-gradient `recompact_train` path. The paper's RoPE,
final-RMSNorm, and identity-initialization fixes remain intentionally absent as documented v1
limitations. The existing shared cluster was reused and left running.

## Phase 3 — evaluation and presentation artifacts

Add:

```text
still/src/still/eval_recurrent.py
```

Evaluate four arms:

1. Full-context Qwen oracle; context grows with depth.
2. Text-window baseline; only the most recent 64 source-token positions remain.
3. Single-step checkpoint applied recurrently with a fixed 64-position cache.
4. Recurrence-aware checkpoint with the same fixed 64-position cache.

Evaluate separately by target age:

- Oldest fact.
- Middle fact.
- Newest fact.
- Overall.

Primary artifact:

```text
X: compaction depth (log scale)
Y: oldest-fact MCQ accuracy
Lines: full context, text64, single-step Still, recurrent Still
```

Required result table:

| Method | KV/text positions | Depth 1 | Depth 8 | Depth 32 | Depth 100 |
|---|---:|---:|---:|---:|---:|
| Full context | grows | | | | |
| Text window | 64 | | | | |
| Single-step Still | 64 | | | | |
| Recurrent Still | 64 | | | | |

Also include:

- Source tokens represented at each depth.
- Effective source-token/slot ratio.
- KL or CE gap to full-context teacher.
- Compaction time per chunk.

Persist raw metrics and plots under:

```text
/persist/cartridges/metrics/
/persist/cartridges/plots/
```

Status: **completed on `team-4-box` (2026-08-01, smoke job 58 and final job 59).**
The final evaluation used Qwen3-4B in bfloat16, the held-out deterministic seed 2903,
eight examples per depth, depths `{1, 2, 4, 8, 16, 32, 100}`, 64-token source chunks,
and an equal fixed budget of 64 positions for text, single-step Still, and recurrent Still.
Each depth has four oldest, two middle, and two newest targets. All 224 method/example
records were produced and validated.

Overall MCQ accuracy:

| Method | KV/text positions | Depth 1 | Depth 8 | Depth 32 | Depth 100 |
|---|---:|---:|---:|---:|---:|
| Full context | grows | 1.000 | 1.000 | 1.000 | 1.000 |
| Text window | 64 | 1.000 | 0.375 | 0.250 | 0.375 |
| Single-step Still | 64 | 1.000 | 1.000 | 0.875 | 1.000 |
| Recurrent Still | 64 | 1.000 | 1.000 | 1.000 | 1.000 |

Oldest-fact MCQ accuracy:

| Method | KV/text positions | Depth 1 | Depth 8 | Depth 32 | Depth 100 |
|---|---:|---:|---:|---:|---:|
| Full context | grows | 1.000 | 1.000 | 1.000 | 1.000 |
| Text window | 64 | 1.000 | 0.000 | 0.000 | 0.000 |
| Single-step Still | 64 | 1.000 | 1.000 | 1.000 | 1.000 |
| Recurrent Still | 64 | 1.000 | 1.000 | 1.000 | 1.000 |

The primary accuracy curve alone understates the recurrence-aware gain because the one-token
classification remains correct after substantial probability drift. At depth 100, the
single-step checkpoint had a mean answer-CE gap of `0.33062` and forward KL of `0.33279`
to the full-context teacher. The recurrent checkpoint reduced those to `0.00014` and
`0.02093`, respectively. Mean compaction time at depth 100 was `51.77 ms/chunk` for the
single-step checkpoint and `44.60 ms/chunk` for the recurrent checkpoint.

The source-token/slot ratio is `1x`, `8x`, `32x`, and `100x` at the four table depths.
Thus the depth-100 learned arms represent 6,400 source tokens using 64 fixed KV positions;
the text arm exposes only its most recent 64 source tokens, while full context grows to
6,400 positions. Recurrent Still was correct on all 56 held-out examples and matched the
full-context oracle at every evaluated depth. Single-step Still missed one oldest example
at depth 4 and one example each at depths 16 and 32, with substantially larger CE/KL drift.

Versioned persistent artifacts (the writer refuses collisions and never overwrites earlier
result plots):

```text
/persist/cartridges/metrics/phase_3_qwen3_4b_fixed64_v1/
/persist/cartridges/plots/phase_3_qwen3_4b_fixed64_v1/
/persist/cartridges/runs/phase_3_qwen3_4b_fixed64_v1.jsonl
```

The validated metrics, CSV, Markdown table, and three PNG plots were also pulled to
`hackathon/reports/phase3/phase_3_qwen3_4b_fixed64_v1/`. Existing plots under
`hackathon/reports/scratchpad_memory/` and the versioned Phase 3 smoke artifacts were
left unchanged. Local verification after Phase 3: **30 Still tests** and **121 Hackathon
tests** passed; Ruff, generated YAML shell syntax, and `git diff --check` were clean.
The persistent cluster was reused and left running.

## Phase 4 — track-compliant Alien API KV memory

### 4.0 Experimental contract

Run `split=""` so all 240 episodes execute in `seq_index` order. The neural arms must be
strictly sequential: episode *i* is scored and finalized, its legal memory events update the
state, and only then may episode *i + 1* start.

The memory writer API accepts explicit `feedback` and `tool_executions` arguments rather than a
whole result record. It may consume only:

- `trace.info["feedback"]`, keeping useful correction sentences verbatim.
- The agent's own tool name, arguments, returned success/error code, and error message.

It must never consume the accepted label, `trace.info["accepted"]`, reward, metric values,
`invoked`, `violated`, `world_traps`, `value_defensible`, the episode answer, or any data loaded
from the fleet JSONL. Those fields may be used only after the run for reporting. Exclude the
`submit_answer` call from tool memory because its arguments are the agent's answer, not an
observation.

Use one deterministic serializer for every memory arm:

```text
REVIEWER: <verbatim correction sentence>
TOOL: <name>(<canonical JSON arguments>) -> ok
TOOL: <name>(<canonical JSON arguments>) -> ERROR:<code>:<message>
```

Ignore `Accepted.` and the contentless "does not follow from the records" rejection. Deduplicate
correction and tool events with content hashes; hashes may control repeated writes but are never
decoded or exposed to the model. No LLM-written summary is used in the primary comparison, so an
OpenAI call cannot become a hidden difference between arms.

### 4.1 Attach the trained Still cache to the Qwen tool agent

Add a `StillQwenBackend` beside `HuggingFaceQwenBackend` and reuse the existing `QwenToolAgent`.
It must:

1. Load `Qwen/Qwen3-4B`, `STILLConfig(num_latents=64, latent_dim=256, num_blocks=2)`, and either
   the single-step or recurrent checkpoint.
2. Render the current episode's system message, live tool conversation, and tool schemas through
   the same Qwen chat template as the cold backend.
3. Pass only those live tokens plus the optional `CompactKVAttachment` to
   `STILLModel.decode_generate`.
4. Reuse the same prior-episode attachment on every tool turn within an episode. Current-episode
   tool results stay live; they enter persistent memory only after `score()` and `finalize()`.
5. Match the cold arm's sampling parameters and per-episode seed exactly.

Load the Phase 2 checkpoint's nested `perceiver` state and validate its recorded latent/model
configuration before generation. Do not route through vLLM or the Still HTTP server for this
experiment; in-process inference preserves tool schemas and makes the attached state explicit.

### 4.2 Recurrent fixed-budget memory state

Add a `RecurrentKVLedger` with this update rule:

```python
for event in legal_events(feedback, tool_executions):
    for chunk in chunks(tokenizer.encode(event), size=64):
        state = model.compact_tokens(chunk) if state is None else model.recompact(state, chunk)
```

After the first event, the state must always contain exactly 64 positions at every layer,
regardless of source tokens or episode count. Track but do not feed these observability fields:

- Total legal source tokens represented.
- Number of recurrent compaction updates.
- Current memory positions.
- Compaction time per episode.
- SHA-256 hashes of ingested events for deduplication and audit.

Checkpoint records and the compact tensors after every episode under `/persist`; resume must
restore the exact next episode, cache, counters, and hash set. The loop must never rebuild the
cache from stored result records during a normal run.

Replace the existing text control's all-history `entries` list with a streaming 64-token window
for the final experiment. This prevents the control from retaining a semantic copy of evicted
text outside its declared memory budget.

### 4.3 Paired arms

Run the same 240 episodes with Qwen3-4B in each arm:

| Arm | Persistent model-visible memory | Purpose |
|---|---:|---|
| Cold | 0 positions | Same-model stateless baseline |
| Text64 | Most recent 64 text tokens | Equal-position non-neural control |
| Still64 single-step | 64 recurrent KV positions | Measures recurrence distribution shift |
| Still64 recurrence-aware | 64 recurrent KV positions | Primary method |
| Full legal text, optional | Grows | Same-model information upper bound; not equal budget |

The official GPT-5.6 scratchpad result remains a clearly labeled external reference line. Do not
describe a Qwen-vs-GPT difference as a memory improvement.

First run episodes 0--29 as a mechanical smoke. Continue to all 240 unless there are parse/runtime
errors, non-finite caches, a position-count violation, forbidden-field ingestion, or a collapse in
typed answer submission. Do not stop merely because early acceptance is flat: useful corrections
and retests accumulate over the stream.

Use one GPU per arm inside the existing team box. Memory arms are sequential internally, but the
arms can run concurrently. Use unique output and checkpoint paths; do not create another SkyPilot
cluster.

### 4.4 Required report

Extend the existing scratchpad reporting logic for the Qwen arms. Produce:

- Overall and rolling-window `preference_accepted` for all 240 episodes.
- Overall and rolling-window `value_correct` and typed-submission rate.
- Mean tool calls and tool calls by stream quarter.
- Teach-then-retest acceptance, with teaching/retest tags joined only during offline reporting.
- Per-preference and per-world-trap breakdowns, also joined only offline.
- Neural source tokens represented, recurrence count, fixed positions, compression ratio, and
  compaction time.
- Paired acceptance deltas with bootstrap confidence intervals against Cold and Text64.

Primary table:

| Method | Acceptance | Final-60 acceptance | Value correct | Mean tool calls | Memory positions |
|---|---:|---:|---:|---:|---:|
| Cold | | | | | 0 |
| Text64 | | | | | 64 |
| Still64 single-step | | | | | 64 |
| Still64 recurrence-aware | | | | | 64 |

Primary plot: rolling acceptance versus episode index. Secondary plots: rolling tool calls and the
already-completed synthetic oldest-fact accuracy versus recurrence depth.

The minimum positive claim is not "infinite memory." It is:

> With Qwen3-4B and the same 64-position budget, recurrent neural memory improves acceptance or
> efficiency over both a cold agent and a 64-token text window on the real continual-learning
> stream.

If that claim is unsupported, report the result as a domain-transfer failure: the compactor
retained template facts at depth 100 but did not turn real reviewer corrections and tool outcomes
into usable agent behavior. That is a valid result and directly motivates task-shaped
distillation.

### 4.5 Verification gates

Before the 240-episode run:

- Unit test that a compact attachment reaches the registered Still attention path.
- Unit test that 100 event updates remain exactly 64 positions with finite K/V and bias tensors.
- Unit test that the legal-event serializer cannot receive a whole result record, preserves
  correction text verbatim, and excludes `submit_answer`.
- Parity test that cold generation through `StillQwenBackend(cache=None)` renders the same live
  chat template and generation arguments as the ordinary Qwen backend.
- Resume test that uninterrupted and checkpoint-resumed event streams produce identical cache
  tensors and counters.
- A 3-episode real smoke with at least one tool result, one scored answer, and no trace error.

After the run, execute the full Hackathon and Still test suites, generate the report from immutable
result files, pull every `/persist/cartridges/phase4/` artifact locally, and only then release the
cluster.

### 4.6 Three-person split

| Teammate | Primary responsibility | Handoff |
|---|---|---|
| 1 | `StillQwenBackend`, checkpoint loading, attachment parity | One real episode decoded with a 64-position cache |
| 2 | Legal event serializer, `RecurrentKVLedger`, checkpoint/resume | Audited fixed-size state after 100 updates |
| 3 | Multi-arm runner, cluster jobs, report and plots | Paired 30-episode smoke, then immutable 240-episode artifacts |

Once the smoke gates pass, teammates 1 and 2 review traces for cache use and leakage while
teammate 3 launches the full arms. Do not split model behavior or tool parsing into separate forks;
all conditions must continue to use the same agent implementation.

### Phase 4 verified status — stopped at the promotion gate

Status: **implementation complete; 240-episode promotion intentionally not launched
(2026-08-01).** The legal-event boundary, streaming Text64 ledger, recurrent fixed-size KV
ledger, `StillQwenBackend`, atomic per-episode checkpoint/resume, immutable result writer,
paired bootstrap report, and versioned plots are implemented. The offline report recomputes every
legal-event hash from the explicit feedback/tool API before joining fleet-only preference and trap
tags. It rejects a result if those hashes do not exactly match the arm's declared memory events.

All four real three-episode jobs (68--71) passed the required gates: scored answers, at least one
non-answer tool result, no trace errors, finite caches, and exactly 64 neural memory positions.
The 30-episode promotion smoke then produced:

| Arm | Completed | Acceptance | Value correct | Typed submissions | Mean tool calls | Gate |
|---|---:|---:|---:|---:|---:|---|
| Cold | 30/30 | 0.167 | 0.467 | 15/30 | 0.667 | pass |
| Text64 | 30/30 | 0.167 | 0.467 | 10/30 | 0.667 | pass |
| Still64 single-step | 30/30 | 0.000 | 0.033 | 1/30 | 0.033 | **fail: typed submission collapse** |
| Still64 recurrence-aware | 8/30 | 0.000 | 0.125 | 1/8 | 0.250 | **fail: malformed cached decode at episode 8** |

The recurrence-aware failure was replayed from the exact episode-8 checkpoint (job 76) and
reproduced the same three parse failures. The first failing rollout is frozen as
`failure_000008.json`. Chat-template parity, registered Still attention, cache finiteness, and the
64-position invariant all passed. The frozen raw output shows the remaining failure is cached
decoding behavior: it begins a `submit_answer` tool call, fails to close it, and repeats
schema-like closing tags to the generation limit. The shared parser and prompt were not changed
for a neural arm.

The fair common-prefix report therefore uses episodes 0--7 from all four immutable arm results:

| Method | Acceptance | Value correct | Typed-submission rate | Mean tool calls | Positions |
|---|---:|---:|---:|---:|---:|
| Cold | 0.125 | 0.625 | 0.375 | 1.000 | 0 |
| Text64 | 0.250 | 0.500 | 0.375 | 0.750 | 64 |
| Still64 single-step | 0.000 | 0.125 | 0.125 | 0.125 | 64 |
| Still64 recurrence-aware | 0.000 | 0.125 | 0.125 | 0.250 | 64 |

The recurrence-aware paired acceptance delta was `-0.125` versus Cold (95% paired bootstrap CI
`[-0.375, 0.000]`) and `-0.250` versus Text64 (`[-0.625, 0.000]`). Lower neural tool counts are
not an efficiency win because they coincide with suppressed tool/answer submission and worse
acceptance. The predeclared positive claim is unsupported; this is the planned domain-transfer
failure outcome, motivating task-shaped distillation from legal feedback and observed tool events.

Persistent and pulled artifacts:

```text
/persist/cartridges/phase4/qwen3_4b_seed17_smoke3_v1/
/persist/cartridges/phase4/qwen3_4b_seed17_v1/
hackathon/reports/phase4/qwen3_4b_seed17_smoke3_v1/
hackathon/reports/phase4/qwen3_4b_seed17_v1/
```

The paired report is under `report_paired_008/`; raw per-arm records and tensor states are locally
ignored because they contain prompts, answers, and traces. Existing scratchpad and Phase 3 plots,
including `oldest_accuracy_vs_depth.png`, were left unchanged. Final local verification: **137
Hackathon tests** and **30 Still tests** passed, Ruff and `git diff --check` were clean. The existing
`team-4-box` remains up with `/persist` intact; no second cluster was created and no 240-episode
jobs were submitted after the gate failure.

## Cluster scheduling

Do not launch a second SkyPilot cluster. Use additional GPUs only inside the team's existing
persistent box.

First check how many GPUs are actually visible. If the existing box still exposes four H100s,
do not tear it down to resize: `/persist` is tied to the box and the resize/relaunch risk is not
worth it before the presentation. If six GPUs are already provisioned, use all six.

Suggested allocation:

| GPU | Work |
|---:|---|
| 0 | Qwen3-4B Cold, then Full legal text if time permits |
| 1 | Qwen3-4B Text64 |
| 2 | Qwen3-4B Still64 single-step |
| 3 | Qwen3-4B Still64 recurrence-aware |
| 4–5, if already available | Report smoke or a repeated-seed robustness run |

Use unique output paths and avoid running two jobs that write the same checkpoint. Model downloads
must use the shared `/persist/hf` cache.

## Phase 4 fallbacks

### If fewer than four GPUs are visible

Run Cold and recurrence-aware Still64 first, then Text64, then single-step. Compare only arms with
the same completed episode prefix; never compare 240 episodes in one arm to 30 in another.

### If compact-cache generation breaks tool calling

Freeze the first failing trace and verify chat-template parity, registered-attention invocation,
cache position count, and output decoding in that order. Do not change the common tool parser or
prompt for only one arm.

### If the current checkpoint shows no real-stream gain

Keep the result. Report synthetic retention success alongside real-domain transfer failure. The
next experiment is task-shaped teacher/student distillation from legal feedback and observed tool
events; do not silently add labels or switch to RAG.

### If a full arm is interrupted

Resume from its per-episode checkpoint. If resume is not bitwise-equivalent in the verification
test, restart that arm rather than reconstructing neural state from the result JSON.

## Explicitly deferred

- Embedding retrieval and RAG.
- Per-item swappable cartridges.
- vLLM or Rust external KV-cache injection.
- Full backpropagation through 100 compaction cycles.
- Hyperbolic embeddings.
- Online Perceiver updates inside the Alien API stream.
- The episode-180 radical-exploration policy.

The exploration plateau is orthogonal: a memory system cannot store knowledge that the policy
never discovers. Treat novelty-triggered exploration as follow-up work, not part of the primary
compaction comparison.

## Presentation narrative

1. **Track task:** Alien API rewards applying lessons from earlier rollouts; the published
   scratchpad reaches `0.421` acceptance versus `0.150` stateless with GPT-5.6.
2. **Agent foundation:** local Qwen completed real tool-using Alien API episodes without
   parse/runtime errors, so value computation and memory representation can be separated.
3. **Compaction blocker:** a single-step compactor sees its own synthetic K/V after the first
   recurrence, creating a state-distribution shift.
4. **Mechanism result:** recurrence-aware Qwen3-4B training retained held-out template facts
   through 100 updates using exactly 64 KV positions and sharply reduced depth-100 KL drift.
5. **Track result:** compare Cold, Text64, single-step Still64, and recurrence-aware Still64 on
   the same 240 ordered Alien API episodes using only legal prior-rollout memory.
6. **Conclusion:** report the real learning curve, value/tool-call diagnostics, and any gap
   between controlled retention and useful continual-learning behavior.

The ideal claim is:

> Recurrence-aware Still lets Qwen3-4B turn more prior-rollout feedback into accepted answers than
> cold inference or a 64-token text window while holding neural memory at 64 KV positions.

If the intervention does not improve the curve, the result is still useful:

> Recurrence-aware Still retains controlled facts for 100 updates, but that ability does not
> transfer into improved Alien API behavior, identifying task-shaped distillation—not additional
> synthetic depth—as the next bottleneck.
