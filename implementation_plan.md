# Revised implementation plan — fixed-budget recurrent KV memory

## Objective and cut line

There are fewer than two hours before the presentation. The goal is no longer to build the
complete retrieval-based Cartridges roadmap. The goal is to leave with two concrete results:

1. A real Alien API result showing that the local Qwen tool-agent integration works.
2. A controlled fixed-budget experiment showing how recurrence-aware training affects factual
   retention over repeated KV-cache compactions.

The main research question is:

> A single-step compactor is trained on raw model K/V but receives its own synthetic K/V after
> the first recurrence. Does training on those recurrent states prevent the depth-2 collapse and
> preserve early facts for more compaction cycles?

The minimum presentation-quality result is an accuracy-versus-compaction-depth plot. Full
240-episode compact-cache integration is explicitly deferred.

## Decisions after the organizer discussion

| Decision | Choice |
|---|---|
| Final experimental model | `Qwen/Qwen3-4B` |
| Existing integration result | Keep the completed Qwen3-8B Phase 1 result |
| Memory design for today | One global fixed-size recurrent KV state |
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

## Remaining schedule

Use deadline-relative checkpoints. When a checkpoint is missed, follow the stated fallback
instead of expanding scope.

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

## Cluster scheduling

Do not launch a second SkyPilot cluster. Use additional GPUs only inside the team's existing
persistent box.

First check how many GPUs are actually visible. If the existing box still exposes four H100s,
do not tear it down to resize: `/persist` is tied to the box and the resize/relaunch risk is not
worth it before the presentation. If six GPUs are already provisioned, use all six.

Suggested allocation:

| GPU | Work |
|---:|---|
| 0 | Qwen3-4B cold Alien API run |
| 1 | Recurrent code smoke, then single-step/recurrent training |
| 2 | Dataset validation and evaluation smoke |
| 3 | Text-memory Alien API control or independent evaluation |
| 4–5, if already available | Latent-count ablation or parallel final evaluation |

Use unique output paths and avoid running two jobs that write the same checkpoint. Model downloads
must use the shared `/persist/hf` cache.

## Hard fallbacks

### If Qwen3-4B does not produce an agent baseline by T−60

Use the verified 8B Phase 1 numbers in the presentation and stop the 4B run. Do not debug the
agent; it is already proven.

### If recurrent training does not start by T−45

Evaluate the single-step checkpoint recurrently and show the failure curve against full context
and text64. A carefully measured collapse is still a valid result and directly motivates the
recurrence-aware dataset.

### If depth-1 training cannot learn

Run a 10-step Qwen3-0.6B smoke to validate plumbing. Present the 4B failure as an implementation
limitation caused by the incomplete Still architecture; do not spend the remaining time porting
all paper fixes.

### If no learned checkpoint is usable by T−20

Freeze code. Present:

- The verified 8B Alien API agent result.
- The fixed-budget benchmark and baselines.
- Untrained/single-step degradation by recurrence depth, if available.
- The implemented recurrence-training path and its first loss/gradient evidence.

## Explicitly deferred

- Embedding retrieval and RAG.
- Per-item swappable cartridges.
- Full 240-episode compact-cache evaluation.
- vLLM or Rust external KV-cache injection.
- Full backpropagation through 100 compaction cycles.
- Hyperbolic embeddings.
- Online Perceiver updates inside the Alien API stream.
- The episode-180 radical-exploration policy.

The exploration plateau is orthogonal: a memory system cannot store knowledge that the policy
never discovers. Treat novelty-triggered exploration as follow-up work, not part of today's
compaction comparison.

## Presentation narrative

1. **Agent foundation:** local Qwen3-8B completed 20 real tool-using Alien API episodes with no
   parse/runtime errors and `value_correct=0.55`.
2. **Observed blocker:** one-step Still consumes raw model K/V during training but synthetic
   Perceiver-produced K/V after the first recurrence, causing a state-distribution shift.
3. **Intervention:** generate training examples at randomized recurrence depths and train the
   compactor on its own drifted states while holding the cache at 64 positions.
4. **Comparison:** full context versus equal-position text window versus single-step and
   recurrence-aware neural compaction.
5. **Conclusion:** report the retention-depth frontier honestly, including failure depth and the
   cost of extending it.

The ideal claim is:

> Recurrence-aware training moves the fixed-budget retention cliff to greater compaction depth
> than a single-step compactor, especially for facts introduced in the first chunk.

If the intervention does not improve the curve, the result is still useful:

> Repeated KV synthesis introduces a severe state-distribution shift that mixed-depth last-step
> training alone does not solve, identifying the need for full recurrent training or stronger
> architectural fixes.
