I’ll use a compact flow diagram here because the two nested loops are much clearer visually than in prose.

The **outer loop** learns across episodes. The **inner loop** lets the model use tools within one episode.

```mermaid
flowchart TD
  subgraph Outer["Outer loop — memory arm: one sequential pass over tasks"]
    start["Load tasks in fixed order"]
    pad["Compose scratchpad\ncorrection ledger + operational notes"]
    prompt["Prefix current task with scratchpad"]
    run["Run one rollout"]
    update_ledger["Append + dedupe eligible\nreviewer corrections"]
    update_notes["Rewrite operational notes\n≤300 words; hard cap 3,000 chars"]
    checkpoint["Checkpoint state"]
    next{"More tasks?"}

    start --> pad --> prompt --> run --> update_ledger --> update_notes --> checkpoint --> next
    next -->|yes| pad
  end

  subgraph Inner["Inner loop — one rollout / one task"]
    request["Send task + system prompt +\ntool schemas to model"]
    calls{"Model requested\nfunction calls?"}
    dispatch["Run every requested local tool\nCRM / Wiki / submit_answer"]
    tool_results["Append tool results to Trace\nsend outputs back to model"]
    finish["Record final answer"]
    evaluate["Local score() + finalize()\nmetrics + Margot feedback"]

    request --> calls
    calls -->|yes| dispatch --> tool_results --> request
    calls -->|no| finish --> evaluate
  end

  run --> request
  evaluate --> update_ledger
```

The inner loop repeats until the model stops requesting tools, or the driver reaches its safety limit. Each cycle is:

1. The model receives the current task, its previous tool outputs, and available tool schemas.
2. It either asks for CRM/wiki tools or stops.
3. The driver executes requested tools locally and returns their JSON results to the model.
4. The model decides whether another tool call is needed.
5. The model calls `submit_answer`, or emits final text; the evaluator scores it and produces feedback.

This is implemented in [`run_rollout()`](/Users/rezamalek/Documents/develop/touch-weights-hack/hackathon/examples/responses_rollouts.py:129). The driver permits up to 12 configured tool-turn batches, but a batch may contain more than one tool call.

The outer loop runs only in the memory condition:

1. Build the scratchpad from all remembered corrections plus the current operational-note summary.
2. Run the inner loop for the next task.
3. Add new reviewer corrections to the permanent ledger.
4. Replace the operational notes with a concise summary of what was observed.
5. Save a checkpoint and move to the next task.

That sequence is in [`run_memory_arm()`](/Users/rezamalek/Documents/develop/touch-weights-hack/hackathon/examples/scratchpad_loop.py:123).

For comparison, the script also runs a separate **stateless arm**: it invokes the same inner rollout for each task but with no scratchpad and may run tasks concurrently. Its purpose is only to show whether the outer-loop memory creates real improvement.