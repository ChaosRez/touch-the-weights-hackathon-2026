# Phase 4 — fixed-budget continual memory

Model: `Qwen/Qwen3-4B` · episodes per arm: 8

| Method | Acceptance | Final-60 acceptance | Value correct | Mean tool calls | Memory positions |
|---|---:|---:|---:|---:|---:|
| Cold | 0.125 | 0.125 | 0.625 | 1.00 | 0 |
| Text64 | 0.250 | 0.250 | 0.500 | 0.75 | 64 |
| Still64 single-step | 0.000 | 0.000 | 0.125 | 0.12 | 64 |
| Still64 recurrence-aware | 0.000 | 0.000 | 0.125 | 0.25 | 64 |

## Paired recurrent deltas

- Versus Cold: -0.125 acceptance (95% paired bootstrap CI -0.375 to +0.000).
- Versus Text64: -0.250 acceptance (95% paired bootstrap CI -0.625 to +0.000).

## Interpretation

The predeclared fixed-budget claim is not supported. Treat this as a domain-transfer failure: the compactor retained synthetic template facts but did not reliably turn real reviewer corrections and tool outcomes into better continual agent behavior.

The GPT-5.6 scratchpad result is an external reference only; Qwen-vs-GPT differences are not memory gains.

Plots in this directory were generated from the four immutable Phase 4 result files.
The existing Phase 3 oldest-fact plot remains at `reports/phase3/phase_3_qwen3_4b_fixed64_v1/plots/oldest_accuracy_vs_depth.png` and was not overwritten.
