# Notices and provenance

This repository is a participant-built research prototype from the
[Touch Weights — Continual Learning Hack](https://luma.com/0fgouohr), organized by Construct Labs
and Alexandria. It is not an official Construct Labs release or a statement of endorsement by the
organizers.

This notice documents where the starting materials came from and which parts were developed or
extended by the hackathon team. It is an attribution record, not an additional license grant.

## Alien API environment

The continual-learning environment, synthetic company data, verifier, toolsets, reference
scratchpad implementation, cluster instructions, and original documentation came from the
organizer's public repository:

- Upstream: <https://github.com/constructlabs/hackathon>
- Preserved brief in this fork: [`HACKATHON_README.md`](HACKATHON_README.md)
- Upstream remote used during development: `https://github.com/constructlabs/hackathon.git`

Those upstream files retain their original copyright and license terms. In particular, see this
repository's [`LICENSE`](LICENSE), which is the license shipped with the environment. The fact that
an upstream repository is publicly readable does not by itself replace or broaden those terms.
Contact Construct Labs for permissions beyond the license text.

## Still neural KV-cache compactor

The Still implementation used as the neural-compaction starting point was developed by Max Meuer
for an earlier hackathon prototype and published at:

- Starting point: <https://github.com/MaxMeuer/still>
- Our event fork: <https://github.com/ChaosRez/still-touch-the-weights-hackathon2026>

Our fork adds the recurrence-focused work used in this project, including deterministic recurrent
synthetic data, differentiable recurrent compaction, mixed-depth training, fixed-budget evaluation,
inference-time recurrence, and associated tests and cluster jobs. Consult the upstream repository
and its authors for the terms governing reuse of the starting implementation; this notice does not
create a new license for their work.

Still implements ideas from the paper **STILL: Neural KV Cache Compaction**
([arXiv:2606.07878](https://arxiv.org/abs/2606.07878)). The paper authors are not affiliated with or
responsible for this hackathon implementation unless stated in their own materials.

## Hackathon-team additions

The three-person team built the Cartridges-specific integration and experiments during the event.
The main additions in this repository are:

- `src/cartridge_memory/`: local Qwen tool agent, memory attachment interfaces, legal-event
  boundary, streaming text ledger, recurrent KV ledger, and checkpoint/resume state.
- `examples/cartridge_loop.py`, `examples/phase4_loop.py`, `examples/phase4_report.py`, and
  `examples/phase4_presentation_plots.py`: tool-agent rollouts, paired evaluation, statistics, and
  presentation artifacts.
- `training/cartridges*.yaml`: reproducible SkyPilot jobs for the Qwen and Still experiments.
- `tests/test_*ledger.py`, `tests/test_still_qwen_backend.py`, and related Phase 1–4 tests.
- `reports/phase3/`, `reports/phase4/`, and `reports/presentation/`: aggregate metrics and plots
  produced by the team's runs.

Git history is the authoritative record for line-level authorship and modifications across all
three contributors.

## Results and model attribution

- Qwen3 model names and checkpoints are provided by their respective publisher and remain subject
  to the publisher's model terms.
- Hugging Face Transformers, PyTorch, SkyPilot, `verifiers`, and other dependencies remain subject
  to their respective licenses.
- Committed result tables and plots state their sample sizes and should not be interpreted as
  claims by the upstream project authors or event organizers.

If any attribution or ownership boundary here is incomplete, please open an issue before reusing
the affected component.
