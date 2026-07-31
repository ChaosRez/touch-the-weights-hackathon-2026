"""The alien-api v4 taskset: pure hydration of the committed episode fleet.

There is **one world** (kestrel) and **one feedbacker** (Margot). ``AlienApiTaskset``
contains no generation code: ``load()`` reads the packaged episodes JSONL — rows built
and certified by ``alien-api-synth`` — and maps them verbatim onto ``AlienApiData``.
Margot's accepted label, her choices for the invoked dimensions, the defensible
enumeration, the budget, and the feedback hints all ride in the row; nothing is derived
at runtime.

``AlienApiTask`` scores **binary acceptance**: the answer must exactly match the stored
``accepted`` label — the defensible value *in Margot's preferred presentation* — for
reward 1.0, else 0.0. Efficiency is pure observability (2026-07-28 change): ``budget``,
``tool_calls``, and ``over_budget`` measure the exploration-collapse axis for the outer
loop but never tax the score.

Weight-0 metrics expose the learning curve the outer loop reads: ``value_correct``
(reached *some* defensible reading — learned the world), ``preference_accepted``
(matched Margot — learned the feedbacker), ``preferences_violated``, plus
``tool_calls`` / ``over_budget`` / ``artifact_tokens``. ``finalize()`` writes her
teaching feedback (voiced cache, template fallback) and the per-episode preference tags
to ``trace.info`` — the outer-loop seam. Scoring never calls a model.

v1 contract notes (unchanged): ``TaskData`` is frozen pydantic; ``@vf.reward``/
``@vf.metric`` methods run via ``Task.score``; ``Task.score`` does not call ``finalize``;
``len(trace.tool_messages)`` is the call count; load-time knobs live on the
``TasksetConfig`` subclass (``--env.taskset.*``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf

from alien_api_env.feedbacker.feedback import Violation, persona_feedback
from alien_api_env.vf.tools import (
    ARTIFACT_VERBOSITY,
    AnswerToolset,
    CrmToolset,
    WikiToolset,
    count_tokens,
    is_submit_answer,
    submitted_value,
)
from alien_api_env.world import value_extract
from alien_api_env.world.store import DATA_DIR, DEFAULT_WORLD

# The one feedbacker. Her profile labeled the fleet at build time; at runtime her name
# only selects the voiced feedback cache.
DEFAULT_PERSONA = "margot"

DEFAULT_DATASET = DATA_DIR / "episodes" / "alien_api_v4.jsonl"

# The answer-channel contract. Framing + format hygiene only: it tells the model that a
# reviewer with latent conventions accepts exactly one presentation and how to submit,
# but never which conventions the reviewer holds — learning those from corrections is
# the task. The `escalate` token is channel documentation (how to escalate), while *when*
# to escalate is a learned preference.
SYSTEM_PROMPT = (
    "You are a back-office data analyst for a company's CRM/ERP system. A reviewer with "
    "firm conventions of their own reads every answer and accepts exactly one source, "
    "scope, format, and naming for it. The records and the SOP/wiki do not always agree, "
    "and several readings of a question can be defended from the data; your reviewer "
    "accepts only the reading that matches their conventions, which you can only learn "
    "from their corrections. Use as few tool calls as possible.\n\n"
    "Submit your final answer by calling the `submit_answer` tool with `value` set to "
    "EXACTLY the answer and nothing else — no explanation, label, markdown, units, "
    "currency symbols, or thousands separators. Numbers are bare integers; words, codes, "
    "ids, names, titles, and dates are the exact string. If you conclude the question "
    "should be escalated rather than answered, submit the value `escalate`. The "
    "`submit_answer` call does not count against your tool-call budget. If you do not "
    "call `submit_answer`, your final message is read as the answer instead."
)


class AlienApiData(vf.TaskData):
    """One fleet episode's wire data, hydrated verbatim from its stored row.

    ``world`` names the stored world the toolsets hydrate (``worlds_root`` overrides the
    packaged data directory — the milestone ``render_root`` analog, threaded through the
    task data because the toolsets run in separate MCP processes). ``invoked`` /
    ``defensible`` / ``accepted`` / ``choices`` / ``feedback_hints`` / ``kind`` are the
    row's fields; none are shown to the model.
    """

    world: str = DEFAULT_WORLD
    worlds_root: str = ""
    budget: int = 4
    invoked: tuple[str, ...] = ()
    world_traps: tuple[str, ...] = ()  # discrete world behaviors the solution path crosses
    defensible: dict[str, str] = {}
    accepted: str = ""
    choices: dict[str, str] = {}
    feedback_hints: dict[str, str] = {}
    kind: str = ""
    split: str = "train"
    persona_id: str = DEFAULT_PERSONA
    artifact_verbosity: int = ARTIFACT_VERBOSITY
    system_prompt: str | None = SYSTEM_PROMPT


class AlienApiTasksetConfig(vf.TasksetConfig):
    """Load-time knobs (``--env.taskset.*``) — the whole surface.

    ``dataset_path`` defaults to the packaged fleet; both it and ``worlds_root`` also
    accept an ``hf://org/name`` dataset reference (needs the ``hf`` extra; token via
    ``HF_TOKEN`` for private repos). ``split`` filters (``train``/``eval``/empty for all).
    There is no seed, no n, no version, no persona knob: the fleet is committed data
    labeled by the one feedbacker.
    """

    dataset_path: str = ""
    split: str = "train"
    worlds_root: str = ""
    # Milestone-style artifact-store reference ("name/version"): resolves BOTH the
    # fleet and the worlds from the digest-verified GCS cache. First-class internal
    # channel; hf:// values on the path knobs are the open-source mirror.
    artifact: str = ""
    artifact_remote: str = ""
    artifact_verbosity: int = ARTIFACT_VERBOSITY


def load_episodes(dataset_path: str = "", split: str = "") -> list[dict]:
    """Read fleet rows (milestone's ``load_records`` shape), optionally split-filtered.

    ``dataset_path`` may be a local file or an ``hf://org/name`` dataset reference (the
    published fleet lives at ``hf://ConstructLabs/alien-api``); empty uses the bundled
    copy shipped in the wheel."""
    from alien_api_env.world.store import (
        DEFAULT_HF_DATASET,
        HF_PREFIX,
        pull_artifact,
        resolve_hf,
    )

    if dataset_path.startswith(HF_PREFIX):
        path = resolve_hf(dataset_path, f"episodes/{DEFAULT_DATASET.name}")
    elif dataset_path:
        path = Path(dataset_path)
    elif DEFAULT_DATASET.exists():
        path = DEFAULT_DATASET  # locally installed; the offline path
    else:
        try:  # first-class remote: the GCS artifact store
            path = Path(pull_artifact()["files"]["episodes.jsonl"])
        except Exception:
            path = resolve_hf(DEFAULT_HF_DATASET, f"episodes/{DEFAULT_DATASET.name}")
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if split and row.get("split") != split:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"no episodes loaded from {path} (split={split!r})")
    return rows


class AlienApiTask(vf.Task[AlienApiData, vf.State, vf.TaskConfig]):
    """One ambiguous back-office question, scored against Margot's stored label."""

    tools = (CrmToolset, WikiToolset, AnswerToolset)

    @staticmethod
    def _answer(task: AlienApiData, trace: vf.Trace) -> str:
        """The effective answer: the last ``submit_answer`` value (typed channel), else
        the last assistant message. Whitespace-stripped; nothing else is normalized —
        presentation is a learned preference, not noise."""
        submitted = submitted_value(trace)
        raw = submitted if submitted is not None else trace.last_reply
        return (raw or "").strip()

    @staticmethod
    def _violations(task: AlienApiData, answer: str) -> tuple[str, ...] | None:
        """The invoked dimensions whose preferred option ``answer`` does not match, or
        ``None`` when the answer carries no defensible value at all.

        Exact-rendering match first; the value extractor as fallback (a defensible value
        wrapped in prose is still attributable). When several combos render the same
        string, the attribution is charitable: the fewest-violations combo wins.
        """
        exact = [k for k, v in task.defensible.items() if v == answer]
        if not exact:
            value = value_extract.match(answer, task.defensible.values())
            if value is None:
                return None
            exact = [k for k, v in task.defensible.items() if v == value]

        def violated(key: str) -> tuple[str, ...]:
            pairs = (part.split("=", 1) for part in key.split("|"))
            return tuple(dim for dim, opt in pairs if task.choices.get(dim) != opt)

        return min((violated(k) for k in exact), key=len)

    @staticmethod
    def _calls_used(trace: vf.Trace) -> int:
        """World reads counting against the budget (the answer channel is exempt)."""
        return sum(1 for m in trace.tool_messages if not is_submit_answer(m.name))

    @vf.reward(weight=1.0)
    async def acceptance(self, task: AlienApiData, trace: vf.Trace) -> float:
        """Binary acceptance: 1.0 iff the answer exactly matches the stored label.

        The exact match enforces both layers at once — the value must be defensible
        *and* presented under Margot's conventions (the label is by construction a
        defensible value in her preferred presentation). Efficiency deliberately does
        NOT scale the reward (2026-07-28): ``budget`` / ``tool_calls`` / ``over_budget``
        remain weight-0 observability for the learning curve.
        """
        return float(self._answer(task, trace) == task.accepted)

    @vf.metric
    async def preference_accepted(self, task: AlienApiData, trace: vf.Trace) -> float:
        """1.0 iff the answer matches the stored label — learned the feedbacker (weight 0)."""
        return float(self._answer(task, trace) == task.accepted)

    @vf.metric
    async def value_correct(self, task: AlienApiData, trace: vf.Trace) -> float:
        """1.0 iff the answer carries *some* defensible value — learned the world,
        independent of whether the presentation matches Margot (weight 0)."""
        return float(self._violations(task, self._answer(task, trace)) is not None)

    @vf.metric
    async def preferences_violated(self, task: AlienApiData, trace: vf.Trace) -> float:
        """How many invoked dimensions the answer's reading missed (weight 0). A
        non-defensible answer counts every invoked dimension — nothing was matched."""
        violations = self._violations(task, self._answer(task, trace))
        return float(len(task.invoked) if violations is None else len(violations))

    @vf.metric
    async def tool_calls(self, trace: vf.Trace) -> float:
        """World-read tool results — the efficiency term's ``calls_used``."""
        return float(self._calls_used(trace))

    @vf.metric
    async def over_budget(self, task: AlienApiData, trace: vf.Trace) -> float:
        """1.0 iff the rollout used more world reads than its budget."""
        return 1.0 if self._calls_used(trace) > task.budget else 0.0

    @vf.metric
    async def artifact_tokens(self, trace: vf.Trace) -> float:
        """Total tokens returned by the world-read tools (the answer channel excluded)."""
        from verifiers.v1.types import content_text

        return float(
            sum(
                count_tokens(content_text(m.content) or "")
                for m in trace.tool_messages
                if not is_submit_answer(m.name)
            )
        )

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """Write Margot's teaching feedback + preference tags to ``trace.info``.

        The outer-loop seam: feedback teaches the violated preference-classes (never the
        answer), in her voice from the frozen cache (template on any miss — still no
        model call, ever). ``Task.score`` does not call this; the real rollout path (and
        the deterministic tests) do. The environment never reads it back.
        """
        data = self.data
        answer = self._answer(data, trace)
        violations = self._violations(data, answer)
        trace.info["accepted"] = data.accepted
        trace.info["invoked"] = list(data.invoked)
        trace.info["world_traps"] = list(data.world_traps)
        trace.info["kind"] = data.kind
        # Disambiguates the non-defensible case for outer-loop consumers: there,
        # ``violated`` lists every invoked dimension by convention ("nothing matched"),
        # which is NOT a set of convention violations (2026-07-28 trace-audit finding).
        trace.info["value_defensible"] = violations is not None
        if violations is None:
            trace.info["violated"] = list(data.invoked)
            trace.info["feedback"] = persona_feedback(
                [], persona_id=data.persona_id, value_defensible=False
            )
            return
        trace.info["violated"] = list(violations)
        trace.info["feedback"] = persona_feedback(
            [Violation(dimension=d, chosen=data.choices[d]) for d in violations],
            persona_id=data.persona_id,
            value_defensible=True,
            hints=data.feedback_hints,
        )


class AlienApiTaskset(vf.Taskset[AlienApiTask, AlienApiTasksetConfig]):
    """Hydrates the committed fleet: no generation, no seeds, no model.

    ``load()`` reads the episodes JSONL at taskset materialization; per-rollout world
    hydration happens in the toolsets' ``setup_task`` (cached per process, offloaded
    with ``asyncio.to_thread``).
    """

    def load(self) -> Iterator[AlienApiTask]:
        cfg = self.config
        dataset_path, worlds_root = cfg.dataset_path, cfg.worlds_root
        if cfg.artifact:
            from alien_api_env.world.store import pull_artifact

            pulled = pull_artifact(cfg.artifact, cfg.artifact_remote)
            dataset_path = dataset_path or str(pulled["files"]["episodes.jsonl"])
            worlds_root = worlds_root or str(pulled["blobs"]["worlds"])
        for row in load_episodes(dataset_path, cfg.split):
            yield AlienApiTask(
                AlienApiData(
                    idx=row["seq_index"],
                    name=row["episodeId"],
                    prompt=row["prompt"],
                    world=row["world"],
                    worlds_root=worlds_root,
                    budget=row["budget"],
                    invoked=tuple(row["invoked"]),
                    world_traps=tuple(row.get("world_traps", ())),
                    defensible=dict(row["defensible"]),
                    accepted=row["accepted"],
                    choices=dict(row["choices"]),
                    feedback_hints=dict(row["feedback_hints"]),
                    kind=row["kind"],
                    split=row["split"],
                    artifact_verbosity=cfg.artifact_verbosity,
                ),
                cfg.task,
            )
