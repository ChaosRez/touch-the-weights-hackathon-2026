"""Endpoint-gated persona sampler: an LLM character commits to a coherent frozen profile.

Build-time only, never in the reward path. A persona (a short character brief — "a
meticulous ex-auditor who distrusts derived fields") is shown the full preference schema
and commits to exactly one option per dimension, in character. The choices are validated
against the schema and **frozen** to ``feedbacker/profiles/<persona_id>.json``; from then
on the profile is plain committed data — the taskset loads it with
``world/profile.load_persona_profile`` and CI never calls a model.

Coherence is the point: one character making all fifteen choices yields correlated
conventions (an auditor who distrusts the stale ARR also distrusts the capped search
listing), which is the generalization prior a smart learner can exploit — predicting
unseen preferences from the character implied by seen corrections.

Gating follows the repo convention: ``VF_PROBE_BASE_URL`` / ``VF_PROBE_API_KEY`` /
``VF_PROBE_MODEL``. Without them the sampler returns ``None`` (SKIPPED, never faked);
the CLI, whose job is to write a file, errors clearly instead.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from alien_api_env.world.preferences import SCHEMA
from alien_api_env.world.profile import PROFILES_DIR, Profile, load_persona_profile, validate_profile

# The environment ships ONE persona: Margot, the feedbacker every sequence reviews under
# by default. Her committed profile (`profiles/margot.json`) and voiced feedback cache
# (`feedback_cache/margot.json`) were authored by the build-time model on 2026-07-27.
# This registry exists so she can be regenerated (or a variant added) with a different
# model via the CLI; it is a catalog of briefs, never of simultaneously-active reviewers.
PERSONA_BRIEFS: dict[str, str] = {
    "margot": (
        "Margot Reinholt, a veteran head of finance operations. You are exacting and "
        "primary-source minded: you distrust derived or denormalized fields, read figures "
        "in the store's native units without rounding, think in the company's fiscal "
        "calendar, count only what is truly won or truly available, cite records by their "
        "immutable ids, and escalate anything empty rather than letting it pass."
    ),
}


def endpoint_configured() -> bool:
    """Whether the live-endpoint env vars are all set."""
    return bool(
        os.environ.get("VF_PROBE_BASE_URL")
        and os.environ.get("VF_PROBE_API_KEY")
        and os.environ.get("VF_PROBE_MODEL")
    )


def _schema_block() -> str:
    lines = []
    for d in SCHEMA:
        cond = f" (applies only to: {d.condition})" if d.condition else ""
        lines.append(
            f"- {d.id} [{d.kind}]{cond}: applies to {d.applies_when}. "
            f"Options: {', '.join(d.options)}"
        )
    return "\n".join(lines)


def persona_prompt(brief: str) -> str:
    """The instruction the persona answers. Public so the runbook can quote it and the
    offline audit can reuse it verbatim."""
    return (
        f"You are {brief}\n\n"
        "You review every answer your analysts produce, and you hold firm conventions "
        "about which source to trust, how to scope definitions, how figures are "
        "presented, what things are called, and when to escalate. For each convention "
        "dimension below, commit to exactly ONE option — the one your character would "
        "genuinely insist on. Stay coherent: your choices should feel like one person's "
        "worldview, not a coin flip per row.\n\n"
        f"{_schema_block()}\n\n"
        "Reply with ONLY a JSON object mapping every dimension id to your chosen option "
        "string, no other text."
    )


def _parse_choices(text: str) -> dict[str, str]:
    """The JSON object in a model reply (tolerates a code fence)."""
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    payload = fence.group(1) if fence else stripped
    if not fence:
        brace = re.search(r"\{.*\}", payload, re.DOTALL)
        if brace:
            payload = brace.group(0)
    choices = json.loads(payload)
    if not isinstance(choices, dict):
        raise ValueError("persona reply is not a JSON object")
    return {str(k): str(v) for k, v in choices.items()}


def generate_persona_profile(
    persona_id: str,
    brief: str | None = None,
    *,
    attempts: int = 3,
    out_dir: Path | None = None,
) -> Profile | None:
    """Sample a persona profile from the configured endpoint and freeze it to JSON.

    Returns the frozen ``Profile``, or ``None`` (SKIPPED) when no endpoint is
    configured. Invalid model replies are retried up to ``attempts`` times, then raise —
    a bad profile is never silently frozen.
    """
    if not endpoint_configured():
        return None
    brief = brief or PERSONA_BRIEFS[persona_id]

    from openai import OpenAI  # lazy: build-time dependency only

    client = OpenAI(
        base_url=os.environ["VF_PROBE_BASE_URL"], api_key=os.environ["VF_PROBE_API_KEY"]
    )
    model = os.environ["VF_PROBE_MODEL"]

    last_error: Exception | None = None
    for _ in range(attempts):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": persona_prompt(brief)}],
        )
        text = response.choices[0].message.content or ""
        try:
            choices = _parse_choices(text)
            validate_profile(choices)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_error = e
            continue
        payload: dict[str, Any] = {
            "persona_id": persona_id,
            "brief": brief,
            "model": model,
            "choices": choices,
        }
        target_dir = out_dir or PROFILES_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{persona_id}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if out_dir is None:
            return load_persona_profile(persona_id)
        return Profile(seed=0, choices=dict(choices), persona_id=persona_id)
    raise ValueError(
        f"persona {persona_id!r} produced no schema-valid profile in {attempts} attempts: {last_error}"
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m alien_api_env.feedbacker.persona")
    ap.add_argument("--persona-id", required=True, help="file stem under feedbacker/profiles/")
    ap.add_argument(
        "--brief",
        default=None,
        help="character brief; defaults to the built-in brief for --persona-id",
    )
    args = ap.parse_args()

    if not endpoint_configured():
        print(
            "ERROR: persona generation needs a live endpoint and was asked to write a "
            "profile.\nSet VF_PROBE_BASE_URL / VF_PROBE_API_KEY / VF_PROBE_MODEL to run.",
        )
        return 2
    if args.brief is None and args.persona_id not in PERSONA_BRIEFS:
        print(
            f"ERROR: no built-in brief for {args.persona_id!r}; pass --brief. "
            f"Built-ins: {sorted(PERSONA_BRIEFS)}"
        )
        return 2
    profile = generate_persona_profile(args.persona_id, args.brief)
    assert profile is not None
    print(f"froze persona profile {args.persona_id!r}:")
    for dim, opt in sorted(profile.choices.items()):
        print(f"  {dim} = {opt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
