"""Build the immutable offline Phase 4 paired-arm report and plots."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

ARMS = ("cold", "text64", "still_single", "still_recurrent")
LABELS = {
    "cold": "Cold",
    "text64": "Text64",
    "still_single": "Still64 single-step",
    "still_recurrent": "Still64 recurrence-aware",
}
COLORS = {
    "cold": "#8f8f8f",
    "text64": "#e69f00",
    "still_single": "#56b4e9",
    "still_recurrent": "#0072b2",
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def rolling(values: list[float], window: int) -> list[float]:
    return [mean(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))]


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    position = probability * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def json_safe(value: Any) -> Any:
    """Replace non-finite display sentinels with strict-JSON nulls."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def paired_bootstrap(
    left: list[float],
    right: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap the paired mean delta ``left - right`` by episode index."""

    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap inputs must be non-empty and equal length")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    rng = random.Random(seed)
    n = len(differences)
    draws = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)
    )
    return {
        "delta": mean(differences),
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
        "bootstrap_samples": samples,
    }


def metric(record: dict[str, Any], name: str) -> float:
    if name == "typed_submission_rate":
        return float(bool(record.get("submitted")))
    return float(record.get("metrics", {}).get(name, 0.0))


def acceptance(records: list[dict[str, Any]]) -> float:
    return mean([metric(record, "preference_accepted") for record in records])


def records_for_tag(
    records: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], field: str, tag: str
) -> list[dict[str, Any]]:
    return [record for record in records if tag in metadata[record["episodeId"]].get(field, [])]


def teach_retest(
    records: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    taught: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = {"teach": [], "retest": []}
    for record in records:
        invoked = set(metadata[record["episodeId"]].get("invoked", []))
        group = "retest" if invoked and invoked <= taught else "teach"
        groups[group].append(record)
        # This is an offline tag only. An accepted answer confirms the invoked dimensions;
        # a rejection's explicit reviewer correction teaches the invoked preference bundle.
        if metric(record, "preference_accepted") == 1.0 or record.get("feedback"):
            taught.update(invoked)
    return groups


def load_arms(specifications: list[str]) -> dict[str, dict[str, Any]]:
    paths: dict[str, Path] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("--arm must be ARM=RESULTS_JSON")
        arm, raw_path = specification.split("=", 1)
        if arm not in ARMS or arm in paths:
            raise ValueError(f"invalid or duplicate arm: {arm}")
        paths[arm] = Path(raw_path)
    missing = set(ARMS) - paths.keys()
    if missing:
        raise ValueError(f"missing Phase 4 arms: {sorted(missing)}")
    data = {arm: json.loads(paths[arm].read_text()) for arm in ARMS}
    for arm, result in data.items():
        if result.get("format") != "phase4-arm-result-v1" or result.get("arm") != arm:
            raise ValueError(f"{arm} is not a matching Phase 4 result")
        records = result.get("records", [])
        if [record.get("index") for record in records] != list(range(len(records))):
            raise ValueError(f"{arm} records are not in contiguous seq_index order")
    lengths = {len(result["records"]) for result in data.values()}
    models = {result["model"] for result in data.values()}
    seeds = {result["seed"] for result in data.values()}
    sequences = {
        tuple(record["episodeId"] for record in result["records"]) for result in data.values()
    }
    if len(lengths) != 1 or len(models) != 1 or len(seeds) != 1 or len(sequences) != 1:
        raise ValueError("Phase 4 arms are not paired on length, model, seed, and episode order")
    return data


def load_fleet(path: Path, episode_ids: set[str]) -> dict[str, dict[str, Any]]:
    metadata = {
        row["episodeId"]: row
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
        if row["episodeId"] in episode_ids
    }
    missing = episode_ids - metadata.keys()
    if missing:
        raise ValueError(f"fleet is missing result episodes: {sorted(missing)[:3]}")
    return metadata


def aggregate(
    data: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    arms = {arm: data[arm]["records"] for arm in ARMS}
    dimensions = sorted({tag for row in metadata.values() for tag in row.get("invoked", [])})
    traps = sorted({tag for row in metadata.values() for tag in row.get("world_traps", [])})
    summary: dict[str, Any] = {}
    teaching = {arm: teach_retest(records, metadata) for arm, records in arms.items()}
    for arm, records in arms.items():
        quarters = [records[start : start + 60] for start in range(0, len(records), 60)]
        final_positions = int(records[-1].get("memory_positions_after", 0)) if records else 0
        source_tokens = int(records[-1].get("memory_source_tokens", 0)) if records else 0
        summary[arm] = {
            "acceptance": acceptance(records),
            "final_60_acceptance": acceptance(records[-60:]),
            "value_correct": mean([metric(record, "value_correct") for record in records]),
            "typed_submission_rate": mean(
                [metric(record, "typed_submission_rate") for record in records]
            ),
            "mean_tool_calls": mean([metric(record, "tool_calls") for record in records]),
            "tool_calls_by_stream_quarter": [
                mean([metric(record, "tool_calls") for record in quarter]) for quarter in quarters
            ],
            "memory_positions": final_positions,
            "source_tokens_represented": source_tokens,
            "compression_ratio": source_tokens / final_positions if final_positions else None,
            "recurrence_count": int(records[-1].get("memory_recurrence_count", 0))
            if records
            else 0,
            "compaction_seconds": sum(
                float(record.get("memory_compaction_seconds", 0.0)) for record in records
            ),
            "teach_retest": {
                group: {
                    "n": len(group_records),
                    "acceptance": acceptance(group_records),
                }
                for group, group_records in teaching[arm].items()
            },
        }

    comparisons: dict[str, Any] = {}
    for arm in ("text64", "still_single", "still_recurrent"):
        comparisons[arm] = {}
        for baseline in ("cold", "text64"):
            if arm == baseline:
                continue
            comparisons[arm][baseline] = {
                "acceptance": paired_bootstrap(
                    [metric(row, "preference_accepted") for row in arms[arm]],
                    [metric(row, "preference_accepted") for row in arms[baseline]],
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + ARMS.index(arm) * 10 + ARMS.index(baseline),
                ),
                "tool_calls": paired_bootstrap(
                    [metric(row, "tool_calls") for row in arms[arm]],
                    [metric(row, "tool_calls") for row in arms[baseline]],
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + 100 + ARMS.index(arm) * 10 + ARMS.index(baseline),
                ),
            }

    by_preference = {
        dimension: {
            arm: {
                "n": len(rows := records_for_tag(records, metadata, "invoked", dimension)),
                "acceptance": acceptance(rows),
            }
            for arm, records in arms.items()
        }
        for dimension in dimensions
    }
    by_world_trap = {
        trap: {
            arm: {
                "n": len(rows := records_for_tag(records, metadata, "world_traps", trap)),
                "acceptance": acceptance(rows),
                "mean_tool_calls": mean([metric(row, "tool_calls") for row in rows]),
            }
            for arm, records in arms.items()
        }
        for trap in [*traps, "no_trap"]
    }
    for arm, records in arms.items():
        no_trap = [record for record in records if not metadata[record["episodeId"]].get("world_traps")]
        by_world_trap["no_trap"][arm] = {
            "n": len(no_trap),
            "acceptance": acceptance(no_trap),
            "mean_tool_calls": mean([metric(row, "tool_calls") for row in no_trap]),
        }

    recurrent = comparisons["still_recurrent"]
    accepted_better = all(
        recurrent[baseline]["acceptance"]["ci95_low"] > 0 for baseline in ("cold", "text64")
    )
    efficient_noninferior = all(
        recurrent[baseline]["acceptance"]["ci95_low"] >= 0
        and recurrent[baseline]["tool_calls"]["ci95_high"] < 0
        for baseline in ("cold", "text64")
    )
    return {
        "format": "phase4-comparison-v1",
        "model": next(iter({result["model"] for result in data.values()})),
        "seed": next(iter({result["seed"] for result in data.values()})),
        "episodes_per_arm": len(arms["cold"]),
        "summary": summary,
        "paired_deltas": comparisons,
        "by_preference": by_preference,
        "by_world_trap": by_world_trap,
        "claim_supported": accepted_better or efficient_noninferior,
        "claim_rule": (
            "recurrent acceptance delta CI is positive versus both baselines, or acceptance "
            "is non-inferior and paired tool-call delta CI is negative versus both"
        ),
    }


def plot_lines(
    axes: Any,
    arms: dict[str, list[dict[str, Any]]],
    value: Callable[[dict[str, Any]], float],
    window: int,
) -> None:
    for arm in ARMS:
        axes.plot(
            rolling([value(record) for record in arms[arm]], window),
            color=COLORS[arm],
            label=LABELS[arm],
            linewidth=2 if arm == "still_recurrent" else 1.6,
        )


def write_plots(
    data: dict[str, dict[str, Any]], metrics: dict[str, Any], out: Path, window: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = {arm: data[arm]["records"] for arm in ARMS}
    for filename, title, ylabel, value, bounded in (
        (
            "rolling_acceptance.png",
            "Acceptance over the continual-learning stream",
            f"preference accepted (rolling {window})",
            lambda row: metric(row, "preference_accepted"),
            True,
        ),
        (
            "rolling_value_correct.png",
            "Value correctness over the continual-learning stream",
            f"value correct (rolling {window})",
            lambda row: metric(row, "value_correct"),
            True,
        ),
        (
            "rolling_submission_rate.png",
            "Typed answer submission over the continual-learning stream",
            f"typed-submission rate (rolling {window})",
            lambda row: metric(row, "typed_submission_rate"),
            True,
        ),
        (
            "rolling_tool_calls.png",
            "Tool calls over the continual-learning stream",
            f"tool calls (rolling {window})",
            lambda row: metric(row, "tool_calls"),
            False,
        ),
    ):
        fig, ax = plt.subplots(figsize=(8.6, 4.5))
        plot_lines(ax, arms, value, window)
        ax.set_xlabel("episode index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if bounded:
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    x = list(range(1, len(metrics["summary"]["cold"]["tool_calls_by_stream_quarter"]) + 1))
    width = 0.18
    for offset, arm in enumerate(ARMS):
        ax.bar(
            [value + (offset - 1.5) * width for value in x],
            metrics["summary"][arm]["tool_calls_by_stream_quarter"],
            width=width,
            color=COLORS[arm],
            label=LABELS[arm],
        )
    ax.set_xlabel("stream quarter (60 episodes)")
    ax.set_ylabel("mean tool calls")
    ax.set_xticks(x)
    ax.set_title("Tool calls by stream quarter")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tool_calls_by_quarter.png", dpi=160)
    plt.close(fig)

    dimensions = list(metrics["by_preference"])
    fig, ax = plt.subplots(figsize=(9, max(5, len(dimensions) * 0.36)))
    y = list(range(len(dimensions)))
    width = 0.19
    for offset, arm in enumerate(ARMS):
        ax.barh(
            [value + (offset - 1.5) * width for value in y],
            [metrics["by_preference"][dimension][arm]["acceptance"] for dimension in dimensions],
            height=width,
            color=COLORS[arm],
            label=LABELS[arm],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(dimensions, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("acceptance")
    ax.set_title("Acceptance by invoked preference")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "by_preference.png", dpi=160)
    plt.close(fig)

    traps = list(metrics["by_world_trap"])
    fig, (accept_ax, calls_ax) = plt.subplots(1, 2, figsize=(13, 4.8))
    x = list(range(len(traps)))
    width = 0.19
    for offset, arm in enumerate(ARMS):
        positions = [value + (offset - 1.5) * width for value in x]
        accept_ax.bar(
            positions,
            [metrics["by_world_trap"][trap][arm]["acceptance"] for trap in traps],
            width=width,
            color=COLORS[arm],
            label=LABELS[arm],
        )
        calls_ax.bar(
            positions,
            [metrics["by_world_trap"][trap][arm]["mean_tool_calls"] for trap in traps],
            width=width,
            color=COLORS[arm],
        )
    for ax in (accept_ax, calls_ax):
        ax.set_xticks(x)
        ax.set_xticklabels(traps, rotation=20, ha="right", fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    accept_ax.set_ylim(0, 1)
    accept_ax.set_ylabel("acceptance")
    accept_ax.set_title("Acceptance by world trap")
    calls_ax.set_ylabel("mean tool calls")
    calls_ax.set_title("Tool calls by world trap")
    accept_ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "by_world_trap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    x = [0, 1]
    width = 0.19
    for offset, arm in enumerate(ARMS):
        values = [
            metrics["summary"][arm]["teach_retest"][group]["acceptance"]
            for group in ("teach", "retest")
        ]
        ax.bar(
            [value + (offset - 1.5) * width for value in x],
            values,
            width=width,
            color=COLORS[arm],
            label=LABELS[arm],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["teach / first exposure", "all invoked preferences retested"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("acceptance")
    ax.set_title("Teach-then-retest acceptance")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "teach_retest.png", dpi=160)
    plt.close(fig)


def write_markdown(metrics: dict[str, Any], out: Path, phase3_plot: str | None) -> None:
    rows = [
        "# Phase 4 — fixed-budget continual memory",
        "",
        f"Model: `{metrics['model']}` · episodes per arm: {metrics['episodes_per_arm']}",
        "",
        "| Method | Acceptance | Final-60 acceptance | Value correct | Mean tool calls | Memory positions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = metrics["summary"][arm]
        rows.append(
            f"| {LABELS[arm]} | {values['acceptance']:.3f} | "
            f"{values['final_60_acceptance']:.3f} | {values['value_correct']:.3f} | "
            f"{values['mean_tool_calls']:.2f} | {values['memory_positions']} |"
        )
    rows.extend(["", "## Paired recurrent deltas", ""])
    for baseline in ("cold", "text64"):
        comparison = metrics["paired_deltas"]["still_recurrent"][baseline]["acceptance"]
        rows.append(
            f"- Versus {LABELS[baseline]}: {comparison['delta']:+.3f} acceptance "
            f"(95% paired bootstrap CI {comparison['ci95_low']:+.3f} to "
            f"{comparison['ci95_high']:+.3f})."
        )
    rows.extend(["", "## Interpretation", ""])
    if metrics["claim_supported"]:
        rows.append(
            "The predeclared fixed-budget claim is supported under the conservative confidence-interval rule."
        )
    else:
        rows.append(
            "The predeclared fixed-budget claim is not supported. Treat this as a domain-transfer "
            "failure: the compactor retained synthetic template facts but did not reliably turn real "
            "reviewer corrections and tool outcomes into better continual agent behavior."
        )
    rows.extend(
        [
            "",
            "The GPT-5.6 scratchpad result is an external reference only; Qwen-vs-GPT differences are not memory gains.",
            "",
            "Plots in this directory were generated from the four immutable Phase 4 result files.",
        ]
    )
    if phase3_plot:
        rows.append(f"The existing Phase 3 oldest-fact plot remains at `{phase3_plot}` and was not overwritten.")
    (out / "report.md").write_text("\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="ARM=RESULTS_JSON")
    parser.add_argument(
        "--fleet",
        default="src/alien_api_env/data/episodes/alien_api_v4.jsonl",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=49017)
    parser.add_argument("--phase3-oldest-fact-plot")
    args = parser.parse_args()
    if args.window < 1 or args.bootstrap_samples < 1:
        raise ValueError("window and bootstrap samples must be positive")
    data = load_arms(args.arm)
    episode_ids = {record["episodeId"] for record in data["cold"]["records"]}
    metadata = load_fleet(Path(args.fleet), episode_ids)
    metrics = aggregate(
        data,
        metadata,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2) + "\n")
    write_plots(data, metrics, out, args.window)
    write_markdown(metrics, out, args.phase3_oldest_fact_plot)
    print(json.dumps({"summary": metrics["summary"], "claim_supported": metrics["claim_supported"]}, indent=2))
    print(f"Phase 4 report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
