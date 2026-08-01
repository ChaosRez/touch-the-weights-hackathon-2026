"""Generate two slide-ready plots for the Phase 3/4 result story."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METHODS = ("cold", "text64", "still_single", "still_recurrent")
LABELS = {
    "cold": "Cold",
    "text64": "Text64",
    "still_single": "Still64\nsingle-step",
    "still_recurrent": "Still64\nrecurrent",
}
COLORS = {
    "cold": "#8b8b8b",
    "text64": "#e69f00",
    "still_single": "#56b4e9",
    "still_recurrent": "#0072b2",
}


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def style_axis(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2, linewidth=0.8)
    axis.set_axisbelow(True)


def synthetic_to_real(
    phase3: dict[str, Any], phase4: dict[str, Any], output: Path
) -> None:
    import matplotlib.pyplot as plt

    depths = [int(depth) for depth in phase3["config"]["depths"]]
    phase3_methods = (
        ("full_context", "Full context", "#222222", "o"),
        ("text_window", "Text window (64)", COLORS["cold"], "s"),
        ("single_step", "Still single-step (64)", COLORS["text64"], "^"),
        ("recurrent", "Still recurrent (64)", "#009e73", "D"),
    )
    fig, (retention, transfer) = plt.subplots(1, 2, figsize=(14.2, 6.1))

    for method, label, color, marker in phase3_methods:
        values = [
            phase3["summary"][method]["by_depth"][str(depth)]["oldest"]["accuracy"]
            for depth in depths
        ]
        retention.plot(
            range(len(depths)),
            values,
            color=color,
            marker=marker,
            linewidth=2.6,
            markersize=7,
            label=label,
        )
    retention.set_xticks(range(len(depths)), depths)
    retention.set_ylim(-0.04, 1.08)
    retention.set_xlabel("Recurrent updates / source chunks")
    retention.set_ylabel("Oldest-fact accuracy")
    retention.set_title("A. Controlled retention succeeds", loc="left", fontweight="bold")
    retention.legend(frameon=False, fontsize=9, loc="lower right")
    retention.text(
        len(depths) - 1,
        0.93,
        "100× source tokens\ninside 64 KV positions",
        ha="right",
        va="top",
        color="#007a59",
        fontsize=10,
        fontweight="bold",
    )
    style_axis(retention)

    metric_names = ("acceptance", "value_correct", "typed_submission_rate")
    metric_labels = ("Acceptance", "Value correct", "Typed submission")
    x = list(range(len(metric_names)))
    width = 0.19
    for offset, method in enumerate(METHODS):
        values = [float(phase4["summary"][method][metric]) for metric in metric_names]
        positions = [position + (offset - 1.5) * width for position in x]
        bars = transfer.bar(
            positions,
            values,
            width=width,
            color=COLORS[method],
            label=LABELS[method].replace("\n", " "),
        )
        for bar, value in zip(bars, values, strict=True):
            transfer.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}".lstrip("0"),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    transfer.set_xticks(x, metric_labels)
    transfer.set_ylim(0, 0.76)
    transfer.set_ylabel("Rate")
    transfer.set_title("B. Real tool-use transfer fails", loc="left", fontweight="bold")
    transfer.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    transfer.text(
        0.02,
        0.97,
        "Fair common prefix: episodes 0–7 (n=8 per arm)",
        transform=transfer.transAxes,
        va="top",
        fontsize=9,
        color="#555555",
    )
    style_axis(transfer)

    fig.suptitle(
        "Fixed-budget memory retained facts—but the KV state did not transfer to agent behavior",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Same Qwen3-4B backbone and 64-position budget. Phase 4 stopped at the declared smoke gate; no 240-episode claim.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.93), w_pad=3.1)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def gate_outcomes(results: dict[str, dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    completed = [int(results[method]["summary"]["n"]) for method in METHODS]
    typed = [int(results[method]["summary"]["typed_submissions"]) for method in METHODS]
    typed_rates = [float(results[method]["summary"]["typed_submission_rate"]) for method in METHODS]
    labels = [LABELS[method] for method in METHODS]
    colors = [COLORS[method] for method in METHODS]
    y = list(range(len(METHODS)))

    fig, (episodes_axis, typed_axis) = plt.subplots(1, 2, figsize=(14.2, 5.7))
    completion_bars = episodes_axis.barh(y, completed, color=colors, height=0.58)
    episodes_axis.axvline(30, color="#555555", linewidth=1.2, linestyle="--")
    episodes_axis.set_xlim(0, 34)
    episodes_axis.set_yticks(y, labels)
    episodes_axis.invert_yaxis()
    episodes_axis.set_xlabel("Episodes completed before gate decision")
    episodes_axis.set_title("A. Promotion smoke outcome", loc="left", fontweight="bold")
    outcomes = (
        "PASS",
        "PASS",
        "FAIL · submission collapse",
        "FAIL · malformed decode @ episode 8",
    )
    for bar, count, outcome in zip(completion_bars, completed, outcomes, strict=True):
        episodes_axis.text(
            min(count + 0.5, 30.5),
            bar.get_y() + bar.get_height() / 2,
            f"{count}/30   {outcome}",
            va="center",
            fontsize=9,
            fontweight="bold" if outcome != "PASS" else "normal",
            color="#9b2226" if outcome != "PASS" else "#333333",
        )
    episodes_axis.grid(axis="x", alpha=0.2)
    episodes_axis.spines[["top", "right"]].set_visible(False)

    typed_bars = typed_axis.barh(y, typed_rates, color=colors, height=0.58)
    typed_axis.axvline(0.10, color="#9b2226", linewidth=1.5, linestyle="--")
    typed_axis.text(
        0.105,
        0.02,
        "10% minimum",
        transform=typed_axis.get_xaxis_transform(),
        color="#9b2226",
        fontsize=9,
        va="bottom",
    )
    typed_axis.set_xlim(0, 0.60)
    typed_axis.set_yticks(y, labels)
    typed_axis.invert_yaxis()
    typed_axis.set_xlabel("Typed submit_answer rate on observed episodes")
    typed_axis.set_title("B. Tool protocol survives only in controls", loc="left", fontweight="bold")
    for bar, count, total, rate in zip(typed_bars, typed, completed, typed_rates, strict=True):
        typed_axis.text(
            rate + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{count}/{total} ({rate:.1%})",
            va="center",
            fontsize=9,
        )
    typed_axis.grid(axis="x", alpha=0.2)
    typed_axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Why Phase 4 stopped before 240 episodes",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Single-step completed 30 episodes but stopped using the answer tool; recurrent Still reproduced an unclosed tool call from the exact episode-8 checkpoint.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92), w_pad=4.0)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3", required=True)
    parser.add_argument("--phase4-paired", required=True)
    parser.add_argument("--cold", required=True)
    parser.add_argument("--text64", required=True)
    parser.add_argument("--still-single", required=True)
    parser.add_argument("--still-recurrent", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    phase3 = load_json(args.phase3)
    phase4 = load_json(args.phase4_paired)
    results = {
        "cold": load_json(args.cold),
        "text64": load_json(args.text64),
        "still_single": load_json(args.still_single),
        "still_recurrent": load_json(args.still_recurrent),
    }
    synthetic_to_real(phase3, phase4, output / "synthetic_to_real_transfer.png")
    gate_outcomes(results, output / "phase4_smoke_gate.png")
    print(f"Presentation plots -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
