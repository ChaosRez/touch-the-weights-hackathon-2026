"""Build the outer-loop scratchpad report for alien-api: plots + aggregate metrics.

Reads the paired-arm results JSON written by ``scratchpad_loop.py`` (240
episodes per arm, ``stateless`` vs ``memory``), joins ``world_traps`` from the installed
fleet, and writes PNG plots plus a metrics JSON into the report directory. The raw
results file is NEVER committed (it embeds fleet prompts and accepted labels); only the
plots and aggregates are.

Acceptance is read from ``preference_accepted``, which is identical to the current
binary reward (1.0 iff the answer equals the Margot-accepted label). The trial that
produced the archived results ran under the since-removed efficiency multiplier, so its
``reward`` field is NOT comparable across the reward change; ``preference_accepted`` is.

Usage:
  uv run --with matplotlib python examples/scratchpad_report.py \
      --results <scratchpad_loop.json> --out reports/my_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
FLEET = (
    REPO / "packages" / "alien-api-env" / "src" / "alien_api_env" / "data"
    / "episodes" / "alien_api_v4.jsonl"
)
WINDOW = 40  # rolling window, episodes


def rolling(vals: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(sum(vals[lo : i + 1]) / (i + 1 - lo))
    return out


def acc(rows: list[dict]) -> float:
    """Binary acceptance (== the current reward): answer equals the accepted label."""
    return sum(r["preference_accepted"] for r in rows) / len(rows) if rows else float("nan")


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}")
        return 1
    if not FLEET.exists():
        print(f"ERROR: installed fleet not found at {FLEET}; run build_fleet.py --install")
        return 1

    data = json.loads(results_path.read_text())
    stateless, memory = data["stateless"], data["memory"]
    if len(stateless) != len(memory):
        print(f"ERROR: arms differ in length ({len(stateless)} vs {len(memory)})")
        return 1
    traps_by_id = {
        row["episodeId"]: row.get("world_traps", [])
        for row in (json.loads(line) for line in FLEET.read_text().splitlines() if line)
    }
    missing = [r["episodeId"] for r in memory if r["episodeId"] not in traps_by_id]
    if missing:
        print(f"ERROR: {len(missing)} episodes missing from the installed fleet: {missing[:3]}")
        return 1
    for arm in (stateless, memory):
        arm.sort(key=lambda r: r["idx"])
        for r in arm:
            r["world_traps"] = traps_by_id[r["episodeId"]]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    C_MEM, C_STL = "#2a7de1", "#b0b0b0"

    # 1. learning curve: rolling acceptance over episode index
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(rolling([r["preference_accepted"] for r in memory], WINDOW), color=C_MEM, lw=2,
            label="scratchpad memory")
    ax.plot(rolling([r["preference_accepted"] for r in stateless], WINDOW), color=C_STL, lw=2,
            label="stateless")
    ax.set_xlabel("episode index")
    ax.set_ylabel(f"acceptance (rolling {WINDOW})")
    ax.set_title("Margot-acceptance over the episode stream")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=150)
    plt.close(fig)

    # 2. efficiency curve: rolling tool calls
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(rolling([r["tool_calls"] for r in memory], WINDOW), color=C_MEM, lw=2,
            label="scratchpad memory")
    ax.plot(rolling([r["tool_calls"] for r in stateless], WINDOW), color=C_STL, lw=2,
            label="stateless")
    ax.set_xlabel("episode index")
    ax.set_ylabel(f"tool calls (rolling {WINDOW})")
    ax.set_title("Tool calls over the episode stream (weight-0 observability)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "tool_calls.png", dpi=150)
    plt.close(fig)

    # 3. per preference dimension (the remembering axis)
    dims = sorted({d for r in memory for d in r["invoked"]})
    by_dim = {
        d: (
            acc([r for r in stateless if d in r["invoked"]]),
            acc([r for r in memory if d in r["invoked"]]),
            len([r for r in memory if d in r["invoked"]]),
        )
        for d in dims
    }
    order = sorted(dims, key=lambda d: by_dim[d][1] - by_dim[d][0])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = range(len(order))
    ax.barh([i + 0.2 for i in y], [by_dim[d][1] for d in order], height=0.38,
            color=C_MEM, label="scratchpad memory")
    ax.barh([i - 0.2 for i in y], [by_dim[d][0] for d in order], height=0.38,
            color=C_STL, label="stateless")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{d}  (n={by_dim[d][2]})" for d in order], fontsize=8)
    ax.set_xlabel("acceptance on episodes invoking the dimension")
    ax.set_title("Preference axis: acceptance per invoked dimension")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(0.98, 0.35))
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out / "by_preference.png", dpi=150)
    plt.close(fig)

    # 4. per world trap (the quirk axis): acceptance + tool calls
    traps = sorted({t for r in memory for t in r["world_traps"]})
    groups = traps + ["no_trap"]

    def rows_for(arm: list[dict], g: str) -> list[dict]:
        if g == "no_trap":
            return [r for r in arm if not r["world_traps"]]
        return [r for r in arm if g in r["world_traps"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = range(len(groups))
    for ax, metric, label in (
        (ax1, lambda rs: acc(rs), "acceptance"),
        (ax2, lambda rs: mean([r["tool_calls"] for r in rs]), "mean tool calls"),
    ):
        ax.bar([i - 0.2 for i in x], [metric(rows_for(stateless, g)) for g in groups],
               width=0.38, color=C_STL, label="stateless")
        ax.bar([i + 0.2 for i in x], [metric(rows_for(memory, g)) for g in groups],
               width=0.38, color=C_MEM, label="scratchpad memory")
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [f"{g.replace('_', chr(10), 1)}\n(n={len(rows_for(memory, g))})" for g in groups],
            fontsize=8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, axis="y")
    ax1.set_ylim(0, 1)
    ax1.set_title("World-quirk axis: acceptance per trap")
    ax2.set_title("World-quirk axis: tool calls per trap")
    ax1.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "by_world_trap.png", dpi=150)
    plt.close(fig)

    # 5. teach-then-retest: first exposure of a dimension vs after its feedback existed
    def split_exposures(arm: list[dict]) -> tuple[list[dict], list[dict]]:
        taught: set[str] = set()
        first, retest = [], []
        for r in arm:
            (retest if all(d in taught for d in r["invoked"]) else first).append(r)
            taught.update(r["violated"])  # feedback teaches exactly the violated dims
            if r["preference_accepted"] == 1.0:
                taught.update(r["invoked"])  # an accepted answer confirms its dims
        return first, retest

    stl_first, stl_re = split_exposures(stateless)
    mem_first, mem_re = split_exposures(memory)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    cats = ["first exposure", "all invoked dims\npreviously taught"]
    ax.bar([0 - 0.2, 1 - 0.2], [acc(stl_first), acc(stl_re)], width=0.38, color=C_STL,
           label="stateless")
    ax.bar([0 + 0.2, 1 + 0.2], [acc(mem_first), acc(mem_re)], width=0.38, color=C_MEM,
           label="scratchpad memory")
    for i, (s, m) in enumerate([(stl_first, mem_first), (stl_re, mem_re)]):
        ax.text(i - 0.2, acc(s) + 0.02, f"n={len(s)}", ha="center", fontsize=8)
        ax.text(i + 0.2, acc(m) + 0.02, f"n={len(m)}", ha="center", fontsize=8)
    ax.set_ylabel("acceptance")
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(cats)
    ax.set_title("Teach-then-retest: does banked feedback transfer?")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out / "teach_retest.png", dpi=150)
    plt.close(fig)

    metrics = {
        "model": data["model"],
        "episodes_per_arm": len(memory),
        "acceptance": {"stateless": round(acc(stateless), 3), "memory": round(acc(memory), 3)},
        "mean_tool_calls": {
            "stateless": round(mean([r["tool_calls"] for r in stateless]), 2),
            "memory": round(mean([r["tool_calls"] for r in memory]), 2),
        },
        "final_window_acceptance": {
            "stateless": round(acc(stateless[-WINDOW:]), 3),
            "memory": round(acc(memory[-WINDOW:]), 3),
        },
        "teach_retest_acceptance": {
            "stateless_first": round(acc(stl_first), 3),
            "stateless_retest": round(acc(stl_re), 3),
            "memory_first": round(acc(mem_first), 3),
            "memory_retest": round(acc(mem_re), 3),
            "n_retest": len(mem_re),
        },
        "by_preference": {
            d: {"stateless": round(v[0], 3), "memory": round(v[1], 3), "n": v[2]}
            for d, v in by_dim.items()
        },
        "by_world_trap": {
            g: {
                "stateless": round(acc(rows_for(stateless, g)), 3),
                "memory": round(acc(rows_for(memory, g)), 3),
                "stateless_calls": round(mean([r["tool_calls"] for r in rows_for(stateless, g)]), 2),
                "memory_calls": round(mean([r["tool_calls"] for r in rows_for(memory, g)]), 2),
                "n": len(rows_for(memory, g)),
            }
            for g in groups
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"\nwrote plots + metrics.json to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
