#!/usr/bin/env python3
"""Figures for the paper and the project report.

Reads the per-seed result JSONs directly rather than the aggregate, so error
breakdowns (which the aggregate does not carry) are available and every figure
reports the same n.

Chart conventions follow one house style: a single value axis per chart (never
two scales), thin marks with a surface gap between adjacent bars, recessive grid
and axes, direct value labels on every bar — the palette's mid-chroma slots sit
below 3:1 against the surface, so labels are the required relief, not decoration.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Validated categorical slots (light mode) — assigned in fixed order, never cycled.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e2e1dd"

ARM_ORDER = [
    ("b0_random", "Random"),
    ("b0_fifo", "FIFO"),
    ("b1_lru", "LRU"),
    ("b2_lfu", "LFU"),
    ("b4_aoi_assumed", "AoI window (assumed)"),
    ("b4_aoi_tuned", "AoI window (tuned)"),
    ("b5_foggycache", "FoggyCache reuse"),
    ("p_proposed", "Proposed"),
]
ERROR_SERIES = [
    ("wrong_place", "wrong place", BLUE),
    ("stale", "stale", ORANGE),
    ("wrong_subject", "wrong subject", AQUA),
    ("wrong_intent", "wrong intent", YELLOW),
]


def style(ax, xlabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)


def load_runs(evaluation: Path) -> Dict[str, Dict[str, List[dict]]]:
    """arm -> regime -> per-seed results."""
    runs: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(evaluation.glob("seed_*/*__*.json")):
        arm, regime = path.stem.split("__")
        with path.open() as handle:
            runs[arm][regime].append(json.load(handle))
    return runs


def mean_of(results: Sequence[dict], key: str, scale: float = 1.0) -> float:
    values = [float(r.get(key, 0.0) or 0.0) for r in results]
    return statistics.fmean(values) * scale if values else float("nan")


def label_bars(ax, bars, values, fmt="{:.1f}", pad=0.6) -> None:
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + pad, bar.get_y() + bar.get_height() / 2,
                fmt.format(value), va="center", ha="left",
                color=INK, fontsize=8.5)


def fig_hit_vs_correct(runs, out: Path) -> None:
    """The money figure: how much of a hit rate is actually right."""
    labels, hit, correct = [], [], []
    for arm, name in ARM_ORDER:
        published = runs.get(arm, {}).get("published", [])
        if not published:
            continue
        labels.append(name)
        hit.append(mean_of(published, "hit_rate", 100))
        correct.append(mean_of(published, "valid_hit_rate", 100))

    positions = range(len(labels))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars_hit = ax.barh([p + height / 2 + 0.02 for p in positions], hit,
                       height=height, color=BLUE, label="cache hit rate")
    bars_ok = ax.barh([p - height / 2 - 0.02 for p in positions], correct,
                      height=height, color=ORANGE, label="answers actually correct")
    label_bars(ax, bars_hit, hit)
    label_bars(ax, bars_ok, correct)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, color=INK, fontsize=9.5)
    ax.set_xlim(0, 118)
    style(ax, "% of requests")
    ax.legend(frameon=False, fontsize=9, ncol=2, labelcolor=INK_SOFT,
              loc="lower left", bbox_to_anchor=(0, 1.01))
    ax.set_title("A cache hit is not a correct answer",
                 color=INK, fontsize=12, loc="left", pad=32)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_verified(runs, ceiling: Optional[float], out: Path) -> None:
    """Usable hits once every arm must deliver correct answers."""
    labels, usable = [], []
    for arm, name in ARM_ORDER:
        verified = runs.get(arm, {}).get("verified", [])
        if not verified:
            continue
        labels.append(name)
        usable.append(mean_of(verified, "valid_hit_rate", 100))

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    colors = [BLUE if name != "Proposed" else ORANGE for name in labels]
    bars = ax.barh(list(range(len(labels))), usable, height=0.62, color=colors)
    label_bars(ax, bars, usable)
    if ceiling is not None:
        ax.axvline(ceiling, color=INK_SOFT, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.text(ceiling - 1.5, len(labels) - 0.4,
                f"clairvoyant ceiling {ceiling:.1f}%", ha="right", va="center",
                color=INK_SOFT, fontsize=8.5)
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels, color=INK, fontsize=9.5)
    ax.set_xlim(0, max(100, (ceiling or 0) + 6))
    style(ax, "% of requests served correctly from cache")
    ax.set_title("Usable hits at equal correctness",
                 color=INK, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_error_taxonomy(runs, out: Path) -> None:
    """What the wrong answers are wrong about."""
    labels, stacks = [], []
    for arm, name in ARM_ORDER:
        published = runs.get(arm, {}).get("published", [])
        if not published:
            continue
        totals = defaultdict(float)
        for result in published:
            requests = float(result.get("cache_hits", 0)) + float(
                result.get("cache_misses", 0))
            breakdown = result.get("error_breakdown", {}) or {}
            for kind, _, _ in ERROR_SERIES:
                totals[kind] += 100.0 * float(breakdown.get(kind, 0)) / max(requests, 1)
        labels.append(name)
        stacks.append({k: totals[k] / len(published) for k, _, _ in ERROR_SERIES})

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    positions = list(range(len(labels)))
    left = [0.0] * len(labels)
    for kind, legend, color in ERROR_SERIES:
        widths = [s[kind] for s in stacks]
        ax.barh(positions, widths, left=left, height=0.62, color=color,
                label=legend, edgecolor=SURFACE, linewidth=1.2)
        left = [l + w for l, w in zip(left, widths)]
    for position, total in zip(positions, left):
        ax.text(total + 0.7, position, f"{total:.0f}", va="center",
                color=INK, fontsize=8.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, color=INK, fontsize=9.5)
    style(ax, "% of requests answered wrongly")
    ax.legend(frameon=False, fontsize=9, ncol=4, labelcolor=INK_SOFT,
              loc="lower left", bbox_to_anchor=(0, 1.01))
    ax.set_title("What the wrong answers are wrong about",
                 color=INK, fontsize=12, loc="left", pad=32)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_latency(runs, out: Path) -> None:
    labels, latency = [], []
    for arm, name in ARM_ORDER:
        verified = runs.get(arm, {}).get("verified", [])
        if not verified:
            continue
        labels.append(name)
        latency.append(mean_of(verified, "average_completion"))

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    colors = [BLUE if name != "Proposed" else ORANGE for name in labels]
    bars = ax.barh(list(range(len(labels))), latency, height=0.62, color=colors)
    label_bars(ax, bars, latency, fmt="{:.2f}", pad=0.12)
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels, color=INK, fontsize=9.5)
    ax.set_xlim(0, max(latency) * 1.14)
    style(ax, "mean task completion time (ms), all answers correct")
    ax.set_title("Latency at equal correctness",
                 color=INK, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_decay_rates(fitted_path: Path, out: Path) -> None:
    """Measured invalidation rates against the values the literature assumes."""
    with fitted_path.open() as handle:
        merged = json.load(handle)["merged"]
    intents = [i for i in merged if merged[i]["lambda_fitted"] > 0]
    measured = [merged[i]["lambda_fitted"] for i in intents]
    assumed = [merged[i]["lambda_hand_set"] for i in intents]

    positions = range(len(intents))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    bars_m = ax.barh([p + height / 2 + 0.02 for p in positions], measured,
                     height=height, color=BLUE, label="measured from mobility")
    bars_a = ax.barh([p - height / 2 - 0.02 for p in positions], assumed,
                     height=height, color=ORANGE, label="assumed by the literature")
    label_bars(ax, bars_m, measured, fmt="{:.3f}", pad=0.004)
    label_bars(ax, bars_a, assumed, fmt="{:.3f}", pad=0.004)
    ax.set_yticks(list(positions))
    ax.set_yticklabels([i.replace("_", " ") for i in intents], color=INK, fontsize=9.5)
    ax.set_xlim(0, max(measured + assumed) * 1.25)
    style(ax, "invalidation rate λ (per second)")
    ax.legend(frameon=False, fontsize=9, ncol=2, labelcolor=INK_SOFT,
              loc="lower left", bbox_to_anchor=(0, 1.01))
    ax.set_title("How fast results really go stale",
                 color=INK, fontsize=12, loc="left", pad=32)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluation", type=Path,
                        default=ROOT / "experiments" / "evaluation")
    parser.add_argument("--fitted-rates", type=Path,
                        default=ROOT / "experiments" / "figures"
                        / "fitted_decay_rates_all30.json")
    parser.add_argument("--clairvoyant", type=Path,
                        default=ROOT / "experiments" / "baselines" / "runs"
                        / "clairvoyant.json")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments" / "figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.evaluation)
    if not runs:
        print(f"No results under {args.evaluation}", file=sys.stderr)
        return 1

    ceiling = None
    if args.clairvoyant.exists():
        with args.clairvoyant.open() as handle:
            ceiling = 100.0 * json.load(handle)["clairvoyant_bound"]

    fig_hit_vs_correct(runs, args.out_dir / "fig_hit_vs_correct.png")
    fig_verified(runs, ceiling, args.out_dir / "fig_verified.png")
    fig_error_taxonomy(runs, args.out_dir / "fig_error_taxonomy.png")
    fig_latency(runs, args.out_dir / "fig_latency.png")
    if args.fitted_rates.exists():
        fig_decay_rates(args.fitted_rates, args.out_dir / "fig_decay_rates.png")

    n = len(next(iter(runs.values()))["published"])
    print(f"Figures written to {args.out_dir} (n≈{n} seeds per arm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
