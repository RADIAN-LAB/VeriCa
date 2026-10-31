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

# Embed TrueType (Type 42) rather than Type 3 fonts: IEEE PDF eXpress rejects Type 3.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import energy_model  # noqa: E402

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
    ("p_proposed", "VeriCa"),
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
    """Published regime, replacing the results table: rates and completion."""
    labels, hit, correct, wrong, completion = [], [], [], [], []
    for arm, name in ARM_ORDER:
        published = runs.get(arm, {}).get("published", [])
        if not published:
            continue
        labels.append(name)
        h = mean_of(published, "hit_rate", 100)
        c = mean_of(published, "valid_hit_rate", 100)
        hit.append(h)
        correct.append(c)
        wrong.append(h - c)
        completion.append(mean_of(published, "average_completion"))

    positions = range(len(labels))
    height = 0.26
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.4, 5.2), sharey=True, layout="constrained",
        gridspec_kw={"width_ratios": [1.9, 1.0]})
    bars_hit = ax1.barh([p + height + 0.02 for p in positions], hit,
                        height=height, color=BLUE, label="hit rate")
    bars_ok = ax1.barh(list(positions), correct,
                       height=height, color=ORANGE, label="correct answers")
    bars_wr = ax1.barh([p - height - 0.02 for p in positions], wrong,
                       height=height, color=AQUA, label="wrong answers")
    label_bars(ax1, bars_hit, hit)
    label_bars(ax1, bars_ok, correct)
    label_bars(ax1, bars_wr, wrong)
    ax1.set_yticks(list(positions))
    ax1.set_yticklabels(labels, color=INK, fontsize=9.5)
    ax1.set_xlim(0, 112)
    style(ax1, "% of requests")
    ax1.legend(frameon=False, fontsize=9, ncol=3, labelcolor=INK_SOFT,
               loc="lower left", bbox_to_anchor=(0, 1.005))

    bars_ms = ax2.barh(list(positions), completion, height=0.62, color=BLUE)
    label_bars(ax2, bars_ms, completion, fmt="{:.2f}", pad=0.12)
    ax2.set_xlim(0, max(completion) * 1.3)
    style(ax2, "completion (ms)")

    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig_verified(runs, ceiling: Optional[float], energy: Dict[str, float],
                 out: Path) -> None:
    """Verified regime, replacing the results table: hits, latency, energy."""
    labels, usable, latency, joules = [], [], [], []
    for arm, name in ARM_ORDER:
        verified = runs.get(arm, {}).get("verified", [])
        if not verified:
            continue
        labels.append(name)
        usable.append(mean_of(verified, "valid_hit_rate", 100))
        latency.append(mean_of(verified, "average_completion"))
        joules.append(energy.get(arm, float("nan")))

    positions = list(range(len(labels)))
    colors = [ORANGE if name == "VeriCa" else BLUE for name in labels]
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(8.4, 4.4), sharey=True, layout="constrained",
        gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})

    bars = ax1.barh(positions, usable, height=0.62, color=colors)
    label_bars(ax1, bars, usable)
    if ceiling is not None:
        ax1.axvline(ceiling, color=INK_SOFT, linestyle=(0, (4, 3)),
                    linewidth=1.2)
        ax1.text(ceiling - 1.5, len(labels) - 0.4,
                 f"clairvoyant {ceiling:.1f}%", ha="right", va="center",
                 color=INK_SOFT, fontsize=8.5)
    ax1.set_yticks(positions)
    ax1.set_yticklabels(labels, color=INK, fontsize=9.5)
    ax1.set_xlim(0, max(106, (ceiling or 0) + 8))
    style(ax1, "usable hits (% of requests)")

    bars = ax2.barh(positions, latency, height=0.62, color=colors)
    label_bars(ax2, bars, latency, fmt="{:.2f}", pad=0.15)
    ax2.set_xlim(0, max(latency) * 1.32)
    style(ax2, "completion (ms)")

    bars = ax3.barh(positions, joules, height=0.62, color=colors)
    label_bars(ax3, bars, joules, fmt="{:.3f}", pad=0.006)
    ax3.set_xlim(0, max(joules) * 1.36)
    style(ax3, "J per correct answer")

    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def energy_per_correct(evaluation: Path) -> Dict[str, float]:
    """Mean J per correct answer per arm, verified regime, physical model."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for seed_dir in sorted(evaluation.glob("seed_*")):
        trace = seed_dir / "tasks.csv"
        if not trace.exists():
            continue
        profile = energy_model.trace_profile(trace)
        for path in seed_dir.glob("*__verified.json"):
            arm = path.stem.split("__")[0]
            with path.open() as handle:
                result = json.load(handle)
            acc[arm].append(
                energy_model.energy_joules(result, profile)["j_per_correct_answer"])
    return {arm: statistics.fmean(v) for arm, v in acc.items()}


SCENARIO_FILES = [
    ("Sparse", "density_v100"),
    ("Congested", "density_v400"),
    ("Bologna", "bologna"),
]
SCENARIO_ARMS = [
    ("b4_aoi_tuned", "AoI window (tuned)"),
    ("b1_lru", "LRU"),
    ("b5_foggycache", "FoggyCache reuse"),
    ("p_proposed", "VeriCa"),
]


def fig_scenarios(runs, out: Path) -> None:
    """Generality across topology and density, replacing the results table."""
    values: Dict[str, Dict[str, float]] = {}
    values["Grid"] = {
        arm: mean_of(runs.get(arm, {}).get("verified", []), "valid_hit_rate", 100)
        for arm, _ in SCENARIO_ARMS
    }
    for scenario, dirname in SCENARIO_FILES:
        path = ROOT / "experiments" / dirname / "aggregated_results.json"
        with path.open() as handle:
            agg = json.load(handle)
        values[scenario] = {
            arm: 100.0 * agg[f"{arm}__verified"]["metrics"]["valid_hit_rate"]["mean"]
            for arm, _ in SCENARIO_ARMS
        }

    series = [("Grid", BLUE), ("Sparse", ORANGE), ("Congested", AQUA),
              ("Bologna", YELLOW)]
    positions = list(range(len(SCENARIO_ARMS)))
    height = 0.19
    offsets = [1.5 * height, 0.5 * height, -0.5 * height, -1.5 * height]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for (scenario, color), offset in zip(series, offsets):
        vals = [values[scenario][arm] for arm, _ in SCENARIO_ARMS]
        bars = ax.barh([p + offset for p in positions], vals, height=height,
                       color=color, label=scenario, edgecolor=SURFACE,
                       linewidth=0.8)
        label_bars(ax, bars, vals, pad=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels([name for _, name in SCENARIO_ARMS], color=INK,
                       fontsize=9.5)
    ax.set_xlim(0, 100)
    style(ax, "usable hits (% of requests), verified regime")
    ax.legend(frameon=False, fontsize=9, ncol=4, labelcolor=INK_SOFT,
              loc="lower left", bbox_to_anchor=(0, 1.005))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
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
    fig.savefig(out.with_suffix(".pdf"))
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
    fig.savefig(out.with_suffix(".pdf"))
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
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def style_line(ax, xlabel: str, ylabel: str) -> None:
    """``style`` for line charts: the value axis is y, so the grid is too."""
    style(ax, xlabel)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)


def end_label(ax, x: float, y: float, text: str, dy: float = 0.0) -> None:
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                va="center", ha="left", color=INK, fontsize=8.5)


TOLERANCE_ARMS = [
    ("p_proposed", "VeriCa", ORANGE, "o"),
    ("b5_foggycache", "FoggyCache reuse", BLUE, "s"),
    ("b1_lru", "LRU", AQUA, "^"),
]


def fig_tolerance(sweep_path: Path, out: Path) -> None:
    """Verified usable hits as every referee tolerance is scaled together."""
    with sweep_path.open() as handle:
        per_scale = json.load(handle)["per_scale"]
    scales = sorted(per_scale, key=float)
    xs = [float(s) for s in scales]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for arm, name, color, marker in TOLERANCE_ARMS:
        ys = [per_scale[s][arm]["verified_usable"] for s in scales]
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        # The lowest series is labelled below its line so the value never
        # sits on the series above it.
        dy = -13 if arm == TOLERANCE_ARMS[-1][0] else 9
        for x, y in zip(xs, ys):
            if x in (xs[0], xs[-1]):
                ax.annotate(f"{y:.1f}", (x, y), xytext=(0, dy),
                            textcoords="offset points", ha="center",
                            color=INK_SOFT, fontsize=8)
        end_label(ax, xs[-1], ys[-1], name)
    ax.axvline(1.0, color=INK_SOFT, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.annotate("tolerances used in the paper", (1.0, 96), xytext=(4, 0),
                textcoords="offset points", color=INK_SOFT, fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.2f}×" for x in xs])
    ax.set_xlim(xs[0] - 0.08, xs[-1] + 0.42)
    ax.set_ylim(0, 100)
    style_line(ax, "tolerance scale (stricter ← → looser)",
               "usable hits (% of requests)")
    ax.legend(frameon=False, fontsize=9, ncol=3, labelcolor=INK_SOFT,
              loc="lower left", bbox_to_anchor=(0, 1.01))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


SPATIAL_FEATURES = [
    ("mean_speed", "mean speed", BLUE, "o"),
    ("edge_speeds", "lane speeds", ORANGE, "s"),
    ("stopped_set", "stopped set", AQUA, "^"),
    ("density", "vehicle count", YELLOW, "D"),
    ("occupancy", "occupancy grid", MAGENTA, "v"),
]


def _bin_mid(label: str) -> float:
    lo, hi = label.split("-")
    return (float(lo) + float(hi)) / 2


def fig_spatial(decorrelation_path: Path, out: Path) -> None:
    """How far each answer feature stays valid when the request is displaced."""
    with decorrelation_path.open() as handle:
        data = json.load(handle)

    fig, (ax_n, ax_c) = plt.subplots(
        1, 2, figsize=(8.4, 4.8), sharey=True,
        gridspec_kw={"width_ratios": [1, 2.2]})

    curve = data["neighbourhood"]["curve"]
    xs = [_bin_mid(b) for b in curve]
    ys = [curve[b]["p_agree"] for b in curve]
    # Neutral ink: this panel's single series is not one of the right panel's
    # features, so it must not borrow one of their colours.
    ax_n.plot(xs, ys, color=INK_SOFT, linewidth=2, marker="o", markersize=6,
              markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax_n.axvline(50, color=INK_SOFT, linewidth=0.8, linestyle=(0, (3, 3)))
    ax_n.annotate("50 m neighbour\nradius", (50, 0.62), xytext=(5, 0),
                  textcoords="offset points", color=INK_SOFT, fontsize=8)
    ax_n.set_xlim(0, 300)
    ax_n.set_ylim(0, 1.05)
    ax_n.set_title("Vehicle neighbourhood", color=INK, fontsize=10, loc="left")
    style_line(ax_n, "displacement (m)", "probability the answer still agrees")

    for key, name, color, marker in SPATIAL_FEATURES:
        curve = data["region_features"][key]["curve"]
        xs = [_bin_mid(b) for b in curve]
        ys = [curve[b]["p_agree"] for b in curve]
        ax_c.plot(xs, ys, color=color, linewidth=2, marker=marker,
                  markersize=6, markevery=2, markeredgecolor=SURFACE,
                  markeredgewidth=1.5, label=name)
    ax_c.axvline(500, color=INK_SOFT, linewidth=0.8, linestyle=(0, (3, 3)))
    ax_c.annotate("cell radius", (500, 1.0), xytext=(5, 0),
                  textcoords="offset points", color=INK_SOFT, fontsize=8)
    ax_c.set_xlim(0, 1700)
    ax_c.set_title("Cell-level observables", color=INK, fontsize=10, loc="left")
    style_line(ax_c, "displacement of the 500 m question region (m)", "")
    ax_c.legend(frameon=False, fontsize=8.5, ncol=3, labelcolor=INK_SOFT,
                loc="upper left", bbox_to_anchor=(0, -0.17))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
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
    parser.add_argument("--tolerance-sweep", type=Path,
                        default=ROOT / "experiments" / "sensitivity_tolerance"
                        / "tolerance_sensitivity.json")
    parser.add_argument("--decorrelation", type=Path,
                        default=ROOT / "experiments" / "scope_decorrelation"
                        / "decorrelation.json")
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

    energy = energy_per_correct(args.evaluation)
    fig_hit_vs_correct(runs, args.out_dir / "fig_hit_vs_correct.png")
    fig_verified(runs, ceiling, energy, args.out_dir / "fig_verified.png")
    fig_scenarios(runs, args.out_dir / "fig_scenarios.png")
    fig_error_taxonomy(runs, args.out_dir / "fig_error_taxonomy.png")
    fig_latency(runs, args.out_dir / "fig_latency.png")
    if args.fitted_rates.exists():
        fig_decay_rates(args.fitted_rates, args.out_dir / "fig_decay_rates.png")
    if args.tolerance_sweep.exists():
        fig_tolerance(args.tolerance_sweep, args.out_dir / "fig_tolerance.png")
    if args.decorrelation.exists():
        fig_spatial(args.decorrelation, args.out_dir / "fig_spatial.png")

    n = len(next(iter(runs.values()))["published"])
    print(f"Figures written to {args.out_dir} (n≈{n} seeds per arm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
