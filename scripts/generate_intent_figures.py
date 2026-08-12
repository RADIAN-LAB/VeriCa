#!/usr/bin/env python3
"""Generate paper-ready plots and paired tests from multi-seed results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments" / "multi_seed" / "aggregated_results.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "figures"

PLOTS = {
    "valid_hit_rate": ("Valid cache-hit rate (%)", 100.0),
    "stale_reuse_rate": ("Stale reuse rate (%)", 100.0),
    "success_rate": ("Task success rate (%)", 100.0),
    "average_completion": ("Average completion time (ms)", 1.0),
    "energy_joules": ("Energy consumption (J)", 1.0),
}


def save_bar(summary: dict, metric: str, output_dir: Path) -> None:
    label, scale = PLOTS[metric]
    methods = list(summary)
    names = [summary[mid]["name"] for mid in methods]
    means = [summary[mid]["metrics"][metric]["mean"] * scale for mid in methods]
    stds = [summary[mid]["metrics"][metric]["std"] * scale for mid in methods]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    positions = np.arange(len(methods))
    ax.bar(positions, means, yerr=stds, capsize=4)
    ax.set_ylabel(label)
    ax.set_xticks(positions, names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"fig_{metric}.png", dpi=300)
    fig.savefig(output_dir / f"fig_{metric}.pdf")
    plt.close(fig)


def save_per_intent_heatmap(summary: dict, output_dir: Path) -> None:
    methods = [mid for mid in summary if summary[mid].get("raw")]
    intent_names = []
    for method_id in methods:
        for run in summary[method_id]["raw"]:
            intent_names.extend(run.get("per_intent", {}).keys())
    intent_names = list(dict.fromkeys(intent_names))
    if not intent_names:
        return

    matrix = np.zeros((len(methods), len(intent_names)))
    for row, method_id in enumerate(methods):
        stale = {name: 0 for name in intent_names}
        hits = {name: 0 for name in intent_names}
        for run in summary[method_id]["raw"]:
            for name, counts in run.get("per_intent", {}).items():
                stale[name] += counts.get("stale_hits", 0)
                hits[name] += counts.get("hits", 0)
        for col, name in enumerate(intent_names):
            matrix[row, col] = 100.0 * stale[name] / hits[name] if hits[name] else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(matrix, aspect="auto", cmap="magma_r", vmin=0)
    ax.set_xticks(np.arange(len(intent_names)), intent_names, rotation=30, ha="right")
    ax.set_yticks(
        np.arange(len(methods)), [summary[mid]["name"] for mid in methods]
    )
    ax.set_title("Stale reuse by intent (%)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.1f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Stale reuse (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_stale_reuse_by_intent.png", dpi=300)
    fig.savefig(output_dir / "fig_stale_reuse_by_intent.pdf")
    plt.close(fig)


def paired_tests(summary: dict) -> dict:
    if "6_coca" not in summary or "7_intent_nlp" not in summary:
        return {}
    coca = summary["6_coca"]["raw"]
    intent = summary["7_intent_nlp"]["raw"]
    by_seed_coca = {int(run["seed"]): run for run in coca}
    by_seed_intent = {int(run["seed"]): run for run in intent}
    seeds = sorted(set(by_seed_coca) & set(by_seed_intent))
    results = {}
    for metric in PLOTS:
        left = np.array([by_seed_coca[s][metric] for s in seeds], dtype=float)
        right = np.array([by_seed_intent[s][metric] for s in seeds], dtype=float)
        differences = right - left
        std = float(np.std(differences, ddof=1)) if len(seeds) > 1 else 0.0
        t_stat = (
            float(np.mean(differences) / (std / math.sqrt(len(seeds))))
            if std > 0
            else None
        )
        p_value = None
        if len(seeds) > 1:
            try:
                from scipy.stats import ttest_rel

                test = ttest_rel(right, left)
                if math.isfinite(float(test.statistic)):
                    t_stat = float(test.statistic)
                if math.isfinite(float(test.pvalue)):
                    p_value = float(test.pvalue)
            except ImportError:
                pass
        results[metric] = {
            "n": len(seeds),
            "mean_difference_intent_minus_coca": float(np.mean(differences)),
            "paired_t": t_stat,
            "p_value": p_value,
        }
    return results


def save_threshold_sensitivity(output_dir: Path) -> None:
    path = ROOT / "experiments" / "sensitivity" / "aggregated_results.json"
    if not path.exists():
        return
    with path.open() as sensitivity_file:
        data = json.load(sensitivity_file)
    thresholds = sorted(float(value) for value in data)
    valid = [100 * data[str(value)]["valid_hit_rate"]["mean"] for value in thresholds]
    stale = [100 * data[str(value)]["stale_reuse_rate"]["mean"] for value in thresholds]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, valid, marker="o", label="Valid hit rate")
    ax.plot(thresholds, stale, marker="s", label="Stale reuse rate")
    ax.set_xlabel("Reuse threshold")
    ax.set_ylabel("Rate (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig_threshold_sensitivity.png", dpi=300)
    fig.savefig(output_dir / "fig_threshold_sensitivity.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open() as summary_file:
        summary = json.load(summary_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in PLOTS:
        save_bar(summary, metric, args.output_dir)
    save_per_intent_heatmap(summary, args.output_dir)
    save_threshold_sensitivity(args.output_dir)
    tests = paired_tests(summary)
    with (args.output_dir / "statistical_tests.json").open("w") as test_file:
        json.dump(tests, test_file, indent=2)
    print(f"Generated figures in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
