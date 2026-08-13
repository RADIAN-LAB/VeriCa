#!/usr/bin/env python3
"""Paired significance tests over the evaluation seeds.

Seeds are independent mobility realisations and every arm sees all of them, so
the comparisons are paired: for each seed, the difference between the proposed
method and a baseline on that same traffic. Pairing removes the between-seed
variance, which is large — traffic density differs run to run — and would
otherwise swamp the effect.

Choices, and why:

* **Wilcoxon signed-rank, not a paired t-test.** Cache hit rates are bounded in
  [0, 1] and their per-seed differences are not plausibly normal. The previous
  code used ``ttest_rel`` on n=1, which cannot produce a p-value at all.
* **Holm-Bonferroni across each metric's family of comparisons.** Testing the
  proposed method against eight baselines on one metric is eight chances to find
  a difference; Holm controls that without Bonferroni's loss of power.
* **Median paired difference with a bootstrap CI, plus Cliff's delta.** A
  p-value says an effect exists, not that it matters. The interval says how big
  it is, and the delta says how consistently it holds.

Falls back to a normal approximation with tie and continuity corrections when
scipy is unavailable, so the numbers can always be regenerated.

Usage:
    python3 scripts/paired_tests.py \
        --results experiments/evaluation/aggregated_results.json \
        --regime verified --output experiments/figures/statistical_tests.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - exercised by whichever branch the environment has
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover
    _scipy_stats = None

PROPOSED = "p_proposed"
BOOTSTRAP_SAMPLES = 10_000
# Metrics where a larger value is better, used only to label the direction.
HIGHER_IS_BETTER = {
    "hit_rate", "valid_hit_rate", "success_rate", "semantic_precision",
}


def wilcoxon_signed_rank(differences: Sequence[float]) -> Optional[float]:
    """Two-sided p-value for the paired differences, or None if undefined."""
    nonzero = [d for d in differences if d != 0.0]
    if len(nonzero) < 6:
        # Below this the test has no power worth reporting; say so rather than
        # emit a number that looks meaningful.
        return None

    if _scipy_stats is not None:
        return float(_scipy_stats.wilcoxon(nonzero).pvalue)

    magnitudes = sorted((abs(d), i) for i, d in enumerate(nonzero))
    ranks = [0.0] * len(nonzero)
    index = 0
    tie_correction = 0.0
    while index < len(magnitudes):
        stop = index
        while (stop + 1 < len(magnitudes)
               and magnitudes[stop + 1][0] == magnitudes[index][0]):
            stop += 1
        average_rank = (index + stop) / 2.0 + 1.0
        group = stop - index + 1
        tie_correction += group ** 3 - group
        for position in range(index, stop + 1):
            ranks[magnitudes[position][1]] = average_rank
        index = stop + 1

    positive = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    n = len(nonzero)
    mean = n * (n + 1) / 4.0
    variance = (n * (n + 1) * (2 * n + 1) - tie_correction / 2.0) / 24.0
    if variance <= 0:
        return None
    z = (abs(positive - mean) - 0.5) / math.sqrt(variance)
    return float(math.erfc(z / math.sqrt(2.0)))


def holm_correction(pvalues: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Holm-Bonferroni step-down adjustment within one family of tests."""
    testable = [(name, p) for name, p in pvalues.items() if p is not None]
    testable.sort(key=lambda item: item[1])
    adjusted: Dict[str, Optional[float]] = {n: None for n in pvalues}
    running = 0.0
    remaining = len(testable)
    for position, (name, p) in enumerate(testable):
        candidate = min(1.0, (remaining - position) * p)
        running = max(running, candidate)   # enforce monotonicity
        adjusted[name] = running
    return adjusted


def bootstrap_ci(
    differences: Sequence[float], confidence: float = 0.95, seed: int = 12345
) -> Tuple[float, float]:
    """Percentile bootstrap interval for the median paired difference."""
    if not differences:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(differences)
    medians = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        sample.sort()
        mid = n // 2
        medians.append(
            sample[mid] if n % 2 else (sample[mid - 1] + sample[mid]) / 2.0
        )
    medians.sort()
    tail = (1.0 - confidence) / 2.0
    return (
        medians[int(tail * BOOTSTRAP_SAMPLES)],
        medians[min(int((1.0 - tail) * BOOTSTRAP_SAMPLES), BOOTSTRAP_SAMPLES - 1)],
    )


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Probability a value from ``a`` exceeds one from ``b``, minus the reverse."""
    if not a or not b:
        return float("nan")
    greater = sum(1 for x in a for y in b if x > y)
    smaller = sum(1 for x in a for y in b if x < y)
    return (greater - smaller) / (len(a) * len(b))


def describe_delta(delta: float) -> str:
    magnitude = abs(delta)
    if math.isnan(delta):
        return "undefined"
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


def paired_values(
    proposed: dict, baseline: dict, metric: str
) -> Tuple[List[float], List[float]]:
    """Per-seed values for both arms, restricted to seeds both completed."""
    shared = sorted(set(proposed["per_seed"]) & set(baseline["per_seed"]), key=int)
    ours, theirs = [], []
    for seed in shared:
        mine = proposed["per_seed"][seed].get(metric)
        yours = baseline["per_seed"][seed].get(metric)
        if mine is None or yours is None:
            continue
        ours.append(float(mine))
        theirs.append(float(yours))
    return ours, theirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--regime", default="verified", choices=["verified", "published"])
    parser.add_argument("--metrics", nargs="+",
                        default=["valid_hit_rate", "average_completion",
                                 "hit_rate", "energy_joules"])
    parser.add_argument(
        "--exclude", nargs="*", default=[],
        help="Arms to leave out. The Holm family must contain exactly the "
             "comparisons that are reported — correcting over a baseline that "
             "never appears in the paper would penalise the reported ones for "
             "a test nobody sees.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.results.open() as handle:
        summary = json.load(handle)

    proposed_key = f"{PROPOSED}__{args.regime}"
    if proposed_key not in summary:
        print(f"No proposed arm for regime {args.regime} in {args.results}",
              file=sys.stderr)
        return 1
    proposed = summary[proposed_key]

    baselines = {
        key: entry for key, entry in summary.items()
        if entry["regime"] == args.regime and entry["arm"] != PROPOSED
        and entry["arm"] not in set(args.exclude)
    }

    report: Dict[str, Dict[str, dict]] = {}
    for metric in args.metrics:
        raw_p: Dict[str, Optional[float]] = {}
        rows: Dict[str, dict] = {}
        for key, baseline in baselines.items():
            ours, theirs = paired_values(proposed, baseline, metric)
            if len(ours) < 2:
                continue
            differences = [a - b for a, b in zip(ours, theirs)]
            low, high = bootstrap_ci(differences)
            delta = cliffs_delta(ours, theirs)
            raw_p[key] = wilcoxon_signed_rank(differences)
            rows[key] = {
                "baseline": baseline["name"],
                "n_pairs": len(differences),
                "proposed_median": statistics_median(ours),
                "baseline_median": statistics_median(theirs),
                "median_difference": statistics_median(differences),
                "ci95_low": low, "ci95_high": high,
                "cliffs_delta": delta,
                "effect": describe_delta(delta),
                "higher_is_better": metric in HIGHER_IS_BETTER,
            }
        adjusted = holm_correction(raw_p)
        for key, row in rows.items():
            row["p_value"] = raw_p.get(key)
            row["p_value_holm"] = adjusted.get(key)
        report[metric] = rows

    print(f"Regime: {args.regime}")
    for metric, rows in report.items():
        print(f"\n{metric}  (proposed − baseline, paired by mobility seed)")
        print(f"  {'baseline':<24}{'median Δ':>11}{'95% CI':>22}"
              f"{'p(Holm)':>10}{'effect':>12}{'n':>4}")
        for row in sorted(rows.values(), key=lambda r: r["median_difference"]):
            p = row["p_value_holm"]
            p_text = "n/a" if p is None else (f"{p:.2g}" if p >= 1e-4 else "<1e-4")
            interval = f"[{row['ci95_low']:.4g}, {row['ci95_high']:.4g}]"
            print(f"  {row['baseline']:<24}{row['median_difference']:>11.4g}"
                  f"{interval:>22}{p_text:>10}{row['effect']:>12}"
                  f"{row['n_pairs']:>4}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(
                {"regime": args.regime, "comparisons": report,
                 "test": "Wilcoxon signed-rank, Holm-corrected per metric",
                 "scipy": _scipy_stats is not None},
                handle, indent=2,
            )
        print(f"\nSaved {args.output}")
    return 0


def statistics_median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return float("nan")
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


if __name__ == "__main__":
    raise SystemExit(main())
