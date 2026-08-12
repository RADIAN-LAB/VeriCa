#!/usr/bin/env python3
"""Fit per-intent freshness decay rates against the trace-grounded oracle.

The freshness model scores a cached result as exp(-lambda[intent] * age). Those
lambdas used to be hand-set constants, and the ground truth used to be sampled
from the very same exponential — so the model could not be wrong. Here the
lifetimes come from observed vehicle mobility instead, and lambda is *estimated*
from them.

Two things are reported, and both belong in the paper:

  * the fitted rate per intent, with a comparison against the hand-set value;
  * how well an exponential actually describes the empirical survival curve,
    since the freshness model assumes memorylessness and that assumption should
    be tested rather than asserted.

Right-censoring is handled explicitly: results still valid at ``--max-age``
contribute exposure time but no failure, which is the standard exponential MLE
lambda = failures / total_exposure.

Usage:
    python3 scripts/fit_decay_rates.py --oracle experiments/oracle/seed_01.pkl
    python3 scripts/fit_decay_rates.py --oracle-dir experiments/oracle --output \
        experiments/figures/fitted_decay_rates.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intent.freshness import DECAY_RATES  # noqa: E402
from intent.oracle import InvalidationOracle  # noqa: E402
from sdn_core.task_taxonomy import INTENT_LABELS  # noqa: E402

DEFAULT_ORACLE_DIR = ROOT / "experiments" / "oracle"


def exponential_mle(
    ages: Sequence[int], censored: int, max_age: int
) -> Optional[float]:
    """MLE of the exponential rate under right-censoring.

    lambda = (number of observed failures) / (total observed exposure time),
    where each censored observation contributes ``max_age`` of exposure.
    """
    exposure = sum(ages) + censored * max_age
    if exposure <= 0 or not ages:
        return None
    return len(ages) / exposure


def exponential_fit_quality(
    ages: Sequence[int], rate: float, max_age: int
) -> Dict[str, float]:
    """How closely the empirical survival curve tracks exp(-rate * age).

    Reports the maximum absolute deviation between the empirical and fitted
    survival functions (a Kolmogorov-Smirnov-style statistic). A large value
    means the memorylessness the freshness model assumes does not hold, which is
    worth stating plainly rather than hiding.
    """
    if not ages:
        return {"max_deviation": float("nan"), "mean_deviation": float("nan")}

    ordered = sorted(ages)
    n = len(ordered)
    deviations = []
    for index, age in enumerate(ordered):
        empirical_survival = 1.0 - (index + 1) / n
        fitted_survival = math.exp(-rate * age)
        deviations.append(abs(empirical_survival - fitted_survival))
    return {
        "max_deviation": max(deviations),
        "mean_deviation": statistics.fmean(deviations),
    }


def collect_samples(
    oracle: InvalidationOracle,
    intent: str,
    max_age: int,
    stride: int,
) -> Tuple[List[int], int]:
    """Gather (observed lifetimes, censored count) for one intent."""
    ages: List[int] = []
    censored = 0
    for (rsu_id, timestep) in oracle.signatures:
        if stride > 1 and timestep % stride:
            continue
        for sample in oracle.survival_samples(intent, rsu_id, timestep, max_age):
            if sample is None:
                censored += 1
            else:
                ages.append(sample)
    return ages, censored


def fit_oracle(
    oracle: InvalidationOracle, max_age: int, stride: int
) -> Dict[str, Dict[str, float]]:
    """Fit every intent against one oracle."""
    fitted: Dict[str, Dict[str, float]] = {}
    for index, intent in enumerate(INTENT_LABELS):
        ages, censored = collect_samples(oracle, intent, max_age, stride)
        total = len(ages) + censored
        rate = exponential_mle(ages, censored, max_age)
        quality = exponential_fit_quality(ages, rate, max_age) if rate else {
            "max_deviation": float("nan"), "mean_deviation": float("nan")
        }
        fitted[intent] = {
            "lambda_fitted": rate if rate is not None else 0.0,
            "lambda_hand_set": DECAY_RATES[index],
            "half_life_s": (math.log(2) / rate) if rate else float("inf"),
            "samples": total,
            "observed_failures": len(ages),
            "censored": censored,
            "censored_fraction": censored / total if total else 0.0,
            "median_lifetime_s": statistics.median(ages) if ages else float("nan"),
            "mean_lifetime_s": statistics.fmean(ages) if ages else float("nan"),
            **quality,
        }
    return fitted


def merge_across_oracles(
    per_oracle: List[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """Average fitted rates across mobility realisations."""
    merged: Dict[str, Dict[str, float]] = {}
    for intent in INTENT_LABELS:
        rates = [fit[intent]["lambda_fitted"] for fit in per_oracle]
        merged[intent] = {
            "lambda_fitted": statistics.fmean(rates),
            "lambda_std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "lambda_hand_set": per_oracle[0][intent]["lambda_hand_set"],
            "n_oracles": len(rates),
        }
        rate = merged[intent]["lambda_fitted"]
        merged[intent]["half_life_s"] = (
            math.log(2) / rate if rate > 0 else float("inf")
        )
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--oracle",
        type=Path,
        nargs="+",
        help="One or more oracle pickles. Pass the calibration seeds explicitly "
             "when fitting rates the policy will use, so no evaluation trace "
             "contributes to a parameter it is later scored against.",
    )
    source.add_argument("--oracle-dir", type=Path, default=None,
                        help=f"Directory of oracle pickles (default: {DEFAULT_ORACLE_DIR})")
    parser.add_argument("--max-age", type=int, default=120,
                        help="Right-censoring horizon in seconds (default: 120)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Sample every Nth timestep (default: 1 = all)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the fitted rates to this JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.oracle:
        paths = list(args.oracle)
    else:
        directory = args.oracle_dir or DEFAULT_ORACLE_DIR
        paths = sorted(directory.glob("*.pkl"))
        if not paths:
            print(f"No oracle pickles under {directory}. "
                  "Build one with scripts/build_oracle.py.", file=sys.stderr)
            return 1

    per_oracle = []
    for path in paths:
        oracle = InvalidationOracle.load(path)
        per_oracle.append(fit_oracle(oracle, args.max_age, args.stride))
        print(f"fitted {path.name} ({len(oracle.signatures)} cell-steps)")

    single = per_oracle[0]
    print(f"\n{'intent':<24}{'lambda_fit':>11}{'lambda_hand':>12}{'ratio':>8}"
          f"{'half-life':>11}{'median':>8}{'cens%':>7}{'KS dev':>8}")
    print("-" * 89)
    for intent in INTENT_LABELS:
        row = single[intent]
        hand = row["lambda_hand_set"]
        ratio = (row["lambda_fitted"] / hand) if hand else float("inf")
        half = row["half_life_s"]
        print(f"{intent:<24}{row['lambda_fitted']:>11.4f}{hand:>12.4f}"
              f"{ratio:>8.1f}"
              f"{(f'{half:.1f}s' if half != float('inf') else 'inf'):>11}"
              f"{row['median_lifetime_s']:>8.1f}"
              f"{100 * row['censored_fraction']:>7.1f}"
              f"{row['max_deviation']:>8.3f}")

    result = {
        "per_oracle": {p.name: fit for p, fit in zip(paths, per_oracle)},
        "merged": merge_across_oracles(per_oracle) if len(per_oracle) > 1 else single,
        "max_age_s": args.max_age,
        "stride": args.stride,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
