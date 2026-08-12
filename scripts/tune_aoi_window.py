#!/usr/bin/env python3
"""Derive refreshing windows for the AoI-aware caching baseline (B4).

B4 serves a cached result while its Age of Information stays inside a window
W(intent) and refreshes it once it ages out. Two window tables are produced,
and the paper should report both:

  assumed  — W from the *hand-set* per-intent decay rates, which is what a
             method designed without measured invalidation data would use. This
             is the honest representation of the AoI-caching literature.
  tuned    — W from the empirical survival function measured on the calibration
             traces, at a target precision tau. This deliberately hands the
             baseline our own measurement contribution, so that any remaining
             gap isolates what key scoping and serve-time verification add
             rather than what better lambdas add.

Reporting only the first would flatter the proposed method; reporting only the
second would hide how the literature actually performs.

Usage:
    python3 scripts/tune_aoi_window.py \
        --oracle experiments/oracle/seed_01.pkl experiments/oracle/seed_02.pkl \
                 experiments/oracle/seed_03.pkl \
        --tau 0.5 0.6 0.7 0.8 0.9 --output-dir experiments/baselines/aoi
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intent.freshness import DECAY_RATES  # noqa: E402
from intent.oracle import InvalidationOracle  # noqa: E402
from sdn_core.task_taxonomy import INTENT_LABELS  # noqa: E402

from fit_decay_rates import collect_samples  # noqa: E402

# Windows are capped so that a never-invalidating class (content_delivery, whose
# measured survival never drops) does not produce an unbounded window that the
# simulator would treat as "never refresh".
MAX_WINDOW_S = 300.0


def survival_curve(ages: Sequence[int], censored: int, max_age: int) -> List[float]:
    """Empirical P(result still correct at age a), a = 0..max_age.

    Censored observations survived the whole horizon, so they count as alive at
    every age; ignoring them would badly understate lifetimes for slow intents.
    """
    total = len(ages) + censored
    if total == 0:
        return [1.0] * (max_age + 1)
    failures_by_age = [0] * (max_age + 2)
    for age in ages:
        if 0 <= age <= max_age:
            failures_by_age[age] += 1

    curve = []
    failed_so_far = 0
    for age in range(max_age + 1):
        failed_so_far += failures_by_age[age]
        curve.append((total - failed_so_far) / total)
    return curve


def window_at_precision(curve: Sequence[float], tau: float) -> float:
    """Largest age whose survival still meets the target precision."""
    window = 0.0
    for age, survival in enumerate(curve):
        if survival >= tau:
            window = float(age)
        else:
            break
    return min(window, MAX_WINDOW_S)


def assumed_windows(tau: float) -> Dict[str, float]:
    """Windows a designer would set from the hand-set decay rates alone."""
    windows = {}
    for index, label in enumerate(INTENT_LABELS):
        rate = DECAY_RATES[index]
        windows[label] = MAX_WINDOW_S if rate <= 0 else min(
            -math.log(tau) / rate, MAX_WINDOW_S
        )
    return windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oracle", type=Path, nargs="+", required=True,
                        help="Calibration oracles only — never an evaluation seed.")
    parser.add_argument("--tau", type=float, nargs="+",
                        default=[0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Target precision levels to emit windows for.")
    parser.add_argument("--max-age", type=int, default=120)
    parser.add_argument("--stride", type=int, default=2,
                        help="Sample every Nth timestep (default 2; the curve is "
                             "an average over many cell-steps and is stable).")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments" / "baselines" / "aoi")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Pool samples across calibration oracles: the window is one number per
    # intent, so it should reflect every calibration realisation at once.
    pooled: Dict[str, tuple[List[int], int]] = {}
    for path in args.oracle:
        oracle = InvalidationOracle.load(path)
        for intent in INTENT_LABELS:
            ages, censored = collect_samples(
                oracle, intent, args.max_age, args.stride
            )
            previous_ages, previous_censored = pooled.get(intent, ([], 0))
            pooled[intent] = (previous_ages + ages, previous_censored + censored)
        print(f"pooled {path.name}")

    curves = {
        intent: survival_curve(ages, censored, args.max_age)
        for intent, (ages, censored) in pooled.items()
    }

    written = []
    for tau in args.tau:
        tuned = {
            intent: window_at_precision(curve, tau)
            for intent, curve in curves.items()
        }
        payload = {
            "windows": tuned,
            "tau": tau,
            "source": "empirical survival, calibration oracles",
            "calibration_oracles": [p.name for p in args.oracle],
        }
        path = args.output_dir / f"tuned_tau{tau:.2f}.json"
        with path.open("w") as handle:
            json.dump(payload, handle, indent=2)
        written.append(path)

        assumed = assumed_windows(tau)
        assumed_path = args.output_dir / f"assumed_tau{tau:.2f}.json"
        with assumed_path.open("w") as handle:
            json.dump(
                {"windows": assumed, "tau": tau, "source": "hand-set decay rates"},
                handle, indent=2,
            )
        written.append(assumed_path)

        print(f"\ntau={tau:.2f}")
        print(f"  {'intent':<24}{'tuned W':>10}{'assumed W':>12}")
        for intent in INTENT_LABELS:
            print(f"  {intent:<24}{tuned[intent]:>10.1f}{assumed[intent]:>12.1f}")

    print(f"\nWrote {len(written)} window tables to {args.output_dir}")
    print("Sweep tau on a calibration seed and carry the best into evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
