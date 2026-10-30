#!/usr/bin/env python3
"""Quantify the circularity bias: the same runs judged two ways.

Prior evaluations that model validity sample each result's lifetime from the
same decay rates the freshness policy uses to decide whether to serve, so the
policy is graded against its own assumption. This script runs the published
LRU and AoI-assumed arms on the evaluation seeds twice — once judged by the
trace-grounded oracle, once with the oracle omitted so correctness falls back
to the self-consistent exponential construction. The runs are deterministic,
so each pair has identical hit streams and the only thing that changes is the
judge. The gap between the two valid-hit-rate columns is the bias quoted in
the problem statement.

Usage:
    python3 scripts/run_circularity_ablation.py --jobs 5
    python3 scripts/run_circularity_ablation.py --aggregate-only
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "experiments" / "evaluation"
ORACLE_DIR = ROOT / "experiments" / "oracle"
OUT_DIR = ROOT / "experiments" / "circularity"
AOI_DIR = ROOT / "experiments" / "baselines" / "aoi"

EVALUATION_SEEDS = list(range(4, 31))

# Published regime only: the ablation is about how the *reported* figure is
# judged, and the verified regime never relies on the sampled lifetimes.
ARMS: Dict[str, List[str]] = {
    "b1_lru": ["--evict-policy", "lru", "--no-fl",
               "--reuse-mode", "coca", "--key-mode", "coca"],
    "b4_aoi_assumed": ["--evict-policy", "lru", "--no-fl",
                       "--reuse-mode", "aoi_window", "--key-mode", "coca",
                       "--aoi-window", str(AOI_DIR / "assumed_tau0.80.json")],
}

JUDGES = ("oracle", "exponential")


def run_arm(seed: int, arm: str, judge: str) -> None:
    trace = EVAL_DIR / f"seed_{seed:02d}" / "tasks.csv"
    if not trace.exists():
        raise FileNotFoundError(f"No task trace for seed {seed}: {trace}")
    output = OUT_DIR / f"seed_{seed:02d}" / f"{arm}__{judge}.json"
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["TASK_GEN_SEED"] = str(seed)
    command = [
        sys.executable, str(ROOT / "scripts" / "run_intent_experiment.py"),
        "--trace", str(trace), "--output", str(output),
        "--seed", str(seed), *ARMS[arm],
    ]
    if judge == "oracle":
        command.extend(["--oracle", str(ORACLE_DIR / f"seed_{seed:02d}.pkl")])
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(f"{arm}/{judge} seed {seed}:\n{result.stdout[-800:]}\n"
                           f"{result.stderr[-1500:]}")


def aggregate() -> dict:
    summary: Dict[str, dict] = {}
    for arm in ARMS:
        pairs = {}
        for seed in EVALUATION_SEEDS:
            paths = {j: OUT_DIR / f"seed_{seed:02d}" / f"{arm}__{j}.json"
                     for j in JUDGES}
            if not all(p.exists() for p in paths.values()):
                continue
            runs = {}
            for judge, path in paths.items():
                with path.open() as handle:
                    runs[judge] = json.load(handle)
            if abs(runs["oracle"]["hit_rate"]
                   - runs["exponential"]["hit_rate"]) > 1e-12:
                raise RuntimeError(
                    f"{arm} seed {seed}: hit rates diverge; the two runs are "
                    "not the same policy execution and cannot be paired.")
            # The frozen evaluation run for this seed/arm, for the
            # load-contamination scan (it should match the fresh oracle run).
            frozen_path = EVAL_DIR / f"seed_{seed:02d}" / f"{arm}__published.json"
            frozen_hit = None
            if frozen_path.exists():
                with frozen_path.open() as handle:
                    frozen_hit = json.load(handle)["hit_rate"]
            pairs[seed] = {
                "hit_rate": runs["oracle"]["hit_rate"],
                "valid_hit_rate_exponential": runs["exponential"]["valid_hit_rate"],
                "valid_hit_rate_oracle": runs["oracle"]["valid_hit_rate"],
                "frozen_evaluation_hit_rate": frozen_hit,
                "frozen_matches": (frozen_hit is not None and
                                   abs(frozen_hit - runs["oracle"]["hit_rate"]) < 1e-12),
            }
        exp_vals = [p["valid_hit_rate_exponential"] for p in pairs.values()]
        ora_vals = [p["valid_hit_rate_oracle"] for p in pairs.values()]
        diffs = [e - o for e, o in zip(exp_vals, ora_vals)]
        summary[arm] = {
            "n": len(pairs),
            "valid_hit_rate_exponential": {
                "mean": statistics.fmean(exp_vals),
                "std": statistics.stdev(exp_vals),
            },
            "valid_hit_rate_oracle": {
                "mean": statistics.fmean(ora_vals),
                "std": statistics.stdev(ora_vals),
            },
            "bias": {
                "mean": statistics.fmean(diffs),
                "std": statistics.stdev(diffs),
                "min": min(diffs), "max": max(diffs),
            },
            "frozen_mismatched_seeds": sorted(
                s for s, p in pairs.items() if not p["frozen_matches"]),
            "per_seed": pairs,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    if not args.aggregate_only:
        jobs = [(seed, arm, judge) for seed in EVALUATION_SEEDS
                for arm in ARMS for judge in JUDGES]
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            futures = {pool.submit(run_arm, *job): job for job in jobs}
            done = 0
            for future in concurrent.futures.as_completed(futures):
                future.result()
                done += 1
                print(f"[{done}/{len(jobs)}] {futures[future]}", flush=True)

    summary = aggregate()
    out = OUT_DIR / "aggregated_results.json"
    with out.open("w") as handle:
        json.dump(summary, handle, indent=2)
    for arm, stats in summary.items():
        print(f"{arm} (n={stats['n']}): exponential judge "
              f"{stats['valid_hit_rate_exponential']['mean']:.3f} vs oracle "
              f"{stats['valid_hit_rate_oracle']['mean']:.3f} "
              f"(bias {stats['bias']['mean']:+.3f}); frozen mismatches: "
              f"{stats['frozen_mismatched_seeds']}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
