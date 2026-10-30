#!/usr/bin/env python3
"""Ablation: a predictive freshness gate layered on top of verification.

The proposed arm serves any context-scoped entry that passes serve-time
verification. This script adds a freshness gate in front of it — an entry is
offered for verification only while its fitted survival probability stays
above the threshold — and runs it on the evaluation seeds under the frozen
protocol (plain LRU, no FL placement, verified regime). The decay rates come
from the calibration split only, and the threshold matches the tau used for
the AoI refreshing windows, so the gate is the best-informed predictor the
calibration data supports. Each seed is paired with the frozen proposed run.

Usage:
    python3 scripts/run_freshness_gate_ablation.py --jobs 6
    python3 scripts/run_freshness_gate_ablation.py --aggregate-only
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
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "experiments" / "evaluation"
ORACLE_DIR = ROOT / "experiments" / "oracle"
OUT_DIR = ROOT / "experiments" / "freshness_gate"
DECAY_RATES = ROOT / "experiments" / "figures" / "fitted_decay_rates_calib.json"
FRESHNESS_THRESHOLD = 0.8

EVALUATION_SEEDS = list(range(4, 31))

GATED_ARGS = [
    "--evict-policy", "lru", "--no-fl",
    "--reuse-mode", "freshness_only", "--key-mode", "context_adaptive",
    "--decay-rates", str(DECAY_RATES),
    "--freshness-threshold", str(FRESHNESS_THRESHOLD),
    "--verify-before-serve",
]


def run_seed(seed: int) -> None:
    trace = EVAL_DIR / f"seed_{seed:02d}" / "tasks.csv"
    if not trace.exists():
        raise FileNotFoundError(f"No task trace for seed {seed}: {trace}")
    output = OUT_DIR / f"seed_{seed:02d}" / "p_proposed_gated__verified.json"
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["TASK_GEN_SEED"] = str(seed)
    command = [
        sys.executable, str(ROOT / "scripts" / "run_intent_experiment.py"),
        "--trace", str(trace), "--output", str(output),
        "--seed", str(seed), *GATED_ARGS,
        "--oracle", str(ORACLE_DIR / f"seed_{seed:02d}.pkl"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(f"seed {seed}:\n{result.stdout[-800:]}\n"
                           f"{result.stderr[-1500:]}")


def aggregate() -> dict:
    pairs: Dict[int, dict] = {}
    for seed in EVALUATION_SEEDS:
        gated = OUT_DIR / f"seed_{seed:02d}" / "p_proposed_gated__verified.json"
        ungated = EVAL_DIR / f"seed_{seed:02d}" / "p_proposed__verified.json"
        if not (gated.exists() and ungated.exists()):
            continue
        with gated.open() as handle:
            g = json.load(handle)
        with ungated.open() as handle:
            u = json.load(handle)
        pairs[seed] = {
            "usable_hit_rate_gated": g["valid_hit_rate"],
            "usable_hit_rate_ungated": u["valid_hit_rate"],
            "average_completion_gated": g["average_completion"],
            "average_completion_ungated": u["average_completion"],
        }

    def stats(key: str) -> dict:
        values = [p[key] for p in pairs.values()]
        return {"mean": statistics.fmean(values), "std": statistics.stdev(values)}

    diffs = [p["usable_hit_rate_ungated"] - p["usable_hit_rate_gated"]
             for p in pairs.values()]
    return {
        "n": len(pairs),
        "decay_rates": str(DECAY_RATES.relative_to(ROOT)),
        "freshness_threshold": FRESHNESS_THRESHOLD,
        "usable_hit_rate_gated": stats("usable_hit_rate_gated"),
        "usable_hit_rate_ungated": stats("usable_hit_rate_ungated"),
        "average_completion_gated": stats("average_completion_gated"),
        "average_completion_ungated": stats("average_completion_ungated"),
        "loss_from_gate": {
            "mean": statistics.fmean(diffs), "std": statistics.stdev(diffs),
            "min": min(diffs), "max": max(diffs),
            "seeds_where_gate_helps": sorted(
                s for s, p in pairs.items()
                if p["usable_hit_rate_gated"] > p["usable_hit_rate_ungated"]),
        },
        "per_seed": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    if not args.aggregate_only:
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            futures = {pool.submit(run_seed, s): s for s in EVALUATION_SEEDS}
            done = 0
            for future in concurrent.futures.as_completed(futures):
                future.result()
                done += 1
                print(f"[{done}/{len(futures)}] seed {futures[future]}", flush=True)

    summary = aggregate()
    out = OUT_DIR / "aggregated_results.json"
    with out.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"n={summary['n']}: gated "
          f"{summary['usable_hit_rate_gated']['mean']:.4f} vs ungated "
          f"{summary['usable_hit_rate_ungated']['mean']:.4f} "
          f"(loss {summary['loss_from_gate']['mean']:+.4f}); "
          f"gate helps on seeds {summary['loss_from_gate']['seeds_where_gate_helps']}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
