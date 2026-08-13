#!/usr/bin/env python3
"""Run the frozen arm set over independent mobility realisations.

Each evaluation seed is a *different SUMO run*, paired with the invalidation
oracle built from that same trace, so confidence intervals measure traffic
variability rather than task-generation noise. Seeds 01-03 are the calibration
split — every fitted parameter (decay rates, AoI windows, reuse thresholds) comes
from those and from nowhere else, so no arm is ever scored on a trace it was
tuned on.

Every arm runs twice:

  published — the method as its own literature defines it. The column that
              matters is how often it answers wrongly, which no prior evaluation
              of these methods reports.
  verified  — the RSU checks an entry against observed state before serving and
              recomputes on mismatch, so every arm delivers only correct answers
              and their latencies are finally comparable.

Usage:
    python3 scripts/run_evaluation.py --jobs 6
    python3 scripts/run_evaluation.py --aggregate-only
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coca_adapter.bootstrap import COCA_ROOT  # noqa: E402

MOBILITY_DIR = COCA_ROOT / "experiments" / "sumo" / "mobility"
ORACLE_DIR = ROOT / "experiments" / "oracle"
RL_MODEL = COCA_ROOT / "experiments" / "models" / "final_rl_trained.pt"
AOI_DIR = ROOT / "experiments" / "baselines" / "aoi"
DEFAULT_OUT = ROOT / "experiments" / "evaluation"

CALIBRATION_SEEDS = [1, 2, 3]

# Plain LRU is the substrate for everything except the CoCa arm, so the
# comparison never depends on unpublished prior work.
LRU_BASE = ["--evict-policy", "lru", "--no-fl"]

ARMS: Dict[str, Dict[str, object]] = {
    "b0_random": {
        "name": "Random",
        "args": ["--evict-policy", "random", "--no-fl",
                 "--reuse-mode", "coca", "--key-mode", "coca"],
    },
    "b0_fifo": {
        "name": "FIFO",
        "args": ["--evict-policy", "fifo", "--no-fl",
                 "--reuse-mode", "coca", "--key-mode", "coca"],
    },
    "b1_lru": {
        "name": "LRU",
        "args": [*LRU_BASE, "--reuse-mode", "coca", "--key-mode", "coca"],
    },
    "b2_lfu": {
        "name": "LFU",
        "args": ["--evict-policy", "lfu", "--no-fl",
                 "--reuse-mode", "coca", "--key-mode", "coca"],
    },
    "b4_aoi_assumed": {
        "name": "AoI window (assumed)",
        "args": [*LRU_BASE, "--reuse-mode", "aoi_window", "--key-mode", "coca",
                 "--aoi-window", str(AOI_DIR / "assumed_tau0.80.json")],
    },
    "b4_aoi_tuned": {
        "name": "AoI window (tuned)",
        "args": [*LRU_BASE, "--reuse-mode", "aoi_window", "--key-mode", "coca",
                 "--aoi-window", str(AOI_DIR / "tuned_tau0.80.json")],
    },
    "b5_foggycache": {
        "name": "FoggyCache reuse",
        "args": [*LRU_BASE, "--reuse-mode", "foggycache", "--key-mode", "coca"],
    },
    "b6_coca": {
        "name": "CoCa (FL+RL)",
        "args": ["--reuse-mode", "coca", "--key-mode", "coca",
                 "--load-rl", str(RL_MODEL)],
    },
    "p_proposed": {
        "name": "Proposed",
        "args": [*LRU_BASE, "--reuse-mode", "coca",
                 "--key-mode", "context_adaptive"],
    },
}

REGIMES = {"published": [], "verified": ["--verify-before-serve"]}

# Reported for both regimes; correctness means different things in each, which
# is why the two are never merged into one table.
METRICS = [
    "hit_rate", "valid_hit_rate", "stale_reuse_rate", "semantic_error_rate",
    "success_rate", "average_completion", "cache_hits", "cache_misses",
    "energy_joules", "verification_failures", "distinct_keys_seen",
]


def run_command(command: List[str], seed: int) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["TASK_GEN_SEED"] = str(seed)
    return subprocess.run(command, capture_output=True, text=True, env=env)


def generate_trace(seed: int, output: Path) -> None:
    """Build this seed's task trace from *its own* mobility realisation."""
    fcd = MOBILITY_DIR / f"seed_{seed:02d}" / "fcd_trace.xml"
    if not fcd.exists():
        raise FileNotFoundError(
            f"No mobility trace for seed {seed}: {fcd}. Build it with "
            "scripts/build_mobility.py in the CoCa tree."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_command([
        sys.executable,
        str(COCA_ROOT / "experiments" / "sumo" / "sumo_to_tasks.py"),
        "--fcd", str(fcd), "--output", str(output),
    ], seed)
    if result.returncode:
        raise RuntimeError(f"Trace generation failed for seed {seed}:\n"
                           f"{result.stderr[-2000:]}")


def run_arm(seed: int, trace: Path, arm: str, regime: str, output: Path,
            max_ts: Optional[int], oracle_dir: Path = ORACLE_DIR,
            aoi_dir: Path = AOI_DIR) -> None:
    oracle = oracle_dir / f"seed_{seed:02d}.pkl"
    if not oracle.exists():
        raise FileNotFoundError(
            f"No oracle for seed {seed}: {oracle}. Build it with "
            "scripts/build_oracle.py --mobility-dir."
        )
    arm_args = [
        str(aoi_dir / Path(a).name) if a.endswith(".json") else a
        for a in ARMS[arm]["args"]
    ]
    command = [
        sys.executable, str(ROOT / "scripts" / "run_intent_experiment.py"),
        "--trace", str(trace), "--output", str(output),
        "--oracle", str(oracle), "--seed", str(seed),
        *arm_args, *REGIMES[regime],
    ]
    if max_ts is not None:
        command.extend(["--max-ts", str(max_ts)])
    result = run_command(command, seed)
    if result.returncode:
        raise RuntimeError(f"{arm}/{regime} seed {seed}:\n"
                           f"{result.stdout[-800:]}\n{result.stderr[-1500:]}")


def aggregate(seeds: List[int], out_dir: Path) -> dict:
    summary: Dict[str, dict] = {}
    for regime in REGIMES:
        for arm in ARMS:
            runs = []
            for seed in seeds:
                path = out_dir / f"seed_{seed:02d}" / f"{arm}__{regime}.json"
                if path.exists():
                    with path.open() as handle:
                        runs.append(json.load(handle))
            if not runs:
                continue
            metrics = {}
            for metric in METRICS:
                values = [float(run.get(metric, 0.0) or 0.0) for run in runs]
                metrics[metric] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "median": statistics.median(values),
                    "min": min(values), "max": max(values), "n": len(values),
                }
            summary[f"{arm}__{regime}"] = {
                "name": ARMS[arm]["name"], "arm": arm, "regime": regime,
                "metrics": metrics,
                # Per-seed values are kept so the paired tests can run later
                # without re-reading every result file.
                "per_seed": {
                    str(seed): {m: run.get(m) for m in METRICS}
                    for seed, run in zip(seeds, runs)
                },
            }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=30,
                        help="Highest mobility seed to use (default: 30).")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--max-ts", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--oracle-dir", type=Path, default=ORACLE_DIR,
                        help="Oracle pickles for this scenario's mobility.")
    parser.add_argument("--aoi-dir", type=Path, default=AOI_DIR,
                        help="B4 window tables tuned on this scenario's "
                             "calibration seeds — windows do not transfer "
                             "between maps.")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--regimes", nargs="+", choices=list(REGIMES),
                        default=list(REGIMES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [s for s in range(1, args.seeds + 1) if s not in CALIBRATION_SEEDS]
    started = time.time()
    failures: List[str] = []

    if not args.aggregate_only:
        traces = {
            seed: args.out_dir / f"seed_{seed:02d}" / "tasks.csv" for seed in seeds
        }
        pending = [s for s in seeds if args.force or not traces[s].exists()]
        if pending:
            print(f"Generating {len(pending)} task traces...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
                for future in concurrent.futures.as_completed({
                    pool.submit(generate_trace, s, traces[s]): s for s in pending
                }):
                    future.result()
            print("Traces ready.")

        work = []
        for seed in seeds:
            for arm in args.arms:
                for regime in args.regimes:
                    output = args.out_dir / f"seed_{seed:02d}" / f"{arm}__{regime}.json"
                    if output.exists() and not args.force:
                        continue
                    work.append((seed, arm, regime, output))

        print(f"{len(work)} runs over {len(seeds)} independent mobility seeds "
              f"({args.jobs} at a time)")
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(run_arm, seed, traces[seed], arm, regime, output,
                            args.max_ts, args.oracle_dir, args.aoi_dir):
                    (seed, arm, regime)
                for seed, arm, regime, output in work
            }
            for future in concurrent.futures.as_completed(futures):
                seed, arm, regime = futures[future]
                done += 1
                try:
                    future.result()
                    print(f"[{done}/{len(work)}] seed {seed:02d} {arm} {regime}")
                except Exception as exc:  # noqa: BLE001 — collect and continue
                    failures.append(f"seed {seed} {arm}/{regime}: {exc}")
                    print(f"[{done}/{len(work)}] seed {seed:02d} {arm} {regime} "
                          f"FAILED", file=sys.stderr)

    summary = aggregate(seeds, args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "aggregated_results.json"
    with output.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nSaved {output} ({time.time() - started:.0f}s)")
    for regime in args.regimes:
        print(f"\n{regime.upper()}")
        print(f"  {'arm':<24}{'hit%':>8}{'correct%':>10}{'ms':>9}{'n':>4}")
        for key, entry in summary.items():
            if entry["regime"] != regime:
                continue
            m = entry["metrics"]
            print(f"  {entry['name']:<24}{100 * m['hit_rate']['mean']:8.1f}"
                  f"{100 * m['valid_hit_rate']['mean']:10.1f}"
                  f"{m['average_completion']['mean']:9.2f}"
                  f"{m['valid_hit_rate']['n']:4d}")

    if failures:
        print(f"\n{len(failures)} run(s) failed:", file=sys.stderr)
        for failure in failures[:20]:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
