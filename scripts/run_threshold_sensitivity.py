#!/usr/bin/env python3
"""Evaluate the intent reuse threshold over repeated SUMO task traces."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coca_adapter.bootstrap import COCA_ROOT

RL_MODEL = COCA_ROOT / "experiments" / "models" / "final_rl_trained.pt"
TRACE_ROOT = ROOT / "experiments" / "multi_seed"
OUTPUT_ROOT = ROOT / "experiments" / "sensitivity"
METRICS = ["valid_hit_rate", "stale_reuse_rate", "success_rate", "cache_hits"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.55, 0.60, 0.65, 0.70, 0.75]
    )
    parser.add_argument("--max-ts", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    all_results = {}
    for threshold in args.thresholds:
        runs = []
        threshold_dir = OUTPUT_ROOT / f"threshold_{threshold:.2f}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(1, args.seeds + 1):
            trace = TRACE_ROOT / f"seed_{seed:02d}" / "tasks.csv"
            if not trace.exists():
                raise FileNotFoundError(
                    f"{trace} is missing; run run_multi_seed_intent.py first"
                )
            output = threshold_dir / f"seed_{seed:02d}.json"
            if args.force or not output.exists():
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_intent_experiment.py"),
                    "--trace",
                    str(trace),
                    "--output",
                    str(output),
                    "--reuse-mode",
                    "intent",
                    "--threshold",
                    str(threshold),
                    "--seed",
                    str(seed),
                    "--load-rl",
                    str(RL_MODEL),
                ]
                if args.max_ts is not None:
                    command.extend(["--max-ts", str(args.max_ts)])
                env = dict(os.environ)
                env["PYTHONPATH"] = str(ROOT)
                result = subprocess.run(
                    command, capture_output=True, text=True, env=env
                )
                if result.returncode:
                    raise RuntimeError(result.stderr[-2000:])
            with output.open() as result_file:
                runs.append(json.load(result_file))

        aggregate = {}
        for metric in METRICS:
            values = [float(run[metric]) for run in runs]
            aggregate[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        all_results[str(threshold)] = aggregate

    with (OUTPUT_ROOT / "aggregated_results.json").open("w") as output_file:
        json.dump(all_results, output_file, indent=2)
    print(f"Saved threshold results to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
