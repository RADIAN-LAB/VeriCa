#!/usr/bin/env python3
"""Run CoCa baselines, Intent-NLP, and ablations over multiple seeds."""

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
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coca_adapter.bootstrap import COCA_ROOT

FCD_TRACE = COCA_ROOT / "experiments" / "sumo" / "output" / "fcd_trace.xml"
RL_MODEL = COCA_ROOT / "experiments" / "models" / "final_rl_trained.pt"
OUTPUT_ROOT = ROOT / "experiments" / "multi_seed"
# Every seed varies task generation over the *same* mobility trace, so one
# invalidation oracle serves them all. Mobility seeds would each need their own.
ORACLE = ROOT / "experiments" / "oracle" / "base.pkl"

METHODS = {
    "1_no_cache": {
        "name": "No Cache",
        "args": ["--reuse-mode", "coca", "--evict-policy", "none", "--no-fl"],
    },
    "2_random": {
        "name": "Random",
        "args": ["--reuse-mode", "coca", "--evict-policy", "random", "--no-fl"],
    },
    "3_lru": {
        "name": "LRU",
        "args": ["--reuse-mode", "coca", "--evict-policy", "lru", "--no-fl"],
    },
    "4_lfu": {
        "name": "LFU",
        "args": ["--reuse-mode", "coca", "--evict-policy", "lfu", "--no-fl"],
    },
    "5_fl_only": {
        "name": "FL-Only",
        "args": ["--reuse-mode", "coca", "--fl-only"],
    },
    "6_coca": {
        "name": "CoCa (FL+RL)",
        "args": ["--reuse-mode", "coca", "--load-rl", str(RL_MODEL)],
    },
    "7_intent_nlp": {
        "name": "Intent-NLP (full)",
        "args": ["--reuse-mode", "intent", "--load-rl", str(RL_MODEL)],
    },
    "8_freshness_only": {
        "name": "Freshness-only",
        "args": ["--reuse-mode", "freshness_only", "--load-rl", str(RL_MODEL)],
    },
    "9_intent_only": {
        "name": "Intent-only",
        "args": ["--reuse-mode", "intent_only", "--load-rl", str(RL_MODEL)],
    },
}

METRICS = [
    "success_rate",
    "average_completion",
    "cache_hits",
    "cache_misses",
    # Under an oracle run, "valid" means the served result was actually the right
    # answer, and "stale" means it was not — for any reason, ageing included.
    "valid_hit_rate",
    "stale_reuse_rate",
    "semantic_cache_hits",
    "semantic_precision",
    "semantic_error_rate",
    "energy_joules",
    "runtime_seconds",
]


def run_command(command: list[str], seed: int) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["TASK_GEN_SEED"] = str(seed)
    return subprocess.run(command, capture_output=True, text=True, env=env)


def generate_trace(seed: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(COCA_ROOT / "experiments" / "sumo" / "sumo_to_tasks.py"),
        "--fcd",
        str(FCD_TRACE),
        "--output",
        str(output),
    ]
    result = run_command(command, seed)
    if result.returncode:
        raise RuntimeError(f"Trace generation failed for seed {seed}:\n{result.stderr}")


def run_method(
    seed: int,
    trace: Path,
    method_id: str,
    output: Path,
    max_ts: Optional[int],
    oracle: Optional[Path] = None,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_intent_experiment.py"),
        "--trace",
        str(trace),
        "--output",
        str(output),
        "--seed",
        str(seed),
        *METHODS[method_id]["args"],
    ]
    if oracle is not None:
        command.extend(["--oracle", str(oracle)])
    if max_ts is not None:
        command.extend(["--max-ts", str(max_ts)])
    result = run_command(command, seed)
    if result.returncode:
        raise RuntimeError(
            f"{method_id} failed for seed {seed}:\n"
            f"{result.stdout[-1000:]}\n{result.stderr[-2000:]}"
        )


def aggregate(seeds: list[int], method_ids: list[str], out_dir: Path) -> dict:
    summary = {}
    for method_id in method_ids:
        runs = []
        for seed in seeds:
            path = out_dir / f"seed_{seed:02d}" / f"{method_id}.json"
            if path.exists():
                with path.open() as result_file:
                    runs.append(json.load(result_file))
        if not runs:
            continue
        metrics = {}
        for metric in METRICS:
            values = [float(run.get(metric, 0.0)) for run in runs]
            metrics[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
        summary[method_id] = {
            "name": METHODS[method_id]["name"],
            "metrics": metrics,
            "raw": runs,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--max-ts", type=int, default=None)
    parser.add_argument(
        "--oracle",
        type=Path,
        default=ORACLE,
        help=f"Invalidation oracle for the mobility trace (default: {ORACLE}). "
             "Built automatically if absent.",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Judge correctness by the self-consistent exponential model "
             "instead. Only meaningful as the ablation that measures how much "
             "that circularity inflates the reported validity.",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Run this many method-seed combinations concurrently (default: 1). "
             "Runs are independent subprocesses; each writes its own FL log.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Where seed_XX/ result directories are written (default: "
             "experiments/multi_seed). Point sweeps and smoke tests at a "
             "scratch directory so canonical results are never overwritten.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
    )
    return parser.parse_args()


def _ensure_oracle(path: Path) -> Path:
    """Return the oracle for the mobility trace, building it if needed."""
    if path.exists():
        print(f"oracle: {path}")
        return path
    from intent.oracle import build_oracle

    print(f"oracle: building from {FCD_TRACE}")
    started = time.time()
    oracle = build_oracle(FCD_TRACE)
    oracle.save(path)
    print(f"oracle: {len(oracle.signatures)} cell-steps "
          f"in {time.time() - started:.1f}s -> {path}")
    return path


def _ensure_traces(seeds: list[int], out_dir: Path, jobs: int, force: bool) -> dict[int, Path]:
    """Generate every seed's task trace, in parallel, before any method runs."""
    traces = {seed: out_dir / f"seed_{seed:02d}" / "tasks.csv" for seed in seeds}
    pending = [s for s in seeds if force or not traces[s].exists()]
    for seed in seeds:
        if seed not in pending:
            print(f"[seed {seed}] trace: already present")

    if not pending:
        return traces

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        futures = {
            pool.submit(generate_trace, seed, traces[seed]): seed for seed in pending
        }
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            future.result()  # re-raise: without a trace the seed cannot run
            print(f"[seed {seed}] trace: generated")
    return traces


def _run_all_methods(
    seeds: list[int],
    traces: dict[int, Path],
    method_ids: list[str],
    out_dir: Path,
    max_ts: Optional[int],
    jobs: int,
    force: bool,
    oracle: Optional[Path] = None,
) -> list[str]:
    """Run every (seed, method) combination, returning a list of failures."""
    work = []
    for seed in seeds:
        for method_id in method_ids:
            output = out_dir / f"seed_{seed:02d}" / f"{method_id}.json"
            if output.exists() and not force:
                print(f"[seed {seed}] {method_id}: already complete")
                continue
            work.append((seed, method_id, output))

    failures: list[str] = []
    if not work:
        return failures

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        futures = {
            pool.submit(run_method, seed, traces[seed], method_id, output,
                        max_ts, oracle):
                (seed, method_id)
            for seed, method_id, output in work
        }
        for future in concurrent.futures.as_completed(futures):
            seed, method_id = futures[future]
            done += 1
            try:
                future.result()
                print(f"[{done}/{len(work)}] seed {seed} {method_id}: ok")
            except Exception as exc:  # noqa: BLE001 — collect and continue
                failures.append(f"seed {seed} {method_id}: {exc}")
                print(f"[{done}/{len(work)}] seed {seed} {method_id}: FAILED",
                      file=sys.stderr)
    return failures


def main() -> int:
    args = parse_args()
    seeds = list(range(1, args.seeds + 1))
    started = time.time()
    failures: list[str] = []

    if not args.aggregate_only:
        oracle = None if args.no_oracle else _ensure_oracle(args.oracle)
        traces = _ensure_traces(seeds, args.out_dir, args.jobs, args.force)
        failures = _run_all_methods(
            seeds, traces, args.methods, args.out_dir, args.max_ts,
            args.jobs, args.force, oracle,
        )

    result = aggregate(seeds, args.methods, args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "aggregated_results.json"
    with output.open("w") as result_file:
        json.dump(result, result_file, indent=2)
    print(f"Saved {output} ({time.time() - started:.1f}s)")
    for method in result.values():
        metrics = method["metrics"]
        n = metrics["valid_hit_rate"]["n"]
        print(
            f"{method['name']:<22} n={n:<3} "
            f"correct={100 * metrics['valid_hit_rate']['mean']:6.2f}% "
            f"wrong={100 * metrics['stale_reuse_rate']['mean']:6.2f}% "
            f"sem_prec={100 * metrics['semantic_precision']['mean']:6.2f}% "
            f"success={100 * metrics['success_rate']['mean']:6.2f}%"
        )

    if failures:
        print(f"\n{len(failures)} run(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        # Aggregates above are computed over whatever completed, so a partial
        # run must not look like a clean one.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
