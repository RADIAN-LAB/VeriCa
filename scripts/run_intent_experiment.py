#!/usr/bin/env python3
"""Run CoCa with a drop-in intent-aware cache, without editing CoCa.

The wrapper loads CoCa's original experiment runner, replaces only the cache
manager constructor and scheduler callable in memory, then appends intent
quality metrics to the original result JSON.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coca_adapter.bootstrap import COCA_ROOT, bootstrap_coca
from coca_adapter.intent_distributed import (
    KEY_MODES,
    REUSE_MODES,
    IntentAwareDistributedCacheManager,
)

bootstrap_coca()

from cache.manager import make_task_key


def load_coca_runner():
    runner_path = COCA_ROOT / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("_coca_run_experiment", runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CoCa runner from {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CoCa with intent/AoI-aware result reuse"
    )
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-mode", choices=sorted(REUSE_MODES), default="intent")
    parser.add_argument(
        "--oracle",
        type=Path,
        default=None,
        help="Invalidation oracle for the mobility trace this task trace came "
             "from (scripts/build_oracle.py). Without it, correctness falls back "
             "to the self-consistent exponential ablation.",
    )
    parser.add_argument(
        "--key-mode",
        choices=sorted(KEY_MODES),
        default="coca",
        help="How cache entries are addressed. 'coca' is the space-blind "
             "baseline key; 'context_rsu' and 'context_zone' scope the key to "
             "the intent and the place the result describes.",
    )
    parser.add_argument(
        "--decay-rates",
        type=Path,
        default=None,
        help="Fitted per-intent decay rates from scripts/fit_decay_rates.py. "
             "Without it the hand-set lambdas are used, which the empirical fit "
             "contradicts by up to 43x.",
    )
    parser.add_argument(
        "--freshness-threshold",
        type=float,
        default=None,
        help="Minimum survival probability required to reuse a result. Higher "
             "values trade cache hits for correct answers.",
    )
    parser.add_argument(
        "--cache-capacity",
        type=int,
        default=None,
        help="Entries per RSU, overriding CoCa's 100. Validity-preserving keys "
             "have a far larger key space than CoCa's 145, so capacity has to be "
             "reported against key cardinality rather than held fixed.",
    )
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ts", type=int, default=None)
    parser.add_argument(
        "--evict-policy",
        choices=["lru", "lfu", "fifo", "random", "none"],
        default=None,
    )
    parser.add_argument("--load-rl", type=Path, default=None)
    parser.add_argument("--no-fl", action="store_true")
    parser.add_argument(
        "--fl-only",
        action="store_true",
        help="Enable FL pre-placement and replace RL decisions with no-op",
    )
    parser.add_argument("--save-rl", type=Path, default=None)
    return parser.parse_args()


def apply_decay_rates(path: Optional[Path]) -> Optional[dict[str, float]]:
    """Install fitted per-intent decay rates in place of the hand-set ones.

    The fit is keyed by intent label; the freshness model is keyed by intent
    index, so the mapping goes through the authoritative taxonomy order. Mutates
    the shared ``DECAY_RATES`` dict because ``FreshnessPolicy`` instances hold a
    reference to it rather than a copy.
    """
    if path is None:
        return None

    from intent.freshness import DECAY_RATES
    from sdn_core.task_taxonomy import INTENT_LABELS as TAXONOMY_LABELS

    with path.open() as handle:
        fitted = json.load(handle)
    table = fitted.get("merged", fitted.get("per_oracle", {}))
    if "merged" not in fitted:
        # A single-oracle fit is stored per file name; take the only entry.
        table = next(iter(table.values())) if table else {}

    applied: dict[str, float] = {}
    for index, label in enumerate(TAXONOMY_LABELS):
        row = table.get(label)
        if row is None:
            continue
        DECAY_RATES[index] = float(row["lambda_fitted"])
        applied[label] = DECAY_RATES[index]
    print(f"Fitted decay rates from {path.name}: "
          + ", ".join(f"{k}={v:.4f}" for k, v in applied.items()))
    return applied


def energy_joules(result: dict[str, Any]) -> float:
    hits = result["cache_hits"]
    misses = result["cache_misses"]
    total = result["total_tasks"]
    success_rate = result["success_rate"]
    return (
        hits * 0.1
        + misses * success_rate * 2.0
        + total * (1.0 - success_rate) * 1.5
    )


def main() -> int:
    args = parse_args()
    os.environ["TASK_GEN_SEED"] = str(args.seed)

    oracle = None
    if args.oracle is not None:
        from intent.oracle import InvalidationOracle

        oracle = InvalidationOracle.load(args.oracle)
        print(f"Loaded oracle {args.oracle.name}: "
              f"{len(oracle.signatures)} cell-steps from {oracle.source}")
    else:
        print("WARNING: no --oracle given; correctness is judged by the "
              "self-consistent exponential model and is not a validity result.",
              file=sys.stderr)

    applied_rates = apply_decay_rates(args.decay_rates)
    if args.freshness_threshold is not None:
        from intent.freshness import freshness_policy

        freshness_policy.threshold = args.freshness_threshold
        print(f"Freshness threshold: {args.freshness_threshold}")

    coca_runner = load_coca_runner()
    # Redirect CoCa's training log into this project; its source tree stays read-only.
    coca_runner.ROOT = ROOT
    captured: dict[str, IntentAwareDistributedCacheManager] = {}

    if args.fl_only:
        class NoOpRLAgent:
            epsilon = 0.0
            step_counter = 0

            def decide(self, _state):
                return "noop"

            def observe(self, _state, _action, _reward, _next_state):
                return None

            def save(self, _path):
                return None

        coca_runner.RLAgent = NoOpRLAgent

    def manager_factory(*factory_args, **factory_kwargs):
        if args.cache_capacity is not None:
            # CoCa passes its capacity by keyword; positionally it is the second
            # argument after rsu_ids.
            if "cache_capacity" in factory_kwargs or len(factory_args) < 2:
                factory_kwargs["cache_capacity"] = args.cache_capacity
            else:
                factory_args = (
                    factory_args[0], args.cache_capacity, *factory_args[2:]
                )
        manager = IntentAwareDistributedCacheManager(
            *factory_args,
            **factory_kwargs,
            reuse_mode=args.reuse_mode,
            threshold=args.threshold,
            seed=args.seed,
            oracle=oracle,
            key_mode=args.key_mode,
        )
        captured["manager"] = manager
        return manager

    original_scheduler = coca_runner.task_scheduling

    def scheduler_with_context(*scheduler_args, **scheduler_kwargs):
        manager = captured["manager"]
        tasks = scheduler_args[0]
        vehicle_attributes = (
            scheduler_args[7]
            if len(scheduler_args) > 7
            else scheduler_kwargs.get("vehicle_attributes", {})
        )
        items = []
        timestamps = []
        for node_id, _task_id, fog_rsu, _final_rsu in tasks:
            attrs = dict(vehicle_attributes.get(node_id, {}))
            attrs["origin_rsu"] = fog_rsu
            # The oracle judges correctness per vehicle, so the requester's
            # identity has to survive the trip through CoCa's key-only lookup.
            attrs["node_id"] = node_id
            attrs.setdefault("task_type", 0)
            attrs.setdefault("priority", 0)
            attrs.setdefault("data_size", 0)
            timestamp = float(attrs.get("time", 0.0))
            timestamps.append(timestamp)
            key = make_task_key(
                attrs["task_type"], attrs["priority"], attrs["data_size"]
            )
            items.append((key, attrs))
        if timestamps:
            manager.set_sim_time(max(timestamps))
        manager.register_tasks(items)
        return original_scheduler(*scheduler_args, **scheduler_kwargs)

    coca_runner.DistributedCacheManager = manager_factory
    coca_runner.task_scheduling = scheduler_with_context

    forwarded = [
        str(COCA_ROOT / "scripts" / "run_experiment.py"),
        "--trace",
        str(args.trace),
        "--output",
        str(args.output),
    ]
    if args.max_ts is not None:
        forwarded.extend(["--max_ts", str(args.max_ts)])
    if args.evict_policy is not None:
        forwarded.extend(["--evict_policy", args.evict_policy])
    if args.load_rl is not None:
        forwarded.extend(["--load_rl", str(args.load_rl)])
    if args.no_fl:
        forwarded.append("--no_fl")
    if args.save_rl is not None:
        forwarded.extend(["--save_rl", str(args.save_rl)])

    old_argv = sys.argv
    try:
        sys.argv = forwarded
        coca_runner.main()
    finally:
        sys.argv = old_argv

    manager = captured["manager"]
    with args.output.open() as result_file:
        result = json.load(result_file)
    result.update(manager.metrics())
    result["energy_joules"] = energy_joules(result)
    result["seed"] = args.seed
    result["reuse_threshold"] = args.threshold
    result["key_mode"] = args.key_mode
    result["decay_rates"] = applied_rates
    result["freshness_threshold"] = args.freshness_threshold
    with args.output.open("w") as result_file:
        json.dump(result, result_file, indent=2)

    print("Intent metrics:")
    print(json.dumps(manager.metrics(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
