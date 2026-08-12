#!/usr/bin/env python3
"""Physical energy accounting for a run, replacing the invented formula.

What it replaces
----------------
``run_intent_experiment.energy_joules`` computes

    hits * 0.1 + misses * success_rate * 2.0 + total * (1 - success_rate) * 1.5

with three flat per-task constants and no physical basis. The middle term
multiplies a count of *misses* by a *success rate*, which corresponds to no
quantity in the system, and the reported "-53% energy" is the hit rate restated
in joules. The same three constants appear in CoCa's two plotting scripts.

The model here
--------------
Energy is charged where the system actually spends it, and every term is derived
from counters the run already records, so results do not need regenerating:

    E = P_cpu  * (T_reactive + T_speculative + T_verify)      compute
      + E_bit  * bits transferred                             radio
      + P_idle * cache residency * wall time                  storage

``T_reactive`` is the time spent computing results for requests the cache could
not answer — one computation per miss. Task computation time is taken from the
system's own timing model (``sdn_core.timing.estimate_computation_time``) in
expectation over the seed's task mix, since per-task realised times are not
recorded in aggregate. That expectation is exact in the deterministic factors
(type, priority, size) and averages the two random ones, whose means are 1.25
(load) and 1.0 (jitter).

``T_speculative`` is what predictive placement costs whether or not the
prediction was right: CoCa issues ~1100 speculative computations per run at 58%
precision, so 42% is waste that any honest energy figure has to show.

Constants
---------
The three constants below are the model's only free parameters, and the
conclusions must not rest on them — ``--sensitivity`` re-runs the comparison at
+/-50% on each. **They are placeholders pending citation**: fill in values from
the edge-server and C-V2X literature before submission, and state them in the
paper with their sources.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coca_path  # noqa: F401,E402 — puts the CoCa tree on sys.path

from sdn_core.task_taxonomy import compute_factor  # noqa: E402
from sdn_core.timing import (  # noqa: E402
    ALPHA,
    COMPUTATION_BASE_TIME,
)

# --- Free parameters (PLACEHOLDERS — cite before submission) ---------------
# Active draw of the RSU's compute element while running a task.
P_CPU_W = 30.0
# Radio energy per transmitted bit, short-range V2X.
E_BIT_J = 1.0e-7
# Idle draw attributable to holding cached results.
P_CACHE_IDLE_W = 1.0

# Mean of the uniform load factor in estimate_computation_time: 1 + U(0, 0.5).
MEAN_LOAD_FACTOR = 1.25
BITS_PER_KB = 8000.0


def expected_computation_ms(task_type: int, priority: int, size_kb: float) -> float:
    """Expected fresh-computation time, mirroring the system's timing model."""
    return (
        COMPUTATION_BASE_TIME
        * compute_factor(task_type)
        * (1.0 / (priority + 1))
        * (1.0 + (size_kb / 1000.0) * 0.5)
        * MEAN_LOAD_FACTOR
    )


def trace_profile(trace: Path) -> Dict[str, float]:
    """Mean per-task computation time and result size for one seed's workload."""
    computation_times: List[float] = []
    result_bits: List[float] = []
    with trace.open() as handle:
        for row in csv.DictReader(handle):
            task_type = int(row["task_type"])
            priority = int(row["priority"])
            size_kb = float(row["data_size_kB"])
            computation_times.append(
                expected_computation_ms(task_type, priority, size_kb)
            )
            # A result is ALPHA times its input, per the delivery model.
            result_bits.append(ALPHA * size_kb * BITS_PER_KB)
    if not computation_times:
        raise ValueError(f"No tasks in {trace}")
    return {
        "mean_computation_ms": statistics.fmean(computation_times),
        "mean_result_bits": statistics.fmean(result_bits),
        "tasks": len(computation_times),
    }


def energy_joules(
    result: dict,
    profile: Dict[str, float],
    p_cpu_w: float = P_CPU_W,
    e_bit_j: float = E_BIT_J,
    p_cache_idle_w: float = P_CACHE_IDLE_W,
) -> Dict[str, float]:
    """Break a run's energy into the terms that produced it."""
    misses = float(result.get("cache_misses", 0))
    hits = float(result.get("cache_hits", 0))
    requests = misses + hits

    reactive_ms = misses * profile["mean_computation_ms"]
    speculative_ms = float(result.get("proactive_compute_ms", 0.0) or 0.0)
    verify_ms = float(result.get("verification_ms_total", 0.0) or 0.0)

    compute_j = p_cpu_w * (reactive_ms + speculative_ms + verify_ms) / 1000.0

    # Every answered request delivers a result to the vehicle; a hit served from
    # another RSU also crosses the backhaul once more.
    delivered_bits = requests * profile["mean_result_bits"]
    radio_j = e_bit_j * delivered_bits

    # Storage is charged over the run's simulated duration, approximated by the
    # residency of the caches during it.
    residency = float(result.get("distinct_keys_resident", 0))
    wall_s = float(result.get("runtime_seconds", 0.0) or 0.0)
    idle_j = p_cache_idle_w * wall_s * (residency / max(residency, 1.0))

    total = compute_j + radio_j + idle_j
    correct = float(result.get("valid_cache_hits", 0)) + misses
    return {
        "energy_j": total,
        "compute_j": compute_j,
        "radio_j": radio_j,
        "idle_j": idle_j,
        "reactive_ms": reactive_ms,
        "speculative_ms": speculative_ms,
        "speculative_share": speculative_ms / (reactive_ms + speculative_ms)
        if (reactive_ms + speculative_ms) else 0.0,
        # The comparable figure: joules spent per answer that was actually right.
        "j_per_correct_answer": total / correct if correct else float("nan"),
        "correct_answers": correct,
    }


def iter_results(paths: Iterable[Path]) -> Iterable[tuple[Path, dict]]:
    for path in paths:
        if path.is_dir():
            yield from iter_results(sorted(path.glob("**/*.json")))
            continue
        if path.name in {"aggregated_results.json", "statistical_tests.json"}:
            continue
        try:
            with path.open() as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            continue
        if "cache_hits" in payload:
            yield path, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--trace", type=Path, default=None,
                        help="Task trace for the workload profile. Defaults to "
                             "tasks.csv beside each result.")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Repeat at +/-50%% on each constant.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles: Dict[Path, Dict[str, float]] = {}
    report: Dict[str, dict] = {}

    for path, result in iter_results(args.results):
        trace = args.trace or (path.parent / "tasks.csv")
        if not trace.exists():
            print(f"skipping {path.name}: no trace at {trace}", file=sys.stderr)
            continue
        if trace not in profiles:
            profiles[trace] = trace_profile(trace)
        report[str(path)] = energy_joules(result, profiles[trace])

    if not report:
        print("No results with a matching task trace.", file=sys.stderr)
        return 1

    print(f"{'run':<46}{'J':>12}{'J/correct':>12}{'spec%':>8}")
    for name, terms in sorted(report.items()):
        label = "/".join(Path(name).parts[-2:])[:44]
        print(f"{label:<46}{terms['energy_j']:>12.1f}"
              f"{terms['j_per_correct_answer']:>12.4f}"
              f"{100 * terms['speculative_share']:>8.1f}")

    if args.sensitivity:
        print("\nSensitivity (J per correct answer, one run per setting)")
        sample_path, sample = next(iter(report.items())), None
        path = Path(sample_path[0])
        with path.open() as handle:
            sample = json.load(handle)
        profile = profiles[args.trace or (path.parent / "tasks.csv")]
        print(f"  {'setting':<28}{'J/correct':>12}")
        for label, kwargs in [
            ("baseline", {}),
            ("P_cpu -50%", {"p_cpu_w": P_CPU_W * 0.5}),
            ("P_cpu +50%", {"p_cpu_w": P_CPU_W * 1.5}),
            ("E_bit -50%", {"e_bit_j": E_BIT_J * 0.5}),
            ("E_bit +50%", {"e_bit_j": E_BIT_J * 1.5}),
            ("P_idle -50%", {"p_cache_idle_w": P_CACHE_IDLE_W * 0.5}),
            ("P_idle +50%", {"p_cache_idle_w": P_CACHE_IDLE_W * 1.5}),
        ]:
            terms = energy_joules(sample, profile, **kwargs)
            print(f"  {label:<28}{terms['j_per_correct_answer']:>12.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(
                {
                    "energy": report,
                    "constants": {
                        "P_CPU_W": P_CPU_W, "E_BIT_J": E_BIT_J,
                        "P_CACHE_IDLE_W": P_CACHE_IDLE_W,
                        "note": "PLACEHOLDERS — cite before submission",
                    },
                },
                handle, indent=2,
            )
        print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
