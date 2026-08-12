#!/usr/bin/env python3
"""Precompute the trace-grounded invalidation oracle for a SUMO mobility trace.

The oracle records the observable state of every RSU cell at every timestep, so
that ``intent.ground_truth`` can decide whether a cached result still answers a
given request. It is derived from vehicle mobility alone and never consults the
freshness model's decay rates.

Usage:
    python3 scripts/build_oracle.py \
        --fcd ../sdn-vanet-optimizer/experiments/sumo/output/fcd_trace.xml \
        --output experiments/oracle/seed_01.pkl

    # every mobility realisation at once
    python3 scripts/build_oracle.py --mobility-dir \
        ../sdn-vanet-optimizer/experiments/sumo/mobility --jobs 4
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coca_path import COCA_ROOT  # noqa: E402
from intent.ground_truth import INTENT_FEATURES, coverage_matrix  # noqa: E402
from intent.oracle import InvalidationOracle, build_oracle  # noqa: E402

DEFAULT_FCD = COCA_ROOT / "experiments" / "sumo" / "output" / "fcd_trace.xml"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "oracle"


def build_one(fcd: Path, output: Path, with_neighbours: bool,
              force: bool) -> tuple[Path, str, float]:
    """Build one oracle. Returns (output, status, seconds)."""
    if output.exists() and not force:
        return output, "skipped (already built)", 0.0
    started = time.time()
    oracle = build_oracle(fcd, with_neighbours=with_neighbours)
    oracle.save(output)
    return output, f"{len(oracle.signatures)} cell-steps", time.time() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fcd", type=Path, default=None,
                        help=f"Single FCD trace (default: {DEFAULT_FCD})")
    source.add_argument("--mobility-dir", type=Path, default=None,
                        help="Directory of seed_XX/fcd_trace.xml to build in bulk")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for a single build")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory for bulk builds (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-neighbours", action="store_true",
                        help="Skip per-vehicle neighbourhoods. Quadratic in cell "
                             "population; only sensor_processing needs them.")
    parser.add_argument("--show-coverage", action="store_true",
                        help="Print the intent coverage matrix and exit")
    return parser.parse_args()


def print_coverage() -> None:
    matrix = coverage_matrix()
    labels = list(matrix)
    width = max(len(label) for label in labels) + 1
    print("Which stored intent (row) can answer which request (column):\n")
    print(" " * width + "".join(f"{label[:6]:>8}" for label in labels))
    for stored in labels:
        cells = "".join(
            f"{'yes' if matrix[stored][req] else '-':>8}" for req in labels
        )
        print(f"{stored:<{width}}{cells}")
    print("\nFeatures each intent's result exposes:")
    for intent, features in INTENT_FEATURES.items():
        listed = ", ".join(sorted(features)) if features else "(none — static content)"
        print(f"  {intent:<24} {listed}")


def main() -> int:
    args = parse_args()

    if args.show_coverage:
        print_coverage()
        return 0

    with_neighbours = not args.no_neighbours
    jobs: list[tuple[Path, Path]] = []

    if args.mobility_dir:
        for seed_dir in sorted(args.mobility_dir.glob("seed_*")):
            fcd = seed_dir / "fcd_trace.xml"
            if fcd.exists():
                jobs.append((fcd, args.output_dir / f"{seed_dir.name}.pkl"))
        if not jobs:
            print(f"No seed_*/fcd_trace.xml found under {args.mobility_dir}",
                  file=sys.stderr)
            return 1
    else:
        fcd = args.fcd or DEFAULT_FCD
        if not fcd.exists():
            print(f"FCD trace not found: {fcd}", file=sys.stderr)
            return 1
        output = args.output or (args.output_dir / f"{fcd.parent.name}.pkl")
        jobs.append((fcd, output))

    started = time.time()
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        futures = {
            pool.submit(build_one, fcd, output, with_neighbours, args.force): output
            for fcd, output in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            output = futures[future]
            try:
                _, status, seconds = future.result()
                suffix = f" in {seconds:.1f}s" if seconds else ""
                print(f"[{output.name}] {status}{suffix}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"[{output.name}] FAILED: {exc}", file=sys.stderr)

    print(f"\n{len(jobs) - failures}/{len(jobs)} oracles built "
          f"({time.time() - started:.1f}s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
