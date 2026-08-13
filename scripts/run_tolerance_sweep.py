#!/usr/bin/env python3
"""Sensitivity of the headline results to the oracle's agreement tolerances.

The tolerances in ``intent.ground_truth`` define what counts as a correct
answer, and they are chosen rather than measured. This sweep re-runs the
decisive arms with every tolerance scaled together — stricter (0.5x, 0.75x) and
looser (1.25x, 1.5x) — and checks the property the paper actually rests on:
that the *ordering* of methods survives any defensible choice of tolerance.
Absolute percentages are expected to move; that is reported as a band, not
hidden.

Scope decisions, and why:

* **Arms: Proposed, FoggyCache, LRU.** The strongest baseline, the classical
  reference, and the proposed design. The AoI arms are excluded because their
  refreshing windows were tuned against the calibration oracles at baseline
  tolerance; sweeping the tolerance under them without re-tuning would measure
  a mis-tuned baseline, and re-tuning per scale would sweep a different
  quantity than everything else.
* **Scale applies at serve time, not to stored oracles.** Oracle pickles store
  raw state signatures; tolerances enter in ``is_correct`` when a run judges or
  verifies. Setting ``GT_TOLERANCE_SCALE`` in each subprocess therefore covers
  both the scoring and the verification path with no oracle rebuild.
* **Scale 1.0 is copied from the main evaluation**, not re-run — identical
  configuration, and re-running it would only add noise to a column that must
  match Table 2.

Usage:
    python3 scripts/run_tolerance_sweep.py --jobs 6
    python3 scripts/run_tolerance_sweep.py --analyze-only
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVALUATION_DIR = ROOT / "experiments" / "evaluation"
ORACLE_DIR = ROOT / "experiments" / "oracle"
DEFAULT_OUT = ROOT / "experiments" / "sensitivity_tolerance"

SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
SEEDS = list(range(4, 14))          # ten evaluation seeds; never calibration
LRU_BASE = ["--evict-policy", "lru", "--no-fl"]

ARMS: Dict[str, List[str]] = {
    "p_proposed": [*LRU_BASE, "--reuse-mode", "coca",
                   "--key-mode", "context_adaptive"],
    "b5_foggycache": [*LRU_BASE, "--reuse-mode", "foggycache",
                      "--key-mode", "coca"],
    "b1_lru": [*LRU_BASE, "--reuse-mode", "coca", "--key-mode", "coca"],
}
REGIMES = {"published": [], "verified": ["--verify-before-serve"]}

# The claim under test: at every scale, this ordering of verified usable hits
# must hold. If it breaks, the paper's conclusion is tolerance-dependent and
# must change — better to learn that here than from a reviewer.
EXPECTED_ORDER = ["p_proposed", "b5_foggycache", "b1_lru"]


def scale_dir(out_dir: Path, scale: float) -> Path:
    return out_dir / f"scale_{scale:.2f}"


def run_one(seed: int, arm: str, regime: str, scale: float, output: Path,
            max_ts: Optional[int]) -> None:
    trace = EVALUATION_DIR / f"seed_{seed:02d}" / "tasks.csv"
    oracle = ORACLE_DIR / f"seed_{seed:02d}.pkl"
    for required in (trace, oracle):
        if not required.exists():
            raise FileNotFoundError(f"{required} (run the main evaluation first)")

    command = [
        sys.executable, str(ROOT / "scripts" / "run_intent_experiment.py"),
        "--trace", str(trace), "--output", str(output),
        "--oracle", str(oracle), "--seed", str(seed),
        *ARMS[arm], *REGIMES[regime],
    ]
    if max_ts is not None:
        command.extend(["--max-ts", str(max_ts)])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["TASK_GEN_SEED"] = str(seed)
    env["GT_TOLERANCE_SCALE"] = str(scale)
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(f"{arm}/{regime} scale {scale} seed {seed}:\n"
                           f"{result.stderr[-1500:]}")


def copy_baseline_scale(out_dir: Path) -> int:
    """Scale 1.0 already exists in the main evaluation; link it in."""
    copied = 0
    target_root = scale_dir(out_dir, 1.0)
    for seed in SEEDS:
        target = target_root / f"seed_{seed:02d}"
        target.mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            for regime in REGIMES:
                name = f"{arm}__{regime}.json"
                source = EVALUATION_DIR / f"seed_{seed:02d}" / name
                if source.exists() and not (target / name).exists():
                    shutil.copy2(source, target / name)
                    copied += 1
    return copied


def analyze(out_dir: Path) -> dict:
    report: Dict[str, dict] = {}
    ordering_holds = True
    print(f"\n{'scale':>6}  {'arm':<16}{'published correct%':>20}"
          f"{'verified usable%':>18}{'n':>4}")
    for scale in SCALES:
        rows = {}
        for arm in ARMS:
            values = {"published": [], "verified": []}
            for seed in SEEDS:
                for regime in REGIMES:
                    path = (scale_dir(out_dir, scale) / f"seed_{seed:02d}"
                            / f"{arm}__{regime}.json")
                    if not path.exists():
                        continue
                    with path.open() as handle:
                        run = json.load(handle)
                    recorded = run.get("tolerance_scale", 1.0)
                    if abs(recorded - scale) > 1e-9:
                        raise RuntimeError(
                            f"{path} judged at scale {recorded}, expected {scale}"
                        )
                    values[regime].append(100 * run["valid_hit_rate"])
            if values["verified"]:
                rows[arm] = {
                    "published_correct": statistics.fmean(values["published"]),
                    "verified_usable": statistics.fmean(values["verified"]),
                    "n": len(values["verified"]),
                }
        for arm, row in rows.items():
            print(f"{scale:>6.2f}  {arm:<16}{row['published_correct']:>20.1f}"
                  f"{row['verified_usable']:>18.1f}{row['n']:>4}")
        ranked = sorted(rows, key=lambda a: -rows[a]["verified_usable"])
        if ranked != [a for a in EXPECTED_ORDER if a in rows]:
            ordering_holds = False
            print(f"        ORDERING VIOLATED at scale {scale}: {ranked}")
        report[f"{scale:.2f}"] = rows

    verdict = "HOLDS at every scale" if ordering_holds else "VIOLATED — see above"
    print(f"\nOrdering (verified usable hits): {verdict}")
    return {"per_scale": report, "expected_order": EXPECTED_ORDER,
            "ordering_holds": ordering_holds, "seeds": SEEDS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--max-ts", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    failures: List[str] = []

    if not args.analyze_only:
        copied = copy_baseline_scale(args.out_dir)
        print(f"scale 1.00: {copied} results copied from the main evaluation")

        work = []
        for scale in SCALES:
            if scale == 1.0:
                continue
            for seed in SEEDS:
                for arm in ARMS:
                    for regime in REGIMES:
                        output = (scale_dir(args.out_dir, scale)
                                  / f"seed_{seed:02d}" / f"{arm}__{regime}.json")
                        if output.exists() and not args.force:
                            continue
                        output.parent.mkdir(parents=True, exist_ok=True)
                        work.append((seed, arm, regime, scale, output))

        print(f"{len(work)} runs across {len(SCALES) - 1} swept scales "
              f"({args.jobs} at a time)")
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(run_one, seed, arm, regime, scale, output,
                            args.max_ts): (seed, arm, regime, scale)
                for seed, arm, regime, scale, output in work
            }
            for future in concurrent.futures.as_completed(futures):
                seed, arm, regime, scale = futures[future]
                done += 1
                try:
                    future.result()
                    print(f"[{done}/{len(work)}] scale {scale} seed {seed:02d} "
                          f"{arm} {regime}")
                except Exception as exc:  # noqa: BLE001 — collect and continue
                    failures.append(str(exc))
                    print(f"[{done}/{len(work)}] scale {scale} seed {seed:02d} "
                          f"{arm} {regime} FAILED", file=sys.stderr)

    report = analyze(args.out_dir)
    output = args.out_dir / "tolerance_sensitivity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved {output} ({time.time() - started:.0f}s)")

    if failures:
        print(f"\n{len(failures)} run(s) failed:", file=sys.stderr)
        for failure in failures[:10]:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
