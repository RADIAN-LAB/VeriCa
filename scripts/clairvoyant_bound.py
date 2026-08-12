#!/usr/bin/env python3
"""Upper bound on how many requests any cache could answer correctly (B7).

Every arm in the evaluation is limited by two different things: how well it
predicts what to keep, and how often a correct answer exists to keep at all.
This separates them. Replaying the trace against the oracle with an unbounded
cache and perfect foresight gives the fraction of requests for which *some*
previously computed result is still correct — a ceiling no eviction policy,
key design or placement strategy can exceed.

Without it, "the proposed method reaches 51.5%" is unanchored: a reviewer cannot
tell whether the remaining 48% is headroom or physics.

What the bound assumes
----------------------
* **Unbounded capacity and perfect foresight.** Nothing is ever evicted, so no
  policy can do better. This is why the result is a bound rather than a method.
* **Results are reusable only within the cell they were computed in.** Serving
  another cell's state is what the error taxonomy calls ``wrong_place``; the
  oracle would reject it, so admitting cross-cell candidates could not raise the
  bound.
* **Per-vehicle intents keep their subject.** A sensor-fusion result carries the
  ego vehicle it was computed for, and the oracle compares closed
  neighbourhoods, so candidates are tried with their own vehicle attached.
* **Only the most recent ``--candidates`` results per (cell, intent) are
  tried.** Scanning further can only find *more* servable requests, so the
  reported number is a conservative estimate of the true ceiling — never an
  optimistic one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intent.ground_truth import covers  # noqa: E402
from intent.oracle import InvalidationOracle  # noqa: E402
from sdn_core import scenario  # noqa: E402
from sdn_core.task_taxonomy import (  # noqa: E402
    INTENT_LABELS,
    intent_idx_of,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--max-age", type=int, default=120,
                        help="Ignore results older than this (seconds).")
    parser.add_argument("--candidates", type=int, default=200,
                        help="Most recent results tried per (cell, intent).")
    parser.add_argument("--max-ts", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    oracle = InvalidationOracle.load(args.oracle)
    print(f"Loaded oracle {args.oracle.name}: {len(oracle.signatures)} cell-steps")

    # (cell, intent) -> recent (timestep, vehicle) results available to reuse.
    available: Dict[Tuple[int, str], Deque[Tuple[int, str]]] = defaultdict(
        lambda: deque(maxlen=args.candidates)
    )
    # Which stored intents can answer each requested intent, precomputed.
    answering: Dict[str, List[str]] = {
        requested: [s for s in INTENT_LABELS if covers(s, requested)]
        for requested in INTENT_LABELS
    }

    total = 0
    servable = 0
    per_intent_total: Dict[str, int] = defaultdict(int)
    per_intent_servable: Dict[str, int] = defaultdict(int)

    with args.trace.open() as handle:
        for row in csv.DictReader(handle):
            timestep = int(float(row["timestamp"]))
            if args.max_ts is not None and timestep > args.max_ts:
                break

            x, y = float(row.get("pos_x", 0.0)), float(row.get("pos_y", 0.0))
            cell = scenario.nearest_rsu(x, y)
            intent = INTENT_LABELS[intent_idx_of(int(row["task_type"]))]
            vehicle = str(row["node_id"])

            total += 1
            per_intent_total[intent] += 1

            hit = False
            for stored_intent in answering[intent]:
                for stored_ts, stored_vehicle in reversed(
                    available[(cell, stored_intent)]
                ):
                    if timestep - stored_ts > args.max_age:
                        break          # deque is in time order; older is worse
                    if oracle.correct(
                        stored_intent, intent, cell, stored_ts, cell, timestep,
                        vehicle_id=stored_vehicle, request_vehicle_id=vehicle,
                    ):
                        hit = True
                        break
                if hit:
                    break

            if hit:
                servable += 1
                per_intent_servable[intent] += 1

            # This task is computed either way, so its result joins the pool.
            available[(cell, intent)].append((timestep, vehicle))

    bound = servable / total if total else 0.0
    print(f"\nRequests: {total}")
    print(f"Correctly servable from some earlier result: {servable} "
          f"({100 * bound:.1f}%)")
    print("\nCeiling per intent:")
    print(f"  {'intent':<24}{'requests':>10}{'servable':>10}{'bound%':>9}")
    for intent in INTENT_LABELS:
        count = per_intent_total[intent]
        if not count:
            continue
        share = per_intent_servable[intent] / count
        print(f"  {intent:<24}{count:>10}{per_intent_servable[intent]:>10}"
              f"{100 * share:>9.1f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as out:
            json.dump(
                {
                    "clairvoyant_bound": bound,
                    "requests": total,
                    "servable": servable,
                    "per_intent": {
                        intent: {
                            "requests": per_intent_total[intent],
                            "servable": per_intent_servable[intent],
                        }
                        for intent in INTENT_LABELS if per_intent_total[intent]
                    },
                    "trace": str(args.trace),
                    "oracle": str(args.oracle),
                    "max_age_s": args.max_age,
                    "candidates_scanned": args.candidates,
                },
                out, indent=2,
            )
        print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
