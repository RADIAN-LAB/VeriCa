#!/usr/bin/env python3
"""Render the paper's two headline tables from a directory of result JSONs.

The evaluation reports every arm twice, and the two regimes answer different
questions:

  as published        — the method as its own literature defines it, with no
                        validity check. The column that matters is how often it
                        answers *wrongly*, which no prior evaluation reports.
  under the check     — every arm refuses what it cannot verify, so all of them
                        deliver only correct answers and their latencies are
                        finally comparable.

Reading a hit rate from the first table against a latency from the second is the
mistake this layout exists to prevent, so they are printed separately.

Usage:
    python3 scripts/summarize_arms.py --results experiments/baselines/runs \
        experiments/lru_base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Display order and labels. Anything found on disk but not listed here is still
# shown, under its file name — silently dropping a result would be worse than an
# ugly row.
KNOWN_ARMS = [
    ("nocache", "No cache"),
    ("random_published", "B0 Random"),
    ("fifo_published", "B0 FIFO"),
    ("lru_published", "B1 LRU"),
    ("lfu_published", "B2 LFU"),
    ("b3_ttl_published", "B3 Fixed TTL"),
    ("b4_assumed", "B4 AoI, assumed W"),
    ("b4_tuned", "B4 AoI, tuned W"),
    ("b5_published", "B5 FoggyCache"),
    ("coca_published", "B6 CoCa (FL+RL)"),
    ("random_verified", "B0 Random"),
    ("fifo_verified", "B0 FIFO"),
    ("lru_verified", "B1 LRU"),
    ("lfu_verified", "B2 LFU"),
    ("b4_tuned_verify", "B4 AoI, tuned W"),
    ("b5_verified", "B5 FoggyCache"),
    ("coca_verified", "B6 CoCa (FL+RL)"),
    ("proposed_on_lru", "P  Proposed"),
    ("clairvoyant", "B7 Clairvoyant bound"),
]
LABELS = dict(KNOWN_ARMS)
ORDER = {name: index for index, (name, _) in enumerate(KNOWN_ARMS)}


def load_results(directories: List[Path]) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            if path.stem.endswith("_fl_training"):
                continue
            try:
                with path.open() as handle:
                    payload = json.load(handle)
            except json.JSONDecodeError:
                print(f"skipping unreadable {path}", file=sys.stderr)
                continue
            if "cache_hits" not in payload:
                continue
            results[path.stem] = payload
    return results


def requests(result: dict) -> int:
    return int(result["cache_hits"]) + int(result["cache_misses"])


def verified(result: dict) -> bool:
    return bool(result.get("verify_before_serve"))


def row(name: str, result: dict) -> tuple:
    total = requests(result) or 1
    return (
        LABELS.get(name, name),
        100.0 * result["hit_rate"],
        100.0 * result.get("valid_hit_rate", 0.0),
        100.0 * result.get("stale_cache_hits", 0) / total,
        result.get("average_completion", float("nan")),
        result.get("validity_model", "?"),
    )


def print_table(title: str, rows: List[tuple], published: bool) -> None:
    print(f"\n{title}")
    if published:
        print(f"  {'arm':<24}{'hit%':>8}{'correct%':>10}{'wrong%':>9}{'ms':>9}")
    else:
        print(f"  {'arm':<24}{'usable hit%':>13}{'wrong%':>9}{'ms':>9}")
    print("  " + "-" * (60 if published else 55))
    for label, hit, correct, wrong, ms, _model in rows:
        if published:
            print(f"  {label:<24}{hit:8.1f}{correct:10.1f}{wrong:9.1f}{ms:9.2f}")
        else:
            print(f"  {label:<24}{correct:13.1f}{wrong:9.1f}{ms:9.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="Directories of result JSONs.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Also write the assembled table as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = load_results(args.results)
    if not results:
        print("No result JSONs found.", file=sys.stderr)
        return 1

    unverified_model: Optional[str] = None
    published_rows, checked_rows = [], []
    for name, result in results.items():
        if result.get("validity_model") != "oracle":
            unverified_model = name
        (checked_rows if verified(result) else published_rows).append(
            (ORDER.get(name, 999), name, row(name, result))
        )

    published_rows.sort(key=lambda item: (item[0], item[1]))
    checked_rows.sort(key=lambda item: (item[0], item[1]))

    print_table(
        "AS PUBLISHED — no validity check; 'wrong%' is what the method delivers",
        [entry for _, _, entry in published_rows], published=True,
    )
    print_table(
        "UNDER THE VALIDITY CHECK — equal correctness, so latency is comparable",
        [entry for _, _, entry in checked_rows], published=False,
    )

    if unverified_model:
        print(f"\nWARNING: {unverified_model} was not judged by the oracle; its "
              "correctness column is not comparable with the rest.",
              file=sys.stderr)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(
                {
                    "as_published": {n: results[n] for _, n, _ in published_rows},
                    "under_check": {n: results[n] for _, n, _ in checked_rows},
                },
                handle, indent=2,
            )
        print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
