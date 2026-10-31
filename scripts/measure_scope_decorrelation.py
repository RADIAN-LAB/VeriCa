#!/usr/bin/env python3
"""Measure how far each intent's answer features stay valid in space.

The context-scoped key assigns each intent a spatial scope: the RSU coverage
cell for cell-level observables, a 100 m zone for neighbourhood-dependent
answers. This script turns that design choice from an argument into a
measurement: using the referee's own agreement predicates and tolerances
(``intent.ground_truth``), it estimates P(agree) between the answer a request
would get *here* and the answer computed a displacement d away, at the same
instant, as a function of d. The distance at which P(agree) crosses 0.5 is the
tolerance-crossing distance — the natural spatial scope of that intent.

Two probes, both on the calibration seeds (01-03) only, per the protocol:

* Neighbourhood probe (sensor_processing): for every vehicle pair at the same
  timestep, Jaccard of closed neighbourhoods against
  NEIGHBOUR_SET_MIN_JACCARD, binned by inter-vehicle distance.

* Region probe (all cell-level intents): circular windows of the coverage
  radius centred on a grid of points; per window a StateSignature; per window
  pair the referee's per-feature agreement, binned by centre distance.
  Occupancy is compared centre-relative (the local picture), since absolute
  grid coordinates of displaced windows cannot coincide by construction.

Usage:
    python3 scripts/measure_scope_decorrelation.py \
        --mobility-dir ../CoCa/experiments/sumo/mobility \
        --out experiments/scope_decorrelation/decorrelation.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coca_path  # noqa: F401,E402 — puts the CoCa tree on sys.path

from sdn_core import scenario  # noqa: E402
from intent import ground_truth as gt  # noqa: E402

COVERAGE_RADIUS_M = 500.0   # RSU coverage radius (Table I of the paper)
GRID_EXTENT_M = 1200.0
CENTRE_STEP_M = 100.0       # window-centre grid pitch
TIMESTEP_STRIDE_S = 20      # sample every 20 s
T_MIN, T_MAX = 100, 500     # skip warm-up
PAIR_BIN_M = 20.0           # neighbour probe bins
REGION_BIN_M = 100.0        # region probe bins
NEIGHBOUR_MAX_D = 300.0

CALIBRATION_SEEDS = ("seed_01", "seed_02", "seed_03")


def parse_fcd(path: Path):
    """Yield (t, vehicles) for sampled integer timesteps."""
    for _, elem in ET.iterparse(str(path), events=("end",)):
        if elem.tag != "timestep":
            continue
        t = float(elem.get("time", "0"))
        if t.is_integer() and T_MIN <= t <= T_MAX and int(t) % TIMESTEP_STRIDE_S == 0:
            vehicles = [
                {
                    "id": v.get("id"),
                    "x": float(v.get("x")),
                    "y": float(v.get("y")),
                    "speed": float(v.get("speed")),
                    "lane": v.get("lane", ""),
                }
                for v in elem.iter("vehicle")
            ]
            yield int(t), vehicles
        elem.clear()


def closed_neighbourhoods(vehicles) -> Dict[str, FrozenSet[str]]:
    """Closed neighbourhood per vehicle, as the referee defines it."""
    out = {}
    for v in vehicles:
        here = (v["x"], v["y"])
        neigh = {
            o["id"] for o in vehicles
            if o["id"] != v["id"]
            and scenario.distance((o["x"], o["y"]), here) <= scenario.NEIGHBOUR_RADIUS_M
        }
        out[v["id"]] = frozenset(neigh | {v["id"]})
    return out


def relative_occupancy(vehicles, cx: float, cy: float) -> FrozenSet[Tuple[int, int]]:
    """Occupancy grid relative to the window centre (translation-invariant)."""
    cell = scenario.OCCUPANCY_CELL_M
    return frozenset(
        (int((v["x"] - cx) // cell), int((v["y"] - cy) // cell)) for v in vehicles
    )


def occupancy_flip_agrees(a: FrozenSet, b: FrozenSet) -> bool:
    changed = a ^ b
    reference = max(len(a), len(b))
    if reference == 0:
        return True
    return len(changed) / reference <= gt.OCCUPANCY_MAX_FLIP_RATIO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mobility-dir", type=Path,
                    default=coca_path.COCA_ROOT / "experiments" / "sumo" / "mobility")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "experiments" / "scope_decorrelation" / "decorrelation.json")
    args = ap.parse_args()

    # bin -> [agree, total]
    neigh_bins: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    region_bins: Dict[str, Dict[int, List[int]]] = {
        f: defaultdict(lambda: [0, 0])
        for f in ("density", "mean_speed", "stopped_set", "edge_speeds", "occupancy")
    }

    centres = [
        (x, y)
        for x in [i * CENTRE_STEP_M for i in range(int(GRID_EXTENT_M / CENTRE_STEP_M) + 1)]
        for y in [i * CENTRE_STEP_M for i in range(int(GRID_EXTENT_M / CENTRE_STEP_M) + 1)]
    ]

    for seed in CALIBRATION_SEEDS:
        fcd = args.mobility_dir / seed / "fcd_trace.xml"
        if not fcd.exists():
            print(f"missing {fcd}", file=sys.stderr)
            continue
        for t, vehicles in parse_fcd(fcd):
            if not vehicles:
                continue

            # --- neighbourhood probe ---
            neigh = closed_neighbourhoods(vehicles)
            for i, a in enumerate(vehicles):
                for b in vehicles[i + 1:]:
                    d = scenario.distance((a["x"], a["y"]), (b["x"], b["y"]))
                    if d > NEIGHBOUR_MAX_D:
                        continue
                    bin_id = int(d // PAIR_BIN_M)
                    agree = gt._jaccard(neigh[a["id"]], neigh[b["id"]]) \
                        >= gt.NEIGHBOUR_SET_MIN_JACCARD
                    neigh_bins[bin_id][1] += 1
                    neigh_bins[bin_id][0] += int(agree)

            # --- region probe ---
            sigs, rel_occ = [], []
            for cx, cy in centres:
                inside = [
                    v for v in vehicles
                    if scenario.distance((v["x"], v["y"]), (cx, cy)) <= COVERAGE_RADIUS_M
                ]
                sigs.append(gt.signature_from_vehicles(0, t, inside, with_neighbours=False))
                rel_occ.append(relative_occupancy(inside, cx, cy))
            for i in range(len(centres)):
                xi, yi = centres[i]
                for j in range(i + 1, len(centres)):
                    xj, yj = centres[j]
                    d = math.hypot(xi - xj, yi - yj)
                    bin_id = int(d // REGION_BIN_M)
                    checks = {
                        "density": gt._density_agrees(sigs[i], sigs[j]),
                        "mean_speed": gt._mean_speed_agrees(sigs[i], sigs[j]),
                        "stopped_set": gt._stopped_set_agrees(sigs[i], sigs[j]),
                        "edge_speeds": gt._edge_speeds_agree(sigs[i], sigs[j]),
                        "occupancy": occupancy_flip_agrees(rel_occ[i], rel_occ[j]),
                    }
                    for feat, ok in checks.items():
                        region_bins[feat][bin_id][1] += 1
                        region_bins[feat][bin_id][0] += int(ok)
        print(f"{seed} done", file=sys.stderr)

    def curve(bins: Dict[int, List[int]], width: float):
        return {
            f"{int(b * width)}-{int((b + 1) * width)}":
            {"p_agree": round(v[0] / v[1], 4), "n": v[1]}
            for b, v in sorted(bins.items()) if v[1] > 0
        }

    def crossing(bins: Dict[int, List[int]], width: float):
        """Centre of the first bin where P(agree) < 0.5, None if never."""
        for b, v in sorted(bins.items()):
            if v[1] > 0 and v[0] / v[1] < 0.5:
                return (b + 0.5) * width
        return None

    # Per-intent curves: an intent agrees when all its features agree.
    # (Assembled from single-feature curves; traffic = density & mean_speed is
    # approximated by their joint evaluation below rather than the product.)
    result = {
        "protocol": {
            "seeds": list(CALIBRATION_SEEDS),
            "timesteps": f"every {TIMESTEP_STRIDE_S}s in [{T_MIN},{T_MAX}]",
            "window_radius_m": COVERAGE_RADIUS_M,
            "tolerances": "referee defaults (GT_TOLERANCE_SCALE=1)",
        },
        "neighbourhood": {
            "curve": curve(neigh_bins, PAIR_BIN_M),
            "crossing_m": crossing(neigh_bins, PAIR_BIN_M),
        },
        "region_features": {
            feat: {
                "curve": curve(bins, REGION_BIN_M),
                "crossing_m": crossing(bins, REGION_BIN_M),
            }
            for feat, bins in region_bins.items()
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))

    print(json.dumps(result["neighbourhood"]["curve"], indent=1))
    print("neighbourhood crossing:", result["neighbourhood"]["crossing_m"], "m")
    for feat, r in result["region_features"].items():
        print(f"{feat:12s} crossing: {r['crossing_m']} m")


if __name__ == "__main__":
    main()
