"""Approximate computation reuse by input similarity (baseline B5).

Adapted from FoggyCache (Guo et al., MobiCom 2018), the canonical cross-device
computation-reuse design: index cached inputs with locality-sensitive hashing,
look up the k nearest cached inputs for a new request, and reuse their output
when that neighbourhood is homogeneous enough to be confident.

Two adaptations are forced by this setting, and both are stated in the paper:

1. **Homogeneity is measured over the neighbours' intent, not their output
   values.** FoggyCache checks that the k nearest cached *results* agree before
   reusing one. Task result payloads are not simulated here — an entry holds
   ``(success, completion_time)``, which is near-identical for every task — so
   checking agreement over those values would pass trivially and hand the
   baseline a free win. Agreement over what the neighbouring tasks *were* is the
   closest faithful analogue and is the stricter of the two.

2. **Position and speed are part of the input vector.** This is deliberately
   generous: it lets the baseline discover spatial locality on its own, which is
   the very thing the proposed key scoping makes explicit. A baseline denied the
   information the proposed method uses is not a baseline.

What this design has no notion of is *validity*: nothing here can express that a
result was correct when computed and is wrong now. That absence is the point of
the comparison, not an oversight in the adaptation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Feature scaling: each component is divided by a plausible full-scale value so
# that Euclidean distance is not dominated by whichever field happens to have
# the largest units. Grid extent, size bucket ceiling, speed and priority range
# come from sdn_core.scenario / task_taxonomy.
POSITION_SCALE_M = 1200.0
DATA_SIZE_SCALE_KB = 1536.0
SPEED_SCALE_KMH = 120.0
PRIORITY_SCALE = 5.0
# Task type enters as a categorical axis: same type or not, weighted so that a
# type mismatch costs more than any single continuous field can.
TYPE_MISMATCH_WEIGHT = 1.0


def feature_vector(attrs: Dict[str, object]) -> Tuple[float, ...]:
    """Normalised input descriptor for one task."""
    def number(*names: str, default: float = 0.0) -> float:
        for name in names:
            if name in attrs and attrs[name] is not None:
                try:
                    return float(attrs[name])
                except (TypeError, ValueError):
                    continue
        return default

    return (
        number("task_type"),
        number("priority") / PRIORITY_SCALE,
        number("data_size", "data_size_kB") / DATA_SIZE_SCALE_KB,
        number("x", "pos_x") / POSITION_SCALE_M,
        number("y", "pos_y") / POSITION_SCALE_M,
        number("speed", "veh_speed_kmh") / SPEED_SCALE_KMH,
    )


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance with task type treated as categorical."""
    if a[0] != b[0]:
        type_cost = TYPE_MISMATCH_WEIGHT
    else:
        type_cost = 0.0
    tail = sum((x - y) ** 2 for x, y in zip(a[1:], b[1:]))
    return math.sqrt(type_cost ** 2 + tail)


@dataclass
class _Entry:
    key: str
    features: Tuple[float, ...]
    intent_idx: int


@dataclass
class ApproximateReuseIndex:
    """LSH-backed nearest-neighbour index over cached task inputs.

    One index per RSU, mirroring that cache's contents. Entries whose key has
    been evicted are pruned lazily at query time rather than eagerly, because
    CoCa evicts inside its own cache without notifying anything here.
    """

    dim: int = 6
    tables: int = 4
    hyperplanes: int = 8
    k: int = 5
    homogeneity: float = 0.6
    max_distance: float = 0.25
    seed: int = 42

    _planes: List[List[Tuple[float, ...]]] = field(default_factory=list, init=False)
    _buckets: List[Dict[int, List[str]]] = field(default_factory=list, init=False)
    _entries: Dict[str, _Entry] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        # Random hyperplanes; the sign pattern of the projections is the hash.
        self._planes = [
            [
                tuple(rng.gauss(0.0, 1.0) for _ in range(self.dim))
                for _ in range(self.hyperplanes)
            ]
            for _ in range(self.tables)
        ]
        self._buckets = [{} for _ in range(self.tables)]

    # ------------------------------------------------------------------

    def _hash(self, table: int, features: Sequence[float]) -> int:
        code = 0
        for bit, plane in enumerate(self._planes[table]):
            projection = sum(p * f for p, f in zip(plane, features))
            if projection >= 0.0:
                code |= 1 << bit
        return code

    def add(self, key: str, features: Sequence[float], intent_idx: int) -> None:
        vector = tuple(float(v) for v in features)
        if key in self._entries:
            # Re-storing the same key: keep one index entry, refresh its vector.
            self._entries[key] = _Entry(key, vector, intent_idx)
            return
        self._entries[key] = _Entry(key, vector, intent_idx)
        for table in range(self.tables):
            self._buckets[table].setdefault(self._hash(table, vector), []).append(key)

    def _neighbourhood(
        self, features: Sequence[float], resident: Iterable[str]
    ) -> List[Tuple[float, _Entry]]:
        """k nearest indexed entries that are still in the cache."""
        resident_set = set(resident)
        seen: set[str] = set()
        scored: List[Tuple[float, _Entry]] = []
        for table in range(self.tables):
            for key in self._buckets[table].get(self._hash(table, features), ()):
                if key in seen:
                    continue
                seen.add(key)
                entry = self._entries.get(key)
                if entry is None:
                    continue
                if key not in resident_set:
                    continue
                scored.append((distance(features, entry.features), entry))
        scored.sort(key=lambda pair: pair[0])
        return scored[: self.k]

    def query(
        self, features: Sequence[float], resident: Iterable[str]
    ) -> Optional[str]:
        """Key whose result may be reused, or None.

        Reuse requires both that the nearest indexed input is close enough and
        that the neighbourhood agrees on what kind of task it is — FoggyCache's
        confidence condition, without which approximate reuse degenerates into
        "serve whatever is nearest".
        """
        neighbours = self._neighbourhood(features, resident)
        if not neighbours:
            return None

        nearest_distance, nearest = neighbours[0]
        if nearest_distance > self.max_distance:
            return None

        counts: Dict[int, int] = {}
        for _, entry in neighbours:
            counts[entry.intent_idx] = counts.get(entry.intent_idx, 0) + 1
        dominant, dominant_count = max(counts.items(), key=lambda item: item[1])
        if dominant_count / len(neighbours) < self.homogeneity:
            return None
        if nearest.intent_idx != dominant:
            # The closest input disagrees with its own neighbourhood; that is
            # exactly the low-confidence case the check exists to reject.
            return None
        return nearest.key

    def forget(self, key: str) -> None:
        self._entries.pop(key, None)

    def __len__(self) -> int:
        return len(self._entries)


__all__ = [
    "ApproximateReuseIndex",
    "distance",
    "feature_vector",
]
