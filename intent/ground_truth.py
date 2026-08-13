"""Trace-grounded ground truth for whether a cached result is still correct.

Why this module exists
----------------------
The previous oracle drew each cached result's lifetime from
``Exp(DECAY_RATES[intent])`` — the very same per-intent lambda that the freshness
policy uses to *score* staleness. The policy was therefore graded against its own
generative assumption, and the headline `valid_hit_rate` was partly true by
construction. Worse, correctness was checked only against the *source* entry's
lifetime, so substituting one task's result for another's could never be wrong:
the mechanism the paper proposes could only ever gain.

Both problems are replaced by one mechanism grounded in the SUMO trace.

The model
---------
A task result is an answer to a question about the physical state of a cell at a
moment in time. Serving a cached result for a new request is *correct* when two
conditions hold:

1. **Coverage.** The stored result must actually contain what the new question
   needs. A cooperative-perception result carries an occupancy grid and so can
   answer a traffic-density question; a traffic-density result cannot answer a
   perception question. This makes the relation asymmetric — and measurable,
   rather than the hand-set similarity matrix it replaces.

2. **Agreement.** Every feature the new question needs must still agree, within
   tolerance, between the state the result was computed from and the state the
   requester is actually in.

Condition 2 subsumes temporal staleness (same cell, later time) *and* semantic
mismatch (different cell, or a different question), so cross-key reuse can now
be wrong. Decay rates become something to *fit* against this ground truth rather
than something to assume — see ``scripts/fit_decay_rates.py``.

Nothing in this module reads ``intent.freshness.DECAY_RATES``. That separation is
the point, and ``tests/test_ground_truth.py`` asserts it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

import coca_path  # noqa: F401 — puts the CoCa tree on sys.path

from sdn_core import scenario  # noqa: E402
from sdn_core.task_taxonomy import INTENT_LABELS  # noqa: E402

# ---------------------------------------------------------------------------
# State features
# ---------------------------------------------------------------------------
# Each feature is one observable a task result may expose. Intents are described
# by which features their result carries.

DENSITY = "density"                 # vehicle count in the cell
MEAN_SPEED = "mean_speed"           # mean speed in the cell (m/s)
EDGE_SPEEDS = "edge_speeds"         # mean speed per outgoing lane
OCCUPANCY = "occupancy"             # coarse occupancy grid over the cell
NEIGHBOURS = "neighbours"           # ids within NEIGHBOUR_RADIUS_M of the vehicle
HAZARD_STATE = "hazard_state"       # stopped/obstructing vehicles on the roadway
PARKING_STATE = "parking_state"     # stopped vehicles read as parking occupancy
CONTENT = "content"                 # the media payload itself

ALL_FEATURES = (
    DENSITY, MEAN_SPEED, EDGE_SPEEDS, OCCUPANCY, NEIGHBOURS,
    HAZARD_STATE, PARKING_STATE, CONTENT,
)

# What each intent's result contains. A result answers its own question, so this
# doubles as what that question requires.
#
# Features are named after the *answer*, not the raw observable, so that two
# different questions over the same observable stay distinct. hazard_detection
# and parking_search both derive from the stopped-vehicle set and share an
# agreement rule, but neither can serve the other: "is the road obstructed" and
# "is there a free space" are different questions.
#
# content_delivery carries CONTENT, which nothing else exposes and which never
# changes — a cached media segment is not invalidated by traffic. Modelling that
# as an *empty* requirement would have been wrong in the other direction: it
# would have let any result at all answer a content request.
INTENT_FEATURES: Dict[str, FrozenSet[str]] = {
    "traffic_analysis":       frozenset({DENSITY, MEAN_SPEED}),
    "hazard_detection":       frozenset({HAZARD_STATE}),
    "route_optimization":     frozenset({EDGE_SPEEDS}),
    "sensor_processing":      frozenset({NEIGHBOURS}),
    # Cooperative perception genuinely subsumes traffic analysis: an occupancy
    # grid carries the density and mean speed a traffic query needs. This is the
    # one substitution the feature model licenses, and it is what makes the
    # empirical substitutability matrix asymmetric.
    "environment_perception": frozenset({OCCUPANCY, DENSITY, MEAN_SPEED}),
    "content_delivery":       frozenset({CONTENT}),
    "parking_search":         frozenset({PARKING_STATE}),
}

# Tolerances: how much a feature may drift before the answer is wrong.
#
# These constants *define* what "correct" means, so every headline number in the
# evaluation depends on them — which makes them the obvious place for a reviewer
# to object. GT_TOLERANCE_SCALE (environment variable, default 1.0) scales all
# of them at once for the sensitivity sweep: values below 1 make the oracle
# stricter, values above 1 looser. For the Jaccard minimums the scale applies to
# the permitted *slack* (1 - minimum), since that is the quantity that plays the
# role of a tolerance; scaling the threshold itself would move it the wrong way.
import os as _os

_SCALE = float(_os.environ.get("GT_TOLERANCE_SCALE", "1.0"))
if not 0.0 < _SCALE <= 4.0:
    raise ValueError(f"GT_TOLERANCE_SCALE={_SCALE} is outside (0, 4]")

TOLERANCE_SCALE = _SCALE

DENSITY_REL_TOLERANCE = 0.20 * _SCALE        # 20% change in vehicle count
MEAN_SPEED_ABS_TOLERANCE_MS = (5.0 / 3.6) * _SCALE   # 5 km/h in m/s
EDGE_SPEED_REL_TOLERANCE = 0.20 * _SCALE     # 20% change on any shared lane
STOPPED_SET_MIN_JACCARD = 1.0 - 0.20 * _SCALE   # hazard/parking set must hold
NEIGHBOUR_SET_MIN_JACCARD = 1.0 - 0.30 * _SCALE # sensor fusion tolerates churn
OCCUPANCY_MAX_FLIP_RATIO = 0.20 * _SCALE     # grid cells that may change


def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Jaccard similarity, with two empty sets treated as identical."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 1.0


# ---------------------------------------------------------------------------
# State signature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateSignature:
    """Observable state of one RSU cell at one timestep."""

    rsu_id: int
    timestep: int
    density: int
    mean_speed: float
    stopped_set: FrozenSet[str]
    edge_speeds: Mapping[str, float]
    occupancy: FrozenSet[Tuple[int, int]]

    # Per-vehicle neighbourhoods, keyed by vehicle id. Only needed for
    # sensor_processing, and only for vehicles present in this cell.
    neighbours: Mapping[str, FrozenSet[str]] = field(default_factory=dict)

    def neighbours_of(self, vehicle_id: Optional[str]) -> FrozenSet[str]:
        if vehicle_id is None:
            return frozenset()
        return self.neighbours.get(str(vehicle_id), frozenset())


def signature_from_vehicles(
    rsu_id: int,
    timestep: int,
    vehicles: Sequence[Mapping[str, object]],
    with_neighbours: bool = True,
) -> StateSignature:
    """Build the signature of one cell from the vehicles inside it.

    Args:
        rsu_id: Cell identifier.
        timestep: Simulation second.
        vehicles: Dicts with ``id``, ``x``, ``y``, ``speed`` and ``lane``.
        with_neighbours: Compute per-vehicle neighbourhoods. Quadratic in the
            cell population, so it can be skipped when sensor_processing is not
            being evaluated.
    """
    speeds = [float(v["speed"]) for v in vehicles]
    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0

    stopped = frozenset(
        str(v["id"]) for v in vehicles
        if float(v["speed"]) <= scenario.STOPPED_SPEED_MS
    )

    lane_speeds: Dict[str, list] = {}
    for vehicle in vehicles:
        lane = str(vehicle.get("lane", ""))
        if lane:
            lane_speeds.setdefault(lane, []).append(float(vehicle["speed"]))
    edge_speeds = {
        lane: sum(values) / len(values) for lane, values in lane_speeds.items()
    }

    occupancy = frozenset(
        scenario.occupancy_cell(float(v["x"]), float(v["y"])) for v in vehicles
    )

    neighbours: Dict[str, FrozenSet[str]] = {}
    if with_neighbours:
        for vehicle in vehicles:
            here = (float(vehicle["x"]), float(vehicle["y"]))
            neighbours[str(vehicle["id"])] = frozenset(
                str(other["id"]) for other in vehicles
                if str(other["id"]) != str(vehicle["id"])
                and scenario.distance(
                    (float(other["x"]), float(other["y"])), here
                ) <= scenario.NEIGHBOUR_RADIUS_M
            )

    return StateSignature(
        rsu_id=rsu_id,
        timestep=timestep,
        density=len(vehicles),
        mean_speed=mean_speed,
        stopped_set=stopped,
        edge_speeds=edge_speeds,
        occupancy=occupancy,
        neighbours=neighbours,
    )


# ---------------------------------------------------------------------------
# Feature agreement
# ---------------------------------------------------------------------------

def _density_agrees(a: StateSignature, b: StateSignature) -> bool:
    if a.density == b.density:
        return True
    reference = max(a.density, b.density)
    if reference == 0:
        return True
    return abs(a.density - b.density) / reference <= DENSITY_REL_TOLERANCE


def _mean_speed_agrees(a: StateSignature, b: StateSignature) -> bool:
    return abs(a.mean_speed - b.mean_speed) <= MEAN_SPEED_ABS_TOLERANCE_MS


def _stopped_set_agrees(a: StateSignature, b: StateSignature) -> bool:
    return _jaccard(a.stopped_set, b.stopped_set) >= STOPPED_SET_MIN_JACCARD


def _edge_speeds_agree(a: StateSignature, b: StateSignature) -> bool:
    shared = set(a.edge_speeds) & set(b.edge_speeds)
    if not shared:
        # No lane in common: a routing answer computed elsewhere says nothing
        # about the lanes this request cares about.
        return not a.edge_speeds and not b.edge_speeds
    for lane in shared:
        left, right = a.edge_speeds[lane], b.edge_speeds[lane]
        reference = max(abs(left), abs(right))
        if reference == 0:
            continue
        if abs(left - right) / reference > EDGE_SPEED_REL_TOLERANCE:
            return False
    return True


def _occupancy_agrees(a: StateSignature, b: StateSignature) -> bool:
    changed = a.occupancy ^ b.occupancy
    reference = max(len(a.occupancy), len(b.occupancy))
    if reference == 0:
        return True
    return len(changed) / reference <= OCCUPANCY_MAX_FLIP_RATIO


def _closed_neighbourhood(
    state: StateSignature, vehicle_id: Optional[str]
) -> FrozenSet[str]:
    """Vehicles covered by a fused sensor result, including the ego vehicle.

    Closed rather than open because the *result* describes everything the fusion
    saw, the ego vehicle included. It also keeps the comparison meaningful across
    vehicles: A's open neighbourhood contains B and B's contains A, so two
    vehicles alone together would score zero overlap on open sets while their
    fused views cover exactly the same two cars.
    """
    if vehicle_id is None:
        return frozenset()
    return state.neighbours_of(vehicle_id) | {str(vehicle_id)}


def _neighbours_agree(
    a: StateSignature,
    b: StateSignature,
    vehicle_id: Optional[str],
    request_vehicle_id: Optional[str] = None,
) -> bool:
    """Whether one vehicle's fused view still answers another's question.

    ``vehicle_id`` produced the cached result; ``request_vehicle_id`` is asking.
    They differ whenever a sensor-fusion result is reused across vehicles, which
    the coarse cache key makes common — and which must be allowed to be wrong.
    """
    requester = request_vehicle_id if request_vehicle_id is not None else vehicle_id
    return _jaccard(
        _closed_neighbourhood(a, vehicle_id),
        _closed_neighbourhood(b, requester),
    ) >= NEIGHBOUR_SET_MIN_JACCARD


def _content_agrees(a: StateSignature, b: StateSignature) -> bool:
    """Media payloads do not depend on traffic state, so they never disagree."""
    return True


# hazard_state and parking_state are distinct answers derived from the same
# observable, so they share an agreement rule while remaining unsubstitutable.
_AGREEMENT = {
    DENSITY: _density_agrees,
    MEAN_SPEED: _mean_speed_agrees,
    HAZARD_STATE: _stopped_set_agrees,
    PARKING_STATE: _stopped_set_agrees,
    EDGE_SPEEDS: _edge_speeds_agree,
    OCCUPANCY: _occupancy_agrees,
    CONTENT: _content_agrees,
}


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def covers(stored_intent: str, requested_intent: str) -> bool:
    """True if a stored_intent result carries what requested_intent needs.

    Asymmetric by design: environment_perception covers traffic_analysis, but
    not the other way round.
    """
    exposed = INTENT_FEATURES.get(stored_intent, frozenset())
    required = INTENT_FEATURES.get(requested_intent, frozenset())
    return required <= exposed


def is_correct(
    stored_intent: str,
    requested_intent: str,
    stored_state: Optional[StateSignature],
    request_state: Optional[StateSignature],
    vehicle_id: Optional[str] = None,
    request_vehicle_id: Optional[str] = None,
) -> bool:
    """Whether serving a cached result for this request yields the right answer.

    Args:
        stored_intent: Intent of the task whose result is cached.
        requested_intent: Intent of the incoming request.
        stored_state: Cell state the cached result was computed from.
        request_state: Cell state the requester is actually in.
        vehicle_id: Vehicle the cached result was computed for, needed only for
            neighbourhood features.
        request_vehicle_id: Vehicle now asking. Defaults to ``vehicle_id``, which
            is the same-vehicle case the survival fit measures.

    Returns:
        True if every feature the request needs is both carried by the stored
        result and still in agreement.
    """
    if not covers(stored_intent, requested_intent):
        return False

    required = INTENT_FEATURES.get(requested_intent, frozenset())
    if not required:
        # An intent with no declared answer features cannot be reasoned about;
        # refuse rather than credit the method by default.
        return False

    if required == {CONTENT}:
        # Static payload: coverage already established identity, and content
        # does not depend on the traffic state, so no signature is needed.
        return True

    if stored_state is None or request_state is None:
        # No ground truth available for one of the two moments. Refuse to call
        # it correct rather than silently crediting the method.
        return False

    for feature in required:
        if feature == NEIGHBOURS:
            if not _neighbours_agree(
                stored_state, request_state, vehicle_id, request_vehicle_id
            ):
                return False
        elif not _AGREEMENT[feature](stored_state, request_state):
            return False
    return True


def coverage_matrix() -> Dict[str, Dict[str, bool]]:
    """Which intents can serve which, by feature coverage alone.

    The empirical substitutability matrix (how often coverage *and* agreement
    both hold) is measured by ``scripts/fit_similarity.py``; this is only its
    structural upper bound.
    """
    return {
        stored: {req: covers(stored, req) for req in INTENT_LABELS}
        for stored in INTENT_LABELS
    }


__all__ = [
    "ALL_FEATURES",
    "CONTENT",
    "DENSITY",
    "EDGE_SPEEDS",
    "HAZARD_STATE",
    "INTENT_FEATURES",
    "MEAN_SPEED",
    "NEIGHBOURS",
    "OCCUPANCY",
    "PARKING_STATE",
    "StateSignature",
    "coverage_matrix",
    "covers",
    "is_correct",
    "signature_from_vehicles",
]
