"""Build and query the trace-grounded invalidation oracle.

Parses a SUMO FCD trace once into a per-(RSU, timestep) signature index, which
``intent.ground_truth`` then uses to decide whether a cached result is still
correct for a given request.

The index is derived purely from vehicle mobility. It never consults the
freshness model's decay rates, so evaluating that model against it is not
circular.
"""
from __future__ import annotations

import pickle
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import coca_path  # noqa: F401 — puts the CoCa tree on sys.path

from sdn_core import scenario  # noqa: E402

from .ground_truth import (  # noqa: E402
    INTENT_FEATURES,
    NEIGHBOURS,
    StateSignature,
    is_correct,
    signature_from_vehicles,
)

# Bump when the signature computation changes, so a stale cached oracle is
# rebuilt rather than silently reused.
ORACLE_FORMAT_VERSION = 1


@dataclass
class InvalidationOracle:
    """Per-(RSU, timestep) state signatures for one mobility trace."""

    signatures: Dict[Tuple[int, int], StateSignature]
    timesteps: List[int]
    source: str
    version: int = ORACLE_FORMAT_VERSION

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def signature(self, rsu_id: int, timestep: float) -> Optional[StateSignature]:
        """Signature for a cell at a timestep, or None if the cell was empty."""
        return self.signatures.get((int(rsu_id), int(timestep)))

    def correct(
        self,
        stored_intent: str,
        requested_intent: str,
        stored_rsu: int,
        stored_ts: float,
        request_rsu: int,
        request_ts: float,
        vehicle_id: Optional[str] = None,
        request_vehicle_id: Optional[str] = None,
    ) -> bool:
        """Whether serving this cached result would give the right answer.

        ``vehicle_id`` is the vehicle the result was computed for and
        ``request_vehicle_id`` the one now asking; they differ when a result is
        reused across vehicles.
        """
        return is_correct(
            stored_intent,
            requested_intent,
            self.signature(stored_rsu, stored_ts),
            self.signature(request_rsu, request_ts),
            vehicle_id=vehicle_id,
            request_vehicle_id=request_vehicle_id,
        )

    def has_state(self, rsu_id: int, timestep: float) -> bool:
        """Whether ground truth exists for this cell-step at all.

        Lets a caller separate "the reuse was wrong" from "the trace cannot say",
        which ``correct`` deliberately collapses into False.
        """
        return self.signature(rsu_id, timestep) is not None

    def survival_age(
        self,
        intent: str,
        rsu_id: int,
        stored_ts: int,
        max_age: int = 120,
        vehicle_id: Optional[str] = None,
    ) -> Optional[int]:
        """Seconds until a result stored here stops answering its own question.

        This is the empirical lifetime the decay-rate fit consumes. Returns None
        if the result is still valid at ``max_age`` (right-censored) or if the
        cell has no signature at ``stored_ts``.
        """
        if self.signature(rsu_id, stored_ts) is None:
            return None
        for age in range(1, max_age + 1):
            later = stored_ts + age
            if self.signature(rsu_id, later) is None:
                # Cell emptied: treat as censored rather than invented data.
                return None
            if not self.correct(
                intent, intent, rsu_id, stored_ts, rsu_id, later,
                vehicle_id=vehicle_id,
            ):
                return age
        return None

    def survival_samples(
        self,
        intent: str,
        rsu_id: int,
        stored_ts: int,
        max_age: int = 120,
    ) -> List[Optional[int]]:
        """Lifetime samples for one cell-step, right-censored values as None.

        Intents whose answer is per-vehicle (sensor_processing, via the
        neighbourhood feature) yield one sample per vehicle present. Evaluating
        them at cell level would compare an empty neighbourhood against another
        empty one, which trivially agrees and would report such results as never
        going stale.
        """
        signature = self.signature(rsu_id, stored_ts)
        if signature is None:
            return []

        if NEIGHBOURS in INTENT_FEATURES.get(intent, frozenset()):
            return [
                self.survival_age(intent, rsu_id, stored_ts, max_age, vehicle_id=vid)
                for vid in signature.neighbours
            ]
        return [self.survival_age(intent, rsu_id, stored_ts, max_age)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "InvalidationOracle":
        with Path(path).open("rb") as handle:
            oracle = pickle.load(handle)
        if getattr(oracle, "version", None) != ORACLE_FORMAT_VERSION:
            raise ValueError(
                f"Oracle at {path} has format version "
                f"{getattr(oracle, 'version', None)}, expected "
                f"{ORACLE_FORMAT_VERSION}. Rebuild with scripts/build_oracle.py."
            )
        return oracle


def iter_timesteps(fcd_path: Path) -> Iterator[Tuple[int, List[Dict[str, object]]]]:
    """Stream (timestep, vehicles) from an FCD trace without holding it all.

    The trace is a few MB here but grows linearly with fleet size and duration,
    and 30 of them are built for the multi-seed evaluation.
    """
    for _event, element in ET.iterparse(str(fcd_path), events=("end",)):
        if element.tag != "timestep":
            continue
        timestep = int(float(element.get("time")))
        vehicles = [
            {
                "id": veh.get("id"),
                "x": float(veh.get("x")),
                "y": float(veh.get("y")),
                "speed": float(veh.get("speed")),
                "lane": veh.get("lane", ""),
            }
            for veh in element.findall("vehicle")
        ]
        yield timestep, vehicles
        element.clear()


def build_oracle(
    fcd_path: Path,
    with_neighbours: bool = True,
    progress_every: int = 0,
) -> InvalidationOracle:
    """Compute the signature index for every (RSU, timestep) in a trace."""
    signatures: Dict[Tuple[int, int], StateSignature] = {}
    timesteps: List[int] = []

    for timestep, vehicles in iter_timesteps(Path(fcd_path)):
        timesteps.append(timestep)
        # Assign each vehicle to its nearest RSU, exactly as the task trace
        # generator does, so a result is attributed to the cell it came from.
        by_rsu: Dict[int, List[Dict[str, object]]] = {}
        for vehicle in vehicles:
            rsu_id = scenario.nearest_rsu(float(vehicle["x"]), float(vehicle["y"]))
            by_rsu.setdefault(rsu_id, []).append(vehicle)

        for rsu_id, cell_vehicles in by_rsu.items():
            signatures[(rsu_id, timestep)] = signature_from_vehicles(
                rsu_id, timestep, cell_vehicles, with_neighbours=with_neighbours
            )

        if progress_every and timestep % progress_every == 0:
            print(f"  ts={timestep} cells={len(signatures)}")

    return InvalidationOracle(
        signatures=signatures,
        timesteps=sorted(timesteps),
        source=str(fcd_path),
    )


__all__ = [
    "ORACLE_FORMAT_VERSION",
    "InvalidationOracle",
    "build_oracle",
    "iter_timesteps",
]
