"""The cache manager's correctness accounting against the trace-grounded oracle.

These tests pin down the two properties the audit demanded: correctness must be
decided by observed mobility rather than by the freshness model's own decay
rates, and substituting one task's result for another's must be *able* to be
wrong.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coca_adapter.intent_distributed import (
    STALE,
    WRONG_INTENT,
    WRONG_PLACE,
    WRONG_SUBJECT,
    IntentAwareDistributedCacheManager,
)
from intent.ground_truth import signature_from_vehicles
from intent.oracle import InvalidationOracle


def _vehicle(vid: str, x: float, y: float, speed: float = 10.0, lane: str = "L1"):
    return {"id": vid, "x": x, "y": y, "speed": speed, "lane": lane}


def _oracle(cells: dict) -> InvalidationOracle:
    """Build an oracle directly from (rsu, timestep) -> vehicle lists."""
    signatures = {
        (rsu, ts): signature_from_vehicles(rsu, ts, vehicles)
        for (rsu, ts), vehicles in cells.items()
    }
    return InvalidationOracle(
        signatures=signatures,
        timesteps=sorted({ts for _, ts in cells}),
        source="synthetic",
    )


def _manager(oracle, mode: str = "intent") -> IntentAwareDistributedCacheManager:
    return IntentAwareDistributedCacheManager(
        rsu_ids=[1, 2],
        cache_capacity=10,
        default_ttl=300,
        reuse_mode=mode,
        seed=7,
        oracle=oracle,
    )


# Positions are chosen so that scenario.nearest_rsu puts them in the intended
# cell: RSU 1 sits at (300, 300) and RSU 2 at (900, 300).
IN_CELL_1 = (0.0, 0.0)
FAR_IN_CELL_1 = (500.0, 100.0)   # same cell, well beyond sensor range of (0, 0)
IN_CELL_2 = (900.0, 300.0)


def _traffic(node_id, x: float, y: float, time: float):
    """A traffic_analysis task issued by one vehicle.

    Task type 5 is TrafficInfo. Type 0 is Navigation, which the taxonomy maps to
    route_optimization — a different question with different state features.
    """
    return {
        "task_type": 5, "priority": 1, "data_size": 100,
        "node_id": node_id, "x": x, "y": y, "time": time, "origin_rsu": 1,
    }


def test_validity_model_reports_which_ground_truth_was_used():
    assert _manager(None).metrics()["validity_model"] == "exponential"
    assert _manager(_oracle({})).metrics()["validity_model"] == "oracle"


def test_hit_is_correct_while_the_cell_state_holds():
    # Same ten vehicles at both moments: a density/speed answer still stands.
    steady = [_vehicle(f"v{i}", 10.0 * i, 0.0) for i in range(10)]
    oracle = _oracle({(1, 10): steady, (1, 12): steady})
    manager = _manager(oracle)

    manager.set_sim_time(10)
    manager.register_tasks([("t", _traffic(1, 0.0, 0.0, 10))])
    manager.store("t", (1, 2.5), 1, replicate=False)

    manager.set_sim_time(12)
    manager.register_tasks([("t", _traffic(1, 0.0, 0.0, 12))])
    assert manager.lookup("t", 1, enable_forwarding=False) == (1, 2.5)
    assert manager.valid_hits == 1
    assert manager.metrics()["error_breakdown"][STALE] == 0


def test_hit_is_stale_once_the_cell_state_moves_on():
    ten = [_vehicle(f"v{i}", 10.0 * i, 0.0) for i in range(10)]
    # Density halves — beyond the 20% tolerance, so the cached answer is wrong.
    five = ten[:5]
    oracle = _oracle({(1, 10): ten, (1, 40): five})
    manager = _manager(oracle, mode="coca")  # no freshness gate: force the hit

    manager.set_sim_time(10)
    manager.register_tasks([("t", _traffic(1, 0.0, 0.0, 10))])
    manager.store("t", (1, 2.5), 1, replicate=False)

    manager.set_sim_time(40)
    manager.register_tasks([("t", _traffic(1, 0.0, 0.0, 40))])
    assert manager.lookup("t", 1, enable_forwarding=False) == (1, 2.5)
    assert manager.stale_hits == 1
    assert manager.metrics()["error_breakdown"][STALE] == 1


def test_correctness_ignores_the_freshness_decay_rates():
    """The whole point: lambda must not be able to change the ground truth."""
    import intent.freshness as freshness

    steady = [_vehicle(f"v{i}", 10.0 * i, 0.0) for i in range(10)]
    oracle = _oracle({(1, 10): steady, (1, 30): steady})
    original = dict(freshness.DECAY_RATES)
    try:
        # Make every intent decay absurdly fast. An oracle-judged hit on an
        # unchanged cell must still be correct.
        for idx in freshness.DECAY_RATES:
            freshness.DECAY_RATES[idx] = 10.0
        manager = _manager(oracle, mode="coca")
        manager.set_sim_time(10)
        manager.register_tasks([("t", _traffic(1, 0.0, 0.0, 10))])
        manager.store("t", (1, 2.5), 1, replicate=False)
        manager.set_sim_time(30)
        manager.register_tasks([("t", _traffic(1, 0.0, 0.0, 30))])
        manager.lookup("t", 1, enable_forwarding=False)
    finally:
        freshness.DECAY_RATES.clear()
        freshness.DECAY_RATES.update(original)

    assert manager.valid_hits == 1
    assert manager.stale_hits == 0


def test_semantic_substitution_across_cells_is_counted_wrong():
    # RSU 1 is busy, RSU 2 is empty: a traffic answer from one does not describe
    # the other, however fresh it is.
    busy = [_vehicle(f"v{i}", 10.0 * i, 0.0) for i in range(10)]
    quiet = [_vehicle("w0", *IN_CELL_2, speed=0.5)]
    oracle = _oracle({(1, 10): busy, (2, 10): quiet})
    manager = _manager(oracle, mode="coca")

    manager.set_sim_time(10)
    manager.register_tasks([("t", _traffic(1, *IN_CELL_1, time=10))])
    manager.store("t", (1, 2.5), 1, replicate=False)

    # A vehicle in RSU 2's cell asks for the same key and is forwarded the entry.
    manager.register_tasks([("t", _traffic(2, *IN_CELL_2, time=10))])
    assert manager.lookup("t", 2, enable_forwarding=True) == (1, 2.5)
    metrics = manager.metrics()
    assert metrics["error_breakdown"][WRONG_PLACE] == 1
    assert metrics["semantic_error_rate"] == 1.0


def test_wrong_intent_hit_is_attributed_to_the_substitution():
    steady = [_vehicle(f"v{i}", 10.0 * i, 0.0) for i in range(10)]
    oracle = _oracle({(1, 10): steady})
    manager = _manager(oracle, mode="coca")

    manager.set_sim_time(10)
    stored = _traffic(1, 0.0, 0.0, 10)                 # traffic_analysis
    incoming = dict(stored, task_type=1, node_id=2)    # hazard_detection
    manager.register_tasks([("shared", stored)])
    manager.store("shared", (1, 2.5), 1, replicate=False)

    # Same cache key, different question — the coarse key makes this collision
    # real, and a traffic result carries no hazard state.
    manager.register_tasks([("shared", incoming)])
    assert manager.lookup("shared", 1, enable_forwarding=False) == (1, 2.5)
    assert manager.metrics()["error_breakdown"][WRONG_INTENT] == 1


def test_sensor_fusion_result_does_not_transfer_between_distant_vehicles():
    # Two clusters far apart in the same cell: their fused views share nothing.
    here = [_vehicle(f"a{i}", 10.0 * i, 0.0) for i in range(4)]
    there = [_vehicle(f"b{i}", FAR_IN_CELL_1[0] + 10.0 * i, FAR_IN_CELL_1[1])
             for i in range(4)]
    oracle = _oracle({(1, 10): here + there})
    manager = _manager(oracle, mode="coca")

    def sensor_task(node_id: str, x: float, y: float):
        return {
            "task_type": 3, "priority": 1, "data_size": 100,   # sensor_processing
            "node_id": node_id, "x": x, "y": y, "time": 10, "origin_rsu": 1,
        }

    manager.set_sim_time(10)
    manager.register_tasks([("s", sensor_task("a0", 0.0, 0.0))])
    manager.store("s", (1, 2.5), 1, replicate=False)

    manager.register_tasks([("s", sensor_task("b0", *FAR_IN_CELL_1))])
    assert manager.lookup("s", 1, enable_forwarding=False) == (1, 2.5)
    assert manager.metrics()["error_breakdown"][WRONG_SUBJECT] == 1


def test_nearby_vehicles_may_share_a_fusion_result():
    cluster = [_vehicle(f"a{i}", 5.0 * i, 0.0) for i in range(4)]
    oracle = _oracle({(1, 10): cluster})
    manager = _manager(oracle, mode="coca")

    def sensor_task(node_id: str):
        return {
            "task_type": 3, "priority": 1, "data_size": 100,
            "node_id": node_id, "x": 0.0, "y": 0.0, "time": 10, "origin_rsu": 1,
        }

    manager.set_sim_time(10)
    manager.register_tasks([("s", sensor_task("a0"))])
    manager.store("s", (1, 2.5), 1, replicate=False)

    manager.register_tasks([("s", sensor_task("a1"))])
    assert manager.lookup("s", 1, enable_forwarding=False) == (1, 2.5)
    assert manager.valid_hits == 1


def test_request_contexts_are_matched_to_their_own_vehicles():
    """Several vehicles share one coarse key in a batch; each keeps its context."""
    here = [_vehicle(f"a{i}", 10.0 * i, 0.0) for i in range(4)]
    there = [_vehicle(f"b{i}", FAR_IN_CELL_1[0] + 10.0 * i, FAR_IN_CELL_1[1])
             for i in range(4)]
    oracle = _oracle({(1, 10): here + there})
    manager = _manager(oracle, mode="coca")

    def sensor_task(node_id: str, x: float, y: float):
        return {
            "task_type": 3, "priority": 1, "data_size": 100,
            "node_id": node_id, "x": x, "y": y, "time": 10, "origin_rsu": 1,
        }

    manager.set_sim_time(10)
    manager.register_tasks([("s", sensor_task("a0", 0.0, 0.0))])
    manager.store("s", (1, 2.5), 1, replicate=False)

    # One batch, one key, two very different requesters: the near one is served
    # correctly, the far one is not.
    manager.register_tasks([
        ("s", sensor_task("a1", 10.0, 0.0)),
        ("s", sensor_task("b0", *FAR_IN_CELL_1)),
    ])
    assert manager.lookup("s", 1, enable_forwarding=False) == (1, 2.5)
    assert manager.lookup("s", 1, enable_forwarding=False) == (1, 2.5)
    assert manager.valid_hits == 1
    assert manager.metrics()["error_breakdown"][WRONG_SUBJECT] == 1


def test_speculative_entries_are_usable_and_belong_to_no_vehicle():
    steady = [_vehicle(f"v{i}", 10.0 * i, 0.0) for i in range(10)]
    oracle = _oracle({(1, 10): steady})
    manager = _manager(oracle, mode="coca")

    manager.set_sim_time(10)
    task = _traffic(1, 0.0, 0.0, 10)
    manager.register_tasks([("t", task)])
    attrs = dict(task, data_size_kB=100)
    assert manager.precompute("t", 1, attrs) is not None

    metadata = manager._metadata[1]["t"]
    assert metadata.vehicle_id is None      # computed for nobody in particular
    assert metadata.cell_rsu == 1

    # It must still be servable — otherwise pre-placement is silently disabled.
    manager.register_tasks([("t", _traffic(2, 0.0, 0.0, 10))])
    assert manager.lookup("t", 1, enable_forwarding=False) is not None
    assert manager.total_hits == 1
