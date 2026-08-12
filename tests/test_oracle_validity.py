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


# ---------------------------------------------------------------------------
# Key scoping and serve-time verification
# ---------------------------------------------------------------------------

def _sensor_task(node_id: str, x: float, y: float, time: float = 10):
    return {
        "task_type": 3, "priority": 1, "data_size": 100,   # sensor_processing
        "node_id": node_id, "x": x, "y": y, "time": time, "origin_rsu": 1,
    }


def test_adaptive_scope_splits_per_vehicle_intents_from_cell_intents():
    """Key scope should follow what the answer actually depends on."""
    manager = _manager(_oracle({}), mode="coca")
    manager.key_mode = "context_adaptive"
    manager.set_sim_time(10)

    # A traffic answer holds across the cell, so it is keyed by the cell.
    traffic = _traffic(1, *IN_CELL_1, time=10)
    manager.register_tasks([("t", traffic)])
    traffic_key = manager._cache_key(
        "t", "traffic_analysis", manager._current_context("t", 1)
    )
    assert ":r" in traffic_key

    # A fusion answer depends on the ego vehicle's neighbourhood, so it is keyed
    # by the local zone instead.
    sensor = _sensor_task("a0", *IN_CELL_1)
    manager.register_tasks([("s", sensor)])
    sensor_key = manager._cache_key(
        "s", "sensor_processing", manager._current_context("s", 1)
    )
    assert ":z" in sensor_key


def test_adaptive_scope_separates_distant_vehicles_in_one_cell():
    manager = _manager(_oracle({}), mode="coca")
    manager.key_mode = "context_adaptive"
    manager.set_sim_time(10)

    near = _sensor_task("a0", *IN_CELL_1)
    far = _sensor_task("b0", *FAR_IN_CELL_1)
    manager.register_tasks([("s", near)])
    near_key = manager._cache_key(
        "s", "sensor_processing", manager._context_of(near)
    )
    far_key = manager._cache_key(
        "s", "sensor_processing", manager._context_of(far)
    )
    assert near_key != far_key


def test_verification_refuses_a_result_the_state_no_longer_supports():
    ten = [_vehicle(f"v{i}", 10.0 * i, 0.0) for i in range(10)]
    five = ten[:5]
    oracle = _oracle({(1, 10): ten, (1, 40): five})
    manager = IntentAwareDistributedCacheManager(
        rsu_ids=[1, 2], cache_capacity=10, default_ttl=300,
        reuse_mode="coca", seed=7, oracle=oracle, verify_before_serve=True,
    )

    manager.set_sim_time(10)
    manager.register_tasks([("t", _traffic(1, *IN_CELL_1, time=10))])
    manager.store("t", (1, 2.5), 1, replicate=False)

    manager.set_sim_time(40)
    manager.register_tasks([("t", _traffic(1, *IN_CELL_1, time=40))])
    # Without verification this is served and counted wrong; with it, refused.
    assert manager.lookup("t", 1, enable_forwarding=False) is None
    assert manager.verification_failures == 1
    assert manager.total_misses == 1
    assert manager.stale_hits == 0


def test_verification_still_serves_a_result_the_state_supports():
    steady = [_vehicle(f"v{i}", 10.0 * i, 0.0) for i in range(10)]
    oracle = _oracle({(1, 10): steady, (1, 12): steady})
    manager = IntentAwareDistributedCacheManager(
        rsu_ids=[1, 2], cache_capacity=10, default_ttl=300,
        reuse_mode="coca", seed=7, oracle=oracle, verify_before_serve=True,
    )

    manager.set_sim_time(10)
    manager.register_tasks([("t", _traffic(1, *IN_CELL_1, time=10))])
    manager.store("t", (1, 2.5), 1, replicate=False)

    manager.set_sim_time(12)
    manager.register_tasks([("t", _traffic(1, *IN_CELL_1, time=12))])
    assert manager.lookup("t", 1, enable_forwarding=False) == (1, 2.5)
    assert manager.valid_hits == 1
    assert manager.verification_failures == 0


def test_verification_refuses_what_it_cannot_observe():
    """A forwarded entry describes another cell, which this RSU cannot check."""
    busy = [_vehicle(f"v{i}", 10.0 * i, 0.0) for i in range(10)]
    quiet = [_vehicle("w0", *IN_CELL_2, speed=0.5)]
    oracle = _oracle({(1, 10): busy, (2, 10): quiet})
    manager = IntentAwareDistributedCacheManager(
        rsu_ids=[1, 2], cache_capacity=10, default_ttl=300,
        reuse_mode="coca", seed=7, oracle=oracle, verify_before_serve=True,
    )

    manager.set_sim_time(10)
    manager.register_tasks([("t", _traffic(1, *IN_CELL_1, time=10))])
    manager.store("t", (1, 2.5), 1, replicate=False)

    manager.register_tasks([("t", _traffic(2, *IN_CELL_2, time=10))])
    assert manager.lookup("t", 2, enable_forwarding=True) is None
    assert manager.verification_failures == 1


def test_verification_delivers_only_correct_answers():
    """The property the whole design rests on: no wrong answer survives."""
    ten = [_vehicle(f"v{i}", 10.0 * i, 0.0) for i in range(10)]
    oracle = _oracle({(1, t): (ten if t < 30 else ten[:4]) for t in range(10, 60)})
    manager = IntentAwareDistributedCacheManager(
        rsu_ids=[1, 2], cache_capacity=10, default_ttl=300,
        reuse_mode="coca", seed=7, oracle=oracle, verify_before_serve=True,
    )

    manager.set_sim_time(10)
    manager.register_tasks([("t", _traffic(1, *IN_CELL_1, time=10))])
    manager.store("t", (1, 2.5), 1, replicate=False)

    for now in range(11, 60):
        manager.set_sim_time(now)
        manager.register_tasks([("t", _traffic(1, *IN_CELL_1, time=now))])
        manager.lookup("t", 1, enable_forwarding=False)

    assert manager.stale_hits == 0
    assert sum(manager.error_kinds.values()) == 0
    assert manager.valid_hits > 0     # and it still serves when it legitimately can


def test_aoi_baseline_never_substitutes_across_keys():
    """B4 addresses content by identity; cross-key reuse would be our mechanism,
    not the baseline's, and charging it for those errors would be unfair."""
    steady = [_vehicle(f"v{i}", 10.0 * i, 0.0) for i in range(10)]
    oracle = _oracle({(1, 10): steady})
    manager = IntentAwareDistributedCacheManager(
        rsu_ids=[1, 2], cache_capacity=10, default_ttl=300,
        reuse_mode="aoi_window", seed=7, oracle=oracle,
        aoi_windows={idx: 30.0 for idx in range(7)},
    )

    manager.set_sim_time(10)
    stored = _traffic(1, *IN_CELL_1, time=10)
    manager.register_tasks([("stored", stored)])
    manager.store("stored", (1, 2.5), 1, replicate=False)

    other = dict(stored, task_type=1, node_id=2)   # a different key and intent
    manager.register_tasks([("other", other)])
    assert manager.lookup("other", 1, enable_forwarding=False) is None
    assert manager.semantic_hits == 0


def test_aoi_window_bounds_age_per_intent():
    steady = [_vehicle(f"v{i}", 10.0 * i, 0.0) for i in range(10)]
    oracle = _oracle({(1, t): steady for t in range(10, 60)})
    manager = IntentAwareDistributedCacheManager(
        rsu_ids=[1, 2], cache_capacity=10, default_ttl=300,
        reuse_mode="aoi_window", seed=7, oracle=oracle,
        aoi_windows={idx: 5.0 for idx in range(7)},
    )

    manager.set_sim_time(10)
    manager.register_tasks([("t", _traffic(1, *IN_CELL_1, time=10))])
    manager.store("t", (1, 2.5), 1, replicate=False)

    manager.set_sim_time(14)                       # inside the window
    manager.register_tasks([("t", _traffic(1, *IN_CELL_1, time=14))])
    assert manager.lookup("t", 1, enable_forwarding=False) == (1, 2.5)

    manager.set_sim_time(20)                       # aged out
    manager.register_tasks([("t", _traffic(1, *IN_CELL_1, time=20))])
    assert manager.lookup("t", 1, enable_forwarding=False) is None
