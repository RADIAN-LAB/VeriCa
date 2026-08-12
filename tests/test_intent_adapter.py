"""Integration tests for the drop-in CoCa cache adapter."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coca_adapter.intent_distributed import IntentAwareDistributedCacheManager


def _manager(mode: str = "intent") -> IntentAwareDistributedCacheManager:
    return IntentAwareDistributedCacheManager(
        rsu_ids=[1, 2],
        cache_capacity=10,
        default_ttl=300,
        reuse_mode=mode,
        seed=7,
    )


def test_semantic_lookup_reuses_cross_type_same_intent():
    # SafetyAlert (1) and Emergency (7) are distinct task types that share the
    # hazard_detection intent, so one's result may serve the other.
    manager = _manager("intent")
    manager.set_sim_time(10)
    manager.register_tasks(
        [
            ("stored", {"task_type": 1, "origin_rsu": 1}),
            ("incoming", {"task_type": 7, "origin_rsu": 1}),
        ]
    )
    manager.store("stored", (1, 2.5), 1, replicate=False)
    manager.set_sim_time(11)

    assert manager.lookup("incoming", 1, enable_forwarding=False) == (1, 2.5)
    assert manager.semantic_hits == 1
    assert manager.total_hits == 1


def test_freshness_mode_rejects_old_hazard_result():
    manager = _manager("freshness_only")
    manager.set_sim_time(10)
    manager.register_tasks([("hazard", {"task_type": 1, "origin_rsu": 1})])
    manager.store("hazard", (1, 2.5), 1, replicate=False)
    manager.set_sim_time(40)

    assert manager.lookup("hazard", 1, enable_forwarding=False) is None
    assert manager.rejected_stale == 1
    assert manager.total_misses == 1


def test_semantic_lookup_never_substitutes_a_different_intent():
    manager = _manager("intent")
    manager.set_sim_time(10)
    manager.register_tasks(
        [
            ("traffic", {"task_type": 0, "origin_rsu": 1}),
            ("hazard", {"task_type": 1, "origin_rsu": 1}),
        ]
    )
    manager.store("traffic", (1, 2.5), 1, replicate=False)

    assert manager.lookup("hazard", 1, enable_forwarding=False) is None
    assert manager.rejected_intent == 1


def test_oracle_metrics_count_stale_reuse_in_coca_mode():
    manager = _manager("coca")
    manager.set_sim_time(10)
    manager.register_tasks([("hazard", {"task_type": 1, "origin_rsu": 1})])
    manager.store("hazard", (1, 2.5), 1, replicate=False)
    manager._metadata[1]["hazard"].valid_until = 11
    manager.set_sim_time(20)

    assert manager.lookup("hazard", 1, enable_forwarding=False) == (1, 2.5)
    assert manager.stale_hits == 1
    assert manager.metrics()["stale_reuse_rate"] == 1.0


def test_simulation_time_zero_is_supported():
    manager = _manager("intent")
    manager.set_sim_time(0)
    manager.register_tasks([("traffic", {"task_type": 0, "origin_rsu": 1})])
    manager.store("traffic", (1, 2.5), 1, replicate=False)

    assert manager.lookup("traffic", 1, enable_forwarding=False) == (1, 2.5)
