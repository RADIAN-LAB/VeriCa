"""The FoggyCache-style approximate reuse baseline (B5).

A baseline is only evidence if it works. These tests check that the index
actually finds near inputs, refuses far ones, and applies the confidence test —
so that when it loses to the proposed design, it loses for the reason claimed
(no notion of validity) rather than because it was broken.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coca_adapter.approximate_reuse import (
    ApproximateReuseIndex,
    distance,
    feature_vector,
)


def _task(task_type=5, priority=1, size=100, x=0.0, y=0.0, speed=30.0):
    return {
        "task_type": task_type, "priority": priority, "data_size": size,
        "x": x, "y": y, "speed": speed,
    }


def test_feature_vector_reads_both_attribute_spellings():
    """The scheduler supplies x/speed; raw trace rows use pos_x/veh_speed_kmh."""
    scheduler_style = feature_vector(_task(x=300.0, y=600.0, speed=60.0))
    trace_style = feature_vector({
        "task_type": 5, "priority": 1, "data_size_kB": 100,
        "pos_x": 300.0, "pos_y": 600.0, "veh_speed_kmh": 60.0,
    })
    assert scheduler_style == trace_style


def test_distance_treats_task_type_as_categorical():
    same = distance(feature_vector(_task(x=0.0)), feature_vector(_task(x=10.0)))
    other = distance(
        feature_vector(_task(task_type=5)), feature_vector(_task(task_type=1))
    )
    assert same < other


def test_reuses_a_near_input():
    index = ApproximateReuseIndex(max_distance=0.25, homogeneity=0.5, k=3)
    for i in range(5):
        index.add(f"near{i}", feature_vector(_task(x=10.0 * i)), intent_idx=0)
    resident = {f"near{i}" for i in range(5)}

    hit = index.query(feature_vector(_task(x=5.0)), resident)
    assert hit is not None


def test_refuses_a_far_input():
    index = ApproximateReuseIndex(max_distance=0.05, homogeneity=0.5, k=3)
    index.add("far", feature_vector(_task(x=0.0, y=0.0)), intent_idx=0)

    # 1200 m away on both axes is a full grid diagonal — not an equivalent input.
    assert index.query(feature_vector(_task(x=1200.0, y=1200.0)), {"far"}) is None


def test_refuses_an_inhomogeneous_neighbourhood():
    """FoggyCache's confidence test: disagreeing neighbours mean no reuse."""
    index = ApproximateReuseIndex(max_distance=5.0, homogeneity=0.9, k=4)
    for i in range(4):
        index.add(f"k{i}", feature_vector(_task(x=1.0 * i)), intent_idx=i)
    resident = {f"k{i}" for i in range(4)}

    assert index.query(feature_vector(_task(x=0.5)), resident) is None


def test_ignores_entries_that_have_been_evicted():
    index = ApproximateReuseIndex(max_distance=5.0, homogeneity=0.5, k=3)
    index.add("gone", feature_vector(_task()), intent_idx=0)

    # CoCa evicts inside its own cache without telling the index; a query with
    # an empty resident set must not resurrect the entry.
    assert index.query(feature_vector(_task()), set()) is None


def test_manager_reuses_across_keys_in_foggycache_mode():
    from coca_adapter.intent_distributed import IntentAwareDistributedCacheManager

    manager = IntentAwareDistributedCacheManager(
        rsu_ids=[1, 2], cache_capacity=10, default_ttl=300,
        reuse_mode="foggycache", seed=7,
        reuse_index_params={"k": 3, "homogeneity": 0.5, "max_distance": 1.0},
    )
    manager.set_sim_time(10)
    stored = _task(x=0.0)
    stored.update({"node_id": 1, "time": 10, "origin_rsu": 1})
    manager.register_tasks([("stored", stored)])
    manager.store("stored", (1, 2.5), 1, replicate=False)

    # A different key whose *input* is nearly the same task, moments later.
    incoming = _task(x=8.0)
    incoming.update({"node_id": 2, "time": 12, "origin_rsu": 1})
    manager.set_sim_time(12)
    manager.register_tasks([("incoming", incoming)])

    assert manager.lookup("incoming", 1, enable_forwarding=False) == (1, 2.5)
    assert manager.semantic_hits == 1


def test_foggycache_has_no_freshness_gate():
    """The property under test: it cannot tell that a result has gone stale."""
    from coca_adapter.intent_distributed import IntentAwareDistributedCacheManager

    manager = IntentAwareDistributedCacheManager(
        rsu_ids=[1, 2], cache_capacity=10, default_ttl=300,
        reuse_mode="foggycache", seed=7,
        reuse_index_params={"k": 3, "homogeneity": 0.5, "max_distance": 1.0},
    )
    manager.set_sim_time(10)
    task = _task()
    task.update({"node_id": 1, "time": 10, "origin_rsu": 1})
    manager.register_tasks([("t", task)])
    manager.store("t", (1, 2.5), 1, replicate=False)

    manager.set_sim_time(10_000)          # ages beyond any plausible lifetime
    later = dict(task, time=10_000)
    manager.register_tasks([("t", later)])
    assert manager.lookup("t", 1, enable_forwarding=False) == (1, 2.5)
    assert manager.rejected_stale == 0
