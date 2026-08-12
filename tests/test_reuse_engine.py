"""Tests for core/reuse_engine.py — ReuseEngine + ReuseDecision."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reuse_engine import ReuseEngine, ReuseDecision


def _task(task_type=0, origin_rsu="rsu_1", time_window="morning"):
    return {"task_type": task_type, "origin_rsu": origin_rsu, "time_window": time_window}


def _cached(intent_idx=0, age=0.0, rsu="rsu_1", time_window="morning"):
    return {
        "intent_idx": intent_idx,
        "stored_at": time.time() - age,
        "context_rsu": rsu,
        "context_time": time_window,
    }


# ---------------------------------------------------------------------------
# Full reuse approval
# ---------------------------------------------------------------------------

def test_reuse_approved_same_intent_fresh_same_context():
    engine = ReuseEngine(threshold=0.65)
    d = engine.decide(_task(task_type=0), _cached(intent_idx=0, age=0.5))
    assert d.should_reuse, f"Expected reuse, got: {d}"
    assert d.reason == "reuse"
    assert d.score >= 0.65


def test_decision_has_all_fields():
    engine = ReuseEngine()
    d = engine.decide(_task(), _cached())
    assert isinstance(d.should_reuse, bool)
    assert 0.0 <= d.score <= 1.0
    assert 0.0 <= d.intent_sim <= 1.0
    assert 0.0 <= d.freshness <= 1.0
    assert 0.0 <= d.context_compat <= 1.0
    assert d.reason in {"reuse", "stale", "intent_mismatch", "context_mismatch", "low_score"}


# ---------------------------------------------------------------------------
# Stale rejection
# ---------------------------------------------------------------------------

def test_stale_hazard_rejected():
    engine = ReuseEngine(freshness_min=0.10)
    # hazard_detection λ=0.20, after 30s score ≈ exp(-6) ≈ 0.0025 < 0.10
    d = engine.decide(_task(task_type=1), _cached(intent_idx=1, age=30.0))
    assert not d.should_reuse
    assert d.reason == "stale"


def test_fresh_hazard_not_stale():
    engine = ReuseEngine(freshness_min=0.10)
    # hazard after 1s: exp(-0.2) ≈ 0.82 > 0.10
    d = engine.decide(_task(task_type=1), _cached(intent_idx=1, age=1.0))
    assert d.freshness > 0.10


# ---------------------------------------------------------------------------
# Intent mismatch rejection
# ---------------------------------------------------------------------------

def test_intent_mismatch_rejected():
    engine = ReuseEngine(intent_min=0.20)
    # task_type=2 (route_optimization intent=2) vs cached intent=3 (sensor_processing)
    # similarity[2][3] = 0.10 < intent_min=0.20
    d = engine.decide(_task(task_type=2), _cached(intent_idx=3, age=0.5))
    assert not d.should_reuse
    assert d.reason == "intent_mismatch"


def test_cross_type_reuse_same_intent():
    # task_type=0 and task_type=5 both map to intent 0 (traffic_analysis)
    engine = ReuseEngine(threshold=0.60)
    d = engine.decide(_task(task_type=5), _cached(intent_idx=0, age=1.0))
    assert d.intent_sim == 1.0, "Same intent should have similarity 1.0"


# ---------------------------------------------------------------------------
# Context compatibility
# ---------------------------------------------------------------------------

def test_different_rsu_reduces_context():
    engine = ReuseEngine()
    d_same = engine.decide(_task(origin_rsu="rsu_1"), _cached(rsu="rsu_1", age=1.0))
    d_diff = engine.decide(_task(origin_rsu="rsu_1"), _cached(rsu="rsu_9", age=1.0))
    assert d_same.context_compat > d_diff.context_compat


def test_missing_context_info_neutral():
    engine = ReuseEngine()
    task = {"task_type": 0}
    entry = {"intent_idx": 0, "stored_at": time.time() - 1.0}
    d = engine.decide(task, entry)
    assert 0.6 <= d.context_compat <= 1.0, "Missing context should be neutral"


# ---------------------------------------------------------------------------
# Reason labels
# ---------------------------------------------------------------------------

def test_reason_reuse_label():
    engine = ReuseEngine(threshold=0.60)
    d = engine.decide(_task(task_type=0), _cached(intent_idx=0, age=0.1))
    assert d.reason == "reuse"


def test_reason_stale_label():
    engine = ReuseEngine(freshness_min=0.10)
    d = engine.decide(_task(task_type=1), _cached(intent_idx=1, age=30.0))
    assert d.reason == "stale"


def test_reason_intent_mismatch_label():
    engine = ReuseEngine(intent_min=0.20)
    d = engine.decide(_task(task_type=2), _cached(intent_idx=3, age=0.5))
    assert d.reason == "intent_mismatch"


# ---------------------------------------------------------------------------
# str representation
# ---------------------------------------------------------------------------

def test_str_representation():
    engine = ReuseEngine()
    d = engine.decide(_task(), _cached())
    s = str(d)
    assert "score=" in s
    assert "reason=" in s


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
