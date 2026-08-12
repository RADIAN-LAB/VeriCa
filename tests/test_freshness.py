"""Tests for intent/freshness.py — AoI-based per-intent freshness model."""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent.freshness import (
    DECAY_RATES,
    FreshnessPolicy,
    aoi,
    freshness_score,
)


# ---------------------------------------------------------------------------
# aoi()
# ---------------------------------------------------------------------------

def test_aoi_zero_age():
    t = 1000.0
    assert aoi(t, now=t) == 0.0


def test_aoi_positive():
    assert aoi(1000.0, now=1010.0) == 10.0


def test_aoi_clamped_to_zero():
    assert aoi(1010.0, now=1000.0) == 0.0


# ---------------------------------------------------------------------------
# freshness_score()
# ---------------------------------------------------------------------------

def test_freshness_at_zero_age_is_one():
    t = 1000.0
    for intent_idx in range(5):
        score = freshness_score(t, intent_idx, now=t)
        assert abs(score - 1.0) < 1e-9, f"intent {intent_idx}: expected 1.0 got {score}"


def test_freshness_decreases_with_age():
    t = 1000.0
    for intent_idx in range(5):
        s1 = freshness_score(t, intent_idx, now=t + 5)
        s2 = freshness_score(t, intent_idx, now=t + 20)
        assert s1 > s2, f"intent {intent_idx}: score should decrease with age"


def test_hazard_decays_faster_than_route():
    t = 1000.0
    age = 10.0
    hazard = freshness_score(t, 1, now=t + age)  # hazard_detection λ=0.20
    route = freshness_score(t, 2, now=t + age)   # route_optimization λ=0.02
    assert hazard < route, "hazard_detection should decay faster than route_optimization"


def test_freshness_half_life():
    """At half-life age, score should be ~0.5."""
    t = 1000.0
    for intent_idx, lam in DECAY_RATES.items():
        half_life = math.log(2) / lam
        score = freshness_score(t, intent_idx, now=t + half_life)
        assert abs(score - 0.5) < 0.01, (
            f"intent {intent_idx}: expected ~0.5 at half-life, got {score:.4f}"
        )


# ---------------------------------------------------------------------------
# FreshnessPolicy
# ---------------------------------------------------------------------------

def test_policy_score_matches_function():
    fp = FreshnessPolicy()
    t = 1000.0
    for intent_idx in range(5):
        age = 5.0
        assert abs(fp.score(t, intent_idx, now=t + age) -
                   freshness_score(t, intent_idx, now=t + age)) < 1e-9


def test_is_fresh_true_when_new():
    fp = FreshnessPolicy(threshold=0.2)
    assert fp.is_fresh(1000.0, 1, now=1000.0)


def test_is_fresh_false_when_stale():
    fp = FreshnessPolicy(threshold=0.2)
    # hazard λ=0.2, threshold 0.2 → stale after -ln(0.2)/0.2 ≈ 8.05s
    assert not fp.is_fresh(1000.0, 1, now=1000.0 + 20.0)


def test_ttl_remaining_positive_when_fresh():
    fp = FreshnessPolicy(threshold=0.2)
    ttl = fp.ttl_remaining(1000.0, 2, now=1000.0)  # route_optimization, brand new
    assert ttl > 0


def test_ttl_remaining_negative_when_stale():
    fp = FreshnessPolicy(threshold=0.2)
    ttl = fp.ttl_remaining(1000.0, 1, now=1000.0 + 30.0)  # hazard, very old
    assert ttl < 0


def test_recommended_ttl_consistent_with_decay():
    fp = FreshnessPolicy(threshold=0.2)
    for intent_idx in range(5):
        ttl = fp.recommended_ttl(intent_idx)
        # At exactly recommended TTL seconds, score should equal threshold
        score = fp.score(1000.0, intent_idx, now=1000.0 + ttl)
        assert abs(score - 0.2) < 0.01, (
            f"intent {intent_idx}: score at recommended_ttl should be ~0.2, got {score:.4f}"
        )


def test_custom_decay_rates():
    custom = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    fp = FreshnessPolicy(decay_rates=custom, threshold=0.5)
    # λ=1, half-life = ln(2) ≈ 0.693s
    score = fp.score(1000.0, 0, now=1000.0 + math.log(2))
    assert abs(score - 0.5) < 0.01


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
