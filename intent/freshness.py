"""AoI-based per-intent freshness model for VANET task-result caching.

Each intent class has a different decay rate (λ) reflecting the physical
dynamics of the underlying vehicular task:

  hazard_detection   → decays fast  (safety-critical, seconds)
  traffic_analysis   → medium decay (traffic changes in ~10-30s)
  route_optimization → slow decay   (routes valid for ~35s)
  sensor_processing  → fast decay   (sensor data updated rapidly)
  environment_perception → slowest  (environment changes slowly)

Formula
-------
  freshness_score(t) = exp(-λ[intent] × AoI(t))

where AoI(t) = t_now - t_stored_at  (age in seconds).
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional

from .classifier import INTENT_LABELS, NUM_INTENTS

# ---------------------------------------------------------------------------
# Per-intent exponential decay rates  λ  (1/second)
# half-life = ln(2) / λ
#
# PROVISIONAL. These are hand-set starting values, superseded by
# ``scripts/fit_decay_rates.py``, which fits λ per intent on a held-out
# calibration split of the SUMO trace against the trace-grounded invalidation
# oracle (``intent/ground_truth.py``). Do not report these values in the paper —
# report the fitted ones, together with how well the exponential model describes
# the empirical invalidation process.
# ---------------------------------------------------------------------------
DECAY_RATES: Dict[int, float] = {
    0: 0.05,    # traffic_analysis       half-life ~14 s
    1: 0.20,    # hazard_detection       half-life ~3.5 s
    2: 0.02,    # route_optimization     half-life ~35 s
    3: 0.10,    # sensor_processing      half-life ~7 s
    4: 0.01,    # environment_perception half-life ~70 s
    5: 0.005,   # content_delivery       half-life ~139 s (near-static payloads)
    6: 0.08,    # parking_search         half-life ~9 s (availability churns fast)
}

# Default threshold below which a result is considered stale
DEFAULT_FRESHNESS_THRESHOLD: float = 0.20


def aoi(stored_at: float, now: Optional[float] = None) -> float:
    """Return Age of Information in seconds (always >= 0)."""
    return max(0.0, (now if now is not None else time.time()) - stored_at)


def freshness_score(
    stored_at: float,
    intent_idx: int,
    now: Optional[float] = None,
) -> float:
    """Continuous freshness score in [0, 1].

    Parameters
    ----------
    stored_at  : Unix timestamp when the result was cached
    intent_idx : intent class index (0–4)
    now        : optional override for current time (useful in tests)

    Returns
    -------
    float in [0, 1] — 1.0 = brand new, approaches 0 as result ages
    """
    idx = intent_idx % NUM_INTENTS
    lam = DECAY_RATES[idx]
    age = aoi(stored_at, now)
    return math.exp(-lam * age)


class FreshnessPolicy:
    """Stateless freshness policy backed by per-intent AoI decay."""

    def __init__(
        self,
        decay_rates: Optional[Dict[int, float]] = None,
        threshold: float = DEFAULT_FRESHNESS_THRESHOLD,
    ) -> None:
        self._rates = decay_rates if decay_rates is not None else DECAY_RATES
        self.threshold = threshold

    def score(self, stored_at: float, intent_idx: int, now: Optional[float] = None) -> float:
        """Freshness score in [0, 1]."""
        idx = intent_idx % NUM_INTENTS
        lam = self._rates.get(idx, DECAY_RATES[idx])
        age = aoi(stored_at, now)
        return math.exp(-lam * age)

    def is_fresh(
        self,
        stored_at: float,
        intent_idx: int,
        threshold: Optional[float] = None,
        now: Optional[float] = None,
    ) -> bool:
        """Return True if freshness score >= threshold."""
        t = threshold if threshold is not None else self.threshold
        return self.score(stored_at, intent_idx, now) >= t

    def ttl_remaining(
        self,
        stored_at: float,
        intent_idx: int,
        threshold: Optional[float] = None,
        now: Optional[float] = None,
    ) -> float:
        """Seconds remaining until score drops below threshold.

        Returns a negative value if already stale.
        """
        t = threshold if threshold is not None else self.threshold
        idx = intent_idx % NUM_INTENTS
        lam = self._rates.get(idx, DECAY_RATES[idx])
        # freshness(age) = exp(-lam * age) >= t
        # => age <= -ln(t) / lam
        if t <= 0.0:
            return float("inf")
        max_age = -math.log(t) / lam
        current_age = aoi(stored_at, now)
        return round(max_age - current_age, 4)

    def recommended_ttl(self, intent_idx: int, threshold: Optional[float] = None) -> float:
        """Static TTL (seconds) inferred from decay rate and threshold."""
        t = threshold if threshold is not None else self.threshold
        idx = intent_idx % NUM_INTENTS
        lam = self._rates.get(idx, DECAY_RATES[idx])
        if t <= 0.0:
            return float("inf")
        return round(-math.log(t) / lam, 2)

    def intent_label(self, intent_idx: int) -> str:
        """Human-readable label for an intent index."""
        return INTENT_LABELS[intent_idx % NUM_INTENTS]


# Module-level singleton
freshness_policy = FreshnessPolicy()

__all__ = [
    "DECAY_RATES",
    "DEFAULT_FRESHNESS_THRESHOLD",
    "FreshnessPolicy",
    "aoi",
    "freshness_score",
    "freshness_policy",
]
