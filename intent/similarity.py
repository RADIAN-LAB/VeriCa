"""Intent similarity matrix and reuse scoring for context-aware caching."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .classifier import IntentClassifier, NUM_INTENTS
from .freshness import FreshnessPolicy, freshness_policy as _default_fp

# Soft similarity matrix[i][j] in [0,1].
#
# PROVISIONAL. Superseded by ``scripts/fit_similarity.py``, which estimates
# entry [i][j] as the empirical probability that an intent-i result is *correct*
# for an intent-j request, measured against the trace-grounded oracle in
# ``intent/ground_truth.py``. Report the measured matrix, not these values.
_SIM: List[List[float]] = [
    # traffic  hazard  route   sensor  environ content parking
    [1.00,     0.30,   0.60,   0.20,   0.40,   0.05,   0.35],  # traffic_analysis
    [0.30,     1.00,   0.20,   0.40,   0.50,   0.05,   0.10],  # hazard_detection
    [0.60,     0.20,   1.00,   0.10,   0.30,   0.05,   0.45],  # route_optimization
    [0.20,     0.40,   0.10,   1.00,   0.60,   0.05,   0.10],  # sensor_processing
    [0.40,     0.50,   0.30,   0.60,   1.00,   0.05,   0.20],  # environment_perception
    [0.05,     0.05,   0.05,   0.05,   0.05,   1.00,   0.05],  # content_delivery
    [0.35,     0.10,   0.45,   0.10,   0.20,   0.05,   1.00],  # parking_search
]


class IntentSimilarity:
    """Compute intent similarity and composite reuse scores."""

    def __init__(
        self,
        classifier: IntentClassifier | None = None,
        freshness_policy: FreshnessPolicy | None = None,
    ):
        from .classifier import intent_classifier
        self._clf = classifier or intent_classifier
        self._fp = freshness_policy or _default_fp

    def similarity(self, i: int, j: int) -> float:
        """Soft similarity between intent index i and j."""
        return _SIM[i % NUM_INTENTS][j % NUM_INTENTS]

    def reuse_score(
        self,
        task_row: Dict[str, Any],
        cached_intent_idx: int,
        context_compat: float = 1.0,
        freshness: float = 1.0,
        stored_at: Optional[float] = None,
        w_intent: float = 0.25,
        w_context: float = 0.40,
        w_freshness: float = 0.35,
    ) -> float:
        """Composite reuse score combining intent similarity, context, and freshness.

        Parameters
        ----------
        task_row          : incoming task attribute dict
        cached_intent_idx : intent index of the cached result
        context_compat    : context compatibility in [0,1]
        freshness         : freshness score in [0,1] — ignored if stored_at is provided
        stored_at         : Unix timestamp when result was cached; if given, freshness
                            is computed automatically via FreshnessPolicy
        w_*               : weights (should sum to 1.0)

        Returns
        -------
        float in [0,1]
        """
        incoming_idx, _ = self._clf.classify(task_row)
        intent_sim = self.similarity(incoming_idx, cached_intent_idx)
        if stored_at is not None:
            freshness = self._fp.score(stored_at, cached_intent_idx)
        score = w_intent * intent_sim + w_context * context_compat + w_freshness * freshness
        return round(min(1.0, max(0.0, score)), 4)

    def should_reuse(
        self,
        task_row: Dict[str, Any],
        cached_intent_idx: int,
        context_compat: float = 1.0,
        freshness: float = 1.0,
        stored_at: Optional[float] = None,
        threshold: float = 0.65,
    ) -> bool:
        """Return True if reuse score >= threshold."""
        return self.reuse_score(
            task_row, cached_intent_idx, context_compat, freshness, stored_at
        ) >= threshold


__all__ = ["IntentSimilarity"]
