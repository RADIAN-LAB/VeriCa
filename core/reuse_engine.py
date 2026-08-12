"""Unified reuse decision engine combining intent similarity and AoI freshness.

Usage
-----
    from core.reuse_engine import ReuseEngine, ReuseDecision

    engine = ReuseEngine()
    decision = engine.decide(task_row, cached_entry)
    if decision.should_reuse:
        return cached_entry["value"]

cached_entry dict keys
----------------------
    intent_idx    : int    — intent class of the cached result
    stored_at     : float  — Unix timestamp when result was cached
    context_rsu   : str    — RSU ID where result was computed (optional)
    context_time  : float  — time-window tag for the result (optional)

task_row dict keys (subset used here)
--------------------------------------
    task_type     : int    — used by IntentClassifier rule-based path
    task_description : str — used by IntentClassifier NLP path (optional)
    origin_rsu    : str    — requesting vehicle's RSU (optional)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from intent.classifier import IntentClassifier
from intent.freshness import FreshnessPolicy, freshness_policy as _default_fp
from intent.similarity import IntentSimilarity, _SIM
from intent.classifier import NUM_INTENTS


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReuseDecision:
    """Outcome of a single reuse evaluation."""

    should_reuse: bool
    score: float
    intent_sim: float
    freshness: float
    context_compat: float
    incoming_intent: int
    cached_intent: int
    reason: str  # reuse | stale | intent_mismatch | context_mismatch | low_score

    def __str__(self) -> str:
        arrow = "✓ REUSE" if self.should_reuse else "✗ SKIP"
        return (
            f"{arrow}  score={self.score:.3f}  "
            f"intent_sim={self.intent_sim:.2f}  "
            f"freshness={self.freshness:.2f}  "
            f"context={self.context_compat:.2f}  "
            f"reason={self.reason}"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ReuseEngine:
    """Gate cache reuse decisions using intent similarity + AoI freshness + context.

    Parameters
    ----------
    threshold       : minimum composite score to allow reuse (default 0.65)
    freshness_min   : minimum freshness score independent of composite (default 0.10)
    intent_min      : minimum intent similarity to allow reuse (default 0.20)
    w_intent        : weight for intent similarity in composite score
    w_context       : weight for context compatibility
    w_freshness     : weight for freshness score
    """

    def __init__(
        self,
        threshold: float = 0.65,
        freshness_min: float = 0.10,
        intent_min: float = 0.20,
        w_intent: float = 0.25,
        w_context: float = 0.40,
        w_freshness: float = 0.35,
        classifier: Optional[IntentClassifier] = None,
        freshness_policy: Optional[FreshnessPolicy] = None,
    ) -> None:
        self.threshold = threshold
        self.freshness_min = freshness_min
        self.intent_min = intent_min
        self.w_intent = w_intent
        self.w_context = w_context
        self.w_freshness = w_freshness
        self._clf = classifier or IntentClassifier()
        self._fp = freshness_policy or _default_fp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        task_row: Dict[str, Any],
        cached_entry: Dict[str, Any],
        now: Optional[float] = None,
    ) -> ReuseDecision:
        """Evaluate whether to reuse a cached result for an incoming task.

        Parameters
        ----------
        task_row      : incoming task attribute dict
        cached_entry  : dict with keys: intent_idx, stored_at, [context_rsu, context_time]
        now           : optional time override (for testing)

        Returns
        -------
        ReuseDecision
        """
        now = time.time() if now is None else now

        # --- Intent classification ---
        incoming_intent, _ = self._clf.classify(task_row)
        cached_intent = int(cached_entry.get("intent_idx", 0)) % NUM_INTENTS
        intent_sim = _SIM[incoming_intent][cached_intent]

        # --- Hard gate: intent similarity too low ---
        if intent_sim < self.intent_min:
            return ReuseDecision(
                should_reuse=False,
                score=0.0,
                intent_sim=intent_sim,
                freshness=0.0,
                context_compat=0.0,
                incoming_intent=incoming_intent,
                cached_intent=cached_intent,
                reason="intent_mismatch",
            )

        # --- Freshness ---
        stored_at = float(cached_entry.get("stored_at", now))
        freshness = self._fp.score(stored_at, cached_intent, now=now)

        # --- Hard gate: result too stale ---
        if freshness < self.freshness_min:
            return ReuseDecision(
                should_reuse=False,
                score=round(self.w_freshness * freshness, 4),
                intent_sim=intent_sim,
                freshness=freshness,
                context_compat=0.0,
                incoming_intent=incoming_intent,
                cached_intent=cached_intent,
                reason="stale",
            )

        # --- Context compatibility ---
        context_compat = self._context_compat(task_row, cached_entry)

        # --- Composite score ---
        score = (
            self.w_intent * intent_sim
            + self.w_context * context_compat
            + self.w_freshness * freshness
        )
        score = round(min(1.0, max(0.0, score)), 4)

        if score < self.threshold:
            reason = "context_mismatch" if context_compat < 0.5 else "low_score"
        else:
            reason = "reuse"

        return ReuseDecision(
            should_reuse=(score >= self.threshold),
            score=score,
            intent_sim=intent_sim,
            freshness=freshness,
            context_compat=context_compat,
            incoming_intent=incoming_intent,
            cached_intent=cached_intent,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _context_compat(
        self,
        task_row: Dict[str, Any],
        cached_entry: Dict[str, Any],
    ) -> float:
        """Simple spatial-temporal context compatibility score in [0, 1].

        Rules
        -----
        - Same RSU → +0.6
        - No RSU info available → neutral 0.8 (benefit of doubt)
        - Time-window match → +0.4 of remainder
        """
        task_rsu = task_row.get("origin_rsu") or task_row.get("destination_rsu")
        cached_rsu = cached_entry.get("context_rsu")

        if task_rsu is None or cached_rsu is None:
            rsu_score = 0.8
        elif task_rsu == cached_rsu:
            rsu_score = 1.0
        else:
            rsu_score = 0.3

        task_tw = task_row.get("time_window")
        cached_tw = cached_entry.get("context_time")

        if task_tw is None or cached_tw is None:
            tw_score = 0.8
        elif task_tw == cached_tw:
            tw_score = 1.0
        else:
            tw_score = 0.4

        return round(0.6 * rsu_score + 0.4 * tw_score, 4)


__all__ = ["ReuseEngine", "ReuseDecision"]
