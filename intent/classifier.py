"""NLP-based intent classifier for VANET task-result caching.

Classifies a task into one of the intent classes using:
  1. Rule-based task_type lookup  (fast path)
  2. TF-IDF cosine similarity over intent description corpus  (NLP path)
  3. Free-text task_description support

The intent registry and the task_type -> intent mapping are both sourced from
``sdn_core.task_taxonomy`` in the CoCa tree, which is the single authoritative
taxonomy. The previous local mapping was ``task_type % 5``, which disagreed with
the semantics the trace generator assigns: Emergency (type 7) was classified as
``route_optimization`` and so received the slowest freshness decay of any class,
while Media streaming (type 2) was classified as ``route_optimization`` too.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

import coca_path  # noqa: F401  — puts the CoCa tree on sys.path

from sdn_core.task_taxonomy import (  # noqa: E402
    INTENT_LABELS as _TAXONOMY_INTENT_LABELS,
    NUM_TASK_TYPES,
    intent_idx_of,
)

# ---------------------------------------------------------------------------
# Intent registry (mirrors the authoritative taxonomy)
# ---------------------------------------------------------------------------

INTENT_LABELS: List[str] = list(_TAXONOMY_INTENT_LABELS)

NUM_INTENTS = len(INTENT_LABELS)

# Description corpus — one sentence per intent, used for TF-IDF NLP path.
# Order must match INTENT_LABELS.
_CORPUS_BY_INTENT: Dict[str, str] = {
    "traffic_analysis":
        "traffic flow density monitoring congestion vehicles count speed measurement",
    "hazard_detection":
        "hazard accident obstacle detection emergency road safety alert warning",
    "route_optimization":
        "route path navigation optimization shortest fastest travel direction",
    "sensor_processing":
        "sensor lidar radar camera fusion raw data processing signal filter",
    "environment_perception":
        "environment perception surroundings map awareness context state coverage",
    "content_delivery":
        "media streaming video content download infotainment playback bitrate segment",
    "parking_search":
        "parking space availability spot vacancy garage kerbside search occupancy",
}

_CORPUS: List[str] = [_CORPUS_BY_INTENT[label] for label in INTENT_LABELS]

# Rule-based mapping: task_type int -> intent index, from the taxonomy.
_TYPE_TO_INTENT: Dict[int, int] = {
    ttype: intent_idx_of(ttype) for ttype in range(NUM_TASK_TYPES)
}

# ---------------------------------------------------------------------------
# Minimal TF-IDF engine (zero external deps)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _build_tfidf(corpus: List[str]) -> List[Dict[str, float]]:
    tokenized = [_tokenize(doc) for doc in corpus]
    N = len(corpus)
    df: Dict[str, int] = {}
    for tokens in tokenized:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    vecs: List[Dict[str, float]] = []
    for tokens in tokenized:
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0 / len(tokens)
        vec = {t: tf[t] * math.log((N + 1) / (df.get(t, 0) + 1) + 1) for t in tf}
        vecs.append(vec)
    return vecs


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    dot = sum(a.get(k, 0.0) * v for k, v in b.items())
    na = math.sqrt(sum(v * v for v in a.values())) + 1e-12
    nb = math.sqrt(sum(v * v for v in b.values())) + 1e-12
    return dot / (na * nb)


_CORPUS_VECS: List[Dict[str, float]] = _build_tfidf(_CORPUS)

# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Classify task intent via NLP (TF-IDF) or rule-based fallback.

    Examples
    --------
    >>> clf = IntentClassifier()
    >>> clf.classify({"task_type": 1})
    (1, 'hazard_detection')
    >>> clf.classify_text("detect obstacle on road")
    (1, 'hazard_detection')
    >>> clf.onehot(2)
    [0.0, 0.0, 1.0, 0.0, 0.0]
    """

    NUM_INTENTS = NUM_INTENTS

    def classify(self, task_row: Dict[str, Any]) -> Tuple[int, str]:
        """Classify from a structured task attribute dict."""
        if "intent_class" in task_row:
            idx = int(task_row["intent_class"]) % NUM_INTENTS
            return idx, INTENT_LABELS[idx]

        if "task_description" in task_row:
            return self.classify_text(str(task_row["task_description"]))

        ttype = int(task_row.get("task_type", 0))
        if ttype in _TYPE_TO_INTENT:
            idx = _TYPE_TO_INTENT[ttype]
            return idx, INTENT_LABELS[idx]

        return self.classify_text(f"task type {ttype}")

    def classify_text(self, text: str) -> Tuple[int, str]:
        """Classify from free-text description using TF-IDF cosine similarity."""
        tokens = _tokenize(text)
        if not tokens:
            return 0, INTENT_LABELS[0]
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0 / len(tokens)
        N = NUM_INTENTS
        query_vec = {
            t: tf[t] * math.log((N + 1) / 2 + 1)
            for t in tf
        }
        scores = [_cosine(query_vec, cv) for cv in _CORPUS_VECS]
        best = int(max(range(NUM_INTENTS), key=lambda i: scores[i]))
        return best, INTENT_LABELS[best]

    def onehot(self, intent_idx: int) -> List[float]:
        """One-hot vector of length NUM_INTENTS."""
        return [1.0 if i == intent_idx else 0.0 for i in range(NUM_INTENTS)]

    def scores(self, text: str) -> List[float]:
        """Return cosine similarity scores against all intents for given text."""
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * NUM_INTENTS
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0 / len(tokens)
        N = NUM_INTENTS
        query_vec = {t: tf[t] * math.log((N + 1) / 2 + 1) for t in tf}
        return [_cosine(query_vec, cv) for cv in _CORPUS_VECS]


# Module-level singleton
intent_classifier = IntentClassifier()

__all__ = ["IntentClassifier", "intent_classifier", "INTENT_LABELS", "NUM_INTENTS"]
