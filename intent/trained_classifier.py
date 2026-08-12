"""Intent classification from text, trained rather than hand-written.

``IntentClassifier`` scores an utterance against one hand-written keyword
sentence per intent. That is string overlap presented as NLP: it cannot learn
that "tailback" and "crawling" both indicate traffic unless someone thinks to
write both words down, and it has no way to weigh evidence that points two ways.

This is a multinomial Naive Bayes classifier over unigrams and bigrams, fitted
on the calibration split and evaluated on held-out seeds. Naive Bayes rather than
something heavier because the vocabulary is small, the training set is modest,
and a linear model on bag-of-words is the standard baseline for short-text intent
classification — a bigger model would be harder to justify than to fit.

No third-party dependency, so the classifier trains and runs wherever the rest of
the pipeline does, and the fitted model is plain JSON that can be inspected.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Add-one smoothing keeps unseen words from zeroing a class outright.
SMOOTHING = 1.0


def tokenize(text: str) -> List[str]:
    """Unigrams plus adjacent bigrams.

    Bigrams matter here: "no rush" and "right now" invert urgency, and "free
    space" means something a bag of single words cannot express.
    """
    words = re.findall(r"[a-z]+", text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


class TrainedIntentClassifier:
    """Multinomial Naive Bayes over tokenised utterances."""

    def __init__(
        self,
        log_prior: Optional[Dict[str, float]] = None,
        log_likelihood: Optional[Dict[str, Dict[str, float]]] = None,
        vocabulary: Optional[List[str]] = None,
        default_log_likelihood: Optional[Dict[str, float]] = None,
    ) -> None:
        self.log_prior = log_prior or {}
        self.log_likelihood = log_likelihood or {}
        self.vocabulary = set(vocabulary or [])
        self.default_log_likelihood = default_log_likelihood or {}

    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls, texts: Sequence[str], labels: Sequence[str]
    ) -> "TrainedIntentClassifier":
        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length")
        if not texts:
            raise ValueError("nothing to train on")

        class_counts: Dict[str, int] = defaultdict(int)
        token_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        totals: Dict[str, int] = defaultdict(int)
        vocabulary: set[str] = set()

        for text, label in zip(texts, labels):
            class_counts[label] += 1
            for token in tokenize(text):
                token_counts[label][token] += 1
                totals[label] += 1
                vocabulary.add(token)

        size = len(vocabulary)
        n = len(texts)
        log_prior = {
            label: math.log(count / n) for label, count in class_counts.items()
        }
        log_likelihood: Dict[str, Dict[str, float]] = {}
        default: Dict[str, float] = {}
        for label in class_counts:
            denominator = totals[label] + SMOOTHING * size
            log_likelihood[label] = {
                token: math.log((count + SMOOTHING) / denominator)
                for token, count in token_counts[label].items()
            }
            # One shared value for every in-vocabulary token this class never
            # saw, so the table stays small instead of |V| x |classes|.
            default[label] = math.log(SMOOTHING / denominator)

        return cls(log_prior, log_likelihood, sorted(vocabulary), default)

    # ------------------------------------------------------------------

    def scores(self, text: str) -> Dict[str, float]:
        """Unnormalised log posterior per intent."""
        result = {}
        for label, prior in self.log_prior.items():
            total = prior
            table = self.log_likelihood.get(label, {})
            fallback = self.default_log_likelihood.get(label, 0.0)
            for token in tokenize(text):
                if token in self.vocabulary:
                    total += table.get(token, fallback)
            result[label] = total
        return result

    def predict(self, text: str) -> str:
        if not self.log_prior:
            raise RuntimeError("classifier has not been fitted")
        return max(self.scores(text).items(), key=lambda item: item[1])[0]

    def accuracy(self, texts: Sequence[str], labels: Sequence[str]) -> float:
        if not texts:
            return float("nan")
        correct = sum(
            1 for text, label in zip(texts, labels) if self.predict(text) == label
        )
        return correct / len(texts)

    def confusion(
        self, texts: Sequence[str], labels: Sequence[str]
    ) -> Dict[str, Dict[str, int]]:
        """True label -> predicted label -> count."""
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for text, label in zip(texts, labels):
            matrix[label][self.predict(text)] += 1
        return {true: dict(row) for true, row in matrix.items()}

    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(
                {
                    "log_prior": self.log_prior,
                    "log_likelihood": self.log_likelihood,
                    "vocabulary": sorted(self.vocabulary),
                    "default_log_likelihood": self.default_log_likelihood,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Path) -> "TrainedIntentClassifier":
        with Path(path).open() as handle:
            payload = json.load(handle)
        return cls(
            payload["log_prior"],
            payload["log_likelihood"],
            payload["vocabulary"],
            payload["default_log_likelihood"],
        )


__all__ = ["TrainedIntentClassifier", "tokenize"]
