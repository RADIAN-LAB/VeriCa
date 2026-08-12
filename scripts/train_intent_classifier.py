#!/usr/bin/env python3
"""Fit and evaluate the text intent classifier on the calibration split.

Trains on utterances synthesised for the calibration seeds and reports accuracy
and a confusion matrix on held-out evaluation seeds. Both numbers belong in the
paper: accuracy on synthetic text is only as meaningful as the text is hard, and
the confusion matrix is where a reader can see whether the errors are the ones
the generator deliberately created (ambiguous phrasings, overlapping vocabulary)
or something the classifier simply cannot do.

Usage:
    python3 scripts/train_intent_classifier.py \
        --train-traces experiments/evaluation/seed_04/tasks.csv \
        --test-traces experiments/evaluation/seed_10/tasks.csv \
        --output experiments/figures/intent_classifier.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intent.text_synthesis import describe_task, true_intent  # noqa: E402
from intent.trained_classifier import TrainedIntentClassifier  # noqa: E402
from sdn_core.task_taxonomy import INTENT_LABELS  # noqa: E402


def load_dataset(
    traces: Sequence[Path], seed: int, ambiguous_rate: float, limit: int
) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for trace in traces:
        with trace.open() as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if limit and index >= limit:
                    break
                texts.append(describe_task(row, seed=seed,
                                           ambiguous_rate=ambiguous_rate))
                labels.append(true_intent(row))
    return texts, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-traces", type=Path, nargs="+", required=True)
    parser.add_argument("--test-traces", type=Path, nargs="+", required=True)
    parser.add_argument("--ambiguous-rate", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=8000,
                        help="Rows per trace (0 = all).")
    parser.add_argument("--model-output", type=Path,
                        default=ROOT / "experiments" / "models" / "intent_text.json")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Different seeds for train and test so the *text* differs too, not just the
    # underlying tasks — otherwise identical utterances appear on both sides.
    train_texts, train_labels = load_dataset(
        args.train_traces, seed=1, ambiguous_rate=args.ambiguous_rate,
        limit=args.limit,
    )
    test_texts, test_labels = load_dataset(
        args.test_traces, seed=2, ambiguous_rate=args.ambiguous_rate,
        limit=args.limit,
    )
    print(f"train: {len(train_texts)} utterances   test: {len(test_texts)}")

    classifier = TrainedIntentClassifier.fit(train_texts, train_labels)
    classifier.save(args.model_output)

    accuracy = classifier.accuracy(test_texts, test_labels)
    confusion = classifier.confusion(test_texts, test_labels)
    print(f"\nHeld-out accuracy: {100 * accuracy:.1f}%")
    print(f"Vocabulary: {len(classifier.vocabulary)} tokens")
    print(f"\nSample utterances:")
    for text, label in list(zip(test_texts, test_labels))[:6]:
        predicted = classifier.predict(text)
        mark = " " if predicted == label else "x"
        print(f"  {mark} {text[:52]:<54} {label} -> {predicted}")

    print("\nConfusion (row = true, column = predicted):")
    present = [label for label in INTENT_LABELS if label in confusion]
    header = "".join(f"{label[:6]:>8}" for label in present)
    print(f"  {'':<24}{header}")
    for true_label in present:
        row = confusion.get(true_label, {})
        total = sum(row.values()) or 1
        cells = "".join(
            f"{100 * row.get(pred, 0) / total:>8.0f}" for pred in present
        )
        print(f"  {true_label:<24}{cells}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(
                {
                    "accuracy": accuracy,
                    "confusion": confusion,
                    "ambiguous_rate": args.ambiguous_rate,
                    "vocabulary_size": len(classifier.vocabulary),
                    "train_utterances": len(train_texts),
                    "test_utterances": len(test_texts),
                    "model": str(args.model_output),
                    "note": "Synthetic text; accuracy is bounded by the "
                            "deliberately ambiguous fraction.",
                },
                handle, indent=2,
            )
        print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
