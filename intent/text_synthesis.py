"""Synthesise natural-language task requests for the intent classifier.

Why this exists
---------------
The evaluation traces carry no text, so ``IntentClassifier.classify`` always took
the ``task_type`` lookup and the TF-IDF path was dead code in every experiment.
The paper claims intent is recovered from natural language; that claim needs
language to recover it from.

Avoiding the circularity trap
-----------------------------
Text generated from an intent label and then classified back to that label
measures the template generator, not the classifier. Four things keep this from
being a tautology, and the paper must state all four as limitations of synthetic
text:

1. **Utterances are built from task context, not from the label.** Speed band,
   urgency, data size and time of day drive the wording, so two tasks with the
   same intent read differently and two tasks with different intents can read
   similarly.
2. **Vocabulary deliberately overlaps between related intents.** "Stopped" and
   "blocked" appear in both hazard and parking phrasings; "slow" and "queue"
   appear in both traffic and routing. A classifier cannot win by keyword
   spotting alone.
3. **No template reuses the classifier's own corpus terms.** Matching against
   ``_CORPUS_BY_INTENT`` would be string comparison dressed up as NLP.
4. **A tunable fraction of utterances is genuinely ambiguous** — phrasings whose
   intent a human could not resolve either. This caps achievable accuracy below
   100%, which is the honest regime; a classifier reported at 99% on synthetic
   text tells the reader nothing.

Surface noise (typos, dropped articles, abbreviations) is applied on top, since
real in-vehicle requests are spoken or typed in a hurry.
"""
from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Mapping, Sequence

import coca_path  # noqa: F401 — puts the CoCa tree on sys.path

from sdn_core.task_taxonomy import INTENT_LABELS, intent_idx_of  # noqa: E402

# Phrasings per intent. Each is a template over slots filled from task context.
# Vocabulary overlap across intents is intentional: see the module docstring.
_TEMPLATES: Dict[str, List[str]] = {
    "traffic_analysis": [
        "how {busy} is it {ahead}",
        "what is the {flow} like on this {road}",
        "check the {queue} up {ahead}",
        "is the {road} {slow} right now",
        "give me the {flow} around here",
    ],
    "hazard_detection": [
        "anything {blocking} the {road} {ahead}",
        "is there a {stopped} vehicle in the way",
        "warn me about {trouble} up {ahead}",
        "check for {trouble} on this {road}",
        "is it {safe} to keep going",
    ],
    "route_optimization": [
        "best way to get {there} from here",
        "find me a {quicker} way {there}",
        "should i turn off before the {queue}",
        "which {road} gets me {there} {quicker}",
        "recalculate my way {there}",
    ],
    "sensor_processing": [
        "merge what the cars around me can see",
        "pull the readings from {nearby} vehicles",
        "combine the feeds from cars {nearby}",
        "what are the vehicles {nearby} picking up",
        "fuse the {nearby} sensor feeds for me",
    ],
    "environment_perception": [
        "build me a picture of what is around",
        "map everything within range",
        "what does the area around me look like",
        "give me the layout of this {area}",
        "show what is occupying the {area}",
    ],
    "content_delivery": [
        "keep playing the {media}",
        "buffer more of this {media}",
        "download the rest of the {media}",
        "stream the next part of the {media}",
        "carry on with the {media}",
    ],
    "parking_search": [
        "anywhere to leave the car {nearby}",
        "find me a {space} around here",
        "is there a free {space} {ahead}",
        "somewhere to stop {nearby}",
        "looking for a {space} near the {area}",
    ],
}

# Slot fillers. Several are shared across intents on purpose.
_SLOTS: Dict[str, Sequence[str]] = {
    "busy": ("busy", "packed", "congested", "heavy"),
    "ahead": ("ahead", "up front", "on the way", "further on"),
    "flow": ("flow", "movement", "conditions", "situation"),
    "road": ("road", "street", "route", "stretch"),
    "queue": ("queue", "backup", "tailback", "line of cars"),
    "slow": ("slow", "crawling", "jammed", "stop start"),
    "blocking": ("blocking", "obstructing", "in the way on"),
    "stopped": ("stopped", "stalled", "broken down", "parked badly"),
    "trouble": ("trouble", "an obstruction", "a problem", "anything dangerous"),
    "safe": ("safe", "clear", "ok"),
    "there": ("there", "to my stop", "to the destination", "to the address"),
    "quicker": ("quicker", "faster", "shorter", "less painful"),
    "nearby": ("nearby", "close by", "around me", "next to me"),
    "area": ("area", "block", "neighbourhood", "district"),
    "media": ("video", "episode", "track", "stream", "playlist"),
    "space": ("space", "spot", "bay", "place to park"),
}

# Phrasings a human could not resolve either. The label stays the task's true
# intent, so these are guaranteed classifier errors — which is the point: they
# put a realistic ceiling on accuracy.
_AMBIGUOUS = (
    "what is going on around here",
    "check the road for me",
    "anything i should know about up ahead",
    "give me an update",
    "look around and tell me",
    "is it clear",
    "what is nearby",
    "have a look at the situation",
)

# Context prefixes driven by the task's own attributes rather than its intent.
_URGENT_PREFIX = ("quick", "urgent", "right now", "asap")
_SLOW_PREFIX = ("while i wait", "no rush", "when you can")

_ABBREVIATIONS = {
    "please": "pls", "you": "u", "are": "r", "about": "abt",
    "right now": "rn", "vehicle": "veh", "and": "n",
}


def _rng_for(row: Mapping[str, object], seed: int) -> random.Random:
    """Deterministic per-task randomness, so a trace is reproducible."""
    identity = f"{seed}|{row.get('node_id')}|{row.get('task_id')}|{row.get('timestamp')}"
    digest = hashlib.sha256(identity.encode()).digest()[:8]
    return random.Random(int.from_bytes(digest, "big"))


def _fill(template: str, rng: random.Random) -> str:
    text = template
    while "{" in text:
        start = text.index("{")
        end = text.index("}", start)
        slot = text[start + 1:end]
        choice = rng.choice(_SLOTS[slot]) if slot in _SLOTS else slot
        text = text[:start] + choice + text[end + 1:]
    return text


def _add_noise(text: str, rng: random.Random) -> str:
    """Typos, abbreviations and dropped articles, as in hurried real input."""
    if rng.random() < 0.25:
        for full, short in _ABBREVIATIONS.items():
            if full in text and rng.random() < 0.5:
                text = text.replace(full, short, 1)
                break
    if rng.random() < 0.2:
        text = text.replace(" the ", " ", 1)
    if rng.random() < 0.15 and len(text) > 6:
        # Transpose two adjacent characters inside a word.
        position = rng.randrange(1, len(text) - 2)
        if text[position] != " " and text[position + 1] != " ":
            text = text[:position] + text[position + 1] + text[position] + text[position + 2:]
    return text


def describe_task(
    row: Mapping[str, object], seed: int = 42, ambiguous_rate: float = 0.12
) -> str:
    """Natural-language request for one task row.

    Args:
        row: Trace row with ``task_type``, ``priority``, ``veh_speed_kmh``.
        seed: Trace-level seed; combined with the task identity so the same
            trace always yields the same text.
        ambiguous_rate: Share of utterances drawn from the unresolvable pool.
    """
    rng = _rng_for(row, seed)
    intent = INTENT_LABELS[intent_idx_of(int(row.get("task_type", 0)))]

    if rng.random() < ambiguous_rate:
        text = rng.choice(_AMBIGUOUS)
    else:
        text = _fill(rng.choice(_TEMPLATES[intent]), rng)

    # Context, not label, decides the framing.
    speed = float(row.get("veh_speed_kmh", 0.0) or 0.0)
    priority = int(row.get("priority", 0) or 0)
    if priority >= 3 and rng.random() < 0.5:
        text = f"{rng.choice(_URGENT_PREFIX)} {text}"
    elif speed < 5.0 and rng.random() < 0.3:
        text = f"{rng.choice(_SLOW_PREFIX)} {text}"

    return _add_noise(text, rng)


def true_intent(row: Mapping[str, object]) -> str:
    """The label the text should be recovered as."""
    return INTENT_LABELS[intent_idx_of(int(row.get("task_type", 0)))]


__all__ = ["describe_task", "true_intent"]
