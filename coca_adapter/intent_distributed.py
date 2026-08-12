"""Intent-aware distributed cache compatible with CoCa's scheduler API.

Whether a served result was actually *right* is judged by the trace-grounded
invalidation oracle (``intent.oracle``) when one is supplied. That oracle is
derived from vehicle mobility alone, so the freshness policy is no longer graded
against the exponential it is built from, and reusing one task's result for
another's question can now be counted as wrong. Passing no oracle falls back to
the old self-consistent exponential sampling, which is retained only as the
``exponential`` ablation that quantifies how much that circularity flattered the
method.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

from .approximate_reuse import ApproximateReuseIndex, feature_vector
from .bootstrap import bootstrap_coca

bootstrap_coca()

from cache.distributed_manager import DistributedCacheManager  # noqa: E402
from cache.manager import is_placeholder  # noqa: E402
from sdn_core import scenario  # noqa: E402

from core.reuse_engine import ReuseEngine  # noqa: E402
from intent.classifier import INTENT_LABELS, IntentClassifier  # noqa: E402
from intent.freshness import DECAY_RATES, freshness_policy  # noqa: E402
from intent.ground_truth import INTENT_FEATURES, NEIGHBOURS, covers  # noqa: E402
from intent.oracle import InvalidationOracle  # noqa: E402


REUSE_MODES = {
    "coca", "intent", "freshness_only", "intent_only", "aoi_window", "foggycache",
}

# Baseline B4, the AoI-aware edge caching literature: a cached item is served
# while its Age of Information stays inside a refreshing window, and refreshed
# once it ages out. Distinct from ``freshness_only`` in what sets the bound —
# a window tuned to the method's own objective on calibration traces, rather
# than a survival function fitted to observed invalidation. That difference is
# the claim under test: whether age is an adequate proxy for validity.
DEFAULT_AOI_WINDOW_S = 10.0
VALIDITY_MODELS = {"oracle", "exponential"}

# How a cache entry is addressed.
#   coca          — CoCa's own key, (task_type, priority, data_size). Blind to
#                   space, which is why most of its hits answer about the wrong
#                   cell; kept as the baseline.
#   context_rsu   — (intent, RSU cell). The granularity the invalidation oracle
#                   measures, so the correctness gain it yields is fully visible.
#   context_zone  — (intent, 100 m occupancy zone). Finer, but measurably too
#                   fine: at ~1200 keys a second request rarely arrives inside
#                   the fitted validity window, so reuse locality collapses.
#   context_adaptive — scope chosen per intent by what the answer depends on.
#                   Cell-level observables (density, mean speed, edge speeds,
#                   stopped set) are constant across an RSU cell, so the cell is
#                   the right scope. A sensor-fusion answer depends on the ego
#                   vehicle's own neighbourhood, so it is only reusable nearby —
#                   the 100 m zone approximates the 50 m neighbour radius.
KEY_MODES = {"coca", "context_rsu", "context_zone", "context_adaptive"}
# Marks a key this class has already rewritten. CoCa's FL advisor learns its
# prediction pool from the coarse keys the runner builds, then hands them back
# to precompute/preplace, so rewriting has to be idempotent.
CONTEXT_KEY_PREFIX = "ctx:"

# Cost of one serve-time validity check, in milliseconds. The check compares a
# handful of scalars and two small sets against state the RSU already maintains
# for its own cell, so it is dominated by a cache-line read, not by computation
# or radio. Two orders of magnitude below CACHE_ACCESS_BASE_TIME (0.5 ms) and
# nearly three below a task computation (25 ms base).
VERIFY_COST_MS = 0.05

# How a hit can be wrong. Kept as named outcomes rather than one "stale" bucket
# because the paper's claim is about semantic reuse specifically: a wrong-intent
# or wrong-subject hit is a cost of the proposed mechanism, while a stale hit is
# the temporal cost every caching scheme already pays.
CORRECT = "correct"
WRONG_INTENT = "wrong_intent"      # stored result does not answer this question
WRONG_PLACE = "wrong_place"        # right question, state of a different cell
WRONG_SUBJECT = "wrong_subject"    # right cell, result belongs to another vehicle
STALE = "stale"                    # right question, right subject, drifted state
UNKNOWN = "unknown"                # trace has no ground truth for this cell-step

ERROR_KINDS = (WRONG_INTENT, WRONG_PLACE, WRONG_SUBJECT, STALE, UNKNOWN)
# Errors attributable to substituting a different question, place, or subject —
# i.e. the price of semantic reuse, as opposed to plain ageing.
SEMANTIC_ERRORS = (WRONG_INTENT, WRONG_PLACE, WRONG_SUBJECT)


@dataclass(frozen=True)
class RequestContext:
    """Where and for whom a task was issued, as the oracle needs it."""

    vehicle_id: Optional[str]
    cell_rsu: int
    timestep: float
    zone: Optional[Tuple[int, int]] = None


@dataclass
class EntryMetadata:
    intent_idx: int
    intent_label: str
    stored_at: float
    context_rsu: int
    task_type: int
    valid_until: float
    # Physical provenance: the cell and vehicle the result was computed for.
    # ``vehicle_id`` is None for speculatively pre-computed entries, which belong
    # to no vehicle — and so cannot correctly answer per-vehicle questions.
    cell_rsu: int = -1
    vehicle_id: Optional[str] = None
    # CoCa's original key for this entry, which is how its task attributes are
    # looked up after the cache key has been rewritten.
    source_key: str = ""

    def as_reuse_entry(self) -> Dict[str, Any]:
        return {
            "intent_idx": self.intent_idx,
            "stored_at": self.stored_at,
            "context_rsu": self.context_rsu,
        }


class IntentAwareDistributedCacheManager(DistributedCacheManager):
    """Drop-in CoCa cache manager with semantic and AoI-aware reuse.

    CoCa itself remains unchanged. The experiment wrapper assigns an instance
    of this class to ``cache.distributed_manager.distributed_cache_manager``.
    """

    def __init__(
        self,
        rsu_ids: list[int],
        cache_capacity: int = 100,
        default_ttl: int = 300,
        eviction_policy: str = "LRU",
        reuse_mode: str = "intent",
        threshold: float = 0.65,
        seed: int = 42,
        oracle: Optional[InvalidationOracle] = None,
        key_mode: str = "coca",
        verify_before_serve: bool = False,
        aoi_windows: Optional[Dict[int, float]] = None,
        reuse_index_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if reuse_mode not in REUSE_MODES:
            raise ValueError(f"reuse_mode must be one of {sorted(REUSE_MODES)}")
        if key_mode not in KEY_MODES:
            raise ValueError(f"key_mode must be one of {sorted(KEY_MODES)}")
        super().__init__(rsu_ids, cache_capacity, default_ttl, eviction_policy)
        self.reuse_mode = reuse_mode
        self.key_mode = key_mode
        self.verify_before_serve = verify_before_serve
        self.aoi_windows = dict(aoi_windows) if aoi_windows else {}
        # One approximate-reuse index per RSU, mirroring that cache's contents.
        index_params = dict(reuse_index_params or {})
        index_params.setdefault("seed", seed)
        self.reuse_indexes: Dict[int, ApproximateReuseIndex] = {
            rid: ApproximateReuseIndex(**index_params) for rid in rsu_ids
        }
        self.oracle = oracle
        self.validity_model = "oracle" if oracle is not None else "exponential"
        # A task result may cross task types within the same semantic intent,
        # but related intents are not interchangeable outputs. The stronger
        # intent gate prevents, for example, a traffic result serving a hazard
        # request merely because both were produced at the same RSU.
        self.engine = ReuseEngine(
            threshold=threshold,
            freshness_min=0.20,
            intent_min=0.80,
        )
        self.classifier = IntentClassifier()
        self.sim_time = 0.0
        self.seed = seed
        self._attrs: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[int, Dict[str, EntryMetadata]] = {
            rid: {} for rid in rsu_ids
        }
        self._preserve_next_store: set[tuple[int, str]] = set()
        # Tasks arrive as a batch but are looked up one at a time, and the cache
        # key is far coarser than a vehicle, so several vehicles in a batch share
        # one key. Queueing the contexts per key in arrival order lets each
        # lookup recover the vehicle that actually issued it — CoCa's scheduler
        # consumes the batch in the same order it was registered.
        self._pending: Dict[str, Deque[RequestContext]] = {}
        self._last_context: Dict[str, RequestContext] = {}
        # Every distinct cache key ever addressed, for the cardinality report.
        self._seen_keys: set[str] = set()

        self.valid_hits = 0
        self.stale_hits = 0
        self.semantic_hits = 0
        self.rejected_stale = 0
        self.rejected_intent = 0
        self.error_kinds: Dict[str, int] = {kind: 0 for kind in ERROR_KINDS}
        # Correctness of *semantic* (cross-key) hits alone, which is the number
        # the proposed mechanism has to defend.
        self.semantic_correct = 0
        self.verifications = 0
        self.verification_failures = 0
        self.per_intent: Dict[int, Dict[str, int]] = {
            idx: defaultdict(int) for idx in range(len(INTENT_LABELS))
        }

    def set_sim_time(self, timestamp: float) -> None:
        self.sim_time = float(timestamp)

    def register_tasks(self, items: Iterable[tuple[str, Dict[str, Any]]]) -> None:
        """Register the batch's task attributes and per-task request contexts.

        CoCa's ``lookup`` receives only a key and an RSU, so everything the
        oracle needs about the requester has to be staged here.
        """
        self._pending = {}
        for key, attrs in items:
            self._attrs[key] = dict(attrs)
            self._pending.setdefault(key, deque()).append(self._context_of(attrs))

    @staticmethod
    def _position_of(attrs: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Vehicle position, however the caller happens to spell it.

        The scheduler wrapper supplies ``x``/``y``; the FL advisor hands back raw
        trace rows, which carry ``pos_x``/``pos_y``.
        """
        for x_name, y_name in (("x", "y"), ("pos_x", "pos_y")):
            if x_name in attrs and y_name in attrs:
                return float(attrs[x_name]), float(attrs[y_name])
        return None

    def _context_of(self, attrs: Dict[str, Any]) -> RequestContext:
        """Physical context of one task: which vehicle, which cell, which second."""
        vehicle = attrs.get("node_id", attrs.get("vehicle_id"))
        position = self._position_of(attrs)
        if position is not None:
            # The cell the vehicle is physically in, which is how the oracle
            # indexes state. ``origin_rsu`` is a scheduling decision and can
            # differ once trajectory prediction forwards a task.
            cell = scenario.nearest_rsu(*position)
            zone = scenario.occupancy_cell(*position)
        else:
            cell = int(attrs.get("origin_rsu", -1))
            zone = None
        timestep = attrs.get("time", attrs.get("timestamp", self.sim_time))
        return RequestContext(
            vehicle_id=None if vehicle is None else str(vehicle),
            cell_rsu=int(cell),
            timestep=float(timestep),
            zone=zone,
        )

    # ------------------------------------------------------------------
    # Cache addressing
    # ------------------------------------------------------------------

    def _intent_label(self, key: str, rsu_id: int) -> str:
        _, label = self.classifier.classify(self._task(key, rsu_id))
        return label

    def _cache_key(
        self, key: str, intent_label: str, context: RequestContext
    ) -> str:
        """Address an entry by what determines its answer, not by task shape.

        CoCa keys on ``(task_type, priority, data_size)``, which says nothing
        about *where* the result is true — the measured cause of 83% of its wrong
        answers. Scoping the key to the intent and the place makes a same-key hit
        a same-question, same-place hit by construction.
        """
        if self.key_mode == "coca" or key.startswith(CONTEXT_KEY_PREFIX):
            self._seen_keys.add(key)
            return key
        per_vehicle = NEIGHBOURS in INTENT_FEATURES.get(intent_label, ())
        zone_scoped = self.key_mode == "context_zone" or (
            self.key_mode == "context_adaptive" and per_vehicle
        )
        if zone_scoped and context.zone is not None:
            zx, zy = context.zone
            rewritten = f"{CONTEXT_KEY_PREFIX}{intent_label}:z{zx}_{zy}"
        else:
            rewritten = f"{CONTEXT_KEY_PREFIX}{intent_label}:r{context.cell_rsu}"
        self._seen_keys.add(rewritten)
        return rewritten

    def _next_context(self, key: str, rsu_id: int) -> RequestContext:
        """Consume the context of the task now being looked up for ``key``."""
        queue = self._pending.get(key)
        if queue:
            context = queue.popleft()
            self._last_context[key] = context
            return context
        return self._current_context(key, rsu_id)

    def _current_context(self, key: str, rsu_id: int) -> RequestContext:
        """Context to attribute a *store* to, without consuming a queued one.

        CoCa computes missed tasks after the lookup loop, so a store carries no
        attributes of its own and arrives out of step with the queue. The most
        recent looked-up context for the key is the closest true provenance —
        same timestep, and the same cell whenever the batch's tasks for that key
        did not span cells.
        """
        remembered = self._last_context.get(key)
        if remembered is not None:
            return remembered
        queue = self._pending.get(key)
        if queue:
            return queue[0]
        return self._context_of(self._task(key, rsu_id))

    def _task(self, key: str, rsu_id: int) -> Dict[str, Any]:
        task = dict(self._attrs.get(key, {}))
        task.setdefault("task_type", 0)
        task.setdefault("origin_rsu", rsu_id)
        return task

    def _sample_valid_until(self, key: str, intent_idx: int) -> float:
        # Fallback ground truth for ``validity_model == "exponential"`` only.
        # Survival P(T > age) = exp(-lambda*age) with the same lambda the
        # freshness policy scores by, which is precisely the circularity the
        # trace-grounded oracle removes; it is kept to measure that bias.
        # Hash sampling keeps it aligned across methods even when their cache
        # operation sequences differ.
        rate = DECAY_RATES[intent_idx]
        if rate <= 0.0:
            # A never-invalidating class — content_delivery, whose fitted rate is
            # exactly zero because a cached media segment is not invalidated by
            # traffic. Sampling a lifetime from it would divide by zero.
            return float("inf")
        identity = f"{self.seed}|{key}|{self.sim_time:.6f}|{intent_idx}".encode()
        sample = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
        uniform = (sample + 0.5) / (2**64)
        lifetime = -math.log1p(-uniform) / rate
        return self.sim_time + lifetime

    def _new_metadata(
        self,
        key: str,
        rsu_id: int,
        context: Optional[RequestContext] = None,
        vehicle_known: bool = True,
    ) -> EntryMetadata:
        task = self._task(key, rsu_id)
        intent_idx, intent_label = self.classifier.classify(task)
        if context is None:
            context = self._current_context(key, rsu_id)
        return EntryMetadata(
            intent_idx=intent_idx,
            intent_label=intent_label,
            stored_at=self.sim_time,
            context_rsu=rsu_id,
            task_type=int(task.get("task_type", 0)),
            # Only the exponential fallback consults this; under an oracle it is
            # unused, so do not pay for the sample.
            valid_until=(
                float("inf") if self.oracle is not None
                else self._sample_valid_until(key, intent_idx)
            ),
            cell_rsu=context.cell_rsu,
            vehicle_id=context.vehicle_id if vehicle_known else None,
            source_key=key,
        )

    # ------------------------------------------------------------------
    # Was serving this entry actually correct?
    # ------------------------------------------------------------------

    def _verify_locally(
        self,
        metadata: EntryMetadata,
        requested_intent_label: str,
        context: RequestContext,
    ) -> bool:
        """Would an RSU, checking before it serves, accept this entry?

        An RSU observes its own cell, so it can compare the state signature an
        entry was computed from against the state it sees now, on exactly the
        features the request needs. That is a local, sub-millisecond check on
        information the RSU already has — not a look at the ground truth.

        The distinction that keeps this honest: the check is restricted to state
        the serving RSU can observe. A cross-RSU forwarded entry cannot be
        verified locally, so it is refused rather than trusted.
        """
        if self.oracle is None:
            return True
        if metadata.cell_rsu != context.cell_rsu:
            # Another cell's state is not observable from here.
            return False
        return self._judge(metadata, requested_intent_label, context) == CORRECT

    def _judge(
        self,
        metadata: EntryMetadata,
        requested_intent_label: str,
        context: RequestContext,
    ) -> str:
        """Classify a hit as correct or as a specific kind of error."""
        if self.oracle is None:
            # Self-consistent fallback: the entry's sampled lifetime, which knows
            # nothing about place, subject, or which question was asked.
            return CORRECT if self.sim_time <= metadata.valid_until else STALE

        stored_ts = int(metadata.stored_at)
        request_ts = int(context.timestep)
        if self.oracle.correct(
            metadata.intent_label,
            requested_intent_label,
            metadata.cell_rsu,
            stored_ts,
            context.cell_rsu,
            request_ts,
            vehicle_id=metadata.vehicle_id,
            request_vehicle_id=context.vehicle_id,
        ):
            return CORRECT

        # Attribute the failure to the most specific thing that differs, so the
        # cost of semantic substitution is not hidden inside a "stale" bucket.
        if not covers(metadata.intent_label, requested_intent_label):
            return WRONG_INTENT
        if not (
            self.oracle.has_state(metadata.cell_rsu, stored_ts)
            and self.oracle.has_state(context.cell_rsu, request_ts)
        ):
            return UNKNOWN
        if metadata.cell_rsu != context.cell_rsu:
            return WRONG_PLACE
        per_vehicle = NEIGHBOURS in INTENT_FEATURES.get(requested_intent_label, ())
        if per_vehicle and metadata.vehicle_id != context.vehicle_id:
            return WRONG_SUBJECT
        return STALE

    def _base_value(self, rsu_id: int, key: str) -> Any | None:
        """Real cached result for a key, or None.

        Reads the backing store directly (rather than via ``lookup``) so that
        probing candidates does not disturb hit/miss accounting. Reservations
        left by the FL advisor are not results, so they are skipped here just as
        ``CacheManager.lookup`` skips them.
        """
        cache = self.rsu_caches[rsu_id]
        item = cache._store.get(key)
        if item is None or is_placeholder(item[1]):
            return None
        return item[1]

    def _decision_allows(
        self,
        task: Dict[str, Any],
        metadata: EntryMetadata,
    ) -> tuple[bool, str]:
        age = max(0.0, self.sim_time - metadata.stored_at)
        if self.reuse_mode == "coca":
            return age <= self.default_ttl, "reuse" if age <= self.default_ttl else "stale"
        if self.reuse_mode == "aoi_window":
            window = self.aoi_windows.get(metadata.intent_idx, DEFAULT_AOI_WINDOW_S)
            fresh = age <= window
            return fresh, "reuse" if fresh else "stale"
        if self.reuse_mode == "foggycache":
            # Approximate reuse decides purely on input similarity; the
            # confidence test already ran in the index. It has no notion of a
            # result going out of date, which is the comparison being made.
            return True, "reuse"
        if self.reuse_mode == "freshness_only":
            fresh = freshness_policy.is_fresh(
                metadata.stored_at, metadata.intent_idx, now=self.sim_time
            )
            return fresh, "reuse" if fresh else "stale"

        decision_now = metadata.stored_at if self.reuse_mode == "intent_only" else self.sim_time
        decision = self.engine.decide(
            task, metadata.as_reuse_entry(), now=decision_now
        )
        return decision.should_reuse, decision.reason

    def _candidates(self, key: str, rsu_id: int, forwarding: bool):
        rsus = [rsu_id]
        if forwarding and self.ENABLE_FORWARDING:
            rsus.extend(r for r in self.rsu_ids if r != rsu_id)

        # Exact matches always have priority.
        for source_rsu in rsus:
            if key in self.rsu_caches[source_rsu]._store:
                yield source_rsu, key, False

        # Exact-key baselines. AoI-aware caching addresses content by identity
        # and bounds its age; it never substitutes one item's result for
        # another's, so letting it fall through to the semantic branch would
        # invent cross-intent reuse the baseline does not do — and then charge
        # it for the resulting wrong answers.
        if self.reuse_mode in {"coca", "freshness_only", "aoi_window"}:
            return

        if self.reuse_mode == "foggycache":
            # Approximate reuse picks one neighbour by input similarity rather
            # than scanning every key, so it yields at most one candidate.
            features = feature_vector(self._task(key, rsu_id))
            for source_rsu in rsus:
                store = self.rsu_caches[source_rsu]._store
                candidate = self.reuse_indexes[source_rsu].query(features, store)
                if candidate is not None and candidate != key:
                    yield source_rsu, candidate, True
            return
        for source_rsu in rsus:
            for candidate_key in list(self.rsu_caches[source_rsu]._store):
                if candidate_key != key:
                    yield source_rsu, candidate_key, True

    def lookup(self, key: str, rsu_id: int, enable_forwarding: bool = True) -> Any | None:
        task = self._task(key, rsu_id)
        incoming_intent, incoming_label = self.classifier.classify(task)
        context = self._next_context(key, rsu_id)
        cache_key = self._cache_key(key, incoming_label, context)

        for source_rsu, candidate_key, semantic in self._candidates(
            cache_key, rsu_id, enable_forwarding
        ):
            metadata = self._metadata[source_rsu].get(candidate_key)
            value = self._base_value(source_rsu, candidate_key)
            if metadata is None or value is None:
                continue
            allowed, reason = self._decision_allows(task, metadata)
            if not allowed:
                if reason == "stale":
                    self.rejected_stale += 1
                elif reason == "intent_mismatch":
                    self.rejected_intent += 1
                continue

            if self.verify_before_serve:
                # The policy would reuse this; the RSU now checks whether the
                # state it was computed from still holds. A failed check costs a
                # signature comparison and falls through to recomputation, which
                # is what makes every delivered answer correct.
                self.verifications += 1
                if not self._verify_locally(metadata, incoming_label, context):
                    self.verification_failures += 1
                    continue

            self.total_hits += 1
            self.per_intent[incoming_intent]["hits"] += 1
            if semantic:
                self.semantic_hits += 1
                self.per_intent[incoming_intent]["semantic_hits"] += 1

            verdict = self._judge(metadata, incoming_label, context)
            if verdict == CORRECT:
                self.valid_hits += 1
                self.per_intent[incoming_intent]["valid_hits"] += 1
                if semantic:
                    self.semantic_correct += 1
                    self.per_intent[incoming_intent]["semantic_correct"] += 1
            else:
                self.stale_hits += 1
                self.error_kinds[verdict] += 1
                self.per_intent[incoming_intent]["stale_hits"] += 1
                self.per_intent[incoming_intent][verdict] += 1

            if source_rsu != rsu_id:
                self.rsu_caches[rsu_id].store(candidate_key, value)
                self._metadata[rsu_id][candidate_key] = metadata
            self._preserve_next_store.add((rsu_id, cache_key))
            return value

        self.total_misses += 1
        self.per_intent[incoming_intent]["misses"] += 1
        return None

    def store(
        self,
        key: str,
        value: Any,
        rsu_id: int,
        ttl: Optional[int] = None,
        admission_threshold: int = 1,
        hot_threshold: int = 2,
        replicate: bool = True,
    ) -> None:
        context = self._current_context(key, rsu_id)
        cache_key = self._cache_key(key, self._intent_label(key, rsu_id), context)

        preserve = (rsu_id, cache_key) in self._preserve_next_store
        if preserve:
            self._preserve_next_store.discard((rsu_id, cache_key))
            old = self._metadata[rsu_id].get(cache_key)
        else:
            old = None

        super().store(
            cache_key,
            value,
            rsu_id,
            ttl=ttl,
            admission_threshold=admission_threshold,
            hot_threshold=hot_threshold,
            replicate=replicate,
        )
        metadata = old or self._new_metadata(key, rsu_id, context=context)
        self._metadata[rsu_id][cache_key] = metadata
        self._index_entry(cache_key, rsu_id, metadata)
        if replicate:
            for neighbor in self.rsu_ids:
                if cache_key in self.rsu_caches[neighbor]._store:
                    self._metadata[neighbor][cache_key] = metadata

    def preplace(self, key: str, rsu_id: int, ttl: Optional[int] = None) -> None:
        context = self._speculative_context(rsu_id, self._task(key, rsu_id))
        cache_key = self._cache_key(key, self._intent_label(key, rsu_id), context)
        existed = cache_key in self.rsu_caches[rsu_id]._store
        super().preplace(cache_key, rsu_id, ttl)
        if not existed and cache_key in self.rsu_caches[rsu_id]._store:
            self._metadata[rsu_id][cache_key] = self._new_metadata(
                key, rsu_id, context=context, vehicle_known=False
            )

    def precompute(
        self,
        key: str,
        rsu_id: int,
        attrs: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> Optional[float]:
        """Attach provenance to a speculatively computed entry.

        CoCa's ``precompute`` writes straight to the backing store, so without
        this override the entry would carry no metadata and every lookup would
        skip it — silently cancelling pre-placement in the intent experiments.
        """
        self._attrs.setdefault(key, dict(attrs))
        context = self._speculative_context(rsu_id, attrs)
        cache_key = self._cache_key(key, self._intent_label(key, rsu_id), context)
        compute_ms = super().precompute(cache_key, rsu_id, attrs, ttl)
        if compute_ms is not None:
            metadata = self._new_metadata(
                key, rsu_id, context=context, vehicle_known=False
            )
            self._metadata[rsu_id][cache_key] = metadata
            self._index_entry(cache_key, rsu_id, metadata)
        return compute_ms

    def _index_entry(
        self, cache_key: str, rsu_id: int, metadata: EntryMetadata
    ) -> None:
        """Record an entry's input descriptor for approximate reuse (B5).

        Indexed for every mode so the baseline can be switched on without
        changing what got cached; only ``foggycache`` ever queries it. The task
        attributes are keyed by CoCa's original key, which is what the index
        describes — the entry's *input*, not its address.
        """
        attrs = self._attrs.get(metadata.source_key)
        if attrs is None:
            return
        self.reuse_indexes[rsu_id].add(
            cache_key, feature_vector(attrs), metadata.intent_idx
        )

    def _speculative_context(
        self, rsu_id: int, attrs: Dict[str, Any]
    ) -> RequestContext:
        """Context of an entry produced for nobody in particular.

        A predicted-hot result is computed at the RSU ahead of any request, so it
        has no ego vehicle — per-vehicle intents therefore cannot be answered
        correctly from it, which is the honest modelling of what speculative
        computation can supply. It is computed for the place demand was last
        observed; failing that, for the RSU's own position.
        """
        position = self._position_of(attrs)
        if position is None:
            position = scenario.RSU_POSITIONS.get(rsu_id)
        return RequestContext(
            vehicle_id=None,
            cell_rsu=rsu_id,
            timestep=self.sim_time,
            zone=scenario.occupancy_cell(*position) if position else None,
        )

    def metrics(self) -> Dict[str, Any]:
        hits = self.total_hits
        attempts = hits + self.total_misses
        semantic_errors = sum(self.error_kinds[kind] for kind in SEMANTIC_ERRORS)
        distinct_keys = {
            key for cache in self.rsu_caches.values() for key in cache._store
        }
        return {
            "reuse_mode": self.reuse_mode,
            "key_mode": self.key_mode,
            "verify_before_serve": self.verify_before_serve,
            "verifications": self.verifications,
            "verification_failures": self.verification_failures,
            # A signature comparison over the handful of features an intent
            # needs. Charged separately because it is paid on every candidate,
            # including the ones that are then refused.
            "verification_ms_total": self.verifications * VERIFY_COST_MS,
            # Key cardinality against cache capacity: the answer to "are these
            # results just a ceiling effect?"
            "distinct_keys_resident": len(distinct_keys),
            "distinct_keys_seen": len(self._seen_keys),
            # Which ground truth judged these hits. "oracle" is the trace-derived
            # one; "exponential" is the self-consistent ablation and its
            # correctness numbers are not comparable to the oracle's.
            "validity_model": self.validity_model,
            "valid_cache_hits": self.valid_hits,
            "stale_cache_hits": self.stale_hits,
            "semantic_cache_hits": self.semantic_hits,
            "semantic_correct_hits": self.semantic_correct,
            "valid_hit_rate": self.valid_hits / attempts if attempts else 0.0,
            "stale_reuse_rate": self.stale_hits / hits if hits else 0.0,
            "hit_error_rate": self.stale_hits / hits if hits else 0.0,
            # Of the cross-key reuses the paper proposes, how many were right.
            "semantic_precision": (
                self.semantic_correct / self.semantic_hits
                if self.semantic_hits else 0.0
            ),
            "semantic_error_rate": semantic_errors / hits if hits else 0.0,
            "error_breakdown": dict(self.error_kinds),
            "rejected_stale": self.rejected_stale,
            "rejected_intent": self.rejected_intent,
            "per_intent": {
                INTENT_LABELS[idx]: dict(counts)
                for idx, counts in self.per_intent.items()
            },
        }


__all__ = ["EntryMetadata", "IntentAwareDistributedCacheManager", "REUSE_MODES"]
