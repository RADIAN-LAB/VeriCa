# intent-nlp-vanet

NLP-based intent classification for context-aware task-result caching in SDN-VANETs.

## Goal

Extend CoCa with an intent layer:
- classify each task's **intent** (e.g. `traffic_analysis`, `hazard_detection`, ...)
- use intent in **reuse decisions**, **FL features**, and **RL state**

## Structure

```
intent-nlp-vanet/
├── intent/
│   ├── __init__.py
│   ├── classifier.py     # NLP intent classifier
│   ├── similarity.py     # Intent similarity matrix + reuse score
│   └── freshness.py      # AoI-based per-intent freshness scoring
├── core/
│   └── reuse_engine.py   # Unified ReuseEngine + ReuseDecision
├── tests/
│   ├── test_classifier.py
│   ├── test_freshness.py
│   └── test_reuse_engine.py
├── requirements.txt
└── README.md
```

---

## Prior Work Survey & Research Gap

### AoI / Freshness in Edge Caching

| Paper | Venue | What they do | Gap vs this work |
|-------|-------|-------------|-----------------|
| Feng et al. | Sensors 2023 | AoI-aware *service* caching at MEC (DNN model parameters) | Caches the model, not the task result; uniform TTL, no per-type decay |
| HotNets 2024 (SIGCOMM) | arXiv 2412.20221 | Argues binary TTL is insufficient for real-time apps; proposes event-driven invalidation | General caching theory, not VANET; no intent layer; no per-type freshness |
| AoI multi-agent scheduling | ScienceDirect 2025 | Minimise AoI in task scheduling via multi-agent RL | Scheduling only, no task-result caching, no intent grouping |

**Key finding:** AoI is used for *service/model* freshness in edge computing, never for *task-result* validity in VANET caching.

### Intent / Semantic Caching

| Paper | Venue | What they do | Gap vs this work |
|-------|-------|-------------|-----------------|
| ContextCache | arXiv 2506.22791 | Semantic cache for LLM multi-turn queries using embeddings | LLM domain only; no VANET; no freshness modeling |
| SlsReuse | arXiv 2511.17262 | Intent-aware serverless function reuse via LLM encoding | Serverless, not vehicular; no per-type decay |
| Dependency-aware VANET offloading | SemanticScholar 2023 | Task dependency-aware offloading + service caching | Service caching, not task-result; no intent; no AoI |

**Key finding:** Semantic/intent caching exists only in LLM and serverless domains — never applied to VANET task-result reuse.

### Task-Result Caching in VANET (from CoCa's 15-paper comparison table)

None of the 15 surveyed papers (Chen 2019 → Ahmed 2025) address:
- Freshness-aware reuse (all use binary TTL or no TTL)
- Intent/semantic grouping for cross-task-type reuse
- Per-task-type staleness tolerance

### Research Gap (what this work fills)

The specific combination of **per-intent AoI freshness decay** + **semantic intent grouping** for **VANET task-result reuse** is not covered by any prior work. The building blocks exist in adjacent areas but have never been combined:

```
AoI for service caching  ──┐
                            ├──► This work: AoI + intent for task-result reuse in VANET
Semantic caching (LLMs)  ──┘
```

CoCa's binary TTL (`default_ttl: 3600`, uniform for all task types) is the direct baseline this work improves upon, supported by the HotNets 2024 argument that TTLs are insufficient for real-time applications like autonomous driving and hazard detection.

---

## Key Formula

```
ReuseScore = w1 × IntentSim(intent_i, intent_j)
           + w2 × ContextCompat(rsu, time_window)
           + w3 × exp(-λ[intent_j] × AoI(stored_at))
```

Per-intent decay rates (λ):

| intent | λ (1/sec) | half-life | Justification |
|--------|-----------|-----------|---------------|
| traffic_analysis | 0.05 | ~14 s | Traffic state changes in 10–30s |
| hazard_detection | 0.20 | ~3.5 s | Safety-critical, rapidly invalidated |
| route_optimization | 0.02 | ~35 s | Routes stay valid for longer |
| sensor_processing | 0.10 | ~7 s | Sensor readings updated rapidly |
| environment_perception | 0.01 | ~70 s | Environment changes slowly |

---

## CoCa Integration (CoCa Remains Read-Only)

The project integrates with the sibling `sdn-vanet-optimizer` repository at
runtime. It does not patch or copy CoCa source code. The wrapper injects
`IntentAwareDistributedCacheManager` into the cache-manager singleton already
used by CoCa's scheduler.

The adapter adds:

- simulation-time AoI freshness (trace timestamps, not wall-clock time)
- semantic lookup after an exact-key miss
- deterministic, seed-controlled content-validity sampling
- valid-hit, stale-reuse, semantic-hit, and per-intent metrics
- four comparable reuse modes: `coca`, `intent`, `freshness_only`,
  and `intent_only`

Set `COCA_ROOT` if the repositories are not siblings:

```bash
export COCA_ROOT=/path/to/sdn-vanet-optimizer
```

## Run One Experiment

```bash
python3 scripts/run_intent_experiment.py \
  --trace ../sdn-vanet-optimizer/experiments/sumo/input/sumo_tasks.csv \
  --output experiments/smoke/intent.json \
  --reuse-mode intent \
  --load-rl ../sdn-vanet-optimizer/experiments/models/final_rl_trained.pt \
  --seed 1
```

Use `--max-ts 20` for a quick integration check.

## Multi-Seed Evaluation

Run a five-seed pilot before the full evaluation:

```bash
python3 scripts/run_multi_seed_intent.py --seeds 5
python3 scripts/generate_intent_figures.py
```

Then produce the paper results:

```bash
python3 scripts/run_multi_seed_intent.py --seeds 30
python3 scripts/run_threshold_sensitivity.py --seeds 5
python3 scripts/generate_intent_figures.py
```

Outputs are written only under this repository:

```text
experiments/
├── multi_seed/
│   ├── seed_01/ ... seed_30/
│   └── aggregated_results.json
├── sensitivity/
└── figures/
    ├── fig_valid_hit_rate.{png,pdf}
    ├── fig_stale_reuse_rate.{png,pdf}
    ├── fig_stale_reuse_by_intent.{png,pdf}
    └── statistical_tests.json
```

The multi-seed comparison includes the six original CoCa methods, the full
Intent-NLP method, and freshness-only and intent-only ablations.

## Current Full-Trace Pilot (Seed 1)

The corrected end-to-end pilot processed 16,341 tasks. These values verify the
pipeline, but they are **not yet paper-level evidence** because `n=1`.

| Method | Valid-hit rate | Stale reuse | Success rate |
|---|---:|---:|---:|
| No Cache | 0.00% | 0.00% | 32.75% |
| Random | 51.10% | 34.48% | 86.40% |
| LRU | 52.51% | 36.89% | 90.01% |
| LFU | 52.51% | 36.89% | 90.01% |
| FL-Only | 75.03% | 21.84% | 96.80% |
| CoCa (FL+RL) | 75.62% | 21.39% | 96.93% |
| Intent-NLP (full) | **85.04%** | **14.93%** | **99.98%** |
| Freshness-only | 77.14% | 18.98% | 96.54% |
| Intent-only | 79.33% | 20.64% | 99.98% |

Against CoCa in this pilot, full Intent-NLP increased valid-hit rate by 9.42
percentage points and reduced stale reuse by 6.46 percentage points. Run 30
seeds before reporting significance or confidence intervals.

## Metric Interpretation

`valid_hit_rate` counts reuse events that occur before a sampled content
invalidation time. `stale_reuse_rate` is the fraction of accepted cache hits
that occur after invalidation. The oracle lifetime is sampled from the same
per-intent exponential process used by the AoI model, with a fixed random seed
for reproducibility. This is a simulation validity model, not measured
real-world ground truth, and should be described as such in the paper.

## Tests

```bash
python3 -m pytest -q
```
