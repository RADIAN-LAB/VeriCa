# VeriCa: Validity-Aware Task-Result Caching for SD-VANETs

Code, mobility configurations and per-seed results for the paper
*VeriCa: Validity-Aware Task-Result Caching for Software-Defined Vehicular
Networks*.

Task-result caching at roadside fog nodes is usually evaluated by hit rate,
but a hit is not a correct answer: a cached result describes one road cell at
one moment. This repository contains

- a **validity referee** built from vehicle mobility alone, which judges every
  served answer and names the cause of each error (wrong place, stale, wrong
  subject, wrong intent);
- **VeriCa**, which combines context-scoped cache keys (intent + place) with
  serve-time verification against the fog node's own cell state;
- the baselines (Random, FIFO, LRU, LFU, AoI refreshing window, FoggyCache
  reuse, clairvoyant bound) and the scripts that produce every table and
  figure in the paper.

## Layout

```
intent/          intent mapping, ground-truth referee, oracle, text classifier
coca_adapter/    VeriCa cache manager and approximate-reuse baseline, injected
                 into the CoCa simulator
core/            reuse engine
scripts/         pipeline: oracle -> calibration -> evaluation -> tables/figures
experiments/     per-seed result JSONs (each carries a provenance block)
tests/           pytest suite
```

## Setup

VeriCa runs on top of the CoCa SD-VANET simulator
([RADIAN-LAB/sdn-vanet-optimizer](https://github.com/RADIAN-LAB/sdn-vanet-optimizer)).
Clone both repositories side by side:

```bash
git clone https://github.com/RADIAN-LAB/sdn-vanet-optimizer.git
git clone https://github.com/RADIAN-LAB/VeriCa.git
cd VeriCa
pip install -r requirements.txt
```

If the two are not siblings, set `COCA_ROOT=/path/to/sdn-vanet-optimizer`.
SUMO is required to regenerate mobility traces.

## Reproducing the results

Seeds 1–3 are the calibration split (every fitted parameter comes from them);
seeds 4–30 are the evaluation split (n = 27).

```bash
# 1. Mobility realisations (in the CoCa repository)
python3 ../sdn-vanet-optimizer/scripts/build_mobility.py --seeds 30

# 2. Validity referee: per-cell, per-second road state from mobility alone
python3 scripts/build_oracle.py \
    --mobility-dir ../sdn-vanet-optimizer/experiments/sumo/mobility --jobs 4

# 3. Calibration on seeds 1-3: lifetimes and AoI windows
python3 scripts/fit_decay_rates.py --oracle experiments/oracle/seed_0{1,2,3}.pkl
python3 scripts/tune_aoi_window.py \
    --oracle experiments/oracle/seed_0{1,2,3}.pkl \
    --tau 0.5 0.6 0.7 0.8 0.9 --output-dir experiments/baselines/aoi

# 4. Evaluation: every arm, both regimes (as published / verified)
python3 scripts/run_evaluation.py --seeds 30 --jobs 6
python3 scripts/clairvoyant_bound.py \
    --trace experiments/multi_seed/seed_01/tasks.csv \
    --oracle experiments/oracle/base.pkl \
    --output experiments/baselines/runs/clairvoyant.json

# 5. Statistics, tables and figures
python3 scripts/paired_tests.py \
    --results experiments/evaluation/aggregated_results.json --regime verified
python3 scripts/summarize_arms.py --results experiments/evaluation
python3 scripts/make_figures.py
```

Additional studies: `run_tolerance_sweep.py` (±50% referee tolerances),
`run_circularity_ablation.py` (self-consistent vs. referee judging),
`measure_scope_decorrelation.py` (spatial validity scales),
`train_intent_classifier.py` (natural-language intent recovery).

## Tests

```bash
python3 -m pytest -q
```
