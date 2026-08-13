#!/bin/zsh
# Phase A2 driver: second topology (Bologna OSM) and density contrast, chained
# behind the tolerance sweep so the machine never runs two evaluations at once.
# Every stage is idempotent — completed runs are skipped — so re-running this
# script after any interruption continues where it stopped.
set -u

INTENT="/Users/kosar/Documents/Ph.D code/windsurf/CoCaandNlp/intent-nlp-vanet"
COCA="/Users/kosar/Documents/Ph.D code/windsurf/CoCaandNlp/sdn-vanet-optimizer"
SUMO_BIN="$HOME/Library/Python/3.9/bin"
BOLOGNA_CFG="$COCA/experiments/sumo/osm/bologna_scenario.json"
ARMS=(b1_lru b4_aoi_assumed b4_aoi_tuned b5_foggycache p_proposed)

cd "$INTENT"
log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- 0. wait for the tolerance sweep to release the CPU --------------------
while pgrep -f run_tolerance_sweep.py >/dev/null; do sleep 60; done
log "tolerance sweep finished; starting A2"

# ---- 1. Bologna (second topology), seeds 4-13, n=10 ------------------------
log "Bologna evaluation"
SCENARIO_CONFIG="$BOLOGNA_CFG" python3 scripts/run_evaluation.py \
    --seeds 13 --jobs 6 \
    --out-dir experiments/bologna \
    --oracle-dir experiments/oracle_bologna \
    --aoi-dir experiments/baselines/aoi_bologna \
    --arms "${ARMS[@]}" \
    >> experiments/bologna/run.log 2>&1
log "Bologna done ($?)"

# ---- 2. density contrast on the grid: sparse (100) and congested (400) -----
for density in 100 400; do
    MOB="$COCA/experiments/sumo/mobility_v$density"
    ORA="experiments/oracle_v$density"
    AOI="experiments/baselines/aoi_v$density"
    OUT="experiments/density_v$density"
    mkdir -p "$OUT"

    log "density $density: mobility"
    PATH="$SUMO_BIN:$PATH" python3 "$COCA/scripts/build_mobility.py" \
        --seeds 13 --jobs 4 --num-vehicles "$density" \
        --out-dir "$MOB" >> "$OUT/run.log" 2>&1

    log "density $density: traces"
    for s in $(seq 1 13); do
        sd=$(printf "seed_%02d" "$s")
        mkdir -p "$OUT/$sd"
        [ -f "$OUT/$sd/tasks.csv" ] && continue
        TASK_GEN_SEED=$s PYTHONPATH="$INTENT" python3 \
            "$COCA/experiments/sumo/sumo_to_tasks.py" \
            --fcd "$MOB/$sd/fcd_trace.xml" \
            --output "$OUT/$sd/tasks.csv" >> "$OUT/run.log" 2>&1
    done

    log "density $density: oracles"
    python3 scripts/build_oracle.py --mobility-dir "$MOB" \
        --output-dir "$ORA" --jobs 4 >> "$OUT/run.log" 2>&1

    log "density $density: B4 windows (calibration seeds only)"
    python3 scripts/tune_aoi_window.py \
        --oracle "$ORA/seed_01.pkl" "$ORA/seed_02.pkl" "$ORA/seed_03.pkl" \
        --tau 0.8 --output-dir "$AOI" >> "$OUT/run.log" 2>&1

    log "density $density: evaluation"
    python3 scripts/run_evaluation.py \
        --seeds 13 --jobs 6 \
        --out-dir "$OUT" --oracle-dir "$ORA" --aoi-dir "$AOI" \
        --arms "${ARMS[@]}" >> "$OUT/run.log" 2>&1
    log "density $density done ($?)"
done

log "A2 complete"
