#!/bin/zsh
# NLP robustness at n=27: the proposed arm with intent recovered from text,
# verified regime, all evaluation seeds. Chained behind the A2 driver so the
# machine never runs two evaluations at once. Idempotent.
set -u
INTENT="/Users/kosar/Documents/Ph.D code/windsurf/CoCaandNlp/intent-nlp-vanet"
cd "$INTENT"
while pgrep -f overnight_a2.sh >/dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] A2 finished; NLP n=27"
mkdir -p experiments/nlp_n27
for s in $(seq 4 30); do
    sd=$(printf "seed_%02d" "$s")
    out="experiments/nlp_n27/${sd}.json"
    [ -f "$out" ] && continue
    PYTHONPATH="$INTENT" TASK_GEN_SEED=$s python3 scripts/run_intent_experiment.py \
        --trace "experiments/evaluation/$sd/tasks.csv" \
        --output "$out" \
        --oracle "experiments/oracle/${sd}.pkl" \
        --seed "$s" --evict-policy lru --no-fl \
        --reuse-mode coca --key-mode context_adaptive \
        --verify-before-serve \
        --classify-from-text experiments/models/intent_text.json \
        >> experiments/nlp_n27/run.log 2>&1 &
    while [ "$(jobs -r | wc -l)" -ge 6 ]; do sleep 10; done
done
wait
echo "[$(date +%H:%M:%S)] NLP n=27 complete: $(ls experiments/nlp_n27/seed_*.json | wc -l)/27"
