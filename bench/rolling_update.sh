#!/usr/bin/env bash
# Measure request loss during a rolling update.
#   bench/rolling_update.sh <deployment> <container> <new-image-or-arg-patch> [rps] [duration]
# Examples:
#   # gateway code rollout
#   bench/rolling_update.sh gateway gateway restart
#   # backend "model version" rollout: swap the GGUF file (Q4 -> Q8)
#   bench/rolling_update.sh llama-small llama 'qwen2.5-0.5b-instruct-q8_0.gguf'
# Requires: kubectl context on the kind cluster, MMAAS_KEY env (an API key), gateway reachable at $GW.
set -euo pipefail
DEPLOY=${1:?deployment}; CONTAINER=${2:?container}; CHANGE=${3:?restart|<gguf-file>}
RPS=${4:-5}; DURATION=${5:-120}
GW=${GW:-http://localhost:8000}; NS=${NS:-mmaas}; MODEL=${MODEL:-fast}
mkdir -p results
OUT=results/rolling_${DEPLOY}_$(date +%Y%m%d_%H%M%S).jsonl

echo "[1/4] baseline check"
curl -fsS "$GW/readyz" >/dev/null

echo "[2/4] load: ${RPS} rps for ${DURATION}s -> $OUT"
python bench/loadgen.py --url "$GW" --key "$MMAAS_KEY" --model "$MODEL" --rps "$RPS" --duration "$DURATION" --out "$OUT" &
LOAD=$!
sleep 15   # steady-state before the change

echo "[3/4] rollout: $DEPLOY ($CHANGE)"
T_ROLL=$(date +%s)
if [[ "$CHANGE" == "restart" ]]; then
  kubectl -n "$NS" rollout restart "deployment/$DEPLOY"
else
  # replace the --hf-file argument value in the container args
  kubectl -n "$NS" get deployment "$DEPLOY" -o json \
    | python -c "
import json,sys
d=json.load(sys.stdin)
for c in d['spec']['template']['spec']['containers']:
    if c['name']=='$CONTAINER':
        a=c['args']; i=a.index('--hf-file'); a[i+1]='$CHANGE'
json.dump(d,sys.stdout)" \
    | kubectl -n "$NS" apply -f -
fi
kubectl -n "$NS" rollout status "deployment/$DEPLOY" --timeout=10m
T_DONE=$(date +%s)
echo "rollout took $((T_DONE - T_ROLL))s"

wait $LOAD
echo "[4/4] analysis"
python bench/analyze_loadgen.py "$OUT" | tee "${OUT%.jsonl}.txt"
echo "rollout window: $T_ROLL .. $T_DONE (unix)" | tee -a "${OUT%.jsonl}.txt"
