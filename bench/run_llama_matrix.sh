#!/usr/bin/env bash
# Launch llama-server in several configurations one after another and benchmark each.
# Needs a native `llama-server` on PATH (brew install llama.cpp / build from source) and `huggingface-cli`.
# Runs on Apple Silicon (Metal) or Linux (CPU/CUDA) — results are machine-specific; record the machine in --label.
#
#   bench/run_llama_matrix.sh results/bench_$(hostname).jsonl
set -euo pipefail
OUT=${1:-results/bench.jsonl}; mkdir -p "$(dirname "$OUT")" models
PORT=${PORT:-8089}; URL="http://127.0.0.1:$PORT/v1"
REPO_T=${REPO_T:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}; REPO_D=${REPO_D:-Qwen/Qwen2.5-0.5B-Instruct-GGUF}
QA=${QA:-bench/qa_small.jsonl}; CONC=${CONC:-1,2,4,8}; NP=${NP:-8}

dl() { huggingface-cli download "$1" "$2" --local-dir models >/dev/null; echo "models/$2"; }
T_Q4=$(dl "$REPO_T" qwen2.5-1.5b-instruct-q4_k_m.gguf)
T_Q8=$(dl "$REPO_T" qwen2.5-1.5b-instruct-q8_0.gguf)
T_F16=$(dl "$REPO_T" qwen2.5-1.5b-instruct-fp16.gguf)
D_Q4=$(dl "$REPO_D" qwen2.5-0.5b-instruct-q4_k_m.gguf)

serve() {  # serve <label> <args...>
  local label=$1; shift
  echo "=== $label"
  llama-server --host 127.0.0.1 --port "$PORT" -np "$NP" -c $((4096 * NP)) --metrics "$@" >"results/server_${label}.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 120); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
  python bench/llama_bench.py --url "$URL" --model bench --label "$label" --concurrency "$CONC" --qa "$QA" --out "$OUT"
  kill "$pid"; wait "$pid" 2>/dev/null || true
}

# 1) quantization: same model, three precisions
serve q4_k_m  -m "$T_Q4"  --alias bench
serve q8_0    -m "$T_Q8"  --alias bench
serve f16     -m "$T_F16" --alias bench
# 2) speculative decoding: 0.5B draft proposing for the 1.5B target (same tokenizer family)
serve spec_q8_draft0.5b -m "$T_Q8" -md "$D_Q4" --draft-max 8 --draft-min 1 --alias bench
# 3) batching: slots=1 vs slots=NP on the same quant (concurrency sweep shows the curve)
NP=1 serve q8_0_np1 -m "$T_Q8" --alias bench

python bench/report.py "$OUT"
