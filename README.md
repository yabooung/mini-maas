# mini-maas

A small, measured **LLM-as-a-Service gateway**: one OpenAI-compatible endpoint in front of several
model backends, with API keys, IP allowlists, token metering, per-key rate limits, model routing with
health-based fallback, Prometheus metrics, and Kubernetes manifests that do zero-loss rolling updates.
Plus the inference benchmarks (quantization, speculative decoding, batching) behind the routing choices.

Built as a public, reproducible reference for the serving layer I have run privately in front of
self-hosted vLLM: the health/fallback design comes from a production incident where a nightly
rebuild starved the inference server of GPU memory and left it down with the status file still
saying "running".

```
client ──Bearer key──▶ gateway ──▶ llama-small   (Qwen2.5-0.5B, priority 10)
                         │   └───▶ llama-medium  (Qwen2.5-1.5B, priority 20, fallback)
                         ├──▶ /tools/<name>/…   (non-LLM services behind the same keys)
                         └──▶ /metrics          (Prometheus) · SQLite usage ledger · JSONL request log
```

## When to use this (and when not to)

This is a small reference implementation, not a replacement for a full gateway product. If you run many providers or need an admin UI, teams, budgets and a Postgres-backed ledger, use [LiteLLM Proxy](https://github.com/BerriAI/litellm) or a managed gateway. Use this when you self-host one to three OpenAI-compatible servers and want keys, metering, fallback and metrics in ~1,500 lines you can read in an afternoon, with no database server.

Status (2026-10): in production in front of one self-hosted vLLM + one MLX fallback; unit tests and `kustomize build` pass in CI. The `docker compose` quick start, the kind rolling-update measurement and the benchmark scripts have **not yet been run end to end** — the measurement tables below are empty until they are.

## What it does

| Concern | Implementation |
|---|---|
| Auth | `Authorization: Bearer mm-…`; keys stored as SHA-256; per-key IP/CIDR allowlist; revocation |
| Rate limit | token bucket per key (capacity = N/min), `429` + `Retry-After` |
| Metering | prompt/completion tokens from the backend's `usage` (streaming via `stream_options.include_usage`); if a backend sends none, counts are **estimated and flagged** (`usage_estimated=1`) so billing drift is visible, not silent |
| Routing | requested model/alias → candidate backends ordered by priority; backends whose context is too small for the (estimated) input are skipped; `413` above a hard limit |
| Fallback | connect error or `5xx` before the first byte → next candidate; backend marked unhealthy after N failures and probed back to health |
| Streaming | SSE passthrough with TTFT measured at the gateway; usage captured from the final chunk |
| Logging | meta log (who/what/when/how long, no prompt text) always; body log opt-in and separate |
| Metrics | requests, latency & TTFT histograms, tokens, estimated-usage rows, fallbacks, rate-limited, auth failures, backend health, in-flight |
| Tools | `/tools/{name}/…` reverse proxy for non-LLM services (e.g. a search index) under the same keys |

## Run it

```bash
# local: gateway + two llama.cpp CPU backends + Prometheus + Grafana
docker compose up -d
docker compose exec gateway mmaas --config /app/gateway.yaml keys create --name dev --rate 120
curl -s localhost:8000/v1/chat/completions -H "Authorization: Bearer mm-…" \
  -d '{"model":"fast","messages":[{"role":"user","content":"hi"}]}' -i | grep -i x-mmaas
# X-MMaaS-Backend: llama-small   X-MMaaS-Model: qwen2.5-0.5b
docker compose exec gateway mmaas --config /app/gateway.yaml usage --since 1h --by backend
```

```bash
# kubernetes (kind): rolling updates with maxUnavailable=0, readiness on /readyz and llama's /health, HPA on the gateway
kind create cluster --name mmaas --config deploy/k8s/kind-config.yaml
docker build -t mini-maas/gateway:dev . && kind load docker-image mini-maas/gateway:dev --name mmaas
kubectl apply -k deploy/k8s
kubectl -n mmaas rollout status deploy/llama-medium --timeout=10m   # first start downloads the GGUF
```

Tests (no models needed — a fake upstream is driven in-process):

```bash
pip install -e .[dev] && pytest -q      # 19 tests: auth, IP, rate limit, routing, fallback, recovery, metering, streaming, tools, metrics
```

## Measurements

> Not yet run — this section is filled from `results/` by the scripts below. Numbers are machine-specific; the machine is recorded with each run.

### Rolling update request loss

`bench/rolling_update.sh` drives constant-rate load through the gateway while a deployment rolls, then counts non-200 responses and the longest gap between successes.

| rollout | rps | duration | requests | failed | longest success gap |
|---|---:|---:|---:|---:|---:|
| gateway `rollout restart` (2 replicas) | | | | | |
| llama-small model file Q4_K_M → Q8_0 (1 replica, maxSurge=1) | | | | | |

### Fallback switch time

Kill `llama-small` under load; measure the gap between its last success and the first success on `llama-medium`.

### Inference: quantization, speculative decoding, batching

`bench/run_llama_matrix.sh` serves Qwen2.5-1.5B-Instruct in several configurations and runs `bench/llama_bench.py` against each.

| config | conc | agg tok/s | per-req decode tok/s p50 | TTFT p50 (s) | latency p50 (s) | latency p95 (s) | QA acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| (pending) | | | | | | | |

## CI/CD

```
push / PR ─▶ ci.yml (ubuntu-latest)                     main ─▶ deploy.yml (self-hosted runner on the VM)
  ruff · pytest · kustomize build                           rsync → /opt/mini-maas  (gateway.yaml, data/ untouched)
  docker build → container smoke (/healthz, /v1/models)     deploy/vm/deploy.sh  MODE=bluegreen
  main: push ghcr.io/<repo>:<sha>, :latest                    build → start idle color (8011|8012) → /readyz + /v1/models
                                                               → nginx upstream swap + reload (no dropped connections)
                                                               → drain 20s → stop old → keep last 3 images
                                                             deploy.sh rollback [<sha>]
```

* `ci.yml` is in this repo. The deploy workflow and the per-host configs (`deploy/vm`, `deploy/dgx`) are site-specific (internal addresses, runner labels) and are kept out of the public tree; the diagram above is what they do.
* CD only runs after CI succeeds on `main` (`workflow_run`); `workflow_dispatch` allows `mode=recreate` (plain `docker run` on :8010, seconds of gap) where nginx is not set up.
* Secrets never enter GitHub: the config and the key/usage ledger live only on the VM; the runner is self-hosted so no SSH key is stored as a secret (same pattern as the service this gateway fronts).
* Readiness gates the switch: a new color that cannot reach any backend never takes traffic, and the previous image tag is recorded for `rollback`.

## Design notes

* **Estimate, then meter.** Token counts for *routing* are a cheap estimate (CJK ≈ 1 token/char, else ≈ 4 chars/token) because routing happens before the backend tokenizes. Token counts for the *ledger* come from the backend. The two are never mixed: estimated ledger rows are flagged and alerted on (`UsageEstimatedRatioHigh`).
* **Fallback only before the first byte.** Once a streamed response has started, the client has state; the gateway finishes or fails that stream rather than silently switching models mid-answer.
* **Health is reported from the data path, not just the probe.** A `5xx`/connect failure marks the backend immediately; the probe only restores it. This keeps the fallback window at one request, not one probe interval.
* **Two log tiers.** Prompt bodies are PII-bearing; they go to a separate, opt-in file with its own retention. The always-on log has everything needed for billing and debugging except the text.
* **One replica = one bucket set.** The rate limiter is in-memory; for N replicas it either over-admits N× or moves to Redis. The ledger is SQLite for the same reason. Both are documented limits, not surprises.

## Layout

```
gateway/     app.py (proxy) · auth.py · router.py · health.py · ratelimit.py · storage.py · metrics.py · reqlog.py · cli.py
tests/       in-process fake upstream (two virtual hosts) + 19 tests
deploy/      compose/ · k8s/ (kustomize, kind, grafana dashboard + provisioning) · prometheus-alerts.yml
bench/       loadgen.py · analyze_loadgen.py · rolling_update.sh · llama_bench.py · run_llama_matrix.sh · qa_small.jsonl · report.py
```

MIT.
