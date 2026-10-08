# mini-maas

[![ci](https://github.com/yabooung/mini-maas/actions/workflows/ci.yml/badge.svg)](https://github.com/yabooung/mini-maas/actions/workflows/ci.yml) · image: `ghcr.io/yabooung/mini-maas:latest`

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

Status (2026-10): in production in front of one self-hosted vLLM + one MLX fallback; unit tests and `kustomize build` pass in CI. The `docker compose` quick start and the kind deployment were run end to end from a fresh clone on 2026-10-09 (Apple Silicon, Docker Desktop 27.5, kind 0.33); that run found and fixed four problems — fixed host ports colliding with local services, API keys living in a per-pod SQLite so half of the requests to a 2-replica Service got `401`, an invalid YAML line from `keys gen`, and failed requests being flagged as estimated usage. The inference benchmark scripts (`bench/run_llama_matrix.sh`) have **not** been run yet.

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
docker compose up -d                 # host ports on 127.0.0.1; override e.g. GRAFANA_PORT=3001 GATEWAY_PORT=8100
docker compose exec gateway mmaas --config /app/gateway.yaml keys create --name dev --rate 120
curl -s localhost:8000/v1/chat/completions -H "Authorization: Bearer mm-…" \
  -d '{"model":"fast","messages":[{"role":"user","content":"hi"}]}' -i | grep -i x-mmaas
# X-MMaaS-Backend: llama-small   X-MMaaS-Model: qwen2.5-0.5b
docker compose exec gateway mmaas --config /app/gateway.yaml usage --since 1h --by backend
```

```bash
# kubernetes (kind): rolling updates with maxUnavailable=0, readiness on /readyz and llama's /health, HPA on the gateway
kind create cluster --name mmaas --config deploy/k8s/kind-config.yaml   # host ports 18000/19090/13000
docker build -t mini-maas/gateway:dev . && kind load docker-image mini-maas/gateway:dev --name mmaas
kubectl create namespace mmaas
# keys are shared by all gateway replicas via a Secret (only SHA-256 hashes are stored)
docker run --rm --entrypoint mmaas mini-maas/gateway:dev keys gen --name dev > /tmp/dev.key   # line 1 = key, line 2 = entry
tail -1 /tmp/dev.key > /tmp/keys.yaml && kubectl -n mmaas create secret generic gateway-keys --from-file=keys.yaml=/tmp/keys.yaml
kubectl apply -k deploy/k8s
kubectl -n mmaas rollout status deploy/llama-medium --timeout=10m   # first start downloads the GGUF
curl -s localhost:18000/v1/chat/completions -H "Authorization: Bearer $(head -1 /tmp/dev.key | cut -d= -f2)"   -d '{"model":"fast","messages":[{"role":"user","content":"hi"}]}'
```

Tests (no models needed — a fake upstream is driven in-process):

```bash
pip install -e .[dev] && pytest -q      # 25 tests: auth, IP, rate limit, routing, fallback, recovery, key pinning, static keys, metering, streaming, tools, metrics
```

## Measurements

> Numbers are machine-specific. Rolling-update and fallback rows: kind / docker compose on an Apple Silicon Mac mini (Docker VM with 14 CPUs, 8 GB), CPU llama.cpp backends, one run each, 2026-10-09.

### Fallback capacity (measured, 2026-10-07)

The fallback backend in production is a Mac mini running `mlx_lm.server` with Qwen3.8-27B 4-bit. Before routing long prompts to it, its real capacity was measured with Korean statute text:

| input | real prompt tokens | cold time (prefill-bound) | result |
|---:|---:|---:|---|
| 19,000 chars | 9,596 | 87 s | 200, correct summary |
| 42,000 chars | 21,180 | 264 s | 200, correct summary |

Korean legal text came out at ~2 chars/token, while the router's pre-estimate counts CJK as ~1 token/char — so the configured 32,768-token limit admits at most ~16k real tokens to the fallback, inside the measured-OK range. Larger requests while the primary is down get `503` instead of being handed to a backend that may not finish them; the client (a report writer) shrinks its input when it sees it is on the fallback tier.

### Rolling update request loss

`bench/rolling_update.sh` drives constant-rate load through the gateway while a deployment rolls, then counts non-200 responses and the longest gap between successes.

| rollout | rps | duration | requests | failed | longest success gap |
|---|---:|---:|---:|---:|---:|
| gateway `rollout restart` (2 replicas) | 5 | 90 s | 450 | 0 | 0.20 s |
| llama-small model file Q4_K_M → Q8_0, backend grace 60 s | 5 | 300 s | 1,500 | 0 | 0.21 s |
| llama-small model file Q4_K_M → Q8_0, backend grace 15 s | 5 | 180 s | 900 | 0 | 0.20 s |

With a 60 s termination grace, 2 of 1,500 requests took ~55 s: they went out on keep-alive connections still pinned to the terminating pod (kube-proxy only re-balances new connections), hung until the pod was killed, and were then retried on a fresh connection by the gateway (`mmaas_connection_retries_total` = 2, no fallback, no failure). With 15 s grace the next rollout had no stalls (max latency 1.05 s). One run each — a stall can still happen, but it is now bounded by ~10 s.

### Fallback switch time

`docker compose stop llama-small`, then one request for the `fast` alias: served by `llama-medium` in 0.17 s end to end (`fallback_from=llama-small` in the ledger). Right after the backend is started again, the probe can still show it healthy for up to one probe interval while the model loads; requests in that window are tried on it, get `503`, and fall back — expected, not an error.

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
