"""Prometheus metrics (process-level registry)."""

from prometheus_client import Counter, Gauge, Histogram

_LAT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)

REQUESTS = Counter("mmaas_requests_total", "Proxied requests",
                   ["path", "backend", "model", "status"])
LATENCY = Histogram("mmaas_request_latency_seconds", "End-to-end request latency",
                    ["backend", "model"], buckets=_LAT_BUCKETS)
TTFT = Histogram("mmaas_ttft_seconds", "Time to first streamed byte",
                 ["backend", "model"], buckets=_LAT_BUCKETS)
TOKENS = Counter("mmaas_tokens_total", "Tokens metered", ["backend", "model", "kind"])
TOKENS_ESTIMATED = Counter("mmaas_tokens_estimated_rows_total",
                           "Requests whose token counts were estimated (backend sent no usage)",
                           ["backend"])
RATELIMITED = Counter("mmaas_ratelimited_total", "Requests rejected by rate limit", ["key"])
AUTH_FAILED = Counter("mmaas_auth_failed_total", "Requests rejected by auth", ["reason"])
FALLBACKS = Counter("mmaas_fallbacks_total", "Requests re-routed after a backend failure",
                    ["from_backend", "to_backend"])
BACKEND_HEALTHY = Gauge("mmaas_backend_healthy", "1 if backend passes health checks", ["backend"])
INFLIGHT = Gauge("mmaas_inflight_requests", "Requests currently being proxied", ["backend"])
