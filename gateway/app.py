"""FastAPI application: OpenAI-compatible proxy with auth, metering, routing, fallback, metrics."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import metrics
from .auth import require_key
from .config import GatewayConfig
from .health import HealthMonitor
from .ratelimit import TokenBucketLimiter
from .reqlog import RequestLogger
from .router import Candidate, RouteError, Router, estimate_request_tokens, estimate_tokens
from .storage import KeyRecord, Storage, UsageRow

_HOP_HEADERS = {"host", "authorization", "content-length", "connection", "transfer-encoding",
                "keep-alive", "accept-encoding"}


@dataclass
class GatewayState:
    cfg: GatewayConfig
    storage: Storage
    limiter: TokenBucketLimiter
    client: httpx.AsyncClient
    health: HealthMonitor
    router: Router
    log: RequestLogger


def _error(status: int, message: str, kind: str, rid: str, **headers: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status,
                        headers={"X-Request-ID": rid, **headers})


def _usage_from(obj) -> tuple[int, int] | None:
    u = obj.get("usage") if isinstance(obj, dict) else None
    if not isinstance(u, dict):
        return None
    try:
        return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
    except (TypeError, ValueError):
        return None


def _completion_text(obj) -> str:
    out = []
    for ch in (obj.get("choices") or []) if isinstance(obj, dict) else []:
        if not isinstance(ch, dict):
            continue
        msg = ch.get("message") or ch.get("delta") or {}
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            out.append(msg["content"])
        elif isinstance(ch.get("text"), str):
            out.append(ch["text"])
    return "".join(out)


def create_app(cfg: GatewayConfig, *, client: httpx.AsyncClient | None = None,
               storage: Storage | None = None, run_health_loop: bool = True) -> FastAPI:
    client = client or httpx.AsyncClient(timeout=None)
    storage = storage or Storage(cfg.storage.sqlite_path)
    health = HealthMonitor(cfg.routing.backends, cfg.health, client)
    for b in cfg.routing.backends:
        metrics.BACKEND_HEALTHY.labels(backend=b.name).set(1)
    state = GatewayState(
        cfg=cfg, storage=storage, limiter=TokenBucketLimiter(), client=client, health=health,
        router=Router(cfg.routing, health), log=RequestLogger(cfg.logging.meta_log, cfg.logging.body_log),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if run_health_loop:
            await health.probe_once()
            health.start()
        try:
            yield
        finally:
            await health.stop()
            await client.aclose()
            state.log.close()
            storage.close()

    app = FastAPI(title="mini-maas gateway", lifespan=lifespan)
    app.state.gw = state

    # ---- plumbing ---------------------------------------------------------

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/readyz")
    async def readyz():
        snap = health.snapshot()
        if any(v["healthy"] for v in snap.values()):
            return {"ready": True, "backends": snap}
        return JSONResponse({"ready": False, "backends": snap}, status_code=503)

    @app.get("/metrics")
    async def prom():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/admin/backends")
    async def backends(_: KeyRecord = Depends(require_key)):
        return health.snapshot()

    @app.get("/v1/models")
    async def models():
        # Unauthenticated on purpose: OpenAI clients (and health probes written for a bare vLLM,
        # e.g. `urlopen(base + "/models")`) call this without a key. The list is not secret on an
        # internal network, and it carries no usage. Completions still require a key.
        return {"object": "list", "data": state.router.served_models()}

    # ---- LLM proxy --------------------------------------------------------

    def finalize(*, rid: str, key: KeyRecord, path: str, model_req: str | None, cand: Candidate | None,
                 status: int, stream: bool, usage: tuple[int, int] | None, est_prompt: int,
                 completion_text: str, t0: float, ttft: float | None, fallback_from: str | None,
                 req_body: dict | None, client_ip: str) -> None:
        latency_ms = (time.perf_counter() - t0) * 1000
        backend = cand.backend.name if cand else None
        model_act = cand.model if cand else None
        if usage is not None:
            p_tok, c_tok, estimated = usage[0], usage[1], False
        elif backend is None or (status >= 400 and status != 499):
            # Failed or unrouted requests are not metered: no tokens, and not flagged as an estimate.
            # 499 (client left mid-stream) is still estimated: the backend did generate those tokens.
            p_tok, c_tok, estimated = 0, 0, False
        else:
            p_tok, c_tok, estimated = est_prompt, estimate_tokens(completion_text), True
            metrics.TOKENS_ESTIMATED.labels(backend=backend).inc()
        storage.record_usage(UsageRow(
            request_id=rid, key_id=key.id, path=path, model_requested=model_req, model_actual=model_act,
            backend=backend, status=status, stream=stream, prompt_tokens=p_tok, completion_tokens=c_tok,
            usage_estimated=estimated, latency_ms=latency_ms, ttft_ms=ttft * 1000 if ttft else None,
            fallback_from=fallback_from,
        ))
        lb, lm = backend or "none", model_act or (model_req or "none")
        metrics.REQUESTS.labels(path=path, backend=lb, model=lm, status=str(status)).inc()
        metrics.LATENCY.labels(backend=lb, model=lm).observe(latency_ms / 1000)
        if ttft is not None:
            metrics.TTFT.labels(backend=lb, model=lm).observe(ttft)
        if status < 400:
            metrics.TOKENS.labels(backend=lb, model=lm, kind="prompt").inc(p_tok)
            metrics.TOKENS.labels(backend=lb, model=lm, kind="completion").inc(c_tok)
        state.log.meta({
            "request_id": rid, "key": key.name, "ip": client_ip, "path": path,
            "model_requested": model_req, "model_actual": model_act, "backend": backend,
            "status": status, "stream": stream, "prompt_tokens": p_tok, "completion_tokens": c_tok,
            "usage_estimated": estimated, "latency_ms": round(latency_ms, 1),
            "ttft_ms": round(ttft * 1000, 1) if ttft else None, "fallback_from": fallback_from,
        })
        if state.log.body_enabled and req_body is not None:
            state.log.body({"request_id": rid, "request": req_body, "completion": completion_text})

    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    async def llm_proxy(request: Request, key: KeyRecord = Depends(require_key)):
        t0 = time.perf_counter()
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        path = request.url.path
        ip = request.state.client_ip
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return _error(400, "request body must be a JSON object", "invalid_request", rid)
        model_req = body.get("model")
        if not isinstance(model_req, str) or not model_req:
            return _error(400, "'model' is required", "invalid_request", rid)
        stream = bool(body.get("stream"))
        est_prompt = estimate_request_tokens(body)
        try:
            cands = state.router.candidates(model_req, est_prompt, key.backends)
        except RouteError as e:
            finalize(rid=rid, key=key, path=path, model_req=model_req, cand=None, status=e.status,
                     stream=stream, usage=None, est_prompt=est_prompt, completion_text="", t0=t0,
                     ttft=None, fallback_from=None, req_body=None, client_ip=ip)
            return _error(e.status, e.message, "routing", rid)
        if stream:
            so = body.get("stream_options")
            body["stream_options"] = {**(so if isinstance(so, dict) else {}), "include_usage": True}
        fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
        fwd_headers["x-request-id"] = rid

        fallback_from: str | None = None
        for cand in cands:
            b = cand.backend
            upstream = dict(body, model=cand.model)
            headers = dict(fwd_headers)
            if b.api_key:
                headers["authorization"] = f"Bearer {b.api_key}"
            url = b.base_url.rstrip("/") + path.removeprefix("/v1")
            metrics.INFLIGHT.labels(backend=b.name).inc()
            resp = None
            # A connection that dies before any response (typically a pooled keep-alive socket to a
            # backend that has just restarted) gets one immediate retry on a fresh connection before
            # we fall back — otherwise the first request after every backend restart lands on the
            # fallback model for no reason. Timeouts and 5xx are not retried here.
            for attempt in (1, 2):
                req = client.build_request("POST", url, json=upstream, headers=headers,
                                           timeout=b.timeout_seconds)
                try:
                    resp = await client.send(req, stream=True)
                    break
                except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError):
                    if attempt == 1:
                        metrics.CONN_RETRIES.labels(backend=b.name).inc()
                        continue
                    resp = None
                except (httpx.HTTPError, OSError):
                    resp = None
                    break
            if resp is None:
                metrics.INFLIGHT.labels(backend=b.name).dec()
                health.report_failure(b.name)
                fallback_from = fallback_from or b.name
                continue
            if resp.status_code >= 500:
                await resp.aclose()
                metrics.INFLIGHT.labels(backend=b.name).dec()
                health.report_failure(b.name)
                fallback_from = fallback_from or b.name
                continue
            health.report_success(b.name)
            if fallback_from:
                metrics.FALLBACKS.labels(from_backend=fallback_from, to_backend=b.name).inc()
            out_headers = {"X-Request-ID": rid, "X-MMaaS-Backend": b.name, "X-MMaaS-Model": cand.model}
            ctype = resp.headers.get("content-type", "application/json")

            if stream and resp.status_code == 200:
                async def gen():
                    ttft: float | None = None
                    usage: tuple[int, int] | None = None
                    text: list[str] = []
                    buf = ""
                    status = 200
                    try:
                        async for chunk in resp.aiter_bytes():
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            yield chunk
                            buf += chunk.decode("utf-8", "replace")
                            while "\n" in buf:
                                line, buf = buf.split("\n", 1)
                                line = line.strip()
                                if not line.startswith("data:"):
                                    continue
                                payload = line[5:].strip()
                                if not payload or payload == "[DONE]":
                                    continue
                                try:
                                    obj = json.loads(payload)
                                except ValueError:
                                    continue
                                usage = _usage_from(obj) or usage
                                text.append(_completion_text(obj))
                    except BaseException:
                        status = 499  # client went away or upstream broke mid-stream
                        raise
                    finally:
                        await resp.aclose()
                        metrics.INFLIGHT.labels(backend=b.name).dec()
                        finalize(rid=rid, key=key, path=path, model_req=model_req, cand=cand,
                                 status=status, stream=True, usage=usage, est_prompt=est_prompt,
                                 completion_text="".join(text), t0=t0, ttft=ttft,
                                 fallback_from=fallback_from, req_body=body, client_ip=ip)
                return StreamingResponse(gen(), status_code=200, media_type=ctype, headers=out_headers)

            raw = await resp.aread()
            await resp.aclose()
            metrics.INFLIGHT.labels(backend=b.name).dec()
            usage, text = None, ""
            if resp.status_code == 200:
                try:
                    obj = json.loads(raw)
                    usage, text = _usage_from(obj), _completion_text(obj)
                except ValueError:
                    pass
            finalize(rid=rid, key=key, path=path, model_req=model_req, cand=cand, status=resp.status_code,
                     stream=False, usage=usage, est_prompt=est_prompt, completion_text=text, t0=t0,
                     ttft=None, fallback_from=fallback_from, req_body=body, client_ip=ip)
            return Response(content=raw, status_code=resp.status_code, media_type=ctype, headers=out_headers)

        finalize(rid=rid, key=key, path=path, model_req=model_req, cand=None, status=502, stream=stream,
                 usage=None, est_prompt=est_prompt, completion_text="", t0=t0, ttft=None,
                 fallback_from=fallback_from, req_body=None, client_ip=ip)
        return _error(502, f"all backends serving '{model_req}' failed", "upstream", rid)

    # ---- tool proxy (non-LLM services behind the same keys/limits) ----------

    @app.api_route("/tools/{tool}/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def tool_proxy(tool: str, rest: str, request: Request, key: KeyRecord = Depends(require_key)):
        t0 = time.perf_counter()
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        target = next((t for t in cfg.tools if t.name == tool), None)
        if target is None:
            raise HTTPException(404, detail={"error": {"message": f"unknown tool '{tool}'", "type": "routing"}})
        url = target.base_url.rstrip("/") + "/" + rest
        if request.url.query:
            url += "?" + request.url.query
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
        headers["x-request-id"] = rid
        content = await request.body()
        try:
            resp = await client.request(request.method, url, content=content, headers=headers, timeout=60)
            status, raw = resp.status_code, resp.content
            ctype = resp.headers.get("content-type", "application/octet-stream")
        except (httpx.HTTPError, OSError) as e:
            status, ctype = 502, "application/json"
            raw = json.dumps({"error": {"message": str(e), "type": "upstream"}}).encode()
        latency_ms = (time.perf_counter() - t0) * 1000
        storage.record_usage(UsageRow(
            request_id=rid, key_id=key.id, path=f"/tools/{tool}", model_requested=None, model_actual=None,
            backend=tool, status=status, stream=False, prompt_tokens=0, completion_tokens=0,
            usage_estimated=False, latency_ms=latency_ms, ttft_ms=None,
        ))
        metrics.REQUESTS.labels(path=f"/tools/{tool}", backend=tool, model="none", status=str(status)).inc()
        metrics.LATENCY.labels(backend=tool, model="none").observe(latency_ms / 1000)
        state.log.meta({"request_id": rid, "key": key.name, "ip": request.state.client_ip,
                        "path": f"/tools/{tool}/{rest}", "backend": tool, "status": status,
                        "latency_ms": round(latency_ms, 1)})
        return Response(content=raw, status_code=status, media_type=ctype,
                        headers={"X-Request-ID": rid, "X-MMaaS-Backend": tool})

    return app
