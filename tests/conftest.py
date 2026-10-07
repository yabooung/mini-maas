"""Test fixtures: a fake OpenAI-compatible upstream (two virtual hosts) and the gateway wired to it
in-process via httpx.ASGITransport — no sockets, no real models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.app import create_app
from gateway.config import GatewayConfig
from gateway.storage import Storage


@dataclass
class UpstreamState:
    down: set[str] = field(default_factory=set)      # virtual hosts returning 503
    send_usage: bool = True                          # include `usage` in responses
    calls: list[dict] = field(default_factory=list)  # (host, body) log


def make_upstream(state: UpstreamState) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def models(request: Request):
        host = request.headers["host"]
        if host in state.down:
            return JSONResponse({"error": "down"}, status_code=503)
        return {"object": "list", "data": []}

    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    async def chat(request: Request):
        host = request.headers["host"]
        body = await request.json()
        state.calls.append({"host": host, "body": body})
        if host in state.down:
            return JSONResponse({"error": "down"}, status_code=503)
        model = body["model"]
        if body.get("stream"):
            def chunks():
                for piece in ("Hel", "lo ", "world"):
                    yield "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "model": model,
                                                 "choices": [{"index": 0, "delta": {"content": piece}}]}) + "\n\n"
                if state.send_usage:
                    yield "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "model": model,
                                                 "choices": [],
                                                 "usage": {"prompt_tokens": 7, "completion_tokens": 3}}) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(chunks(), media_type="text/event-stream")
        out = {"id": "y", "object": "chat.completion", "model": model,
               "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello world"}}]}
        if state.send_usage:
            out["usage"] = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        return out

    @app.api_route("/echo/{rest:path}", methods=["GET", "POST"])
    async def echo(rest: str, request: Request):
        return {"host": request.headers["host"], "method": request.method, "path": rest,
                "query": request.url.query, "body": (await request.body()).decode()}

    return app


CONFIG = {
    "storage": {"sqlite_path": ":memory:"},
    "logging": {"meta_log": None, "body_log": None},
    "health": {"interval_seconds": 1000, "timeout_seconds": 1, "unhealthy_after": 1},
    "ratelimit": {"default_per_minute": 1000},
    "routing": {
        "backends": [
            {"name": "llama-small", "base_url": "http://llama-small/v1", "models": ["qwen2.5-0.5b"],
             "aliases": {"fast": "qwen2.5-0.5b"}, "max_input_tokens": 64, "priority": 10},
            {"name": "llama-medium", "base_url": "http://llama-medium/v1", "models": ["qwen2.5-1.5b"],
             "aliases": {"fast": "qwen2.5-1.5b", "default": "qwen2.5-1.5b"}, "max_input_tokens": 8192,
             "priority": 20},
        ],
        "max_input_tokens_hard": 16384,
    },
    "tools": [{"name": "jlawcite", "base_url": "http://jlawcite/echo"}],
}


@pytest.fixture
def upstream_state() -> UpstreamState:
    return UpstreamState()


@pytest.fixture
async def gw(upstream_state):
    cfg = GatewayConfig.model_validate(CONFIG)
    upstream_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=make_upstream(upstream_state)))
    storage = Storage(":memory:")
    app = create_app(cfg, client=upstream_client, storage=storage, run_health_loop=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        yield c, app.state.gw
    await upstream_client.aclose()
    storage.close()


@pytest.fixture
def key(gw):
    _, state = gw
    plain, rec = state.storage.create_key("test")
    return plain, rec


def auth(plain: str) -> dict:
    return {"Authorization": f"Bearer {plain}"}
