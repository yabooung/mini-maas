import json

import pytest

from gateway.router import estimate_tokens
from tests.conftest import auth

CHAT = {"model": "fast", "messages": [{"role": "user", "content": "hi"}]}


# ---- auth --------------------------------------------------------------------

async def test_missing_key_401(gw):
    c, _ = gw
    r = await c.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 401


async def test_invalid_and_revoked_key_401(gw, key):
    c, state = gw
    plain, rec = key
    assert (await c.post("/v1/chat/completions", json=CHAT, headers=auth("mm-nope"))).status_code == 401
    assert state.storage.revoke_key(rec.id)
    assert (await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))).status_code == 401


async def test_ip_allowlist_403(gw):
    c, state = gw
    plain, _ = state.storage.create_key("locked", ip_allow=["10.0.0.0/8"])
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    assert r.status_code == 403
    plain2, _ = state.storage.create_key("open", ip_allow=["127.0.0.1"])
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain2))
    assert r.status_code == 200


# ---- rate limit --------------------------------------------------------------

async def test_rate_limit_429_with_retry_after(gw):
    c, state = gw
    plain, _ = state.storage.create_key("slow", rate_per_minute=2)
    codes = [(await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


# ---- routing + fallback ------------------------------------------------------

async def test_alias_routes_to_highest_priority_backend(gw, key, upstream_state):
    c, _ = gw
    plain, _ = key
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    assert r.status_code == 200
    assert r.headers["X-MMaaS-Backend"] == "llama-small"
    assert r.headers["X-MMaaS-Model"] == "qwen2.5-0.5b"
    assert upstream_state.calls[-1]["body"]["model"] == "qwen2.5-0.5b"  # alias rewritten upstream


async def test_long_input_skips_small_context_backend(gw, key, upstream_state):
    c, _ = gw
    plain, _ = key
    body = {"model": "fast", "messages": [{"role": "user", "content": "x " * 200}]}  # ~100 tokens > 64
    r = await c.post("/v1/chat/completions", json=body, headers=auth(plain))
    assert r.status_code == 200 and r.headers["X-MMaaS-Backend"] == "llama-medium"


async def test_fallback_on_5xx_and_backend_marked_unhealthy(gw, key, upstream_state):
    c, state = gw
    plain, rec = key
    upstream_state.down.add("llama-small")
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    assert r.status_code == 200 and r.headers["X-MMaaS-Backend"] == "llama-medium"
    rows = state.storage.usage_rows(0, rec.id)
    assert rows[-1]["backend"] == "llama-medium" and rows[-1]["fallback_from"] == "llama-small"
    assert state.health.is_healthy("llama-small") is False  # unhealthy_after=1
    # next request goes straight to medium without touching small
    n = len(upstream_state.calls)
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    assert r.headers["X-MMaaS-Backend"] == "llama-medium"
    assert [x["host"] for x in upstream_state.calls[n:]] == ["llama-medium"]
    # recovery: probe sees it up again
    upstream_state.down.clear()
    await state.health.probe_once()
    assert state.health.is_healthy("llama-small") is True


async def test_pinned_key_never_falls_back(gw, upstream_state):
    c, state = gw
    plain, rec = state.storage.create_key("experiment", backends=["llama-small"])
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    assert r.status_code == 200 and r.headers["X-MMaaS-Backend"] == "llama-small"
    upstream_state.down.add("llama-small")
    n = len(upstream_state.calls)
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    assert r.status_code == 502  # tried its only backend, failed; medium never contacted
    assert all(x["host"] == "llama-small" for x in upstream_state.calls[n:])
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    assert r.status_code == 503  # now marked unhealthy -> no candidate for this key
    # an unpinned key still falls back
    plain2, _ = state.storage.create_key("service")
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain2))
    assert r.status_code == 200 and r.headers["X-MMaaS-Backend"] == "llama-medium"
    # pinning can be lifted
    assert state.storage.set_backends(rec.id, None)
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    assert r.status_code == 200 and r.headers["X-MMaaS-Backend"] == "llama-medium"


def test_storage_migrates_old_ledger(tmp_path):
    import sqlite3
    from gateway.storage import Storage
    p = tmp_path / "old.db"
    con = sqlite3.connect(p)
    con.executescript("CREATE TABLE api_keys (id INTEGER PRIMARY KEY, key_hash TEXT UNIQUE NOT NULL,"
                      " prefix TEXT NOT NULL, name TEXT NOT NULL, ip_allow TEXT, rate_per_minute INTEGER,"
                      " created_at REAL NOT NULL, revoked_at REAL);"
                      " INSERT INTO api_keys VALUES (1,'h','mm-x','old',NULL,NULL,0,NULL);")
    con.commit()
    con.close()
    st = Storage(p)
    assert st.get_key(1).backends is None
    assert st.set_backends(1, ["dgx-vllm"]) and st.get_key(1).backends == ("dgx-vllm",)
    st.close()


async def test_all_backends_down_502_and_readyz_503(gw, key, upstream_state):
    c, state = gw
    plain, _ = key
    upstream_state.down.update({"llama-small", "llama-medium"})
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    assert r.status_code == 502
    assert (await c.get("/readyz")).status_code == 503


async def test_stale_connection_retried_on_same_backend(gw, key, upstream_state, monkeypatch):
    """A connection that dies before any response (e.g. pooled socket to a restarted backend) is retried once
    on the same backend instead of falling back — found while verifying the compose stack on a Mac."""
    import httpx as _httpx
    c, state = gw
    plain, rec = key
    real_send = state.client.send
    calls = {"n": 0}

    async def flaky_send(req, **kw):
        if "llama-small" in str(req.url) and req.url.path.endswith("/chat/completions") and calls["n"] == 0:
            calls["n"] += 1
            raise _httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return await real_send(req, **kw)

    monkeypatch.setattr(state.client, "send", flaky_send)
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    assert r.status_code == 200 and r.headers["X-MMaaS-Backend"] == "llama-small"
    row = state.storage.usage_rows(0, rec.id)[-1]
    assert row["fallback_from"] is None and calls["n"] == 1
    assert state.health.is_healthy("llama-small")


async def test_failed_requests_are_not_metered(gw, key, upstream_state):
    c, state = gw
    plain, rec = key
    await c.post("/v1/chat/completions", json={**CHAT, "model": "nope"}, headers=auth(plain))
    row = state.storage.usage_rows(0, rec.id)[-1]
    assert (row["status"], row["prompt_tokens"], row["completion_tokens"], row["usage_estimated"]) == (404, 0, 0, 0)
    upstream_state.down.update({"llama-small", "llama-medium"})
    await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    row = state.storage.usage_rows(0, rec.id)[-1]
    assert (row["status"], row["prompt_tokens"], row["usage_estimated"]) == (502, 0, 0)


async def test_unknown_model_404_and_hard_limit_413(gw, key):
    c, _ = gw
    plain, _ = key
    r = await c.post("/v1/chat/completions", json={**CHAT, "model": "nope"}, headers=auth(plain))
    assert r.status_code == 404
    huge = {"model": "default", "messages": [{"role": "user", "content": "가" * 20000}]}
    r = await c.post("/v1/chat/completions", json=huge, headers=auth(plain))
    assert r.status_code == 413


async def test_models_lists_only_healthy_backends(gw, key, upstream_state):
    c, state = gw
    plain, _ = key
    r = await c.get("/v1/models")  # no key: probes written for bare vLLM must keep working
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert ids == {"qwen2.5-0.5b", "qwen2.5-1.5b", "fast", "default"}
    state.health.report_failure("llama-medium")
    r = await c.get("/v1/models", headers=auth(plain))
    assert {m["id"] for m in r.json()["data"]} == {"qwen2.5-0.5b", "fast"}


# ---- metering ----------------------------------------------------------------

async def test_usage_recorded_from_backend_counts(gw, key):
    c, state = gw
    plain, rec = key
    r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    assert r.json()["choices"][0]["message"]["content"] == "Hello world"
    row = state.storage.usage_rows(0, rec.id)[-1]
    assert (row["prompt_tokens"], row["completion_tokens"], row["usage_estimated"]) == (7, 3, 0)
    assert row["stream"] == 0 and row["status"] == 200 and row["latency_ms"] > 0


async def test_streaming_passthrough_and_usage_from_final_chunk(gw, key, upstream_state):
    c, state = gw
    plain, rec = key
    async with c.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}, headers=auth(plain)) as r:
        assert r.status_code == 200 and r.headers["X-MMaaS-Backend"] == "llama-small"
        text = (await r.aread()).decode()
    pieces = [json.loads(ln[5:]) for ln in text.splitlines() if ln.startswith("data:") and "[DONE]" not in ln]
    assert "".join(_c(p) for p in pieces) == "Hello world"
    assert upstream_state.calls[-1]["body"]["stream_options"] == {"include_usage": True}
    row = state.storage.usage_rows(0, rec.id)[-1]
    assert (row["prompt_tokens"], row["completion_tokens"], row["usage_estimated"], row["stream"]) == (7, 3, 0, 1)
    assert row["ttft_ms"] is not None and row["ttft_ms"] > 0


async def test_usage_estimated_when_backend_sends_none(gw, key, upstream_state):
    c, state = gw
    plain, rec = key
    upstream_state.send_usage = False
    await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    row = state.storage.usage_rows(0, rec.id)[-1]
    assert row["usage_estimated"] == 1
    assert row["completion_tokens"] == estimate_tokens("Hello world") > 0
    async with c.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}, headers=auth(plain)) as r:
        await r.aread()
    row = state.storage.usage_rows(0, rec.id)[-1]
    assert row["usage_estimated"] == 1 and row["completion_tokens"] == estimate_tokens("Hello world")


async def test_usage_report_groups(gw, key):
    c, state = gw
    plain, _ = key
    for _ in range(3):
        await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    rep = state.storage.usage_report(0, "key")
    assert rep[0]["grp"] == "test" and rep[0]["requests"] == 3 and rep[0]["prompt_tokens"] == 21
    rep = state.storage.usage_report(0, "backend")
    assert rep[0]["grp"] == "llama-small"


# ---- tools proxy -------------------------------------------------------------

async def test_tool_proxy_requires_key_and_forwards(gw, key, upstream_state):
    c, state = gw
    plain, rec = key
    assert (await c.get("/tools/jlawcite/search?q=x")).status_code == 401
    r = await c.post("/tools/jlawcite/search?q=x", content=b"{}", headers=auth(plain))
    assert r.status_code == 200
    assert r.json() == {"host": "jlawcite", "method": "POST", "path": "search", "query": "q=x", "body": "{}"}
    assert (await c.get("/tools/nope/x", headers=auth(plain))).status_code == 404
    row = state.storage.usage_rows(0, rec.id)[-1]
    assert row["path"] == "/tools/jlawcite" and row["backend"] == "jlawcite"


# ---- metrics -----------------------------------------------------------------

async def test_metrics_endpoint_exposes_counters(gw, key):
    c, _ = gw
    plain, _ = key
    await c.post("/v1/chat/completions", json=CHAT, headers=auth(plain))
    body = (await c.get("/metrics")).text
    assert "mmaas_requests_total" in body and "mmaas_tokens_total" in body and "mmaas_backend_healthy" in body


def _c(p):
    ch = p.get("choices") or []
    return ch[0]["delta"].get("content", "") if ch else ""


@pytest.mark.parametrize("text,lo,hi", [("hello world", 2, 4), ("가나다라마", 5, 5), ("", 0, 0)])
def test_estimate_tokens(text, lo, hi):
    assert lo <= estimate_tokens(text) <= hi
