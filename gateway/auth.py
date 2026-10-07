"""Bearer-key authentication, IP allowlist, and per-key rate limiting as one FastAPI dependency."""

from __future__ import annotations

import ipaddress
import math

from fastapi import HTTPException, Request

from . import metrics
from .storage import KeyRecord


def client_ip(request: Request, trust_proxy: bool) -> str:
    if trust_proxy:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "0.0.0.0"


def ip_allowed(ip: str, allow: tuple[str, ...] | None) -> bool:
    if not allow:
        return True
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for entry in allow:
        try:
            if "/" in entry:
                if addr in ipaddress.ip_network(entry, strict=False):
                    return True
            elif addr == ipaddress.ip_address(entry):
                return True
        except ValueError:
            continue
    return False


def _deny(status: int, reason: str, message: str, headers: dict | None = None) -> HTTPException:
    metrics.AUTH_FAILED.labels(reason=reason).inc()
    return HTTPException(status_code=status, detail={"error": {"message": message, "type": reason}},
                         headers=headers)


async def require_key(request: Request) -> KeyRecord:
    gw = request.app.state.gw
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise _deny(401, "missing_key", "missing bearer token")
    rec = gw.storage.verify_key(header[7:].strip())
    if rec is None:
        raise _deny(401, "invalid_key", "invalid API key")
    if rec.revoked:
        raise _deny(401, "revoked_key", "API key has been revoked")
    ip = client_ip(request, gw.cfg.listen.trust_proxy)
    if not ip_allowed(ip, rec.ip_allow):
        raise _deny(403, "ip_not_allowed", f"client IP {ip} is not allowed for this key")
    per_minute = rec.rate_per_minute or gw.cfg.ratelimit.default_per_minute
    ok, retry_after = gw.limiter.allow(rec.id, per_minute)
    if not ok:
        metrics.RATELIMITED.labels(key=rec.name).inc()
        raise HTTPException(
            status_code=429,
            detail={"error": {"message": f"rate limit of {per_minute}/min exceeded",
                              "type": "rate_limited"}},
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )
    request.state.client_ip = ip
    return rec
