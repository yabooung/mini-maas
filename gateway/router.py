"""Model routing: requested model -> ordered list of candidate backends."""

from __future__ import annotations

from dataclasses import dataclass

from .config import BackendCfg, RoutingCfg
from .health import HealthMonitor


@dataclass(frozen=True)
class Candidate:
    backend: BackendCfg
    model: str  # backend's own model name (alias resolved)


class RouteError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def estimate_tokens(text: str) -> int:
    """Cheap pre-routing estimate. CJK characters count ~1 token; other text ~4 chars/token.

    This is only used to pick a backend whose context fits; the usage ledger uses the
    backend-reported token counts whenever they are available.
    """
    cjk = 0
    other = 0
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF or 0xF900 <= o <= 0xFAFF:
            cjk += 1
        else:
            other += 1
    return cjk + (other + 3) // 4


def estimate_request_tokens(body: dict) -> int:
    if "messages" in body:
        parts = []
        for m in body.get("messages") or []:
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):  # multimodal content parts
                parts.extend(p.get("text", "") for p in c if isinstance(p, dict))
        return estimate_tokens("\n".join(parts))
    p = body.get("prompt", "")
    if isinstance(p, list):
        p = "\n".join(map(str, p))
    return estimate_tokens(str(p))


class Router:
    def __init__(self, cfg: RoutingCfg, health: HealthMonitor):
        self._cfg = cfg
        self._health = health

    def candidates(self, requested_model: str, input_tokens: int,
                   allowed: tuple[str, ...] | None = None) -> list[Candidate]:
        """`allowed` pins a caller to specific backends (e.g. experiments that must not silently fall back
        to a different quantization). Outside that set nothing is tried, so the caller gets 503 instead."""
        if input_tokens > self._cfg.max_input_tokens_hard:
            raise RouteError(413, f"input of ~{input_tokens} tokens exceeds hard limit "
                                  f"{self._cfg.max_input_tokens_hard}")
        served: list[Candidate] = []
        fits: list[Candidate] = []
        for b in sorted(self._cfg.backends, key=lambda b: b.priority):
            if allowed is not None and b.name not in allowed:
                continue
            m = b.resolve_model(requested_model)
            if m is None:
                continue
            served.append(Candidate(b, m))
            if input_tokens <= b.max_input_tokens:
                fits.append(Candidate(b, m))
        if not served:
            raise RouteError(404, f"model '{requested_model}' is not served by any backend")
        if not fits:
            raise RouteError(413, f"input of ~{input_tokens} tokens exceeds every backend serving "
                                  f"'{requested_model}'")
        healthy = [c for c in fits if self._health.is_healthy(c.backend.name)]
        if not healthy:
            raise RouteError(503, f"no healthy backend for '{requested_model}'")
        return healthy

    def served_models(self) -> list[dict]:
        out: dict[str, set[str]] = {}
        for b in self._cfg.backends:
            if not self._health.is_healthy(b.name):
                continue
            for n in b.served_names:
                out.setdefault(n, set()).add(b.name)
        return [{"id": n, "object": "model", "owned_by": "mini-maas", "backends": sorted(bs)}
                for n, bs in sorted(out.items())]
