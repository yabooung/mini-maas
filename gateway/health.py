"""Background backend health monitor.

Probes GET {base_url}/models every `interval_seconds`. A backend is unhealthy after
`unhealthy_after` consecutive failures; a single success restores it. The proxy also
reports failures directly so fallback happens faster than the probe interval.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from .config import BackendCfg, HealthCfg
from . import metrics


class HealthMonitor:
    def __init__(self, backends: list[BackendCfg], cfg: HealthCfg, client: httpx.AsyncClient):
        self._backends = {b.name: b for b in backends}
        self._cfg = cfg
        self._client = client
        self._failures: dict[str, int] = {b.name: 0 for b in backends}
        self._healthy: dict[str, bool] = {b.name: True for b in backends}  # optimistic until first probe
        self._last_probe: dict[str, float] = {}
        self._task: asyncio.Task | None = None

    # ---- state ------------------------------------------------------------

    def is_healthy(self, name: str) -> bool:
        return self._healthy.get(name, False)

    def snapshot(self) -> dict[str, dict]:
        return {
            n: {"healthy": self._healthy[n], "consecutive_failures": self._failures[n],
                "last_probe": self._last_probe.get(n)}
            for n in self._backends
        }

    def report_failure(self, name: str) -> None:
        self._failures[name] = self._failures.get(name, 0) + 1
        if self._failures[name] >= self._cfg.unhealthy_after:
            self._set(name, False)

    def report_success(self, name: str) -> None:
        self._failures[name] = 0
        self._set(name, True)

    def _set(self, name: str, ok: bool) -> None:
        self._healthy[name] = ok
        metrics.BACKEND_HEALTHY.labels(backend=name).set(1 if ok else 0)

    # ---- probing ----------------------------------------------------------

    async def probe_once(self) -> None:
        async def one(b: BackendCfg) -> None:
            self._last_probe[b.name] = time.time()
            try:
                r = await self._client.get(f"{b.base_url.rstrip('/')}/models",
                                           timeout=self._cfg.timeout_seconds)
                if r.status_code < 500:
                    self.report_success(b.name)
                else:
                    self.report_failure(b.name)
            except (httpx.HTTPError, OSError):
                self.report_failure(b.name)

        await asyncio.gather(*(one(b) for b in self._backends.values()))

    async def _loop(self) -> None:
        while True:
            await self.probe_once()
            await asyncio.sleep(self._cfg.interval_seconds)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
