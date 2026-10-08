"""Configuration model (loaded from gateway.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ListenCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    trust_proxy: bool = False  # honor X-Forwarded-For for IP allowlists


class StorageCfg(BaseModel):
    sqlite_path: str = "./data/gateway.db"


class LoggingCfg(BaseModel):
    meta_log: str | None = "./data/requests.jsonl"
    body_log: str | None = None


class HealthCfg(BaseModel):
    interval_seconds: float = 5.0
    timeout_seconds: float = 2.0
    unhealthy_after: int = 2


class RateLimitCfg(BaseModel):
    default_per_minute: int = 60


class BackendCfg(BaseModel):
    name: str
    base_url: str
    models: list[str] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    max_input_tokens: int = 8192
    priority: int = 100
    api_key: str | None = None  # sent upstream as Bearer if the backend itself requires one
    timeout_seconds: float = 120.0

    def resolve_model(self, requested: str) -> str | None:
        """Return the backend's own model name for a requested name, or None if unserved."""
        if requested in self.models:
            return requested
        target = self.aliases.get(requested)
        if target is not None and target in self.models:
            return target
        return None

    @property
    def served_names(self) -> list[str]:
        return sorted(set(self.models) | set(self.aliases))


class RoutingCfg(BaseModel):
    backends: list[BackendCfg] = Field(default_factory=list)
    max_input_tokens_hard: int = 16384


class ToolCfg(BaseModel):
    name: str
    base_url: str


class AuthCfg(BaseModel):
    # YAML list of {name, sha256, rate_per_minute?, backends?, ip_allow?}. Loaded at startup into the
    # local ledger, so every replica accepts the same keys (mount it from a Kubernetes Secret).
    # Generate entries with `mmaas keys gen --name ...`; only the hash is stored.
    static_keys_file: str | None = None


class GatewayConfig(BaseModel):
    listen: ListenCfg = Field(default_factory=ListenCfg)
    auth: AuthCfg = Field(default_factory=AuthCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    health: HealthCfg = Field(default_factory=HealthCfg)
    ratelimit: RateLimitCfg = Field(default_factory=RateLimitCfg)
    routing: RoutingCfg = Field(default_factory=RoutingCfg)
    tools: list[ToolCfg] = Field(default_factory=list)

    def backend(self, name: str) -> BackendCfg:
        for b in self.routing.backends:
            if b.name == name:
                return b
        raise KeyError(name)


def load_config(path: str | Path) -> GatewayConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return GatewayConfig.model_validate(raw)
