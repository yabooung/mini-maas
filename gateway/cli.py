"""`mmaas` command line: key management, usage reports, serve."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

from .config import load_config
from .storage import Storage


def _since(spec: str) -> float:
    m = re.fullmatch(r"(\d+)([smhd])", spec)
    if not m:
        raise SystemExit(f"bad --since '{spec}' (use e.g. 30m, 24h, 7d)")
    n, unit = int(m.group(1)), m.group(2)
    return time.time() - n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mmaas")
    p.add_argument("--config", default="gateway.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    keys = sub.add_parser("keys").add_subparsers(dest="kcmd", required=True)
    kc = keys.add_parser("create")
    kc.add_argument("--name", required=True)
    kc.add_argument("--ip", help="comma-separated IPs/CIDRs allowed to use this key")
    kc.add_argument("--rate", type=int, help="requests per minute (default from config)")
    keys.add_parser("list")
    kr = keys.add_parser("revoke")
    kr.add_argument("--id", type=int, required=True)

    us = sub.add_parser("usage")
    us.add_argument("--since", default="24h")
    us.add_argument("--by", choices=["key", "model", "backend", "day"], default="key")
    us.add_argument("--json", action="store_true")

    sv = sub.add_parser("serve")
    sv.add_argument("--host")
    sv.add_argument("--port", type=int)

    a = p.parse_args(argv)
    cfg = load_config(a.config)

    if a.cmd == "serve":
        import uvicorn
        from .app import create_app
        uvicorn.run(create_app(cfg), host=a.host or cfg.listen.host, port=a.port or cfg.listen.port)
        return 0

    st = Storage(cfg.storage.sqlite_path)
    try:
        if a.cmd == "keys":
            if a.kcmd == "create":
                ips = [s.strip() for s in a.ip.split(",")] if a.ip else None
                plain, rec = st.create_key(a.name, ips, a.rate)
                print(f"id={rec.id} name={rec.name} key={plain}")
                print("store this key now; it is not recoverable later", file=sys.stderr)
            elif a.kcmd == "list":
                for r in st.list_keys():
                    flag = "REVOKED" if r.revoked else "active"
                    print(f"{r.id}\t{r.prefix}…\t{r.name}\t{flag}\trate={r.rate_per_minute or 'default'}"
                          f"\tip={','.join(r.ip_allow) if r.ip_allow else 'any'}")
            elif a.kcmd == "revoke":
                print("revoked" if st.revoke_key(a.id) else "no active key with that id")
        elif a.cmd == "usage":
            rows = st.usage_report(_since(a.since), a.by)
            if a.json:
                print(json.dumps(rows, ensure_ascii=False, indent=1))
            else:
                print(f"{a.by:<24}{'req':>8}{'prompt':>10}{'compl':>10}{'est':>6}{'err':>6}{'avg_ms':>9}")
                for r in rows:
                    print(f"{str(r['grp']):<24}{r['requests']:>8}{r['prompt_tokens'] or 0:>10}"
                          f"{r['completion_tokens'] or 0:>10}{r['estimated_rows'] or 0:>6}{r['errors']:>6}"
                          f"{(r['avg_latency_ms'] or 0):>9.0f}")
    finally:
        st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
