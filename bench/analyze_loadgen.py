"""Summarize a loadgen.jsonl: loss count, latency percentiles, backend switch gaps.

    python bench/analyze_loadgen.py results/rolling.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(q * (len(xs) - 1))))
    return xs[k]


def main(path: str) -> None:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows.sort(key=lambda r: r["i"])
    status = Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"] == 200]
    lat = [r["latency_s"] for r in ok]
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    print(f"requests={len(rows)} ok={len(ok)} failed={len(rows) - len(ok)} by_status={dict(status)}")
    if lat:
        print(f"latency_s p50={pct(lat, .5):.3f} p95={pct(lat, .95):.3f} p99={pct(lat, .99):.3f} "
              f"max={max(lat):.3f} mean={statistics.mean(lat):.3f}")
    if ttft:
        print(f"ttft_s    p50={pct(ttft, .5):.3f} p95={pct(ttft, .95):.3f}")
    # backend switches (fallback / rollout) in request order
    prev, switches = None, []
    for r in ok:
        b = r.get("backend")
        if prev is not None and b != prev:
            switches.append((r["i"], prev, b, r["t_start"]))
        prev = b
    if switches:
        print("backend switches (request#, from -> to):")
        for i, a, b, _ in switches:
            print(f"  #{i}: {a} -> {b}")
    # longest gap between consecutive successes = outage window seen by clients
    gaps = [(ok[j]["t_start"] - ok[j - 1]["t_start"], ok[j - 1]["i"], ok[j]["i"]) for j in range(1, len(ok))]
    if gaps:
        g, a, b = max(gaps)
        print(f"longest gap between successes: {g:.2f}s (between #{a} and #{b})")
    failed_ids = [r["i"] for r in rows if r["status"] != 200]
    if failed_ids:
        print(f"failed request ids: {failed_ids[:50]}{' ...' if len(failed_ids) > 50 else ''}")


if __name__ == "__main__":
    main(sys.argv[1])
