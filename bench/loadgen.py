"""Constant-rate load generator with per-request outcome log.

Used for two measurements:
  * rolling-update request loss: run this, trigger `kubectl rollout`, count non-200s.
  * fallback switch time: run this, kill a backend, measure the gap between the last
    success on backend A and the first success on backend B.

    python bench/loadgen.py --url http://localhost:8000 --key mm-... --model fast \
        --rps 5 --duration 120 --out results/rolling.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter

import httpx

PROMPT = [{"role": "user", "content": "Reply with the single word: ok"}]


async def one(client: httpx.AsyncClient, url: str, key: str, model: str, i: int, out, stats: Counter,
              max_tokens: int, stream: bool):
    t0 = time.perf_counter()
    rec = {"i": i, "t_start": time.time()}
    try:
        body = {"model": model, "messages": PROMPT, "max_tokens": max_tokens, "stream": stream}
        if stream:
            ttft = None
            async with client.stream("POST", f"{url}/v1/chat/completions", json=body,
                                     headers={"Authorization": f"Bearer {key}"}) as r:
                async for _ in r.aiter_bytes():
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                rec.update(status=r.status_code, backend=r.headers.get("x-mmaas-backend"), ttft_s=ttft)
        else:
            r = await client.post(f"{url}/v1/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {key}"})
            rec.update(status=r.status_code, backend=r.headers.get("x-mmaas-backend"))
    except httpx.HTTPError as e:
        rec.update(status=0, error=type(e).__name__)
    rec["latency_s"] = time.perf_counter() - t0
    stats[rec["status"]] += 1
    out.write(json.dumps(rec) + "\n")
    out.flush()


async def main_async(a):
    stats: Counter = Counter()
    tasks: list[asyncio.Task] = []
    interval = 1.0 / a.rps
    out = open(a.out, "w", encoding="utf-8")
    try:
        async with httpx.AsyncClient(timeout=a.timeout) as client:
            t_end = time.perf_counter() + a.duration
            i = 0
            next_t = time.perf_counter()
            while time.perf_counter() < t_end:
                tasks.append(asyncio.create_task(one(client, a.url, a.key, a.model, i, out, stats,
                                                     a.max_tokens, a.stream)))
                i += 1
                next_t += interval
                await asyncio.sleep(max(0.0, next_t - time.perf_counter()))
                if i % max(1, int(a.rps * 10)) == 0:
                    print(f"[{i}] {dict(stats)}", flush=True)
            await asyncio.gather(*tasks)
    finally:
        out.close()
    total = sum(stats.values())
    ok = stats.get(200, 0)
    print(json.dumps({"requests": total, "ok": ok, "failed": total - ok, "by_status": dict(stats)}, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--key", required=True)
    p.add_argument("--model", default="fast")
    p.add_argument("--rps", type=float, default=5)
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--out", default="loadgen.jsonl")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
