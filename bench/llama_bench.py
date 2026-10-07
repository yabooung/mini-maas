"""Client-side inference benchmark against any OpenAI-compatible endpoint (llama-server, vLLM, or the gateway).

Measures, per configuration:
  * single-stream: TTFT, decode tokens/s (from streamed chunks), prompt tokens/s (from usage + TTFT)
  * concurrency sweep: aggregate tokens/s and per-request latency at N parallel streams
  * (optional) small QA accuracy, to see whether a quantization level changes answers

    python bench/llama_bench.py --url http://localhost:8081/v1 --model qwen2.5-0.5b \
        --label q4_k_m --concurrency 1,2,4,8 --qa bench/qa_small.jsonl --out results/bench.jsonl

Each run appends one JSON line per (label, concurrency) to --out; summarize with bench/report.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time

import httpx

GEN_PROMPT = ("Write a detailed, multi-paragraph explanation of how a hash table works, "
              "including collisions, load factor and resizing.")


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, round(q * (len(xs) - 1))))]


async def stream_once(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int,
                      headers: dict) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    usage = None
    text = []
    async with client.stream("POST", f"{url}/chat/completions", json=body, headers=headers) as r:
        r.raise_for_status()
        buf = ""
        async for raw in r.aiter_bytes():
            if ttft is None:
                ttft = time.perf_counter() - t0
            buf += raw.decode("utf-8", "replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                try:
                    obj = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    c = (ch.get("delta") or {}).get("content")
                    if c:
                        text.append(c)
                        n_chunks += 1
    total = time.perf_counter() - t0
    c_tok = (usage or {}).get("completion_tokens") or n_chunks
    p_tok = (usage or {}).get("prompt_tokens")
    return {"ttft_s": ttft, "total_s": total, "completion_tokens": c_tok, "prompt_tokens": p_tok,
            "decode_tok_s": (c_tok - 1) / (total - ttft) if ttft and total > ttft and c_tok > 1 else None,
            "prompt_tok_s": p_tok / ttft if p_tok and ttft else None, "text": "".join(text)}


async def sweep(client, url, model, headers, concurrency: int, rounds: int, max_tokens: int) -> dict:
    results = []
    t0 = time.perf_counter()
    for _ in range(rounds):
        batch = await asyncio.gather(*(stream_once(client, url, model, GEN_PROMPT, max_tokens, headers)
                                       for _ in range(concurrency)))
        results.extend(batch)
    wall = time.perf_counter() - t0
    toks = sum(r["completion_tokens"] for r in results)
    return {
        "concurrency": concurrency, "requests": len(results), "wall_s": round(wall, 2),
        "aggregate_tok_s": round(toks / wall, 2),
        "per_request_decode_tok_s_p50": pct([r["decode_tok_s"] for r in results if r["decode_tok_s"]], .5),
        "ttft_s_p50": pct([r["ttft_s"] for r in results], .5), "ttft_s_p95": pct([r["ttft_s"] for r in results], .95),
        "latency_s_p50": pct([r["total_s"] for r in results], .5),
        "latency_s_p95": pct([r["total_s"] for r in results], .95),
        "prompt_tok_s_p50": pct([r["prompt_tok_s"] for r in results if r["prompt_tok_s"]], .5),
    }


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _match(answer: str, expected: str) -> bool:
    a, e = answer.strip().lower(), expected.strip().lower()
    if e in a:
        return True
    nums = _NUM.findall(a)
    return bool(nums) and e in nums


async def qa(client, url, model, headers, path: str) -> dict:
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    correct = 0
    wrong = []
    for it in items:
        r = await stream_once(client, url, model, it["q"] + "\nAnswer with only the answer.", 32, headers)
        ok = _match(r["text"], it["a"])
        correct += ok
        if not ok:
            wrong.append({"q": it["q"], "expected": it["a"], "got": r["text"][:80]})
    return {"qa_n": len(items), "qa_correct": correct, "qa_acc": round(correct / len(items), 3), "qa_wrong": wrong}


async def main_async(a):
    headers = {"Authorization": f"Bearer {a.key}"} if a.key else {}
    async with httpx.AsyncClient(timeout=600) as client:
        await stream_once(client, a.url, a.model, "warmup", 8, headers)  # warm the model / caches
        rows = []
        for c in [int(x) for x in a.concurrency.split(",")]:
            s = await sweep(client, a.url, a.model, headers, c, a.rounds, a.max_tokens)
            s.update(label=a.label, model=a.model, url=a.url, max_tokens=a.max_tokens, ts=time.time())
            print(json.dumps(s), flush=True)
            rows.append(s)
        if a.qa:
            q = await qa(client, a.url, a.model, headers, a.qa)
            q.update(label=a.label, model=a.model, url=a.url, ts=time.time(), kind="qa")
            print(json.dumps({k: v for k, v in q.items() if k != "qa_wrong"}), flush=True)
            rows.append(q)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="OpenAI-compatible base, e.g. http://host:8081/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, help="configuration label, e.g. q4_k_m / q8_0 / f16 / spec-0.5b")
    p.add_argument("--key")
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--qa")
    p.add_argument("--out")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
