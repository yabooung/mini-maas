"""Turn bench.jsonl into a Markdown table (paste into README)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def main(path: str) -> None:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    sweeps = [r for r in rows if r.get("kind") != "qa"]
    qas = {r["label"]: r for r in rows if r.get("kind") == "qa"}
    by_label: dict[str, list] = defaultdict(list)
    for r in sweeps:
        by_label[r["label"]].append(r)
    print("| config | conc | agg tok/s | per-req decode tok/s p50 | TTFT p50 (s) | latency p50 (s) | latency p95 (s) | QA acc |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, rs in by_label.items():
        for r in sorted(rs, key=lambda x: x["concurrency"]):
            q = qas.get(label)
            acc = f"{q['qa_correct']}/{q['qa_n']}" if q and r["concurrency"] == rs[0]["concurrency"] else ""
            print(f"| {label} | {r['concurrency']} | {r['aggregate_tok_s']:.1f} | "
                  f"{(r['per_request_decode_tok_s_p50'] or 0):.1f} | {(r['ttft_s_p50'] or 0):.3f} | "
                  f"{(r['latency_s_p50'] or 0):.2f} | {(r['latency_s_p95'] or 0):.2f} | {acc} |")


if __name__ == "__main__":
    main(sys.argv[1])
