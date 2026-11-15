#!/usr/bin/env python3
"""Mesure latence / débit : python client/benchmark.py --url http://<pi-ip>:8000 -n 10 -c 1"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

PROMPTS = [
    "Why is the sky blue?",
    "Explain the law of large numbers in two sentences.",
    "Give me three tips to learn Python.",
    "What is a Raspberry Pi?",
    "Summarize what the GGUF format is.",
]


def one(url: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    ttft, stats = None, {}
    # No session_id : each request is an independant conversation
    with requests.post(f"{url}/api/chat", json={"message": prompt, "stream": True, "use_rag": False},
                       stream=True, timeout=600) as r:
        if r.status_code != 200:
            return {"ok": False, "why": f"HTTP {r.status_code}"}
        for line in r.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            if ev["type"] == "token" and ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            elif ev["type"] == "done":
                stats = ev["stats"]
            elif ev["type"] == "error":
                return {"ok": False, "why": ev["detail"]}
    return {
        "ok": True,
        "ttft_ms": ttft,
        "total_ms": (time.perf_counter() - t0) * 1000,
        "tokens_per_s": stats.get("tokens_per_s"),
    }


def summarize(name: str, values: list[float]) -> None:
    if not values:
        print(f"{name:>14}: n/a")
        return
    v = sorted(values)
    print(f"{name:>14}: mean {statistics.mean(v):8.1f} | p50 {statistics.median(v):8.1f} | max {v[-1]:8.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("-n", type=int, default=10, help="number of requests")
    p.add_argument("-c", type=int, default=1, help="concurrent clients")
    args = p.parse_args()

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.n)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.c) as ex:
        results = list(ex.map(lambda pr: one(args.url, pr), prompts))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    print(f"\n{len(ok)}/{len(results)} OK | wall {wall:.1f}s | {len(ok) / wall:.2f} req/s")
    for r in results:
        if not r["ok"]:
            print("  failed:", r["why"])
    summarize("TTFT (ms)", [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None])
    summarize("Total (ms)", [r["total_ms"] for r in ok])
    summarize("Tokens/s", [r["tokens_per_s"] for r in ok if r["tokens_per_s"] is not None])


if __name__ == "__main__":
    main()