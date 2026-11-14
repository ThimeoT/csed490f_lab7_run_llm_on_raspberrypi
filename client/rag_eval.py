#!/usr/bin/env python3
"""Evaluate retrieval (hit@1 / hit@3) and compare responses with/without RAG.
   python client/rag_eval.py --url http://<pi-ip>:8000 [--no-generate]"""
import argparse
import json
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--questions", default=str(Path(__file__).with_name("eval_questions.json")))
    ap.add_argument("--no-generate", action="store_true", help="retrieval only (fast)")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    n_in = hit1 = hit3 = n_out = rejected = 0

    for item in questions:
        r = requests.get(f"{args.url}/api/rag/search", params={"q": item["q"], "k": 3}, timeout=60)
        r.raise_for_status()
        data = r.json()
        results, min_score = data["results"], data["min_score"]
        sources = [x["source"] for x in results]
        top = results[0]["score"] if results else None
        expected = item.get("expected_source")

        print(f"\nQ: {item['q']}")
        if expected:
            n_in += 1
            h1 = bool(sources) and sources[0] == expected
            h3 = expected in sources
            hit1 += h1
            hit3 += h3
            print(f"  retrieved: {sources} (top score {top}) | hit@1={h1} hit@3={h3}")
        else:
            n_out += 1
            ok = top is None or top < min_score
            rejected += ok
            print(f"  out-of-scope: top score {top} (threshold {min_score}) | correctly rejected={ok}")
        if item.get("answer"):
            print(f"  expected: {item['answer']}")

        if args.no_generate:
            continue
        for label, flag in (("RAG", True), ("no RAG", False)):
            resp = requests.post(f"{args.url}/api/chat", timeout=600,
                                 json={"message": item["q"], "stream": False, "use_rag": flag})
            body = resp.json()
            stats = body.get("stats", {})
            print(f"  [{label}] total {stats.get('total_ms')} ms, retrieval {stats.get('retrieval_ms')} ms\n"
                  f"    {str(body.get('reply', body))[:300]!r}")

    print(f"\nRetrieval: hit@1 {hit1}/{n_in} | hit@3 {hit3}/{n_in}")
    print(f"Out-of-scope rejected: {rejected}/{n_out}")


if __name__ == "__main__":
    main()