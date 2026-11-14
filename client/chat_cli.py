#!/usr/bin/env python3
"""Client terminal : python client/chat_cli.py --url http://<pi-ip>:8000"""
import argparse
import json

import requests


def fmt(v, spec=".0f"):
    return format(v, spec) if v is not None else "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    args = p.parse_args()

    sid = None
    print("Commands: /new = reset conversation, /bye = quit")
    while True:
        try:
            text = input("\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/bye":
            break
        if text == "/new":
            if sid:
                requests.delete(f"{args.url}/api/sessions/{sid}", timeout=10)
            sid = None
            print("[new conversation]")
            continue

        body = {"message": text, "session_id": sid, "stream": True}
        with requests.post(f"{args.url}/api/chat", json=body, stream=True, timeout=300) as r:
            if r.status_code != 200:
                print(f"[HTTP {r.status_code}] {r.text}")
                continue
            for line in r.iter_lines():
                if not line:
                    continue
                ev = json.loads(line)
                if ev["type"] == "meta":
                    sid = ev["session_id"]
                elif ev["type"] == "token":
                    print(ev["content"], end="", flush=True)
                elif ev["type"] == "error":
                    print(f"\n[error] {ev['detail']}")
                elif ev["type"] == "done":
                    s = ev["stats"]
                    print(f"\n[ttft {fmt(s.get('ttft_ms'))} ms | "
                          f"{fmt(s.get('tokens_per_s'), '.1f')} tok/s]")


if __name__ == "__main__":
    main()