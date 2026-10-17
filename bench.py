#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible chat endpoint (streaming).

Measures:
  - TTFT            time to first token (prefill latency)
  - total           end-to-end latency
  - decode t/s      output tokens per second (decode throughput)
  - prefill t/s     prompt tokens per second (prefill throughput)

Output is a compact table of median values. Use -q/--quiet to print
only the metrics row.

Usage:
  ./bench.py                       # sensible defaults
  ./bench.py --prompt-words 512 --max-tokens 512 --runs 5
  ./bench.py --concurrency 4 --runs 8
  ./bench.py -q                    # metrics row only
"""
import argparse
import json
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DEFAULT_BASE = "http://127.0.0.1:2001/v1"
DEFAULT_MODEL = "halogen-qwen3.8-27b"

# A fixed sentence repeated to build prompts of a target length.
_SENT = ("The quick brown fox jumps over the lazy dog near the riverbank "
         "while the sun sets behind the distant mountains and the wind "
         "carries the scent of pine across the quiet valley. ")


def make_prompt(words: int) -> str:
    reps = max(1, words // len(_SENT.split()))
    return (_SENT * reps)[: words * 5]


def run_one(base: str, model: str, prompt: str, max_tokens: int,
            temperature: float, api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    out_tokens = None
    prompt_tokens = None
    err = None

    try:
        with requests.post(f"{base}/chat/completions", json=payload,
                           headers=headers, stream=True, timeout=600) as r:
            if r.status_code != 200:
                err = f"HTTP {r.status_code}: {r.read()[:300]!r}"
            else:
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data:"):
                        continue
                    data = raw[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    u = obj.get("usage")
                    if u:
                        out_tokens = u.get("completion_tokens", out_tokens)
                        prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                    ch = obj.get("choices") or []
                    if ch:
                        c = (ch[0].get("delta") or {}).get("content")
                        if c:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            n_chunks += 1
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"

    total = time.perf_counter() - t0
    if err:
        return {"error": err}

    if out_tokens is None:  # fallback: ~1 token per content chunk
        out_tokens = n_chunks
    decode_time = max(total - (ttft or total), 1e-6)
    decode_tps = (out_tokens - 1) / decode_time if out_tokens and out_tokens > 1 else 0.0
    prefill_tps = (prompt_tokens / ttft) if (prompt_tokens and ttft) else None

    return {
        "ttft_s": ttft,
        "total_s": total,
        "out_tokens": out_tokens,
        "prompt_tokens": prompt_tokens,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
    }


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"errors": [r["error"] for r in results]}

    def col(k):
        return [r[k] for r in ok if r.get(k) is not None]

    ttft = col("ttft_s")
    total = col("total_s")
    dec = col("decode_tps")
    pre = col("prefill_tps")
    out_tok = sum(r["out_tokens"] for r in ok)
    wall = max(r["total_s"] for r in ok)

    return {
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "ttft_ms": st.median(ttft) * 1e3 if ttft else None,
        "total_ms": st.median(total) * 1e3 if total else None,
        "decode_tps": st.median(dec) if dec else None,
        "prefill_tps": st.median(pre) if pre else None,
        "out_tokens": out_tok,
        "wall_s": wall,
        "agg_tps": out_tok / wall if wall else 0.0,
        "errors": [r["error"] for r in results if "error" in r],
    }


def print_table(rows: list[list[str]], quiet: bool = False) -> None:
    headers = ["TTFT ms", "Latency ms", "Decode t/s", "Prefill t/s",
               "Prompt w", "Model"]
    all_rows = [headers] + rows
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(headers))]

    def line(cells):
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    if not quiet:
        print(line(headers))
        divider = line(["-" * w for w in widths])
        print("".join("-" if ch != "|" else "|" for ch in divider))
    for row in rows:
        print(line(row))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--prompt-words", type=int, default=256,
                    help="approx prompt length in words")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="tokens to generate per request")
    ap.add_argument("--runs", type=int, default=3,
                    help="sequential runs for latency/throughput")
    ap.add_argument("--concurrency", type=int, default=0,
                    help="parallel requests (0 = skip)")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="print only the metrics row")
    args = ap.parse_args()

    prompt = make_prompt(args.prompt_words)

    def row(s: dict) -> list[str]:
        def f(v):
            return f"{v:.0f}" if v is not None else "n/a"
        return [
            f(s.get("ttft_ms")),
            f(s.get("total_ms")),
            f(s.get("decode_tps")),
            f(s.get("prefill_tps")),
            str(args.prompt_words),
            args.model,
        ]

    # Sequential: latency + single-stream throughput
    seq = [run_one(args.base, args.model, prompt, args.max_tokens,
                   args.temp, args.api_key) for _ in range(args.runs)]
    s = summarize(seq)
    if s.get("n_ok", 0) == 0:
        print("ALL RUNS FAILED:")
        for e in s.get("errors", []):
            print("  ", e)
        raise SystemExit(1)

    rows = [row(s)]

    # Concurrency: aggregate throughput
    if args.concurrency > 0:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(run_one, args.base, args.model, prompt,
                              args.max_tokens, args.temp, args.api_key)
                    for _ in range(args.concurrency)]
            conc = [f.result() for f in as_completed(futs)]
        c = summarize(conc)
        if c.get("n_ok", 0) == 0:
            for e in c.get("errors", []):
                print("  ", e)
            raise SystemExit(1)
        rows.append(row(c))

    print_table(rows, quiet=args.quiet)


if __name__ == "__main__":
    main()
