#!/usr/bin/env python3
"""M5 — concurrency sweep against the serving endpoint.

    python scripts/06_bench.py --url http://127.0.0.1:8000 --concurrency 1,4,8,16,32 \\
                               --requests-per-level 32 --tag fp8

What M5 measures, and why it is different from M1:

  M1 measured ONE request at a time (TTFT 35 ms, TPOT 34 ms). A server is a
  scheduling problem: with C requests in flight, aggregate throughput rises while
  per-request latency degrades, and the *shape* of that trade-off is the deliverable.
  So the gate moves to p50/p95/p99 of TTFT and TPOT at each concurrency level.

Two design choices that keep the comparison honest:

  * `/v1/completions` with a RAW prompt, not `/v1/chat/completions`. The chat endpoint
    applies Qwen3's template, which enables thinking by default — the model then emits
    a variable-length monologue before answering (measured earlier: 13-token prompt →
    ~400 tokens). Raw completions give a controlled prompt and controlled output, which
    is what a latency comparison needs.
  * TTFT is timed at the FIRST streamed chunk, not from the total. A non-streaming
    call cannot show TTFT at all, which is why the streaming path is required.

Peak VRAM is sampled device-wide via NVML minus the idle baseline (M1 decision 7, as
amended: per-process accounting is unavailable under WSL2).
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path("/home/jiming/llm-quant-serve")
OUT_DIR = REPO / "results" / "serving"

FILLER = (
    "The engineering team reviewed the quantization results and compared the "
    "measured accuracy across precision tiers, noting that the dominant memory "
    "term at long context is the attention cache rather than the weights. "
)


class VramSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.peak_bytes = 0
        self._handle = None
        self._pynvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._handle is not None

    def snapshot(self) -> int:
        return int(self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used) if self.available else 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self.available:
                try:
                    self.peak_bytes = max(self.peak_bytes, self.snapshot())
                except Exception:
                    pass
            time.sleep(0.02)

    def stop(self) -> None:
        self._stop_event.set()


def one_request(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    """Streaming request; returns per-request timing."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    t0 = time.perf_counter()
    first = None
    stamps = []
    n_tokens = 0
    with requests.post(f"{url}/v1/completions", json=body, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if text:
                now = time.perf_counter()
                if first is None:
                    first = now
                stamps.append(now)
                n_tokens += 1
    t_end = time.perf_counter()
    gaps = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
    return {
        "ttft_ms": (first - t0) * 1000 if first else None,
        "tpot_ms": statistics.mean(gaps) if gaps else None,
        "tokens": n_tokens,
        "wall_ms": (t_end - t0) * 1000,
    }


def pct(vals: list[float], q: float) -> float | None:
    return float(statistics.quantiles(vals, n=100, method="inclusive")[int(q) - 1]) if len(vals) > 1 else (vals[0] if vals else None)


def level(url: str, model: str, concurrency: int, n_requests: int, max_tokens: int,
          sampler: VramSampler) -> dict:
    prompt = FILLER * 8                      # ~200 tokens of controlled prompt
    results: list[dict] = []
    lock = threading.Lock()
    errors: list[str] = []

    def worker(k: int) -> None:
        for i in range(n_requests // concurrency):
            try:
                r = one_request(url, model, prompt, max_tokens, timeout=120)
                with lock:
                    r["worker"] = k
                    results.append(r)
            except Exception as e:
                with lock:
                    errors.append(f"{type(e).__name__}: {str(e)[:120]}")

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ttft = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    tpot = [r["tpot_ms"] for r in results if r["tpot_ms"] is not None]
    tokens = sum(r["tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": errors[:3],
        "n_errors": len(errors),
        "wall_s": round(wall, 2),
        "aggregate_tok_s": round(tokens / wall, 1) if wall else None,
        "ttft_ms": {"p50": pct(ttft, 50), "p95": pct(ttft, 95), "p99": pct(ttft, 99),
                    "mean": statistics.mean(ttft) if ttft else None, "n": len(ttft)},
        "tpot_ms": {"p50": pct(tpot, 50), "p95": pct(tpot, 95), "p99": pct(tpot, 99),
                    "mean": statistics.mean(tpot) if tpot else None, "n": len(tpot)},
        "per_request_tokens_mean": round(tokens / len(results), 1) if results else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen3-8B-fp8-modelopt")
    ap.add_argument("--concurrency", default="1,4,8,16,32")
    ap.add_argument("--requests-per-level", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tag", default="fp8")
    args = ap.parse_args()

    levels = [int(c) for c in args.concurrency.split(",")]
    sampler = VramSampler()
    idle = sampler.snapshot()
    sampler.start()

    # warm up the server: the first call includes kernel autotuning (measured in M1:
    # the first generate took 42 s against 2 s for every later one)
    print(f"warmup: {args.warmup} requests")
    for _ in range(args.warmup):
        try:
            one_request(args.url, args.model, FILLER * 8, 16, timeout=180)
        except Exception as e:
            print("  warmup error:", type(e).__name__, str(e)[:100])

    rows = []
    print(f"\n{'conc':>5s} {'req':>4s} {'tok/s':>8s} {'ttft p50':>9s} {'ttft p95':>9s} "
          f"{'tpot p50':>9s} {'tpot p95':>9s}")
    for c in levels:
        r = level(args.url, args.model, c, args.requests_per_level, args.max_tokens, sampler)
        rows.append(r)
        print(f"{c:5d} {r['requests']:4d} {r['aggregate_tok_s'] or 0:8.1f} "
              f"{r['ttft_ms']['p50'] or 0:9.1f} {r['ttft_ms']['p95'] or 0:9.1f} "
              f"{r['tpot_ms']['p50'] or 0:9.2f} {r['tpot_ms']['p95'] or 0:9.2f}"
              + (f"   errors={r['n_errors']}" if r["n_errors"] else ""))

    sampler.stop()
    peak = sampler.peak_bytes / 2**30 if sampler.available else None
    # NOTE: `idle` here is sampled when the benchmark starts, i.e. AFTER the server has
    # already loaded its weights. The delta therefore captures only the KV growth during
    # the sweep (≈0), not the server's footprint. The absolute peak is the meaningful
    # figure for comparing servers; the delta is kept for completeness.
    doc = {
        "kind": "concurrency_sweep", "tag": args.tag, "url": args.url, "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"requests_per_level": args.requests_per_level,
                   "max_tokens": args.max_tokens, "warmup": args.warmup,
                   "prompt_tokens_approx": 200, "endpoint": "/v1/completions (raw prompt)"},
        "peak_vram_gib": {
            "device_peak_absolute": round(peak, 3) if peak else None,
            "baseline_at_bench_start": round(idle / 2**30, 3),
            "growth_during_sweep": round(peak - idle / 2**30, 3) if peak else None,
            "caveat": "baseline taken with the server already resident; use the absolute "
                      "peak to compare servers, and the engine-reported KV pool for cache "
                      "capacity",
        },
        "levels": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"sweep_{args.tag}.json"
    p.write_text(json.dumps(doc, indent=2))
    print(f"\npeak VRAM: {doc['peak_vram_gib']['device_peak_absolute']} GiB absolute "
          f"(baseline at bench start {doc['peak_vram_gib']['baseline_at_bench_start']} GiB)")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
