#!/usr/bin/env python3
"""M5 — the transfer table: quantization freed weight memory -> cache -> concurrency.

    python scripts/11_transfer_table.py

Reads the two concurrency sweeps and the two servers' engine-reported KV sizing, and
states the chain explicitly:

    weights shrink  ->  KV pool grows  ->  more tokens of cache
                    ->  more concurrent sequences at a fixed context
                    ->  measured throughput and latency at equal concurrency
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path("/home/jiming/llm-quant-serve")
SERVING = REPO / "results" / "serving"

WEIGHTS_GIB = {"fp8": 8.79, "bf16": 15.26}   # measured in M2 / M0
CONTEXT = 4096                                # server max_seq_len
PREFILL_TOKENS = 200                          # benchmark prompt


def kv_from_log(path: Path) -> dict:
    txt = path.read_text(errors="replace")
    tok = re.search(r"max_tokens=(\d+)", txt)
    b = re.search(r"max_gpu_total_bytes=(\d+)", txt)
    frac = re.search(r"free_gpu_memory_fraction=([0-9.]+)", txt)
    return {
        "kv_tokens": int(tok.group(1)) if tok else None,
        "kv_gib": round(int(b.group(1)) / 2**30, 2) if b else None,
        "free_fraction": float(frac.group(1)) if frac else None,
    }


def main() -> int:
    fp8 = json.loads((SERVING / "sweep_fp8.json").read_text())
    bf16 = json.loads((SERVING / "sweep_bf16.json").read_text())
    kv_fp8 = kv_from_log(Path("/home/jiming/serve_fp8.log"))
    kv_bf16 = kv_from_log(Path("/home/jiming/serve_bf16.log"))

    print("=" * 78)
    print("STEP 1 — weights")
    print("=" * 78)
    saved = WEIGHTS_GIB["bf16"] - WEIGHTS_GIB["fp8"]
    print(f"  BF16 {WEIGHTS_GIB['bf16']:.2f} GiB  ->  FP8 {WEIGHTS_GIB['fp8']:.2f} GiB   "
          f"(saved {saved:.2f} GiB, {100*saved/WEIGHTS_GIB['bf16']:.0f} %)")

    print()
    print("=" * 78)
    print("STEP 2 — what the freed memory became: KV cache")
    print("=" * 78)
    print(f"  same free_gpu_memory_fraction = {kv_fp8['free_fraction']}")
    print(f"  BF16: {kv_bf16['kv_tokens']:>7,} tokens  ({kv_bf16['kv_gib']} GiB)")
    print(f"  FP8 : {kv_fp8['kv_tokens']:>7,} tokens  ({kv_fp8['kv_gib']} GiB)")
    ratio = kv_fp8["kv_tokens"] / kv_bf16["kv_tokens"]
    print(f"  -> KV capacity x{ratio:.2f}")

    print()
    print("=" * 78)
    print("STEP 3 — concurrency capacity at the served context length")
    print("=" * 78)
    for label, kv in (("BF16", kv_bf16), ("FP8", kv_fp8)):
        n_full = kv["kv_tokens"] // CONTEXT
        print(f"  {label}: {n_full:>3} concurrent sequences of {CONTEXT} tokens "
              f"(or {kv['kv_tokens'] // PREFILL_TOKENS:,} of the benchmark's "
              f"{PREFILL_TOKENS}-token prompts)")

    print()
    print("=" * 78)
    print("STEP 4 — measured, at equal concurrency")
    print("=" * 78)
    print(f"  {'conc':>5s} {'BF16 tok/s':>11s} {'FP8 tok/s':>10s} {'speedup':>8s} "
          f"{'BF16 TTFT p50':>14s} {'FP8 TTFT p50':>13s} "
          f"{'BF16 TPOT p50':>14s} {'FP8 TPOT p50':>13s}")
    rows = []
    for lv_b, lv_f in zip(bf16["levels"], fp8["levels"]):
        c = lv_b["concurrency"]
        tb, tf = lv_b["aggregate_tok_s"], lv_f["aggregate_tok_s"]
        sp = tf / tb if tb else None
        rows.append({"concurrency": c, "bf16_tok_s": tb, "fp8_tok_s": tf,
                     "speedup": round(sp, 2) if sp else None,
                     "bf16_ttft_p50": lv_b["ttft_ms"]["p50"], "fp8_ttft_p50": lv_f["ttft_ms"]["p50"],
                     "bf16_tpot_p50": lv_b["tpot_ms"]["p50"], "fp8_tpot_p50": lv_f["tpot_ms"]["p50"]})
        print(f"  {c:5d} {tb:11.1f} {tf:10.1f} {sp:7.2f}x "
              f"{lv_b['ttft_ms']['p50']:13.1f} {lv_f['ttft_ms']['p50']:12.1f} "
              f"{lv_b['tpot_ms']['p50']:13.2f} {lv_f['tpot_ms']['p50']:12.2f}")

    print()
    print("=" * 78)
    print("STEP 5 — honest caveats")
    print("=" * 78)
    print(f"  peak VRAM: BF16 {bf16['peak_vram_gib']['device_peak_absolute']} GiB, "
          f"FP8 {fp8['peak_vram_gib']['device_peak_absolute']} GiB — nearly identical,")
    print("  because free_gpu_memory_fraction lets each server take what is available.")
    print("  The saving appears as CACHE CAPACITY, not as lower total VRAM usage.")
    print(f"  FP8 TTFT is WORSE at high concurrency ({fp8['levels'][-1]['ttft_ms']['p50']:.0f} ms "
          f"vs {bf16['levels'][-1]['ttft_ms']['p50']:.0f} ms) despite higher throughput:")
    print("  a larger cache admits more concurrent requests, so prefill queues longer.")

    out = SERVING / "transfer_table.json"
    out.write_text(json.dumps({
        "kind": "quantization_transfer",
        "weights_gib": WEIGHTS_GIB, "saved_gib": round(saved, 2),
        "kv": {"bf16": kv_bf16, "fp8": kv_fp8, "ratio": round(ratio, 2)},
        "context": CONTEXT, "measured": rows,
        "peak_vram": {"bf16": bf16["peak_vram_gib"], "fp8": fp8["peak_vram_gib"]},
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
