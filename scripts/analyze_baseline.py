#!/usr/bin/env python3
"""Per-bucket breakdown of a baseline result.

    python scripts/analyze_baseline.py results/baseline_bf16/<run>.json [more...]

Why this exists: the workload mixes three prompt lengths, so a pooled TTFT
percentile is a mixture of three different regimes. Pooling hides the mechanism
(prefill cost grows with prompt length; decode cost does not). Splitting by
bucket turns one number into an explanation, and checks the hypothesis that
TTFT is prefill-dominated while TPOT is roughly length-invariant.

Also reports per-repetition drift, which is how a thermal ramp inside a run
would announce itself.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METHOD = "linear"  # must match the spec, or the numbers are not comparable
BUCKETS = ("short", "medium", "long")


def pct(vals, q):
    return float(np.percentile(vals, q, method=METHOD)) if vals else float("nan")


def analyze(path: Path) -> None:
    d = json.loads(path.read_text())
    samples = d.get("per_sample", [])
    cfg = d.get("config", {})
    print(f"\n{'='*78}\n{path.name}   tier={d.get('tier')}   "
          f"max_new_tokens={cfg.get('max_new_tokens')}   thinking={cfg.get('thinking')}")

    by_bucket = defaultdict(list)
    for s in samples:
        by_bucket[s["bucket"]].append(s)

    print(f"\n{'bucket':8s} {'n':>4s} {'prompt_tok':>11s} {'gen_tok':>8s} "
          f"{'TTFT p50':>9s} {'TTFT p95':>9s} {'TPOT p50':>9s} {'TPOT p95':>9s} {'tok/s':>7s}")
    for b in BUCKETS:
        rows = by_bucket.get(b, [])
        if not rows:
            continue
        pt = [r["prompt_tokens"] for r in rows]
        gt = [r["generated_tokens"] for r in rows]
        ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
        tpot = [r["tpot_ms"] for r in rows if r["tpot_ms"] is not None]
        tps = np.mean([r["generated_tokens"] / (r["wall_ms"] / 1000.0) for r in rows])
        print(f"{b:8s} {len(rows):4d} {np.mean(pt):11.0f} {np.mean(gt):8.1f} "
              f"{pct(ttft,50):9.1f} {pct(ttft,95):9.1f} {pct(tpot,50):9.2f} "
              f"{pct(tpot,95):9.2f} {tps:7.1f}")

    # Pooled, for contrast with the per-bucket rows
    all_ttft = [s["ttft_ms"] for s in samples if s["ttft_ms"] is not None]
    all_tpot = [s["tpot_ms"] for s in samples if s["tpot_ms"] is not None]
    print(f"{'POOLED':8s} {len(samples):4d} {'':11s} {'':8s} "
          f"{pct(all_ttft,50):9.1f} {pct(all_ttft,95):9.1f} {pct(all_tpot,50):9.2f} "
          f"{pct(all_tpot,95):9.2f}")

    # Is TTFT proportional to prompt length? Prefill is ~O(n); decode is ~O(1) per token.
    shorts = [r["ttft_ms"] for r in by_bucket.get("short", []) if r["ttft_ms"]]
    longs = [r["ttft_ms"] for r in by_bucket.get("long", []) if r["ttft_ms"]]
    n_short = np.mean([r["prompt_tokens"] for r in by_bucket.get("short", [])]) if shorts else 0
    n_long = np.mean([r["prompt_tokens"] for r in by_bucket.get("long", [])]) if longs else 0
    if shorts and longs and n_long > n_short:
        slope = (np.median(longs) - np.median(shorts)) / (n_long - n_short)
        intercept = np.median(shorts) - slope * n_short
        print(f"\nprefill fit: TTFT ≈ {intercept:.1f} ms + {slope*1000:.1f} µs x prompt_tokens")
        print(f"  (extrapolated fixed overhead {intercept:.1f} ms is framework and launch cost)")

    tpot_short = [r["tpot_ms"] for r in by_bucket.get("short", []) if r["tpot_ms"]]
    tpot_long = [r["tpot_ms"] for r in by_bucket.get("long", []) if r["tpot_ms"]]
    if tpot_short and tpot_long:
        delta = (np.median(tpot_long) - np.median(tpot_short)) / np.median(tpot_short) * 100
        print(f"\nTPOT short vs long: {np.median(tpot_short):.2f} vs {np.median(tpot_long):.2f} ms "
              f"({delta:+.1f}%)  -> decode is {'length-invariant' if abs(delta) < 5 else 'length-dependent'}")

    # Drift across repetitions
    print("\nper-repetition TTFT p50 (thermal/ramp detector):")
    by_rep = defaultdict(list)
    for s in samples:
        if s["ttft_ms"] is not None:
            by_rep[s["rep"]].append(s["ttft_ms"])
    for rep in sorted(by_rep):
        print(f"  rep {rep+1}: {pct(by_rep[rep],50):7.1f} ms  (n={len(by_rep[rep])})")

    print(f"\nrealised samples: {len(samples)}   peak VRAM (authoritative): "
          f"{d.get('metrics',{}).get('peak_vram_gib',{}).get('authoritative')} GiB   "
          f"thermal peak {d.get('thermal',{}).get('peak_temp_c')} C / "
          f"{d.get('thermal',{}).get('max_power_w')} W")


for arg in sys.argv[1:]:
    analyze(Path(arg))
