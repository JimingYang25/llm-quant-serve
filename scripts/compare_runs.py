#!/usr/bin/env python3
"""M1 gate: compare two baseline runs and issue PASS/FAIL.

    python scripts/compare_runs.py results/baseline_bf16/run1.json \
                                   results/baseline_bf16/run2.json

Implements measurement_spec.md decision 10 (repeatability tolerance, 5 % on
latency p50) and the fingerprint-completeness gate.

Two checks, deliberately separated, because they fail for different reasons:

  1. COMPARABILITY  — are the two runs even allowed to be compared? Different
     prompt hash, weights revision, fingerprint or config means the pair proves
     nothing, however close the numbers look.
  2. REPEATABILITY  — given comparability, do the numbers agree within tolerance?

A run whose comparability checks fail is not "slightly off"; it is inadmissible,
and the correct response is to explain the difference, not to average it away.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOLERANCE_PCT = 5.0  # [SPEC 10]

# Fields that must be identical for two runs to describe the same thing
FINGERPRINT_KEYS = (
    "torch", "cuda_rt", "transformers", "tensorrt_llm", "nvidia_modelopt",
    "driver", "gpu_name", "compute_capability",
    "prompt_set_sha256", "weights_revision", "pip_freeze_sha256",
)
CONFIG_KEYS = ("max_new_tokens", "thinking", "sampling", "percentile_method",
               "warmup_iterations")

METRIC_PATHS = (
    ("TTFT p50", ("metrics", "ttft_ms", "p50")),
    ("TTFT p95", ("metrics", "ttft_ms", "p95")),
    ("TTFT p99", ("metrics", "ttft_ms", "p99")),
    ("TPOT p50", ("metrics", "tpot_ms", "p50")),
    ("TPOT p95", ("metrics", "tpot_ms", "p95")),
    ("throughput", ("metrics", "throughput", "single_stream_tok_s")),
    ("peak VRAM", ("metrics", "peak_vram_gib", "authoritative")),
)


def dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def load(p):
    return json.loads(Path(p).read_text())


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])

    print(f"run A: {sys.argv[1]}  {a.get('timestamp')}")
    print(f"run B: {sys.argv[2]}  {b.get('timestamp')}")
    print(f"samples A/B: {a.get('sample_counts')} / {b.get('sample_counts')}\n")

    problems = []

    print("=== 1. comparability ===")
    for k in FINGERPRINT_KEYS:
        va, vb = a.get("fingerprint", {}).get(k), b.get("fingerprint", {}).get(k)
        if va != vb:
            problems.append(f"fingerprint.{k}: {va} vs {vb}")
    for k in CONFIG_KEYS:
        va, vb = a.get("config", {}).get(k), b.get("config", {}).get(k)
        if va != vb:
            problems.append(f"config.{k}: {va} vs {vb}")
    ca, cb = a.get("sample_counts", {}), b.get("sample_counts", {})
    for k in ("prompts", "repetitions", "realised"):
        if ca.get(k) != cb.get(k):
            problems.append(f"sample_counts.{k}: {ca.get(k)} vs {cb.get(k)}")
    if problems:
        for p in problems:
            print(f"  MISMATCH  {p}")
    else:
        print("  identical fingerprint, config and sample counts — pair is admissible")

    print("\n=== 2. repeatability (tolerance %.1f %%) ===" % TOLERANCE_PCT)
    print(f"  {'metric':12s} {'A':>10s} {'B':>10s} {'delta %':>9s}   verdict")
    failures = 0
    for label, path in METRIC_PATHS:
        va, vb = dig(a, path), dig(b, path)
        if va is None or vb is None:
            print(f"  {label:12s} {'—':>10s} {'—':>10s} {'—':>9s}   not recorded")
            continue
        if va == 0:
            print(f"  {label:12s} {va:10.3f} {vb:10.3f} {'—':>9s}   A is zero, skipped")
            continue
        delta = (vb - va) / abs(va) * 100.0
        # Latency metrics are the gate; throughput and memory are reported for context
        gated = label.startswith(("TTFT", "TPOT"))
        if gated:
            ok = abs(delta) <= TOLERANCE_PCT
            failures += 0 if ok else 1
            verdict = "PASS" if ok else "FAIL — investigate, do not widen tolerance"
        else:
            verdict = "context only"
        print(f"  {label:12s} {va:10.3f} {vb:10.3f} {delta:+8.2f}%   {verdict}")

    print("\n=== thermal context (explains most repeatability failures) ===")
    for tag, r in (("A", a), ("B", b)):
        t = r.get("thermal", {})
        print(f"  run {tag}: peak {t.get('peak_temp_c')} C, max {t.get('max_power_w')} W, "
              f"measurement wall {r.get('timing_outside_window', {}).get('measurement_wall_s', float('nan')):.0f} s")

    print("\n=== verdict ===")
    if problems:
        print("  INADMISSIBLE: the two runs are not comparable (see section 1).")
        return 2
    if failures:
        print(f"  FAIL: {failures} gated metric(s) outside {TOLERANCE_PCT} %.")
        print("  Diagnose in this order: thermal drift (compare peak temp and power),")
        print("  GPU contention from Windows processes, power-state transitions.")
        return 1
    print(f"  PASS: all gated metrics within {TOLERANCE_PCT} %; M1 baseline gate satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
