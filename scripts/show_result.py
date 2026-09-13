#!/usr/bin/env python3
"""Print a result file in readable form.

    python scripts/show_result.py                       # newest file
    python scripts/show_result.py results/baseline_bf16/<file>.json

Kept as a script rather than an inline command because inspection happens often
and the shell is a poor place to write JSON walking code.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def newest() -> Path:
    files = sorted((REPO / "results").rglob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit("no result files found")
    return files[-1]


path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest()
d = json.loads(path.read_text())

print(f"file        : {path}")
print(f"tier/runtime: {d.get('tier')} / {d.get('runtime')}")
print(f"timestamp   : {d.get('timestamp')}")
print(f"samples     : {d.get('sample_counts')}")
print(f"config      : {json.dumps(d.get('config', {}), ensure_ascii=False)}")

m = d.get("metrics", {})
for key, label, unit in (("ttft_ms", "TTFT", "ms"), ("tpot_ms", "TPOT", "ms")):
    s = m.get(key, {})
    if s:
        print(f"{label:11s} : p50 {s.get('p50', float('nan')):8.2f} {unit}   "
              f"p95 {s.get('p95', float('nan')):8.2f}   "
              f"p99 {s.get('p99', float('nan')):8.2f}   "
              f"mean {s.get('mean', float('nan')):8.2f}   n={s.get('n')}")

tp = m.get("throughput", {})
if tp:
    print(f"throughput  : {tp.get('single_stream_tok_s', float('nan')):.2f} tok/s "
          f"| total tokens {m.get('generated_tokens_total')}")

vr = m.get("peak_vram_gib", {})
if vr:
    print(f"peak VRAM   : {vr.get('authoritative')} GiB  "
          f"(device {vr.get('device_peak')} - idle {vr.get('idle_device')}, "
          f"method: {vr.get('method')})")
    if vr.get("limitation"):
        print(f"              limitation: {vr['limitation']}")

th = d.get("thermal", {})
print(f"thermal     : peak {th.get('peak_temp_c')} C, max {th.get('max_power_w')} W")
print(f"outside win : {json.dumps(d.get('timing_outside_window', {}))}")

fp = d.get("fingerprint", {})
print("fingerprint :")
for k in ("torch", "cuda_rt", "transformers", "tensorrt_llm", "nvidia_modelopt",
          "driver", "gpu_name", "compute_capability",
          "prompt_set_sha256", "weights_revision", "pip_freeze_sha256"):
    if k in fp:
        print(f"   {k:20s} {fp[k]}")

samples = d.get("per_sample", [])
if samples:
    print(f"\nper-sample (first 5 of {len(samples)}):")
    for s in samples[:5]:
        print(f"   {s.get('prompt_id'):4s} {s.get('bucket'):6s} "
              f"prompt_tok={s.get('prompt_tokens'):5d} gen={s.get('generated_tokens'):4d} "
              f"ttft={s.get('ttft_ms'):7.1f} ms tpot={s.get('tpot_ms') and round(s['tpot_ms'],2)} ms")
