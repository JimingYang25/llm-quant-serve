#!/usr/bin/env python3
"""Why is the mixed NVFP4 export (9.57 GiB) LARGER than the FP8 export (8.79 GiB)?

Measured facts to explain:
    BF16                      15.26 GiB
    FP8 W8A8                   8.79 GiB
    mixed NVFP4 + 5 protected  9.57 GiB   <- larger than FP8, despite being 4-bit
    uniform NVFP4              ?          <- not yet exported

Naive arithmetic predicts the mixed export should be ~7 GiB, so the storage format
must not be 0.5 bytes/parameter. This inspects the actual tensors: dtype histogram
and bytes by dtype, per safetensors shard.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

CASES = {
    "bf16": "/home/jiming/models/Qwen3-8B",
    "fp8": "/home/jiming/models/Qwen3-8B-fp8-modelopt",
    "nvfp4_mixed": "/home/jiming/models/Qwen3-8B-nvfp4_mixed-modelopt",
}

try:
    from safetensors import safe_open
except ImportError:
    raise SystemExit("safetensors not installed")

for label, path in CASES.items():
    d = Path(path)
    if not d.exists():
        print(f"\n{label}: MISSING ({path})")
        continue
    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        print(f"\n{label}: no safetensors")
        continue

    by_dtype: dict[str, int] = defaultdict(int)
    by_dtype_count: dict[str, int] = defaultdict(int)
    total_bytes = 0
    samples: dict[str, list[str]] = defaultdict(list)

    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                t = f.get_slice(name)
                dtype = str(t.get_dtype())
                shape = t.get_shape()
                n = 1
                for s in shape:
                    n *= s
                nbytes = n * t.get_dtype().__sizeof__() if hasattr(t.get_dtype(), "__sizeof__") else 0
                # torch dtypes do not expose sizeof reliably here; use a size map
                size_map = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
                            "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
                            "F8_E4M3": 1, "F8_E5M2": 1, "F8_E4M3FN": 1, "BOOL": 1}
                nbytes = n * size_map.get(dtype, 2)
                by_dtype[dtype] += nbytes
                by_dtype_count[dtype] += 1
                total_bytes += nbytes
                if len(samples[dtype]) < 3:
                    samples[dtype].append(name)

    disk = sum(p.stat().st_size for p in shards) / 2**30
    print(f"\n=== {label} ({path}) ===")
    print(f"  shards: {len(shards)}   on-disk: {disk:.2f} GiB   "
          f"tensor bytes: {total_bytes/2**30:.2f} GiB")
    for dtype, b in sorted(by_dtype.items(), key=lambda kv: -kv[1]):
        pct = 100 * b / total_bytes if total_bytes else 0
        print(f"    {dtype:10s} {by_dtype_count[dtype]:5d} tensors  "
              f"{b/2**30:6.2f} GiB  ({pct:5.1f}%)   e.g. {samples[dtype][0] if samples[dtype] else ''}")
