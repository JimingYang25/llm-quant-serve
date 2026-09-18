#!/usr/bin/env python3
"""M4 sensitivity probe — CORRECTED design.

    python scripts/09_sensitivity.py --blocks 0 --ppl-tokens 4096 --smoke
    python scripts/09_sensitivity.py                       # candidate blocks, 32768 tokens

Why this is a rewrite, not a patch. The first version produced a ranking that was
monotone in PROBE ORDER rather than in block identity, and three independent
confounds were found:

  1. BUDGET MISMATCH (my error). The probe scored 8192 tokens while the oracle used
     32752, then compared the two. Measured: BF16 perplexity is 10.6146 at 8192
     tokens and 9.3190 at 32752 — a 14 % swing with only the budget changed. Any
     cross-budget comparison is invalid (results/accuracy/budget_sensitivity_bf16.json).

  2. DISABLE/ENABLE IS NOT IDEMPOTENT. `disable_quantizer` followed by
     `enable_quantizer` moved perplexity from 11.05 to 18.53 — one cycle, same
     config, same tokens. Cycling it 36 times is what produced the fake ranking.

  3. LAZY EXTENSION LOAD. ModelOpt's CUDA extension loads on first use, so the first
     probes ran on the CPU fallback and the rest on CUDA — a silent numerical switch
     part-way through the experiment.

The corrected method follows EZTrain's width_map instead: **protection is baked into
the quantization CONFIG, and every candidate is a fresh quantization from BF16
weights.** No disable/enable cycling, therefore no state to corrupt. The extension is
forced to load before any measurement, so the numerical path is constant throughout.

Cost: one fresh load + quantize + evaluate per candidate block (~1 minute each).
Probing all 36 is ~35 minutes; the default candidate list is a spread of 12.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("CXX", "/usr/bin/g++")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import modelopt_ext_patch  # noqa: E402,F401

BF16_DIR = "/home/jiming/models/Qwen3-8B"
CALIB = REPO / "data" / "calibration" / "calib_128.json"
OUT = REPO / "results" / "quantized" / "sensitivity_nvfp4_v2.json"

# A spread of candidates rather than all 36 by default: first and last blocks are the
# classic sensitivity hotspots, and every 6th block covers the middle cheaply.
DEFAULT_BLOCKS = [0, 1, 6, 12, 18, 24, 30, 33, 34, 35]


def main() -> int:
    import importlib.util
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import modelopt.torch.quantization as mtq

    spec = importlib.util.spec_from_file_location("acc07", REPO / "scripts" / "07_eval_accuracy.py")
    acc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acc)

    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=",".join(str(b) for b in DEFAULT_BLOCKS))
    ap.add_argument("--recipe", default="NVFP4_DEFAULT_CFG")
    ap.add_argument("--ppl-tokens", type=int, default=32768)
    ap.add_argument("--ppl-window", type=int, default=2048)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.smoke:
        args.blocks = "0"
        args.ppl_tokens = 4096

    # ---- force the numerical path to settle BEFORE any measurement ---------
    print("loading ModelOpt CUDA extension (must happen before measuring) …")
    from modelopt.torch.quantization.extensions import get_cuda_ext
    t0 = time.perf_counter()
    get_cuda_ext(raise_if_failed=False)
    print(f"  extension ready in {time.perf_counter() - t0:.1f}s")

    tok = AutoTokenizer.from_pretrained(BF16_DIR)
    texts = json.loads(CALIB.read_text())["texts"]

    windows, corpus_sha = acc.build_ppl_windows(tok, args.ppl_window, args.ppl_tokens)
    print(f"perplexity budget: {args.ppl_tokens} tokens, corpus sha={corpus_sha[:16]}…")

    def forward_loop(m):
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=512)["input_ids"].cuda()
            with torch.no_grad():
                m(input_ids=ids, attention_mask=torch.ones_like(ids))

    def fresh_model():
        m = AutoModelForCausalLM.from_pretrained(
            BF16_DIR, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
        m.eval()
        return m

    def run_config(protect_block: int | None) -> float:
        """Fresh BF16 model → build a config → quantize → measure. No state carried."""
        model = fresh_model()
        cfg = copy.deepcopy(getattr(mtq, args.recipe))
        if protect_block is not None:
            # Protect by CONFIG, not by toggling: these modules keep full precision.
            for pattern in (f"*model.layers.{protect_block}*",
                            f"*layers.{protect_block}.*"):
                cfg["quant_cfg"][pattern] = {"enable": False}
        model = mtq.quantize(model, cfg, forward_loop=forward_loop)
        p = acc.perplexity(model, windows)["perplexity"]
        del model
        torch.cuda.empty_cache()
        return p

    print("\nbaseline: uniform quantized, nothing protected")
    uniform = run_config(None)
    print(f"  uniform {args.recipe}: ppl {uniform:.4f}")

    blocks = [int(b) for b in args.blocks.split(",")]
    rows = []
    for b in blocks:
        p = run_config(b)
        rows.append({"block": b, "protected_ppl": round(p, 4),
                     "recovered": round(uniform - p, 4)})
        print(f"  protect block {b:2d}: ppl {p:8.4f}   recovers {uniform - p:+.4f}")
        # incremental write: a crash must not discard completed probes
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "kind": "sensitivity_probe_v2", "recipe": args.recipe,
            "ppl_tokens": args.ppl_tokens, "corpus_sha256": corpus_sha,
            "uniform_ppl": round(uniform, 4), "blocks": blocks,
            "method": "protection baked into the quantization config; fresh quantize per candidate",
            "results": rows,
        }, indent=2))

    ranked = sorted(rows, key=lambda r: -r["recovered"])
    print("\n=== ranking (most sensitive first) ===")
    for r in ranked:
        print(f"  block {r['block']:2d}   recovers {r['recovered']:+.4f} ppl")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
