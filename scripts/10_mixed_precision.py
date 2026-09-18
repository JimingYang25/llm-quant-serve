#!/usr/bin/env python3
"""M4 step 3 — mixed precision: NVFP4 everywhere, high precision where it hurts.

    python scripts/10_mixed_precision.py --protect 1,12,34,0,35 --ppl-only
    python scripts/10_mixed_precision.py --protect 1,12,34,0,35 --mmlu --export

The allocation problem, stated plainly: uniform NVFP4 costs +2.80 % perplexity
(9.5799 vs 9.3190). The sensitivity probe ranked which blocks carry that damage.
This script spends the budget it just measured: those blocks keep full precision,
everything else stays 4-bit.

Protection goes through the QUANTIZATION CONFIG, and the model is quantized fresh —
never by toggling quantizers on a live model. The first probe version cycled
disable/enable and drifted from 11.05 to 18.53 perplexity in a single cycle.

Reported against three references so the position of this tier is unambiguous:

    BF16 oracle          9.3190    (the accuracy ceiling)
    FP8 W8A8             9.3434    (+0.26 %, 8.8 GiB)
    uniform NVFP4        9.5799    (+2.80 %, ~4.5 GiB)
    mixed (this run)      ?        (should land between FP8 and uniform 4-bit)
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
ACC_DIR = REPO / "results" / "accuracy"
OUT_DIR = REPO / "results" / "quantized"

GATE16_PPL_MAX = 9.785      # decision 16: +5 % over the 9.3190 oracle
GATE16_MMLU_MIN = 0.572     # decision 16: -4 pp


def main() -> int:
    import importlib.util
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import modelopt.torch.quantization as mtq

    spec = importlib.util.spec_from_file_location("acc07", REPO / "scripts" / "07_eval_accuracy.py")
    acc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acc)

    ap = argparse.ArgumentParser()
    ap.add_argument("--protect", required=True, help="comma-separated block indices")
    ap.add_argument("--recipe", default="NVFP4_DEFAULT_CFG")
    ap.add_argument("--ppl-tokens", type=int, default=32768)
    ap.add_argument("--mmlu-limit", type=int, default=500)
    ap.add_argument("--ppl-only", action="store_true")
    ap.add_argument("--mmlu", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--tag", default="nvfp4_mixed")
    args = ap.parse_args()

    protect = [int(x) for x in args.protect.split(",")] if args.protect else []
    print(f"protecting blocks: {protect or '(none — uniform)'}")

    # extension first: the numerical path must be settled before any measurement
    from modelopt.torch.quantization.extensions import get_cuda_ext
    get_cuda_ext(raise_if_failed=False)

    tok = AutoTokenizer.from_pretrained(BF16_DIR)
    texts = json.loads(CALIB.read_text())["texts"]

    model = AutoModelForCausalLM.from_pretrained(
        BF16_DIR, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
    model.eval()

    def forward_loop(m):
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=512)["input_ids"].cuda()
            with torch.no_grad():
                m(input_ids=ids, attention_mask=torch.ones_like(ids))

    cfg = copy.deepcopy(getattr(mtq, args.recipe))
    for b in protect:
        for pattern in (f"*model.layers.{b}*", f"*layers.{b}.*"):
            cfg["quant_cfg"][pattern] = {"enable": False}
    print(f"quantizing with {args.recipe} + {len(protect)} protected blocks …")
    t0 = time.perf_counter()
    model = mtq.quantize(model, cfg, forward_loop=forward_loop)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    result = {
        "tier": args.tag, "kind": "mixed_precision",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recipe": args.recipe, "protected_blocks": protect,
        "calibration": json.loads(CALIB.read_text())["meta"],
        "config": {"ppl_tokens": args.ppl_tokens, "mmlu_limit": args.mmlu_limit},
        "metrics": {},
    }

    # ---- perplexity (always) ----------------------------------------------
    windows, corpus_sha = acc.build_ppl_windows(tok, 2048, args.ppl_tokens)
    ppl = acc.perplexity(model, windows)
    o = json.loads((ACC_DIR / "bf16_20260914T072304Z_oracle.json").read_text())
    oracle_ppl = o["metrics"]["perplexity"]["perplexity"]
    rel = (ppl["perplexity"] / oracle_ppl - 1) * 100
    result["metrics"]["perplexity"] = {**ppl, "corpus_sha256": corpus_sha,
                                       "oracle": oracle_ppl, "rel_pct": round(rel, 3),
                                       "gate_16_max": GATE16_PPL_MAX,
                                       "pass": ppl["perplexity"] <= GATE16_PPL_MAX}
    print(f"\nperplexity {ppl['perplexity']:.4f}  vs oracle {oracle_ppl:.4f}  "
          f"= {rel:+.2f}%   (gate +5% -> {'PASS' if result['metrics']['perplexity']['pass'] else 'FAIL'})")

    # ---- MMLU (optional; it is the coarse guard) --------------------------
    if args.mmlu:
        items, sample_sha = acc.load_mmlu(args.mmlu_limit)
        m = acc.mmlu(model, tok, items)
        oracle_mmlu = o["metrics"]["mmlu"]["accuracy"]
        result["metrics"]["mmlu"] = {
            **{k: v for k, v in m.items() if k != "predictions"},
            "sample_sha256": sample_sha, "oracle": oracle_mmlu,
            "delta_pp": round((m["accuracy"] - oracle_mmlu) * 100, 2),
            "gate_16_min": GATE16_MMLU_MIN,
            "pass": m["accuracy"] >= GATE16_MMLU_MIN,
        }
        print(f"MMLU {m['accuracy']:.4f} vs oracle {oracle_mmlu:.4f} "
              f"({result['metrics']['mmlu']['delta_pp']:+.2f} pp) -> "
              f"{'PASS' if result['metrics']['mmlu']['pass'] else 'FAIL'}")

    if args.export:
        from modelopt.torch.export import export_hf_checkpoint
        d = Path.home() / "models" / f"Qwen3-8B-{args.tag}-modelopt"
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        export_hf_checkpoint(model, export_dir=str(d))
        size = sum(p.stat().st_size for p in d.glob("*.safetensors")) / 2**30
        result["export"] = {"dir": str(d), "gib": round(size, 2),
                            "seconds": round(time.perf_counter() - t0, 1)}
        print(f"exported {size:.2f} GiB -> {d}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = OUT_DIR / f"{args.tag}_{stamp}.json"
    p.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
