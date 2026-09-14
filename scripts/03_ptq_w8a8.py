#!/usr/bin/env python3
"""M2 — post-training quantization to W8A8 (and later FP8 / 4-bit), with the gate.

    # shape check: 8 calibration samples, 20 MMLU items, no export (a few minutes)
    python scripts/03_ptq_w8a8.py --method int8 --smoke

    # the real M2 run (this is the one command M2 needs)
    python scripts/03_ptq_w8a8.py --method int8 --tag w8a8

What it does, in order:

  1. Loads the frozen BF16 checkpoint.
  2. Runs calibration — a few hundred samples of real text through the model so
     every quantizer can observe the actual range of values it must represent.
     This is the whole substance of PTQ: quantization scales are chosen from
     observed statistics, not from theory.
  3. Inserts fake-quantization (QDQ) and evaluates the model.
  4. Exports the quantized checkpoint for later serving, and writes a result file
     containing the accuracy deltas against the BF16 oracle plus a gate verdict.

Two design decisions worth understanding:

  * Accuracy is measured on the IN-MEMORY quantized model, using exactly the same
    functions that produced the oracle (imported from 07_eval_accuracy.py). This is
    what makes the comparison legitimate: same evaluator, same questions, same
    corpus. The exported checkpoint is for serving, not for scoring.
  * The calibration corpus is a DIFFERENT source from the evaluation corpus
    (measurement_spec.md decision 13). Both are hashed and recorded, so no later
    reader has to trust that claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# MUST precede any modelopt import: makes ModelOpt's CUDA extensions buildable rather
# than silently falling back to the CPU implementation (see the module for the
# two-defect diagnosis). Without it, FP8 produces NaN and INT8 numbers are suspect.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelopt_ext_patch  # noqa: E402,F401

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = Path("/home/jiming/models/Qwen3-8B")
ACCURACY_DIR = REPO / "results" / "accuracy"
OUT_DIR = REPO / "results" / "quantized"

# ---- M2 gates, agreed before the measurement existed -----------------------
# Perplexity is dense (32k tokens) and can resolve sub-percent change; MMLU at
# N=500 resolves only ~4 percentage points, so it acts as a coarse guard.
GATE_PPL_REL_PCT = 2.0     # perplexity may rise by at most this much
GATE_MMLU_DROP_PP = 4.0    # MMLU may fall by at most this many percentage points

# ---- quantization recipes --------------------------------------------------
RECIPES = {
    "int8": "INT8_DEFAULT_CFG",            # W8A8, per-tensor int8
    "int8_sq": "INT8_SMOOTHQUANT_CFG",     # W8A8 with activation smoothing
    "fp8": "FP8_DEFAULT_CFG",              # W8A8 float8 (E4M3), Blackwell native
    "nvfp4": "NVFP4_DEFAULT_CFG",          # 4-bit float, Blackwell
    "int4_awq": "INT4_AWQ_CFG",            # W4A16 weight-only
}

CALIB_CANDIDATES = [
    ("tatsu-lab/alpaca", None, "train"),
    ("databricks/databricks-dolly-15k", None, "train"),
    ("Salesforce/wikitext", "wikitext-103-raw-v1", "train"),   # fallback, same family as eval
]


def load_accuracy_module():
    """Import 07_eval_accuracy.py by path so its mathematics is reused verbatim."""
    path = REPO / "scripts" / "07_eval_accuracy.py"
    spec = importlib.util.spec_from_file_location("acc07", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ==========================================================================
# Calibration corpus
# ==========================================================================
def load_calibration_texts(n: int) -> tuple[list[str], dict]:
    from datasets import load_dataset
    for name, config, split in CALIB_CANDIDATES:
        try:
            ds = load_dataset(name, config, split=split) if config else \
                 load_dataset(name, split=split)
            rows = []
            for r in ds:
                if "text" in r and r["text"]:
                    rows.append(r["text"])
                elif {"instruction", "output"} <= set(r.keys()):
                    parts = [r.get("instruction", ""), r.get("input", ""), r.get("output", "")]
                    rows.append("\n".join(p for p in parts if p))
                if len(rows) >= n:
                    break
            if rows:
                blob = "\n\n".join(rows)
                return rows, {
                    "dataset": name, "config": config, "split": split,
                    "n_samples": len(rows), "corpus_sha256": sha256_text(blob),
                    "chars": len(blob),
                }
        except Exception as e:
            print(f"  calibration source {name} unavailable ({type(e).__name__}); trying next")
    raise RuntimeError("no calibration corpus could be loaded")


def make_forward_loop(tok, texts: list[str], seq_len: int):
    """One forward pass per calibration sample; no labels, no generation.

    Batch size 1 keeps memory predictable and avoids the pad==eos trap. The loop is
    what the quantizers observe, so its content is part of the measurement — hence
    the recorded hash.
    """
    batches = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=seq_len)["input_ids"]
        batches.append(ids.to("cuda"))

    def forward_loop(model):
        for ids in batches:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids))
    return forward_loop, len(batches)


# ==========================================================================
# Export
# ==========================================================================
def export_checkpoint(model, export_dir: Path) -> str:
    from modelopt.torch.export import export_hf_checkpoint
    export_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    export_hf_checkpoint(model, export_dir=str(export_dir))
    return f"{time.perf_counter() - t0:.1f}s -> {export_dir}"


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="M2 post-training quantization")
    ap.add_argument("--method", default="int8", choices=sorted(RECIPES))
    ap.add_argument("--calib-samples", type=int, default=None,
                    help="calibration samples (default: 8 in smoke mode, 128 otherwise)")
    ap.add_argument("--calib-seq-len", type=int, default=512)
    ap.add_argument("--export-dir", default="")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--oracle", default="", help="BF16 accuracy file to compare against")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 8 calibration samples, 20 MMLU items, no export")
    ap.add_argument("--control", action="store_true",
                    help="calibrate and insert quantizers, then DISABLE them: the result "
                         "must reproduce the BF16 oracle. Separates 'the quantization "
                         "scheme is bad' from 'the measurement pipeline is broken'.")
    ap.add_argument("--ppl-only", action="store_true", help="skip MMLU (fast iteration)")
    ap.add_argument("--mmlu-only", action="store_true", help="skip perplexity")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    # --smoke reduces EVALUATION cost (fewer MMLU items, smaller perplexity budget,
    # no 8 GB export). It must not silently shrink the calibration set, because the
    # number of calibration samples is the main determinant of quantization quality:
    # confining calibration to 8 samples produced a +55 % perplexity change, which
    # says nothing about W8A8 and everything about calibrating on eight short texts.
    # `None` as the sentinel is what makes an explicit --calib-samples 128 survive.
    if args.calib_samples is None:
        args.calib_samples = 8 if args.smoke else 128
    if args.smoke:
        args.skip_export = True

    tag = args.tag or args.method
    print(f"method={args.method} ({RECIPES[args.method]})  tag={tag}  smoke={args.smoke}")

    acc = load_accuracy_module()

    # ---- oracle to compare against ----------------------------------------
    oracle_path = Path(args.oracle) if args.oracle else None
    if oracle_path is None:
        cands = sorted(ACCURACY_DIR.glob("bf16_*oracle*.json"))
        oracle_path = cands[-1] if cands else None
    oracle = json.loads(oracle_path.read_text()) if oracle_path else None
    if oracle is None:
        print("WARNING: no BF16 oracle found; deltas cannot be computed. "
              "Run scripts/07_eval_accuracy.py --tier bf16 first.")
    else:
        print(f"oracle: {oracle_path.name}  ppl={oracle['metrics']['perplexity']['perplexity']:.4f} "
              f"mmlu={oracle['metrics']['mmlu']['accuracy']:.4f}")

    # ---- model -------------------------------------------------------------
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
    )
    model.eval()
    print(f"loaded BF16 model in {time.perf_counter() - t0:.1f}s")

    # ---- calibration -------------------------------------------------------
    print(f"\ncalibration: loading {args.calib_samples} samples")
    texts, calib_meta = load_calibration_texts(args.calib_samples)
    print(f"  source={calib_meta['dataset']} split={calib_meta['split']} "
          f"n={calib_meta['n_samples']} sha={calib_meta['corpus_sha256'][:16]}…")
    forward_loop, n_batches = make_forward_loop(tok, texts, args.calib_seq_len)

    # ---- quantize ----------------------------------------------------------
    import modelopt.torch.quantization as mtq
    cfg = getattr(mtq, RECIPES[args.method])
    print(f"\nquantizing with {RECIPES[args.method]} ({n_batches} calibration passes, "
          f"seq_len {args.calib_seq_len})")
    t0 = time.perf_counter()
    model = mtq.quantize(model, cfg, forward_loop=forward_loop)
    quant_s = time.perf_counter() - t0
    print(f"quantization applied in {quant_s:.1f}s")
    try:
        mtq.print_quant_summary(model)
    except Exception as e:
        print("  (summary unavailable:", type(e).__name__, ")")

    # ---- control: disable the quantizers, keep everything else identical -----
    # If this does not reproduce the BF16 oracle, the pipeline (not the scheme) is at
    # fault, and every quantization number produced by this script is suspect.
    if args.control:
        from modelopt.torch.quantization import disable_quantizer
        disable_quantizer(model, "*")   # wildcard: every quantizer in the model
        print("\nCONTROL MODE: quantizers inserted but disabled — "
              "the result should match the BF16 oracle (perplexity ≈ 9.32)")

    # ---- evaluate with the SAME functions that produced the oracle ---------
    # Deliberately not using 07's CLI: the mathematics is imported and reused verbatim,
    # which is what makes the comparison against the oracle legitimate.
    print("\naccuracy on the quantized model:")
    ppl_budget = 32768 if not args.smoke else 4096
    mmlu_limit = 500 if not args.smoke else 20

    ppl = None
    if not args.mmlu_only:
        windows, corpus_sha = acc.build_ppl_windows(tok, 2048, ppl_budget)
        ppl = acc.perplexity(model, windows)
        print(f"  perplexity {ppl['perplexity']:.4f} over {ppl['tokens_scored']} tokens")

    mmlu_res = None
    if not args.ppl_only:
        items, sample_sha = acc.load_mmlu(mmlu_limit)
        mmlu_res = acc.mmlu(model, tok, items)
        print(f"  MMLU {mmlu_res['accuracy']:.4f} ± {mmlu_res['standard_error']:.4f} "
              f"(n={mmlu_res['n_questions']}, chance {mmlu_res['chance_level']:.2f})")

    # ---- export ------------------------------------------------------------
    export_note = "skipped"
    if not args.skip_export:
        export_dir = Path(args.export_dir) if args.export_dir else \
            Path.home() / "models" / f"Qwen3-8B-{tag}-modelopt"
        print(f"\nexporting quantized checkpoint")
        export_note = export_checkpoint(model, export_dir)
        print(f"  exported in {export_note}")

    # ---- gate --------------------------------------------------------------
    # A verdict is only admissible if BOTH sides were measured the same way. Smoke
    # mode scales the evaluation down (4 096 perplexity tokens, 20 MMLU items) while
    # the oracle used 32 752 tokens and 500 items, so comparing them measures the
    # change in sample size, not the change in the model. That mistake nearly put a
    # "-16.2 pp MMLU" conclusion in the record when the 20-item confidence interval
    # is ±22 pp.
    verdict = {"status": "NO_ORACLE", "checks": [], "comparable": False}
    if oracle is not None:
        o_ppl = oracle["metrics"]["perplexity"]
        o_mmlu = oracle["metrics"]["mmlu"]
        comparable = True
        reasons = []
        if ppl is not None and ppl["tokens_scored"] != o_ppl["tokens_scored"]:
            comparable = False
            reasons.append(f"perplexity tokens {ppl['tokens_scored']} vs oracle "
                           f"{o_ppl['tokens_scored']}")
        if mmlu_res is not None and mmlu_res["n_questions"] != o_mmlu["n_questions"]:
            comparable = False
            reasons.append(f"MMLU items {mmlu_res['n_questions']} vs oracle "
                           f"{o_mmlu['n_questions']}")
        if args.ppl_only or args.mmlu_only:
            comparable = False
            reasons.append("partial evaluation (--ppl-only/--mmlu-only)")

        checks = []
        if ppl is not None:
            rel = (ppl["perplexity"] - o_ppl["perplexity"]) / o_ppl["perplexity"] * 100.0
            checks.append({"metric": "perplexity", "oracle": o_ppl["perplexity"],
                           "measured": ppl["perplexity"], "delta_rel_pct": round(rel, 3),
                           "limit_pct": GATE_PPL_REL_PCT, "pass": rel <= GATE_PPL_REL_PCT})
        if mmlu_res is not None:
            drop = (o_mmlu["accuracy"] - mmlu_res["accuracy"]) * 100.0
            checks.append({"metric": "mmlu", "oracle": o_mmlu["accuracy"],
                           "measured": mmlu_res["accuracy"], "delta_pp": round(-drop, 3),
                           "limit_drop_pp": GATE_MMLU_DROP_PP,
                           "pass": drop <= GATE_MMLU_DROP_PP})

        all_pass = all(c["pass"] for c in checks) if checks else False
        verdict = {
            "status": ("PASS" if all_pass else "FAIL") if comparable else "PROVISIONAL",
            "comparable": comparable,
            "not_comparable_because": reasons,
            "checks": checks,
        }

        print("\n=== M2 gate ===")
        for c in checks:
            if c["metric"] == "perplexity":
                print(f"  perplexity {c['oracle']:.4f} -> {c['measured']:.4f}  "
                      f"{c['delta_rel_pct']:+.2f}% (limit +{c['limit_pct']}%)  "
                      f"{'PASS' if c['pass'] else 'FAIL'}")
            else:
                print(f"  MMLU       {c['oracle']:.4f} -> {c['measured']:.4f}  "
                      f"{c['delta_pp']:+.2f} pp (limit -{c['limit_drop_pp']} pp)  "
                      f"{'PASS' if c['pass'] else 'FAIL'}")
        print(f"  VERDICT: {verdict['status']}")
        if not comparable:
            print("  NOT ADMISSIBLE: this run was evaluated differently from the oracle.")
            for r in reasons:
                print(f"    - {r}")
            print("  Re-run without --smoke for a verdict that counts.")

    # ---- record ------------------------------------------------------------
    result = {
        "tier": tag,
        "method": args.method,
        "recipe": RECIPES[args.method],
        "kind": "quantized_accuracy",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": {
            "torch": torch.__version__,
            "transformers": acc.package_version("transformers"),
            "nvidia_modelopt": acc.package_version("nvidia-modelopt"),
            "gpu_name": torch.cuda.get_device_name(0),
            "pip_freeze_sha256": acc.fingerprint(MODEL_DIR)["pip_freeze_sha256"],
            "weights_digest": acc.fingerprint(MODEL_DIR)["weights_digest"],
            "prompt_set_sha256": json.loads(
                (REPO / "prompts" / "prompt_set.stats.json").read_text()
            )["prompt_set_sha256"],
        },
        "config": {
            "calibration": calib_meta,
            "calib_seq_len": args.calib_seq_len,
            "smoke": args.smoke,
            "quantization_seconds": round(quant_s, 1),
            "export": export_note,
        },
        "metrics": {
            "perplexity": ({**ppl, "corpus_sha256": corpus_sha} if ppl else None),
            "mmlu": ({**mmlu_res, "sample_sha256": sample_sha} if mmlu_res else None),
        },
        "oracle_file": oracle_path.name if oracle_path else None,
        "gate": verdict,
    }
    for k in ("perplexity", "mmlu"):
        if result["metrics"][k]:
            result["metrics"][k].pop("predictions", None)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    smoke = "_smoke" if args.smoke else ""
    path = OUT_DIR / f"{tag}_{stamp}{smoke}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    # PROVISIONAL is not a failure: the run is simply not admissible as a verdict.
    if verdict["status"] in ("PASS", "NO_ORACLE", "PROVISIONAL"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
