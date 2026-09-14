#!/usr/bin/env python3
"""Verify a quantized checkpoint: files, execution, and equivalence done properly.

    python scripts/check_quantized_checkpoint.py                     # FP8 vs BF16
    python scripts/check_quantized_checkpoint.py --candidate <dir>

Why the equivalence metric is NOT token agreement over a free-running generation:

    Greedy decoding is a feedback loop. A tiny numeric difference flips the argmax
    at some step, and from that step onward EVERY token differs — the two sequences
    have diverged and never re-converge. Measured example: FP8 vs BF16 agreed on a
    long prefix and then differed by one word, which drove overall token agreement
    to 27 % while both outputs remained correct English answering the same question.

    The informative statistics are instead:
      * prefix agreement  — how many leading tokens match (where divergence begins)
      * teacher-forced agreement — same prefix, compare the argmax at each position,
        which does not compound (planned; needs logits from the runtime)

REQUIRES the `if __name__ == "__main__":` guard — the TRT-LLM LLM API spawns MPI
workers and Open MPI aborts if the entry module runs code at import time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("CXX", "/usr/bin/g++")

BF16_DIR = "/home/jiming/models/Qwen3-8B"
PROMPTS = [
    "Name three differences between INT8 and FP8 quantization.",
    "What does time to first token measure?",
    "Explain in one sentence why the KV cache grows with context length.",
]
MAX_TOKENS = 64
KV_FRACTION = 0.35   # pinned so both models allocate comparably; auto-sizing would
                     # otherwise give each model a different share of free memory and
                     # make the memory comparison meaningless


def dir_size_gib(path: str) -> float:
    return sum(p.stat().st_size for p in Path(path).glob("*.safetensors")) / 2**30


def describe(path: str) -> None:
    d = Path(path)
    print(f"  {d.name}: {dir_size_gib(path):.2f} GiB, "
          f"{len(list(d.glob('*.safetensors')))} shards")
    for f in ("hf_quant_config.json", "config.json"):
        if (d / f).exists():
            cfg = json.loads((d / f).read_text())
            q = cfg.get("quantization") or cfg.get("quantization_config")
            if q:
                print(f"    {f}: quant_algo={q.get('quant_algo', q.get('quant_method'))} "
                      f"kv_cache={q.get('kv_cache_quant_algo')} "
                      f"exclude={q.get('exclude_modules', q.get('ignore'))}")


def run(path: str, label: str) -> dict:
    from tensorrt_llm import LLM, SamplingParams

    import time
    t0 = time.perf_counter()
    # NOTE: no KV-cache sizing argument. TRT-LLM 1.2.1 validates LLM(**kwargs)
    # against an internal schema and rejects `kv_cache_free_gpu_memory_fraction`
    # ("LLM got invalid argument"), leaving the engine to auto-size the paged cache
    # from free memory. That makes a raw VRAM comparison between two models unfair —
    # each grabs a different share — so memory is NOT compared here; it is an M5
    # measurement with an explicit KV configuration.
    llm = LLM(model=path, tokenizer=BF16_DIR, max_seq_len=2048, max_batch_size=1)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, SamplingParams(max_tokens=MAX_TOKENS, temperature=0))
    gen_s = time.perf_counter() - t0

    toks = [list(o.outputs[0].token_ids) for o in outs]
    texts = [o.outputs[0].text for o in outs]
    n = sum(len(t) for t in toks)
    print(f"  [{label}] load {load_s:.1f}s | {n} tokens in {gen_s:.1f}s "
          f"({n/gen_s:.1f} tok/s)")
    print(f"  [{label}] output[0]: {texts[0][:120]!r}")
    del llm
    return {"tokens": toks, "texts": texts, "tok_s": n / gen_s, "load_s": load_s}


def prefix_agreement(a: list[list[int]], b: list[list[int]]) -> dict:
    """How far each pair of generations agrees before the first divergence."""
    first_div, matched, total = [], 0, 0
    for x, y in zip(a, b):
        k = 0
        for i, (p, q) in enumerate(zip(x, y)):
            if p != q:
                break
            k += 1
        first_div.append(k)
        matched += k
        total += max(len(x), len(y))
    return {
        "first_divergence_positions": first_div,
        "mean_prefix_tokens_matched": sum(first_div) / len(first_div) if first_div else 0,
        "raw_token_agreement_pct": 100 * matched / total if total else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="/home/jiming/models/Qwen3-8B-fp8-modelopt")
    ap.add_argument("--reference", default=BF16_DIR)
    args = ap.parse_args()

    print("=" * 74)
    print("LEVEL 1 — files")
    print("=" * 74)
    describe(args.reference)
    describe(args.candidate)
    ref_gib, cand_gib = dir_size_gib(args.reference), dir_size_gib(args.candidate)
    print(f"  reduction: {100*(1 - cand_gib/ref_gib):.1f} %")

    print()
    print("=" * 74)
    print("LEVEL 2 — accuracy: see results/quantized/*.json (M2 gate)")
    print("=" * 74)

    print()
    print("=" * 74)
    print("LEVEL 3 — execution under TRT-LLM")
    print("=" * 74)
    cand = run(args.candidate, "candidate")
    ref = run(args.reference, "reference")

    print()
    print("=" * 74)
    print("LEVEL 4 — equivalence (prefix divergence, NOT raw agreement)")
    print("=" * 74)
    pa = prefix_agreement(cand["tokens"], ref["tokens"])
    print(f"  common prefix per prompt : {pa['first_divergence_positions']} tokens "
          f"(of {MAX_TOKENS})")
    print(f"  raw token agreement      : {pa['raw_token_agreement_pct']:.1f} % "
          f"<- misleading metric, shown for contrast")
    print(f"  speedup                  : {cand['tok_s']/ref['tok_s']:.2f}x "
          f"({cand['tok_s']:.1f} vs {ref['tok_s']:.1f} tok/s)")

    out = Path("/home/jiming/llm-quant-serve/results/quantized") / "equivalence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "candidate": args.candidate, "reference": args.reference,
        "size_gib": {"candidate": cand_gib, "reference": ref_gib},
        "speed_tok_s": {"candidate": cand["tok_s"], "reference": ref["tok_s"]},
        "prefix_agreement": pa,
        "max_tokens": MAX_TOKENS, "kv_fraction": KV_FRACTION,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
