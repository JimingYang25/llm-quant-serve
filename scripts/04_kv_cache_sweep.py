#!/usr/bin/env python3
"""M3 — KV cache quantization and context-length sweep.

    python scripts/04_kv_cache_sweep.py --smoke     # 2 contexts x 2 dtypes, quick
    python scripts/04_kv_cache_sweep.py             # full 4 x 3 grid

What this measures and why it is the whole point of M3:

  M2 quantized the WEIGHTS (42 % smaller artifact). M3 quantizes the OTHER memory
  term — the KV cache — which the weights could not touch. The cache grows linearly
  with every token, so at long context it overtakes the weights and becomes the
  real bottleneck. The deliverable is the curve: for each KV precision, peak VRAM
  as a function of context length.

  KV cache is a RUNTIME choice, not a checkpoint property. The export recorded
  `kv_cache_quant_algo: None`, so the same FP8 checkpoint is run three times with
  KvCacheConfig(dtype=...) = 'auto' (FP16) / 'int8' / 'fp8'.

Two measurement decisions that keep this honest:

  * max_tokens is PINNED across dtypes, so all three cache configurations hold the
    same number of tokens and any VRAM difference is genuinely the precision change,
    not the auto-sizer grabbing different amounts of free memory.
  * Peak VRAM comes from NVML (device peak minus idle baseline), the method agreed
    in M1 after per-process accounting turned out to be unavailable under WSL2.

What this deliberately does NOT do: TTFT/TPOT under concurrency. That is M5, where
`trtllm-bench` reports it authoritatively. Wall time and tok/s are reported here as
context, not as the gated metric.

REQUIRES `if __name__ == "__main__":` (TRT-LLM spawns MPI workers).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("CXX", "/usr/bin/g++")

FP8_DIR = "/home/jiming/models/Qwen3-8B-fp8-modelopt"
BF16_DIR = "/home/jiming/models/Qwen3-8B"
OUT = Path("/home/jiming/llm-quant-serve/results/quantized/kv_sweep.json")

MAX_NEW_TOKENS = 128
MAX_SEQ_LEN = 34000          # enough for 32k context + 128 output
POOL_TOKENS = 34000          # pinned KV pool, identical across dtypes
KV_BYTES_PER_TOKEN = 144     # 2 x 36 layers x 8 kv_heads x 128 x 2 bytes (FP16)

FILLER = (
    "The research team described a systematic method for evaluating the accuracy of "
    "quantized language models under varying context lengths, and reported that the "
    "dominant source of memory growth at long sequences was the attention cache. "
)


class VramSampler(threading.Thread):
    """Device-wide peak VRAM via NVML; per-process accounting is unavailable in WSL2."""

    def __init__(self, poll_interval: float = 0.02) -> None:
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self._poll_interval = float(poll_interval)
        self.peak_bytes = 0
        self._handle = None
        self._pynvml = None
        self._last_error: str | None = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:
            self._last_error = str(exc)

    @property
    def available(self) -> bool:
        return self._handle is not None

    @property
    def error(self) -> str | None:
        return self._last_error

    def _read_used(self) -> int:
        return int(self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)

    def snapshot_bytes(self) -> int:
        return self._read_used() if self.available else 0

    def run(self) -> None:
        if not self.available:
            return
        while not self._stop_event.is_set():
            try:
                used = self._read_used()
                if used > self.peak_bytes:
                    self.peak_bytes = used
            except Exception as exc:
                if self._last_error is None:
                    self._last_error = str(exc)
            time.sleep(self._poll_interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=1.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def make_prompt_token_ids(tok, n_tokens: int) -> list[int]:
    """A coherent filler tiled to exactly n_tokens. Raw ids (not chat-templated):
    this sweep measures the mechanism, and the template's fixed overhead would only
    blur the token counts."""
    ids = tok(FILLER, add_special_tokens=False)["input_ids"]
    ids = (ids * (n_tokens // len(ids) + 1))[:n_tokens]
    return list(ids)


def run_cell(model, prompt_ids: list[int], max_new_tokens: int) -> dict:
    from tensorrt_llm import SamplingParams
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0)
    t0 = time.perf_counter()
    outs = model.generate([{"prompt_token_ids": prompt_ids}], sp)
    wall_s = time.perf_counter() - t0
    o = outs[0].outputs[0]
    return {"wall_s": wall_s, "generated": len(o.token_ids),
            "tok_s": len(o.token_ids) / wall_s if wall_s else 0.0,
            "finish_reason": o.finish_reason}


def write_results(args, results: list[dict], dtypes: list[str], contexts: list[int]) -> None:
    """Write after every cell. A KV-cache dtype can CRASH the worker process rather
    than raise (nvfp4 on sm_120 aborted the whole run), and a crash must not discard
    the cells that already succeeded."""
    doc = {
        "tier": "fp8", "kind": "kv_cache_sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": FP8_DIR, "kv_pool_tokens": POOL_TOKENS,
        "kv_bytes_per_token_fp16": KV_BYTES_PER_TOKEN,
        "max_new_tokens": MAX_NEW_TOKENS, "smoke": args.smoke,
        "requested_kv_dtypes": dtypes, "requested_contexts": contexts,
        "cells": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="1024,4096,16384,32768")
    # Discovered the hard way: with an FP8 checkpoint, TRT-LLM 1.2.1 accepts ONLY
    # ('fp8', 'nvfp4', 'auto') for the KV cache dtype. 'int8' raises:
    #   ValueError: Overriding KV cache quantization with an invalid type "int8".
    # So the grid is fp16(auto) / fp8 / nvfp4 — and nvfp4 doubles as an early probe
    # of whether the Blackwell 4-bit cache path works on sm_120 (an M4 question).
    ap.add_argument("--kv-dtypes", default="auto,fp8,nvfp4")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.smoke:
        args.contexts = "1024,2048"
        args.kv_dtypes = "auto,fp8"

    contexts = [int(c) for c in args.contexts.split(",")]
    dtypes = args.kv_dtypes.split(",")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BF16_DIR)

    sampler = VramSampler()
    idle = sampler.snapshot_bytes() if sampler.available else 0
    if sampler.available:
        sampler.start()

    results = []
    try:
        for dtype in dtypes:
            from tensorrt_llm import LLM
            from tensorrt_llm.llmapi import KvCacheConfig

            kv = KvCacheConfig(max_tokens=POOL_TOKENS, dtype=dtype)
            t0 = time.perf_counter()
            try:
                llm = LLM(model=FP8_DIR, tokenizer=BF16_DIR, max_seq_len=MAX_SEQ_LEN,
                          max_batch_size=1, max_num_tokens=MAX_SEQ_LEN,
                          kv_cache_config=kv)
            except Exception as e:
                # A dtype the stack refuses (e.g. a 4-bit cache path unavailable on
                # this architecture) is a finding, not a crash. Record and continue.
                reason = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"\n[kv={dtype}] LOAD FAILED: {reason}")
                for nctx in contexts:
                    results.append({
                        "kv_dtype": dtype, "context_tokens": nctx, "generated": 0,
                        "tok_s": 0.0, "wall_s": None, "finish_reason": "load_failed",
                        "peak_vram_gib": None,
                        "predicted_kv_gib": round(nctx * KV_BYTES_PER_TOKEN / 2**30, 3),
                        "error": reason,
                    })
                continue
            load_s = time.perf_counter() - t0
            print(f"\n[kv={dtype}] loaded in {load_s:.1f}s")

            for nctx in contexts:
                prompt_ids = make_prompt_token_ids(tok, nctx)
                before = sampler.snapshot_bytes() if sampler.available else 0
                try:
                    cell = run_cell(llm, prompt_ids, MAX_NEW_TOKENS)
                except Exception as e:
                    # One failing cell must not destroy the sweep: the failure itself
                    # is a result (e.g. a context the engine refuses without an
                    # explicit max_num_tokens budget).
                    reason = f"{type(e).__name__}: {str(e)[:160]}"
                    results.append({
                        "kv_dtype": dtype, "context_tokens": nctx,
                        "generated": 0, "tok_s": 0.0, "wall_s": None,
                        "finish_reason": "error", "peak_vram_gib": None,
                        "predicted_kv_gib": round(nctx * KV_BYTES_PER_TOKEN / 2**30, 3),
                        "error": reason,
                    })
                    print(f"  ctx={nctx:6d}  ERROR  {reason[:110]}")
                    continue
                peak = sampler.peak_bytes / 2**30 if sampler.available else None
                results.append({
                    "kv_dtype": dtype, "context_tokens": nctx,
                    "generated": cell["generated"], "tok_s": round(cell["tok_s"], 1),
                    "wall_s": round(cell["wall_s"], 2),
                    "finish_reason": cell["finish_reason"],
                    "peak_vram_gib": round(peak - idle / 2**30, 3) if peak else None,
                    "predicted_kv_gib": round(nctx * KV_BYTES_PER_TOKEN / 2**30, 3),
                })
                print(f"  ctx={nctx:6d}  gen={cell['generated']:3d}  "
                      f"{cell['tok_s']:5.1f} tok/s  peak={results[-1]['peak_vram_gib']} GiB  "
                      f"[{cell['finish_reason']}]")
                write_results(args, results, dtypes, contexts)   # survive a crash
            del llm
    finally:
        sampler.stop()

    doc = {
        "tier": "fp8", "kind": "kv_cache_sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": FP8_DIR, "kv_pool_tokens": POOL_TOKENS,
        "kv_bytes_per_token_fp16": KV_BYTES_PER_TOKEN,
        "max_new_tokens": MAX_NEW_TOKENS, "smoke": args.smoke,
        "cells": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
