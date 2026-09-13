#!/usr/bin/env python3
"""M1 baseline oracle — Qwen3-8B BF16, full prompt set, latency and memory.

This file is the executable form of docs/measurement_spec.md. Constants are
tagged [SPEC n] with the decision number they implement; if the spec and this
file ever disagree, the spec wins and this file is the bug.

Design rules, in the order they matter:

  1. Loading and first-call compilation happen BEFORE t0 is taken. The load is
     ~4.4 s and the first generate() call selects cuBLAS kernels, so neither may
     fall inside a measurement window.
  2. Warmup is positional, not statistical: it runs before the loop, not as the
     first N samples discarded afterwards. Discarding samples does not remove
     the contamination, because the arrays and kernels are already warm for the
     later prompts.
  3. TTFT is measured at the first decoding step, not from a streamed chunk,
     because chunk granularity biases TTFT upward.
  4. Peak VRAM comes from NVML, not from torch.cuda.max_memory_allocated. The
     torch allocator cannot see TensorRT's allocations, so using it would make
     the later quantized tiers look artificially better [SPEC 7].
  5. Provenance is read from disk (prompt hash, weight shards, pip freeze) and
     asserted, never typed by hand.

Usage:
    # fast shape check, seconds
    python scripts/02_baseline_bf16.py --prompts 3 --reps 1

    # full run (30 prompts x 5 reps, roughly 10-20 minutes)
    python scripts/02_baseline_bf16.py

Environment variables:
    NVIDIA_MEM_SAMPLE_MS   NVML sampling interval, default 20 ms
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = Path("/home/jiming/models/Qwen3-8B")
PROMPT_SET = REPO / "prompts" / "prompt_set.json"
PROMPT_STATS = REPO / "prompts" / "prompt_set.stats.json"
RESULTS_DIR = REPO / "results" / "baseline_bf16"
HF_REF = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/refs/main"

# --------------------------------------------------------------------------
# Frozen protocol constants — see docs/measurement_spec.md
# --------------------------------------------------------------------------
TIER = "bf16"
RUNTIME = "hf_transformers"
MAX_NEW_TOKENS = 128        # [SPEC 2] latency runs
WARMUP_ITERATIONS = 3       # [SPEC 8]
REPETITIONS = 5             # [SPEC 9]
THINKING = False            # [SPEC 5] thinking disabled for every tier
SEED = 0                    # [SPEC 4] inert while greedy, recorded anyway
PERCENTILE_METHOD = "linear"  # must be stated, or p95/p99 are not comparable
DEVICE = "cuda"


# ==========================================================================
# Provenance — read, do not declare
# ==========================================================================
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pip_freeze_sha256() -> str:
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True, check=True).stdout
    return hashlib.sha256(out.encode()).hexdigest()


def weights_digest() -> tuple[str, str]:
    """Return (revision, digest over the checkpoint's shard digests)."""
    revision = HF_REF.read_text().strip() if HF_REF.exists() else "unknown"
    sums = MODEL_DIR / "SHA256SUMS"
    digest = "unavailable"
    if sums.exists():
        digest = hashlib.sha256(
            "\n".join(sorted(sums.read_text().splitlines())).encode()
        ).hexdigest()
    return revision, digest


def package_version(name: str) -> str:
    import importlib.metadata as md
    try:
        return md.version(name)
    except Exception:
        return "absent"


def build_fingerprint(prompt_sha: str) -> dict:
    """[SPEC 4] Everything needed to decide whether two result files are comparable."""
    revision, w_digest = weights_digest()
    return {
        "torch": torch.__version__,
        "cuda_rt": torch.version.cuda,
        "transformers": package_version("transformers"),
        "tensorrt": package_version("tensorrt"),
        "tensorrt_llm": package_version("tensorrt_llm"),
        "nvidia_modelopt": package_version("nvidia-modelopt"),
        "driver": _driver_version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": "sm_%d%d" % torch.cuda.get_device_capability(0),
        "pip_freeze_sha256": pip_freeze_sha256(),
        "prompt_set_sha256": prompt_sha,
        "weights_revision": revision,
        "weights_digest": w_digest,
    }


def _driver_version() -> str:
    try:
        import pynvml
        pynvml.nvmlInit()
        v = pynvml.nvmlSystemGetDriverVersion()
        return v.decode() if isinstance(v, bytes) else str(v)
    except Exception:
        return "unavailable"


# ==========================================================================
# Workload — resolve the prompt set exactly as its contract specifies
# ==========================================================================
def load_prompt_set() -> list[dict]:
    data = json.loads(PROMPT_SET.read_text(encoding="utf-8"))
    docs = data["documents"]
    resolved = []
    for p in data["prompts"]:
        paragraphs = []
        for ref in p.get("doc_refs", []):
            section = docs[ref]["paragraphs"]
            if p.get("paragraph_slice"):
                start, end = p["paragraph_slice"]
                section = section[start:end]
            paragraphs.extend(section)
        paragraphs.append(p["instruction"])
        resolved.append({
            "id": p["id"],
            "bucket": p["bucket"],
            "language": p["language"],
            "text": "\n\n".join(paragraphs),
        })
    return resolved


def build_inputs(tok, text: str) -> dict:
    """Chat template with thinking disabled [SPEC 5].

    We request the attention mask explicitly: Qwen3 has no dedicated pad token,
    so pad_token_id == eos_token_id and the mask cannot be inferred. At batch 1
    nothing is padded, but relying on that would break the moment batching is
    introduced, and the failure mode is plausible-looking wrong output.
    """
    kwargs = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    try:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}], enable_thinking=THINKING, **kwargs
        )
    except TypeError:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], **kwargs)
    return {k: v.to(DEVICE) for k, v in ids.items()}


# ==========================================================================
# Instrumentation
# ==========================================================================
class StepTimer(LogitsProcessor):
    """One timestamp per decoding step.

    The first invocation happens after the prefill forward pass has produced the
    first token, so t[0] - t_start is TTFT. Subsequent differences are
    inter-token latencies, whose mean is TPOT.
    """

    def __init__(self) -> None:
        self.timestamps: list[float] = []

    def __call__(self, input_ids, scores):
        self.timestamps.append(time.perf_counter())
        return scores


class VramSampler(threading.Thread):
    """[SPEC 7] Peak GPU memory sampled through NVML.

    Two numbers are kept: the device total (comparable across machines, but
    includes every other process, e.g. the Windows desktop compositor in WSL)
    and this process's own usage. The idle baseline is taken before the model
    loads so the device figure can later be corrected for background load.
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        # NOTE: do not name this `_stop`. threading.Thread defines an internal
        # method called `_stop`, and an instance attribute with that name shadows
        # it, making Thread.join() raise "'Event' object is not callable".
        self._stop_event = threading.Event()
        self.peak_process_bytes = 0
        self.peak_device_bytes = 0
        self.peak_temp_c = 0
        self.max_power_w = 0
        self._handle = None
        self._pid = os.getpid()
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._pynvml = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def snapshot(self) -> dict:
        if not self.available:
            return {}
        p = self._pynvml
        return {
            "device_used_bytes": p.nvmlDeviceGetMemoryInfo(self._handle).used,
            "temp_c": p.nvmlDeviceGetTemperature(self._handle, p.NVML_TEMPERATURE_GPU),
            "power_w": p.nvmlDeviceGetPowerUsage(self._handle) / 1000.0,
            "clock_mhz": p.nvmlDeviceGetClockInfo(self._handle, p.NVML_CLOCK_SM),
        }

    def run(self) -> None:
        if not self.available:
            return
        p = self._pynvml
        while not self._stop_event.is_set():
            try:
                self.peak_device_bytes = max(
                    self.peak_device_bytes,
                    p.nvmlDeviceGetMemoryInfo(self._handle).used,
                )
                for proc in p.nvmlDeviceGetComputeRunningProcesses(self._handle):
                    if proc.pid == self._pid and proc.usedGpuMemory:
                        self.peak_process_bytes = max(
                            self.peak_process_bytes, proc.usedGpuMemory
                        )
                self.peak_temp_c = max(
                    self.peak_temp_c,
                    p.nvmlDeviceGetTemperature(self._handle, p.NVML_TEMPERATURE_GPU),
                )
                self.max_power_w = max(
                    self.max_power_w, p.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                )
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()


def percentiles(values: list[float], qs=(50, 95, 99)) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    out = {f"p{q}": float(np.percentile(arr, q, method=PERCENTILE_METHOD)) for q in qs}
    out["mean"] = float(arr.mean())
    out["n"] = int(arr.size)
    return out


# ==========================================================================
# Measurement
# ==========================================================================
def warmup(model, tok, prompts: list[dict]) -> None:
    """[SPEC 8] Runs before any timing starts. Output is discarded entirely."""
    print(f"warmup: {WARMUP_ITERATIONS} iterations (outside every timed window)")
    for i in range(WARMUP_ITERATIONS):
        ids = build_inputs(tok, prompts[i % len(prompts)]["text"])
        with torch.inference_mode():
            model.generate(**ids, do_sample=False, max_new_tokens=16,
                           use_cache=True, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    print("warmup done")


def measure_one(model, tok, text: str, max_new_tokens: int) -> dict:
    """One timed request. Everything inside is the measurement window."""
    ids = build_inputs(tok, text)
    prompt_tokens = int(ids["input_ids"].shape[-1])
    timer = StepTimer()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **ids,
            do_sample=False,                 # greedy [SPEC 3]; no sampling knobs passed
            max_new_tokens=max_new_tokens,
            logits_processor=[timer],        # per-step timestamps
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    generated = out[0][prompt_tokens:]
    text_out = tok.decode(generated, skip_special_tokens=True)

    # [SPEC 5] a hard failure, never a warning: if thinking leaked, every latency
    # number in this file is meaningless because output length stops being controlled.
    if "Thinking Process" in text_out or " thinking" in text_out:
        raise RuntimeError(
            "thinking mode leaked into the workload — refusing to record this run"
        )

    ttft = (timer.timestamps[0] - t0) * 1000.0 if timer.timestamps else None
    gaps = [(b - a) * 1000.0 for a, b in zip(timer.timestamps, timer.timestamps[1:])]
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(generated.shape[-1]),
        "steps_observed": len(timer.timestamps),
        "ttft_ms": ttft,
        "tpot_ms": float(np.mean(gaps)) if gaps else None,
        "wall_ms": (t_end - t0) * 1000.0,
        "sample_output": text_out[:120],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="M1 BF16 baseline oracle")
    ap.add_argument("--prompts", type=int, default=0,
                    help="use only the first N prompts (0 = all)")
    ap.add_argument("--reps", type=int, default=REPETITIONS)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--tag", default="", help="label for the output file")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    # ---- provenance, asserted before anything is measured -----------------
    prompt_sha = sha256_file(PROMPT_SET)
    stats = json.loads(PROMPT_STATS.read_text(encoding="utf-8")) \
        if PROMPT_STATS.exists() else {}
    expected = stats.get("prompt_set_sha256")
    if expected and prompt_sha != expected:
        print(f"ABORT: prompt set changed.\n  measured {prompt_sha}\n  frozen   {expected}\n"
              f"Re-run scripts/check_prompt_set.py and re-freeze before measuring.",
              file=sys.stderr)
        return 2
    print(f"prompt set sha256 {prompt_sha[:16]}… (matches frozen stats)")

    prompts = load_prompt_set()
    if args.prompts:
        prompts = prompts[: args.prompts]
    print(f"workload: {len(prompts)} prompts × {args.reps} reps "
          f"(max_new_tokens={args.max_new_tokens}, thinking={THINKING})")

    # ---- NVML comes up before the model, so the idle baseline is real ------
    sampler = VramSampler(interval_s=float(os.environ.get("NVIDIA_MEM_SAMPLE_MS", 20)) / 1000)
    idle = sampler.snapshot() if sampler.available else {}
    if sampler.available:
        sampler.start()
        print(f"idle device memory {idle.get('device_used_bytes', 0)/2**30:.2f} GiB, "
              f"{idle.get('temp_c', '?')} °C")
    else:
        print("WARNING: NVML unavailable — VRAM figures will be reported as unavailable")

    # ---- load, then warm up, then start the clock -------------------------
    t_load0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    tok.padding_side = "left"        # correct side for decoder-only batching
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map=DEVICE, low_cpu_mem_usage=True
    )
    model.eval()
    # Qwen3 ships sampling defaults (temperature, top_p, top_k) in its own
    # generation_config.json. Greedy decoding ignores them, and transformers warns
    # about the contradiction on every call. Clearing them makes the effective
    # configuration match the spec (greedy, no sampling parameters) and keeps the
    # log free of a warning a future reader would have to interpret.
    for _k in ("temperature", "top_p", "top_k"):
        if getattr(model.generation_config, _k, None) is not None:
            setattr(model.generation_config, _k, None)
    load_s = time.perf_counter() - t_load0
    print(f"model loaded in {load_s:.1f}s (this is outside every timed window)")

    warmup(model, tok, prompts)

    fingerprint = build_fingerprint(prompt_sha)
    samples: list[dict] = []
    t_run0 = time.perf_counter()
    for rep in range(args.reps):
        for p in prompts:
            r = measure_one(model, tok, p["text"], args.max_new_tokens)
            r.update(rep=rep, prompt_id=p["id"], bucket=p["bucket"])
            samples.append(r)
        tpos = [s["ttft_ms"] for s in samples if s["ttft_ms"]]
        print(f"  rep {rep+1}/{args.reps} done — ttft p50 so far "
              f"{np.percentile(tpos, 50, method=PERCENTILE_METHOD):.1f} ms "
              f"({len(samples)} samples)")
    total_wall_s = time.perf_counter() - t_run0

    sampler.stop()
    sampler.join(timeout=1.0)

    # ---- aggregate -------------------------------------------------------
    ttft = [s["ttft_ms"] for s in samples if s["ttft_ms"] is not None]
    tpot = [s["tpot_ms"] for s in samples if s["tpot_ms"] is not None]
    tokens = sum(s["generated_tokens"] for s in samples)
    idle_gib = idle.get("device_used_bytes", 0) / 2**30 if idle else 0.0
    peak_device = sampler.peak_device_bytes / 2**30 if sampler.available else None
    peak_process = sampler.peak_process_bytes / 2**30 if sampler.available else None
    # [SPEC 7, as amended] Per-process NVML accounting returns nothing under WSL2
    # driver passthrough, so the authoritative figure is the device-wide peak minus
    # the idle baseline captured before the model was loaded. It is coarser than a
    # per-process reading, but it is measured identically for every tier, which is
    # what makes tiers comparable — and comparability is the requirement, not precision.
    peak_authoritative = (
        round(peak_device - idle_gib, 3) if peak_device is not None else None
    )

    result = {
        "tier": TIER,
        "runtime": RUNTIME,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": fingerprint,
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "thinking": THINKING,
            "sampling": "greedy",
            "seed": SEED,
            "batch_size": 1,
            "warmup_iterations": WARMUP_ITERATIONS,
            "percentile_method": PERCENTILE_METHOD,
            "model_dir": str(MODEL_DIR),
        },
        "sample_counts": {
            "prompts": len(prompts),
            "repetitions": args.reps,
            "realised": len(samples),
        },
        "metrics": {
            "ttft_ms": percentiles(ttft),
            "tpot_ms": percentiles(tpot),
            "generated_tokens_total": tokens,
            "throughput": {
                "single_stream_tok_s": float(tokens / total_wall_s),
                "aggregate_tok_s": float(tokens / total_wall_s),
                "note": "identical at batch size 1; separated for M5",
            },
            "peak_vram_gib": {
                "authoritative": peak_authoritative,
                "method": "NVML device peak minus idle baseline",
                "device_peak": round(peak_device, 3) if peak_device is not None else None,
                "idle_device": round(idle_gib, 3),
                "process_peak": round(peak_process, 3) if peak_process else None,
                "units": "GiB (bytes / 2**30)",
                "limitation": "NVML per-process accounting is unavailable under WSL2 driver "
                              "passthrough, so the device-wide figure minus the idle baseline "
                              "is used; the method is identical for every tier, which is what "
                              "makes tiers comparable.",
            },
        },
        "thermal": {
            "peak_temp_c": sampler.peak_temp_c or None,
            "max_power_w": sampler.max_power_w or None,
            "start": idle,
        },
        "timing_outside_window": {
            "model_load_s": load_s,
            "measurement_wall_s": total_wall_s,
        },
        "per_sample": samples,
        "notes": "M1 BF16 oracle. HF transformers latency is not the deployment oracle; "
                 "see docs/measurement_spec.md decision 15.",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"{stamp}{tag}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    m = result["metrics"]
    print("\n=== summary ===")
    print(f"  TTFT  p50 {m['ttft_ms'].get('p50', float('nan')):.1f} ms   "
          f"p95 {m['ttft_ms'].get('p95', float('nan')):.1f} ms   "
          f"p99 {m['ttft_ms'].get('p99', float('nan')):.1f} ms   n={m['ttft_ms'].get('n', 0)}")
    print(f"  TPOT  p50 {m['tpot_ms'].get('p50', float('nan')):.2f} ms   "
          f"p95 {m['tpot_ms'].get('p95', float('nan')):.2f} ms")
    print(f"  throughput {m['throughput']['single_stream_tok_s']:.1f} tok/s   "
          f"peak VRAM {m['peak_vram_gib']['authoritative']:.2f} GiB "
          f"(device {m['peak_vram_gib']['device_peak']:.2f} - idle {m['peak_vram_gib']['idle_device']:.2f})")
    print(f"  thermal peak {result['thermal']['peak_temp_c']} °C, "
          f"{result['thermal']['max_power_w'] and round(result['thermal']['max_power_w'])} W")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
