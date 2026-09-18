# LLM Quantize + Serve

**English** | [中文](README.zh-CN.md) ・ Report: **English** | [中文](docs/report.zh-CN.md)

Quantization and deployment study of **Qwen3-8B** on a single **RTX 5090 Laptop (24 GB, Blackwell sm_120)** under WSL2 — with a frozen measurement protocol, verified oracles, and **negative results reported as first-class findings**.

Every number below is traceable to a committed script and a result file in `results/`. Where a measurement was later found to be invalid, the correction is recorded rather than quietly replaced.

---

## Headline results

| Tier | Perplexity (32752 tokens) | vs BF16 | MMLU | Artifact | Verdict |
|---|---|---|---|---|---|
| **BF16 (oracle)** | 9.3190 | — | 0.6260 (N=2000) | 15.26 GiB | baseline |
| **FP8 W8A8** | **9.3434** | **+0.26 %** | 0.6180 | **8.79 GiB** | ✅ **deployment tier** |
| Mixed NVFP4 + 5 protected blocks | 9.4634 | +1.55 % | 0.5490 (N=2000) | 7.25 GiB | ❌ −7.70 pp MMLU |
| Uniform NVFP4 (W4A4) | 9.5799 | +2.80 % | — | ≈5.6 GiB | reference |
| INT8 + SmoothQuant | 14.0729 | **+51 %** | — | — | ❌ unusable |
| INT8 per-tensor | 19.7043 | **+111 %** | — | — | ❌ unusable |

**Conclusion: on this model and stack, 8-bit is the floor.** Sub-8-bit activation quantization cannot be rescued by protecting sensitive layers, and INT8's per-tensor activation scaling fails catastrophically while FP8 — which keeps exponent bits — costs 0.26 %. The mechanism is the activation **outlier channel**: observed `amax` values of 1152–4016 on `down_proj` inputs destroy a single shared integer scale, and NVFP4 survives them better than INT8 only because of sub-block microscaling.

## Serving: what quantization is actually worth

Measured with `trtllm-serve` (OpenAI-compatible) under a concurrency sweep:

| Concurrency | BF16 tok/s | FP8 tok/s | Speedup | BF16 TPOT p50 | FP8 TPOT p50 |
|---|---|---|---|---|---|
| 1 | 44.2 | 75.0 | **1.70×** | 23.04 ms | **13.47 ms** |
| 4 | 177.0 | 298.3 | 1.69× | 22.26 | 12.87 |
| 8 | 342.9 | 514.4 | 1.50× | 23.03 | 15.19 |
| 16 | 642.9 | 982.4 | 1.53× | 24.53 | 15.71 |
| 32 | 1131.3 | 1668.0 | 1.47× | 27.57 | 18.26 |

**The transfer chain, measured end to end:**

```
weights        15.26 GiB → 8.79 GiB      (42 % smaller, 6.47 GiB freed)
KV cache       34,333   → 78,170 tokens  (×2.28 at the same memory fraction)
concurrency         8   → 19 sequences   (at 4096-token context)
throughput                 +47 % at 32 concurrent requests
per-token latency          −40 %
```

**Caveat, stated up front:** peak VRAM is nearly identical (21.97 vs 21.58 GiB) because the server takes whatever memory is free. **The saving appears as cache capacity and throughput, not as lower VRAM usage.**

## KV cache: a trade-off with a crossover

fp8 KV cache vs fp16 KV cache, FP8 weights, batch 1:

| Context | fp16 KV | fp8 KV | ratio |
|---|---|---|---|
| 1 024 | 71.7 tok/s | 53.7 | **0.75** |
| 4 096 | 64.0 | 51.1 | 0.80 |
| 16 384 | 31.1 | 32.7 | **1.05** |
| 32 768 | 17.8 | 22.6 | **1.27** |

Quantizing the cache **costs 25 % throughput below ~16k context and gains 27 % above it** — dequantization overhead dominates when the cache is small; memory bandwidth dominates when it is large. The pool itself halves (4.67 → 2.33 GiB for 33,984 tokens). **Whether to quantize the KV cache is a deployment decision that depends on your context distribution, not a default.**

## The most important methodological finding

```
perplexity : 9.3190 → 9.4634   =  +1.55 %     "nearly lossless"
MMLU       : 0.6260 → 0.5490   =  −7.70 pp    badly damaged   (paired, N=2000, z = 4.96)
```

**A dense metric under-reported a task-level injury by a wide margin.** The project had designated perplexity as "the real gate" because it is dense and resolves sub-percent change, demoting MMLU to a coarse guard. This measurement reversed that: perplexity cannot see narrow, deep damage such as degraded reasoning. **Both metrics are required, and both gates must pass.**

---

## What went wrong (and why it is in the repo)

Nearly every hard-won result here came with a bug that produced *plausible numbers instead of an error*. Each is documented with its evidence:

| # | Failure | Signature | Fix |
|---|---|---|---|
| 1 | WSL mirrored networking blackholes SYNs to unlistened loopback ports | `import tensorrt_llm` hung forever; `mpirun` stuck in `SYN_SENT` to `127.0.0.1:6001` | switch to NAT |
| 2 | ModelOpt's CUDA extensions never built | FP8 produced `NaN`; INT8 silently used the CPU fallback | `CXX` must name a real `g++`, and `-std=c++20` must be in **`extra_cuda_cflags`** (the failing files are `.cu`) |
| 3 | Hugging Face endpoint set after import | every dataset retried `huggingface.co` 5× then fell back to cache | `HF_ENDPOINT` must precede `transformers` (hub reads it at import) |
| 4 | **Wildcard over-matching** | "protect 5 blocks" protected **14** (`*layers.1*` also matches `layers.10–19`); the 4-bit export came out **larger** than FP8 | one pattern, ending at a component boundary |
| 5 | `disable_quantizer` → `enable_quantizer` is not idempotent | one cycle moved perplexity 11.05 → 18.53 | fresh quantization per configuration, never toggle a live model |
| 6 | Perplexity compared across token budgets | BF16 alone swings 9.17 → 10.61 → 8.40 → 9.32 for 4k/8k/16k/32k | budget is part of the measurement |
| 7 | Lazy CUDA-extension load mid-experiment | half the probes ran on a different numerical path | force the extension to load before measuring |
| 8 | Token agreement as an equivalence metric | 27 % "failure" while both outputs were coherent — greedy decoding compounds one flipped token | teacher-forced per-position agreement |
| 9 | Thinking mode leaking into measurements | 13-token prompt → 400-token reply; output length uncontrolled | `enable_thinking=False`, passed through token ids |
| 10 | Per-process NVML unavailable under WSL2 | VRAM reported as `NaN` | device-wide peak minus idle baseline |

**The recurring lesson:** in this stack, mistakes do not announce themselves — they report believable numbers.

---

## Repository layout

```
prompts/          frozen 30-prompt workload (v1.1) + measured token stats + SHA-256
docs/
  measurement_spec.md   the protocol: 16 decisions, gates, amendments
  report.md             full technical report
  model_provenance.md   weight digests, revision, derived memory constants
  glossary.md           every abbreviation used here
  how_to_run.md         cheat sheet: commands per milestone, where each number is generated
  ROADMAP.md            M0–M6 with every finding, including corrections
  *.zh-CN.md            Chinese counterparts; ROADMAP.en.md for the Chinese-primary log
scripts/
  00_env_check.sh       environment acceptance test
  02_baseline_bf16.py   latency oracle (load → warmup → timed loop)
  03_ptq_w8a8.py        PTQ: calibrate → quantize → evaluate → export → gate
  04_kv_cache_sweep.py  KV dtype × context sweep
  06_bench.py           concurrency sweep against the serving endpoint
  07_eval_accuracy.py   perplexity + MMLU
  09_sensitivity.py     which blocks carry the 4-bit damage
  10_mixed_precision.py mixed-precision tier (protection baked into the config)
  11_transfer_table.py  quantization → cache → concurrency
  modelopt_ext_patch.py makes ModelOpt's CUDA extensions buildable
  compare_runs.py       the 5 % repeatability gate
  analyze_baseline.py   per-bucket breakdown and prefill fit
results/          raw JSON for every measurement above
```

## Reproducing

```bash
conda create -n llmquant python=3.12 && conda activate llmquant

# environment (see docs/ROADMAP.md §1 for the full audit and deviation ledger)
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com

bash scripts/00_env_check.sh                 # environment acceptance test
python scripts/check_prompt_set.py           # workload hash must match the spec
python scripts/02_baseline_bf16.py --tag run1   # latency oracle (~10 min)
python scripts/07_eval_accuracy.py --tier bf16  # accuracy oracle (~5 min)
python scripts/03_ptq_w8a8.py --method fp8 --tag fp8   # quantize + gate
python scripts/04_kv_cache_sweep.py          # KV cache sweep
python scripts/06_bench.py                   # serving sweep (needs trtllm-serve running)
```

**Hardware:** the numbers assume one 24 GB Blackwell GPU. Latency figures are single-GPU, batch-1 unless stated.

## Limitations

- **Single GPU, single machine.** No tensor parallelism, no multi-node.
- **A laptop power cap (175 W) and a 24 GB ceiling.** Back-to-back baseline runs sat at **+4.89 %** of a 5 % repeatability gate — **0.11 pp of headroom** — the thermal margin is thin and is documented rather than hidden.
- **MMLU's resolution at N=500 is ±4.3 pp**; only the N=2000 paired comparison (−7.70 pp, z=4.96) supports the 4-bit conclusion.
- **The sensitivity ranking is incomplete.** Ten of 36 blocks were validly probed before the wildcard bug was found; the protection set is the best of those ten, not proven optimal.
- **NVFP4 weights work; NVFP4 KV cache hard-crashes the worker process** on this stack.

## License

MIT (see `LICENSE`). Models and third-party binaries keep their own licenses; **no model weights or engines are committed to this repository.**
