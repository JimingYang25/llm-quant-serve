# Measurement Spec — LLM Quantize + Serve

> **Status:** DECISIONS APPLIED (2026-09-13) — awaiting owner sign-off.
> **Authority:** once signed, no benchmark number produced outside this protocol is admissible.
> **Decision log:** the fifteen decisions below were proposed by the agent and applied at the owner's instruction; items 5, 12, 14 and 15 are structural and are marked as such.

---

## 0. Purpose and scope

Define what is measured, how, at which layer, and with what tolerance, so that the BF16 / INT8 / FP8 / 4-bit tiers are comparable to one another.

**Out of scope:** quantization itself, engine tuning, serving optimization (M2–M5).

**Governing principle:** a number is admissible only if the protocol that produced it was frozen before it was produced. Two properties are non-negotiable:

- **The latency oracle runs in the same execution path as the quantized tiers** (decision 15).
- **Accuracy is measured from checkpoints; engines are verified by equivalence** (decision 14).

---

## 1. Workload

| Field | Decision | Value |
|---|---|---|
| Prompt set path | 1 | `prompts/prompt_set.json` (**version 1.1**) |
| Prompt count | 1 | **30** |
| Length buckets | 1 | **measured, not nominal**: short 19–26 tok (median **24**); medium 230–360 (median **264**); long 884–1000 (median **938**) |
| Composition | 1 | English technical, Chinese, code, short reasoning, long-context summarisation |
| Measured token stats | 1 | `prompts/prompt_set.stats.json` — per-prompt counts, document references and slices, structural checks |
| Prompt set SHA-256 | 1 | **`77763576f2f044c8128fe5e3302a2def6dec52742cc2a4b3a8e93860c7674b0e`** (v1.1) — required verbatim in every result file |

**Decision 1 — revision after measurement (recorded, not hidden).** The nominal targets were ~16 / ~256 / ~1024 tokens. Measurement with the real Qwen3-8B tokenizer showed v1.0 landing at medians of **24 / 31 / 461**: the medium bucket was indistinguishable from the short bucket and the long bucket was 2.2× under target, which would have left the workload with only two usable buckets and no mid-length coverage. Rather than pad with filler prose, v1.1 makes medium prompts reuse 2–3 paragraphs of a topically related document and long prompts pair two documents. Re-measurement: 24 / 264 / 938, all three inside their target ranges. **Lesson recorded:** nominal bucket sizes are a hypothesis; only the tokenizer settles it.
| `max_new_tokens` | 2 | **128** (latency runs) · **256** (quality runs) |
| Sampling | 3 | **Greedy**: `temperature=0`, `top_p=1` |
| Seed | 4 | **0** (recorded; inert while greedy) |
| **Thinking mode** | **5** | **DISABLED for all tiers** |

**Decision 5 — justification (structural).** With thinking enabled, Qwen3 emits a variable-length internal monologue, so generated length differs by an order of magnitude across prompts. TPOT and throughput are functions of how much is generated, so tier-to-tier differences would then reflect emergent verbosity rather than quantization; every run also becomes several times more expensive. Disabling thinking makes the workload a controlled length. **Stated cost:** all reported accuracy is *non-thinking-mode* accuracy, and this must be stated in the report and in any comparison with published results.

**Decision 3 — justification.** Greedy decoding removes sampling variance from the comparison. Without it, distinguishing a real accuracy delta from sampling noise would require several times the sample count.

---

## 2. Metrics

| Metric | Definition | Layer | Statistic |
|---|---|---|---|
| TTFT | Request submission → first output token | **HTTP client (primary)**; engine-reported value recorded as a secondary column | p50, p95, p99 |
| TPOT / ITL | Mean gap between successive output tokens | same | p50, p95, p99 |
| Throughput | Generated tokens per second | same | **single-stream and aggregate reported separately** |
| Peak VRAM | Maximum device memory during the run | **NVML device peak minus idle baseline** (decision 7, as amended); per-process figure recorded when the platform reports it | max |

**Decision 7 — justification (VRAM source).** `torch.cuda.max_memory_allocated` observes only the PyTorch allocator. TensorRT-LLM allocates engine and KV-cache memory outside it, so that metric would report lower values for the quantized tiers for reasons unrelated to quantization — a systematic bias in the same direction the experiment is trying to measure. NVML sampling is slightly coarser but applies identically to every runtime. **Uniformity across runtimes takes precedence over per-runtime precision.**

**Decision 7 — amendment, measured (2026-09-13).** Per-process NVML accounting is **not available under WSL2 driver passthrough**: `nvmlDeviceGetComputeRunningProcesses` returned no entry for our own PID while the device-wide figure read 16.56 GiB. The authoritative figure is therefore **device-wide peak minus the idle baseline captured before loading** — measured **16.56 − 0.96 = 15.60 GiB** for BF16, against 15.25 GiB of weights (consistent, as expected at batch 1). The method is identical for every tier, which preserves exactly the comparability this decision exists to protect. The per-process figure is still recorded when the platform reports it. The result field is `peak_vram_gib`, carrying its own `method` and `limitation` strings so a reader never has to infer how the number was obtained.

**Prohibited:** reporting a mean without percentiles; presenting aggregate throughput as single-stream; mixing measurement layers across tiers.

---

## 3. Measurement discipline

| Rule | Value |
|---|---|
| Warmup iterations discarded | **3** (decision 8) |
| Samples per configuration | **30 prompts × 5 repetitions = 150**; the realised sample count is recorded in every result file (decision 9) |
| Synchronisation | `torch.cuda.synchronize()` immediately before and after each timed region |
| Excluded from the timed window | model load, `import tensorrt_llm` (measured 7.6 s), engine build, CUDA-graph capture, first-call JIT |
| Repeatability tolerance | **5 %** on latency p50 between two independent runs (decision 10) |
| Single-session rule | BF16 and all quantized tiers measured in one session, at comparable temperature and power state |
| GPU contention | verify the GPU is otherwise idle before timing (idle baseline observed: ~676 MiB held by Windows processes) |
| Recorded per run | GPU temperature, power draw, clock, `.wslconfig` networking mode, timestamp |

**Decision 9 — justification and limitation.** 150 samples yield a p95 that is well determined and a **p99 that is approximately the second-worst observation**. p99 is therefore reported but labelled *provisional* until the M5 concurrency sweeps, which generate many more samples. If a credible p99 is needed earlier, increase **repetitions per prompt**, not the number of prompts.

**Decision 10 — justification.** 5 % is tight enough to detect genuine drift (thermal, power-state, contention) while tolerating ordinary run-to-run variation. **If two runs disagree by more than 5 %, the response is to diagnose the cause, never to widen the tolerance.**

---

## 4. Environment fingerprint

Required in every result file:

```
torch.__version__              torch.version.cuda
tensorrt.__version__           tensorrt_llm.__version__
nvidia-modelopt version        transformers.__version__
driver version + CUDA ceiling
pip freeze sha256              prompt set sha256
```

**Standing constraint.** The `transformers` version must be identical across all tiers: it governs tokenisation and chat-template rendering, so a version change can alter the tokens submitted to the model and thereby confound any accuracy delta. A packaging change may separate quantization from serving; it may not straddle the measurement.

---

## 5. Accuracy evaluation

| Field | Decision | Value |
|---|---|---|
| Primary metric | 11 | **Perplexity** on a **wikitext-103 test slice** (first N documents; hash recorded) |
| Secondary metric | 12 | **MMLU subset, log-likelihood scoring, N = 500** |
| Reporting | 13 | Absolute value **and** delta versus the BF16 oracle, oracle measured in the same session |
| Calibration/eval separation | 13 | Different sources; both hashes listed in the report |

**Decision 11 — justification.** Wikitext-103 perplexity is the convention in quantization work, which makes your numbers directly comparable to published tables rather than only to themselves.

**Decision 12 — justification (structural).** With thinking disabled (decision 5), GSM8K degrades sharply and the resulting baseline is compressed and noisy — you would be measuring quantization damage against a weak reference. MMLU scored by **log-likelihood** does not depend on generation at all, is stable, is standard in quantization papers, and is sensitive to weight damage. **Stated power:** N = 500 resolves roughly a 2 % difference; N = 1000 is preferable if the schedule allows, and whichever is used must be stated rather than implied.

---

## 6. Experimental design rules

**Decision 14 — the accuracy artefact is the quantized checkpoint, not the engine (structural).**

Accuracy is measured by **one evaluator only** — HF `transformers` + `lm-eval-harness` — loading each tier's ModelOpt checkpoint. TensorRT engines cannot be evaluated this way, so:

- **Accuracy column:** checkpoints, one evaluator, all tiers.
- **Engine verification:** run a fixed prompt subset through the engine and compare generated tokens against the same checkpoint run, requiring **≥ 95 % token agreement** (the threshold itself is provisional and is revisited at M5).

This keeps accuracy and performance measurements from contaminating one another.

**Decision 15 — the latency oracle runs in the same execution path as the tiers (structural).**

The BF16 latency reference is built with **TensorRT-LLM's own unquantized path**, i.e. the same runtime the quantized tiers will be deployed in. HF `transformers` latency is recorded as a **separate, clearly labelled column** and is **never** used as the speedup denominator, because most of the apparent gain would then be attributable to the runtime change rather than to quantization.

---

## 7. Result file schema

Every file under `results/` must be self-describing:

```json
{
  "tier": "bf16 | w8a8 | fp8 | w4a16",
  "runtime": "tensorrt_llm | hf_transformers",
  "evaluator": "lm_eval_harness@<version> (accuracy runs only)",
  "timestamp": "ISO-8601",
  "fingerprint": { "torch": "", "cuda_rt": "", "tensorrt": "", "tensorrt_llm": "",
                   "modelopt": "", "transformers": "", "driver": "", "pip_freeze_sha256": "" },
  "prompt_set_sha256": "",
  "config": { "max_new_tokens": 128, "thinking": false, "sampling": "greedy", "seed": 0 },
  "sample_counts": { "prompts": 30, "repetitions": 5, "realised": 150 },
  "metrics": {
    "ttft_ms":    { "p50": 0, "p95": 0, "p99": 0, "p99_status": "provisional" },
    "tpot_ms":    { "p50": 0, "p95": 0, "p99": 0 },
    "throughput": { "single_stream_tok_s": 0, "aggregate_tok_s": 0 },
    "peak_vram_gb": 0
  },
  "thermal": { "temp_c": 0, "power_w": 0, "clock_mhz": 0 },
  "accuracy": { "perplexity_wikitext103": 0, "mmlu_loglikelihood_n500": 0 },
  "engine_equivalence": { "token_agreement_pct": 0, "threshold": 95 },
  "notes": ""
}
```

---

## 8. Gate thresholds

| Gate | Threshold | Active from |
|---|---|---|
| Baseline repeatability | latency p50 within **5 %** across two runs | M1 |
| Fingerprint completeness | present in **100 %** of result files | M1 |
| Prompt-set hash | present and matching the committed set | M1 |
| `pip check` | only the documented deviation **D1** | every milestone |
| Engine equivalence | ≥ 95 % token agreement with the checkpoint | M5 |
| **Quantization — perplexity** | relative increase ≤ **2 %** vs the BF16 oracle (9.3190 → must stay ≤ **9.505**); dense metric over 32 752 scored tokens, resolves sub-percent change | **M2** |
| **Quantization — MMLU** | drop ≤ **4 percentage points** vs the oracle (0.6120 → must stay ≥ **0.572**); coarse guard only | **M2** |
| Quantization — latency / VRAM | measured and **reported but not gated** until M5, where the deployment runtime is in play | M2 |

**Why the MMLU gate is 4 pp and not 1 pp (measured, not assumed).** With N = 500 the oracle's standard error is 0.0218, giving a 95 % resolution of **±4.3 percentage points** (`results/accuracy/bf16_*_oracle.json`). A 1–2 pp MMLU drop is therefore *statistically invisible* at this sample size, and gating on it would mean gating on noise. Only drops larger than the noise floor are meaningful. **Perplexity carries the real gate** — it is dense and low-variance — while MMLU exists to catch a catastrophic failure that perplexity might miss.

---

## 9. Deliverables implied by this spec

| Artefact | Path | Owner |
|---|---|---|
| Prompt set (30 prompts) | `prompts/prompt_set.json` | owner |
| Baseline oracle runner | `scripts/02_baseline_bf16.py` | owner |
| Benchmark sweep | `scripts/06_bench.py` | owner |
| Accuracy evaluation | `scripts/07_eval_accuracy.py` | owner |
| Gate aggregator | `scripts/08_gate.py` | owner |
| Oracle results (two runs) | `results/baseline_bf16/` | owner |

---

## 10. Sign-off checklist

- [x] All fifteen decisions carry a value
- [ ] Prompt set exists, is committed, and its SHA-256 is recorded
- [ ] Owner has reviewed decisions **5, 7, 12, 14, 15** (the consequential ones) and accepts the stated costs
- [ ] Reviewer approved the spec **before** any benchmark script was written
- [ ] Owner sign-off recorded below

**Sign-off:** `__________________` (owner) · `__________________` (reviewer)
