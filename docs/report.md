# Quantization and Deployment of Qwen3-8B on a Single 24 GB Blackwell GPU

**English** | [中文](report.zh-CN.md)

Technical report. Every figure is reproducible from a committed script and a result
file under `results/`; corrections to earlier findings are recorded rather than
retracted silently.

---

## 1. Scope and question

**Question.** For an 8-billion-parameter instruction-tuned language model served on one
RTX 5090 Laptop (24 GB, `sm_120`, 175 W cap) under WSL2, what is the cheapest precision
that still meets an accuracy gate — and what does that saving buy in service capacity?

**Motivation.** Quantization claims are usually reported as a size reduction and an
accuracy delta. Both are easy to produce and easy to get wrong: the accuracy delta
depends on the evaluation budget, and the size reduction does not by itself translate
into service capacity. This study fixes a protocol first, then measures the tiers
against it, and finally measures the *service-level* consequence.

**What is deliberately excluded.** Fine-tuning, QAT, multi-GPU, and multi-node serving.
This is a single-GPU post-training study.

---

## 2. Method: freeze the instrument before measuring

The full protocol is `docs/measurement_spec.md` — 16 decisions, each recorded before the
measurement it governs, with three dated amendments. The load-bearing ones:

| Decision | Choice | Why |
|---|---|---|
| Workload | 30 prompts, 3 length buckets, frozen with SHA-256 | prompt length is the axis that separates prefill from decode cost |
| Thinking mode | **disabled** | enabled, output length varies ~10×, so latency stops being a controlled variable |
| Latency oracle | same execution path as the tiers | otherwise "speedup" would partly measure a runtime change |
| Accuracy oracle | one evaluator across all tiers | different evaluators confound the delta |
| Peak VRAM | NVML device peak minus idle baseline | per-process accounting is unavailable under WSL2 (amendment) |
| Repeatability | two runs within **5 %** on latency p50 | thermal drift, not code, dominates run-to-run variation |
| Quantization gate | perplexity ≤ **+2 %**, MMLU ≥ **−4 pp** (FP8); ≤ **+5 %** for 4-bit | fixed before the runs, so it cannot be relaxed after them |

**Measured instrument properties (M1).** Against the BF16 baseline:

```
TTFT p50   35.3 ms      TPOT p50   33.94 ms      throughput 29.2 tok/s
TTFT ≈ 29.1 ms + 66.4 µs × prompt_tokens          (two independent runs, slope within 4 %)
perplexity 9.3190 (32 752 tokens)                 MMLU 0.6120 ± 0.0218 (N = 500)
peak VRAM  15.98 GiB                              KV cache 144 KiB/token
```

The prefill fit is worth noting: it is a two-point result reproduced twice, and it
predicts the whole latency-versus-context curve, which M3 and M5 then confirmed
independently.

**Two runs agreed within 5 %** on every gated metric — but the worst metric sat at
**+4.89 %**, i.e. 0.11 pp inside the gate. On a 175 W laptop the thermal margin *is* the
repeatability budget. That is a limitation of the platform, recorded rather than
engineered around.

---

## 3. Results

### 3.1 Weight precision (M2, M4)

| Tier | Perplexity | Δ | MMLU | Artifact |
|---|---|---|---|---|
| BF16 oracle | 9.3190 | — | 0.6260 (N=2000) | 15.26 GiB |
| **FP8 W8A8** | **9.3434** | **+0.26 %** | 0.6180 | **8.79 GiB** |
| Mixed NVFP4 + 5 protected | 9.4634 | +1.55 % | 0.5490 (N=2000) | 7.25 GiB |
| Uniform NVFP4 (W4A4) | 9.5799 | +2.80 % | — | ≈5.6 GiB |
| INT8 + SmoothQuant | 14.0729 | +51.0 % | — | — |
| INT8 per-tensor | 19.7043 | +111.4 % | — | — |

**FP8 passes both gates and is the deployment tier.** The artifact is 42 % smaller; the
residual 58 % is embeddings and `lm_head`, whose quantizers are disabled, plus the FP8
scales.

**INT8 fails catastrophically, and the mechanism is visible in the calibration log.**
Per-tensor activation scaling collapses onto the largest observed value: measured `amax`
for `down_proj` inputs reached **1152** (per-tensor INT8) and up to **4016** in one layer,
so a single shared scale is spent on a few outlier channels and the rest of the
distribution is crushed. SmoothQuant moves the difficulty into the weights and halves the
damage (+111 % → +51 %) without making it acceptable.

**FP8 survives the same outliers because it keeps exponent bits** — an outlier costs
mantissa precision, not range. NVFP4 survives them because of 16-element microscaling,
at a real cost (+2.80 %).

**Sensitivity-guided protection helps but does not close the gap.** Protecting the most
damaged blocks (fresh quantization per candidate, protection baked into the config)
recovers 8 of the 10 pp of uniform-4-bit MMLU loss while costing ~1.3 GiB. The tier still
fails the gate.

### 3.2 KV cache precision (M3)

| Context | fp16 KV | fp8 KV | ratio |
|---|---|---|---|
| 1 024 | 71.7 tok/s | 53.7 | 0.75 |
| 4 096 | 64.0 | 51.1 | 0.80 |
| 16 384 | 31.1 | 32.7 | 1.05 |
| 32 768 | 17.8 | 22.6 | 1.27 |

Pool: **4.67 GiB (fp16) vs 2.33 GiB (fp8) for 33,984 tokens** — exactly half, matching
the derived constant (2 × 36 layers × 8 kv_heads × 128 × 2 B = 144 B/token).

The two regimes are mechanistically distinct: below ~16k the dequantization cost in the
attention kernel dominates and the small cache makes memory irrelevant; above it,
attention is memory-bandwidth-bound and halving the byte count wins. **The decision
belongs to the deployment, not to a default.**

Two stack constraints were found by trying rather than by reading: with an FP8
checkpoint, `KvCacheConfig(dtype=...)` accepts only `('fp8', 'nvfp4', 'auto')` — **INT8 KV
is rejected** — and the `nvfp4` KV path **hard-crashes the worker process**, which is why
it is documented as a finding rather than included in the sweep.

### 3.3 Serving (M5)

| Concurrency | BF16 tok/s | FP8 tok/s | Speedup | BF16 TTFT p50 | FP8 TTFT p50 | BF16 TPOT p50 | FP8 TPOT p50 |
|---|---|---|---|---|---|---|---|
| 1 | 44.2 | 75.0 | 1.70× | 31.3 | 31.8 | 23.04 | 13.47 |
| 4 | 177.0 | 298.3 | 1.69× | 53.1 | 61.3 | 22.26 | 12.87 |
| 8 | 342.9 | 514.4 | 1.50× | 57.7 | 69.9 | 23.03 | 15.19 |
| 16 | 642.9 | 982.4 | 1.53× | 62.9 | 80.1 | 24.53 | 15.71 |
| 32 | 1131.3 | 1668.0 | 1.47× | 89.0 | 117.4 | 27.57 | 18.26 |

**The transfer chain:**

```
weights        15.26 → 8.79 GiB        (6.47 GiB freed)
KV cache       34,333 → 78,170 tokens  (×2.28 at the same free_gpu_memory_fraction = 0.85)
concurrency         8 → 19 sequences   (at the 4096-token served context)
throughput                +47 %        (at 32 concurrent requests)
per-token latency         −40 %        (TPOT 23.0 → 13.5 ms at batch 1)
```

**The scheduling law is visible in the FP8 sweep itself:** throughput rises 22× from
concurrency 1 to 32 while TPOT degrades only 1.35× and TTFT 3.7×. Continuous batching
trades tail latency for aggregate throughput, and the ratio between the two is the
operating point a deployment chooses.

**Two honest observations.** Peak VRAM is nearly identical between tiers (21.97 vs
21.58 GiB) because each server takes whatever memory is free — **the saving is capacity,
not footprint**. And FP8's TTFT is *worse* at high concurrency (117 vs 89 ms) despite
higher throughput, because a larger cache admits more concurrent requests and prefill
queues longer.

---

## 4. The methodological finding that mattered most

```
perplexity : 9.3190 → 9.4634  = +1.55 %      "nearly lossless"
MMLU       : 0.6260 → 0.5490  = −7.70 pp     severely damaged
                                (paired on identical 2000 items, z = 4.96)
```

**A dense metric under-reported a task-level injury.** The protocol had designated
perplexity the primary gate precisely because it is dense, low-variance, and resolves
sub-percent change, with MMLU demoted to a coarse guard. This result reverses that
ranking: perplexity is sensitive to *broad, shallow* damage and blind to *narrow, deep*
damage such as degraded reasoning. The two metrics are complementary, and **a tier is
only acceptable if both gates pass**.

Two further instrument lessons, both measured:

- **Perplexity is only comparable within one token budget.** BF16 on the same corpus
  scores 9.1705 / 10.6146 / 8.3988 / 9.3190 at 4k / 8k / 16k / 32k — a ±14 % swing from
  the budget alone.
- **MMLU at N = 500 has ±4.3 pp resolution.** The same model scored 0.5560 on one
  500-item subset and 0.5080 on the first 500 of another — a 4.8 pp swing from item
  selection. Only the N = 2000 paired comparison supports a conclusion.

---

## 5. What went wrong

Ten defects were found and fixed; each produced **plausible numbers instead of an error**,
which is the defining hazard of this stack. Full evidence in `docs/ROADMAP.md`.

| # | Defect | Symptom | Root cause | Fix |
|---|---|---|---|---|
| 1 | MPI hang | `import tensorrt_llm` never returned (1712 s) | WSL mirrored networking blackholes SYNs to unlistened loopback ports; `mpirun` stuck in `SYN_SENT` to `127.0.0.1:6001` | NAT networking |
| 2 | ModelOpt extensions never built | FP8 = `NaN`; INT8 silently on the CPU fallback | torch resolves the compiler as `c++` and crashes its own check; the failing units are `.cu`, so `-std=c++20` must be in `extra_cuda_cflags` | `CXX` + both flag lists |
| 3 | Hub endpoint ignored | 5 retries per dataset | `HF_ENDPOINT` set after `transformers` imported it | set it first |
| 4 | **Wildcard over-matching** | "5 protected blocks" protected **14**; the 4-bit export was **larger** than FP8 (9.57 vs 8.79 GiB) | `*layers.1*` also matches `layers.10–19` | one pattern ending at a component boundary |
| 5 | Non-idempotent toggling | one `disable`/`enable` cycle moved ppl 11.05 → 18.53 | quantizer state is not restored | fresh quantization per configuration |
| 6 | Cross-budget comparison | a false "+18.6 %" | compared an 8192-token measurement to a 32752-token oracle | budget is part of the measurement |
| 7 | Lazy extension load | half the probes on a different numerical path | ModelOpt loads its CUDA extension on first use | preload it |
| 8 | Invalid equivalence metric | "27 % agreement" verdict while both outputs were coherent | greedy decoding compounds a single flipped token | teacher-forced per-position comparison |
| 9 | Thinking mode leaking | 13-token prompt → 400-token output | template defaults to thinking on; a text prompt lets the engine re-template | `enable_thinking=False` via token ids |
| 10 | Per-process VRAM unavailable | `NaN` | WSL2 driver passthrough exposes no per-PID accounting | device peak minus idle |

Defect 4 deserves emphasis: it was found by **noticing that an artifact was larger than it
should be**, then counting tensor dtypes inside it. The measurement that exposed the bug
was not the accuracy measurement but the *size* measurement — a reminder that
inconsistency between two independent views of the same claim is the cheapest detector of
error available.

---

## 6. Limitations

1. **Single GPU, batch-1 latency figures** unless a concurrency level is stated.
2. **Thermal margin is thin:** the baseline pair passed a 5 % gate by 0.11 pp on a 175 W
   capped laptop. Cross-session comparisons without a temperature record are not
   trustworthy.
3. **The sensitivity ranking is incomplete:** 10 of 36 blocks were validly probed before
   the wildcard defect was found. The protected set is the best of those ten.
4. **MMLU conclusions rest on N = 2000 for the paired comparison only**; the N = 500
   numbers elsewhere carry ±4.3 pp and are reported as context.
5. **Absolute accuracy is "thinking-disabled" accuracy** and is not comparable to
   published numbers produced with reasoning enabled.
6. **NVFP4 weights work; the NVFP4 KV cache crashes** on this stack version.
7. **Engine-equivalence testing (decision 14) was amended but not executed** — the
   teacher-forced comparison remains open work.

---

## 7. Conclusions

1. **FP8 W8A8 is the correct tier for this model, stack and GPU.** +0.26 % perplexity,
   MMLU unchanged within noise, 42 % smaller, 1.5–1.7× throughput, ~40 % lower per-token
   latency, and 2.28× the KV capacity.
2. **Sub-8-bit is not reachable here.** INT8 fails catastrophically; NVFP4 works in the
   weights (+2.80 %) but loses 7.70 pp of MMLU even with sensitivity-guided protection —
   and the protection that recovers most of it also gives back most of the size advantage.
3. **KV cache quantization is context-dependent**, profitable above ~16k and a 25 %
   throughput tax below it.
4. **Dense and task metrics must both be gated.** Perplexity alone would have approved a
   tier that loses 7.7 points of MMLU.
5. **In this stack, failures report numbers rather than errors.** Every one of the ten
   defects above was found by cross-checking two independent views of a claim, not by an
   exception.

---

## 8. Artifacts

| Artifact | Path |
|---|---|
| Protocol, 16 decisions, amendments | `docs/measurement_spec.md` |
| Weight provenance and derived constants | `docs/model_provenance.md` |
| Milestone log, including all corrections | `docs/ROADMAP.md` |
| Cheat sheet: commands and where numbers are generated | `docs/how_to_run.md` |
| Glossary | `docs/glossary.md` |
| Raw results, one JSON per measurement | `results/` |
| Frozen workload with measured token stats | `prompts/prompt_set.json`, `.stats.json` |

*Every document above has a counterpart in the other language (`*.zh-CN.md`; `ROADMAP.en.md` for the Chinese-primary log), and each carries a language switcher at the top. `model_provenance.md` is a generated digest and remains English-only.*

