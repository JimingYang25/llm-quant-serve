# Cheat sheet — how to run this project

Written for the case where the detail has become overwhelming. If you read only one
file before running anything, read this one. The full reasoning lives in
`docs/measurement_spec.md`; you do not need to hold it in your head.

---

## 1. The whole project in four numbers

| Number | BF16 value | Produced by |
|---|---|---|
| Perplexity (wikitext-103 slice) | **9.3190** | `scripts/07_eval_accuracy.py` |
| MMLU accuracy (N=500) | **0.6120 ± 0.0218** | `scripts/07_eval_accuracy.py` |
| TTFT p50 / TPOT p50 | **35 ms / 34 ms** | `scripts/02_baseline_bf16.py` |
| Peak VRAM | **15.98 GiB** | `scripts/02_baseline_bf16.py` |

Everything else in the project is either a way of producing these, or a way of
proving they are trustworthy. Quantization (M2 onward) changes these numbers; the
job is to change them **only** as much as we can defend.

---

## 2. The one rule to remember

> **If the environment, the prompt set, the weights, or the measuring script changes,
> the old numbers are no longer comparable.**

That is why every result file carries hashes. You do not need to memorise the
version matrix — the files carry it.

---

## 3. What to run, and what to look at

### Before any measurement (30 seconds)

```bash
bash scripts/00_env_check.sh          # environment sane: torch, TRT, MPI, CUDA family
python scripts/check_prompt_set.py    # prompt set unchanged; prints its hash
```

Check: no errors, and the printed hash matches the one in `docs/measurement_spec.md`.

### M1 — the BF16 oracle (already done once)

```bash
python scripts/02_baseline_bf16.py --tag run1      # ~10 min
python scripts/02_baseline_bf16.py --tag run2      # ~10 min, same session
python scripts/compare_runs.py results/baseline_bf16/*run1.json \
                               results/baseline_bf16/*run2.json
python scripts/analyze_baseline.py results/baseline_bf16/*run1.json
```

Check exactly two things:
1. `compare_runs.py` says **PASS** (all gated metrics within 5 %).
2. `analyze_baseline.py` shows TTFT rising with prompt length and TPOT roughly flat.

### Accuracy, for any tier

```bash
python scripts/07_eval_accuracy.py --tier bf16 --smoke    # seconds: shape check
python scripts/07_eval_accuracy.py --tier bf16            # ~5 min: real numbers
```

Check: MMLU is far above 0.25 (chance). If it is near 0.25, the prompt or thinking
mode is wrong — do not record the run.

### Inspecting any result

```bash
python scripts/show_result.py                          # newest file
python scripts/show_result.py results/accuracy/<file>.json
```

---

## 4. Where each number is generated in the code

### `scripts/02_baseline_bf16.py` — latency and memory

| Line | What it does | Number it produces |
|---|---|---|
| 68–73 | Protocol constants: `MAX_NEW_TOKENS=128`, `WARMUP=3`, `REPS=5`, greedy, percentile method | the settings behind every figure |
| 110 | `build_fingerprint()` | the version block in each result file |
| 143 | `load_prompt_set()` | resolves `doc_refs` + `paragraph_slice` into prompt text |
| 165 | `build_inputs()` | applies the chat template with thinking **disabled** |
| 186 | `class StepTimer` | timestamps of each decoding step → **TTFT** and **TPOT** |
| 202 | `class VramSampler` | **peak VRAM**, temperature, power |
| 277 | `percentiles()` | **p50 / p95 / p99** |
| 290 | `warmup()` | runs before timing; produces no number on purpose |
| 302 | `measure_one()` | one request's TTFT/TPOT/wall time; asserts thinking did not leak |
| 345 | `main()` | the order: load → warm up → start clock → loop → write JSON |

### `scripts/07_eval_accuracy.py` — accuracy

| Line | What it does | Number it produces |
|---|---|---|
| 50–54 | `PPL_WINDOW=2048`, `PPL_TOKENS=32768`, `MMLU_N=500`, seed | the settings behind both metrics |
| 114 | `build_ppl_windows()` | the fixed text slice and its hash |
| 154 | `perplexity()` | **perplexity** (token-weighted, so it is mathematically correct) |
| 173 | `load_mmlu()` | the 500 sampled questions and their hash |
| 184 | `make_prompt()` | question + choices, thinking disabled — **the line that fixed the 0.25** |
| 211 | `score_choices()` | log-probability of " A"/" B"/" C"/" D" |
| 228 | `mmlu()` | **accuracy**, standard error, chance level |

### `scripts/compare_runs.py` — the gate

| Line | What it does |
|---|---|
| 26 | `TOLERANCE_PCT = 5.0` — the decision-10 threshold, the only number you must satisfy |
| 29–35 | what must be *identical* for two runs to be comparable at all |
| 37 | which metrics are gated (TTFT, TPOT) and which are context only |
| 60 | `main()` — prints PASS/FAIL and the diagnosis order if it fails |

### `scripts/analyze_baseline.py` — explanation, not measurement

| Line | What it does |
|---|---|
| 32 | `analyze()` — per-bucket table, prefill fit, per-repetition drift |

This one produces no gated number. It exists to explain *why* the numbers look the
way they do — that is where the "TTFT ≈ 29 ms + 66 µs × prompt_tokens" line comes from.

---

## 5. If something looks wrong

| Symptom | First check |
|---|---|
| MMLU ≈ 0.25 | `07_eval_accuracy.py:184` — thinking must be disabled |
| TTFT in seconds, not milliseconds | `02_baseline_bf16.py:345` — is `t0` after load and warmup? |
| Two runs disagree by >5 % | compare `thermal` in both JSONs; close GPU apps; re-run |
| Hash mismatch on start-up | the prompt set changed — re-freeze before measuring |
| `import tensorrt_llm` hangs | MPI/loopback issue; see ROADMAP §1.5–1.6 |
