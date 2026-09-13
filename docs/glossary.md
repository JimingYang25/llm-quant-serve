# Glossary — LLM Quantization & Serving

Every abbreviation used in this project, with its expansion, a plain-language meaning, and where you will actually meet it. Terms are grouped by theme, not alphabetically, so that related ideas sit together.

---

## A. Quantization: the core vocabulary

| Term | Expansion | What it means | Where you meet it |
|---|---|---|---|
| **PTQ** | Post-Training Quantization | Convert an already-trained model to lower precision using a small sample of text, with no retraining | M2 — your first quantized tier |
| **QAT** | Quantization-Aware Training | Simulate low precision *during* training so the model learns to tolerate it | Not used here; contrast with PTQ |
| **Calibration** | — | Running a few hundred text samples through the model to record the range of values each tensor actually takes | M2 — determines every quantization scale |
| **Calibration set** | — | The sample corpus used for that. Must not overlap the evaluation set, or the reported accuracy is inflated | M2 |
| **Scale / zero-point** | — | The two numbers in the affine mapping: `real ≈ scale × integer + zero_point` | Explains why granularity matters |
| **per-tensor / per-channel / per-group** | — | How many scales one tensor gets: one for the whole tensor, one per output channel, or one per small group of weights | M4 — group size is a key knob |
| **Weight-only vs weight+activation** | — | Whether only weights are quantized (activations stay 16-bit) or both | The single most important distinction in the whole project |
| **W8A8 / W4A16** | Weight 8-bit, Activation 8-bit / Weight 4-bit, Activation 16-bit | Compact notation for a quantization recipe | M2 (W8A8), M4 (W4A16) |
| **INT8** | 8-bit integer | Uniform integer format; needs a calibrated range | M2 |
| **FP8 E4M3 / E5M2** | 8-bit float, 4 exponent + 3 mantissa bits / 5 + 2 | Floating format: wide dynamic range, few mantissa bits. E4M3 for weights/activations, E5M2 for gradients | M3 — Blackwell has native FP8 |
| **FP4 / NVFP4 / MXFP4** | 4-bit float (NVIDIA scheme / OCP microscaling scheme) | 4-bit floating formats; NVFP4 is NVIDIA's Blackwell-era variant | M4 — availability on sm_120 must be *measured* |
| **Outlier channel** | — | A handful of activation channels with magnitudes far larger than the rest, which ruin a single shared scale | Why activation quantization is harder than weight quantization |
| **AWQ** | Activation-aware Weight Quantization | Protects the weights most important to activations before quantizing to 4 bits | M4 fallback if NVFP4 fails |
| **GPTQ** | Generative Pre-trained Transformer Quantization (a PTQ method) | Layer-by-layer weight quantization that minimizes output error | M4 alternative to AWQ |
| **SmoothQuant** | — | Migrates activation difficulty into the weights by rescaling, making W8A8 easier | M2 optional improvement |
| **Fake quantization / QDQ** | Quantize-DeQuantize | Insert quantize+dequantize ops so a float model behaves *as if* quantized — for measuring damage without deploying | Sensitivity probing (M4) |
| **KV cache quantization** | Key/Value cache | Quantizing the cached attention keys and values, i.e. the part of memory that grows with context length | M3 — often the real memory bottleneck |
| **Perplexity** | — | `exp(mean negative log-likelihood)`: a continuous measure of how well the model predicts a text corpus. Lower is better | M1 — accuracy oracle |
| **Accuracy delta** | — | The difference in a metric between a quantized tier and the unquantized reference | The number your report is built on |

---

## B. Serving and performance metrics

| Term | Expansion | What it means | Where you meet it |
|---|---|---|---|
| **Oracle / baseline** | — | The unquantized reference measurement everything else is compared against | M1 — you produce two (accuracy, latency) |
| **TTFT** | Time To First Token | Delay from request submission to the first output token; dominated by prompt processing | M1 spec, M5 sweeps |
| **TPOT** | Time Per Output Token | Average time per token *after* the first; governs how the answer "flows" to the user | M1 spec, M5 |
| **ITL** | Inter-Token Latency | Same quantity as TPOT, named from the streaming perspective | Appears in TRT-LLM's own reports |
| **Throughput** | — | Tokens per second. Must always be qualified: single-stream or aggregate across all requests | M5 |
| **Prefill / decode** | — | The two phases: reading the prompt (compute-bound, parallel) and generating tokens (memory-bound, sequential) | Explains TTFT vs TPOT |
| **Chunked prefill** | — | Splitting a long prompt into pieces so a giant prefill does not stall other requests' decode | Read-back question in M1 |
| **Continuous batching** | — | Admitting new requests as soon as a slot frees, instead of waiting for the whole batch | M5 |
| **KV cache** | Key/Value cache | Stored attention state for already-processed tokens; memory grows linearly with context length | M3 — the context-length sweep |
| **Paged KV cache / PagedAttention** | — | Storing the KV cache in fixed-size blocks to eliminate fragmentation | M5 |
| **CUDA Graph** | — | Recording a fixed kernel sequence once and replaying it, to remove per-step launch overhead | M5 optional gain |
| **Concurrency / batch size** | — | Number of requests in flight / number of sequences processed together | M5 sweeps: 1, 4, 8, 16, 32 |
| **Context length** | — | Total tokens in prompt + generated output | M3 sweep: 1k, 4k, 16k, 32k |
| **p50 / p95 / p99** | 50th / 95th / 99th percentile | The value below which that fraction of measurements falls. p99 exposes tail latency | Every results table |
| **Warmup** | — | Iterations discarded before measuring, so cold caches, JIT compilation and clock ramping are excluded | M1 discipline |
| **Peak VRAM** | Video RAM | Maximum GPU memory allocated during the run | M1 metric, M3 bottleneck analysis |
| **Engine** | TensorRT engine | A compiled, GPU-architecture-specific artifact produced from a model; not portable across TRT versions or architectures | M5 |
| **TRT-LLM** | TensorRT-LLM | NVIDIA's LLM inference stack (two modes: build an engine, or run a checkpoint directly) | The project's runtime |
| **ONNX** | Open Neural Network Exchange | A model interchange format | Optional export path |
| **ORT** | ONNX Runtime | A runtime that executes ONNX graphs; can apply INT8 quantization itself | Alternative to TRT |
| **OpenAI-compatible API** | — | An HTTP interface matching OpenAI's schema, so existing clients work unchanged | M5 serving layer |

---

## C. Accuracy evaluation

| Term | Expansion | What it means | Where you meet it |
|---|---|---|---|
| **lm-eval-harness** | Language Model Evaluation Harness | The standard open-source framework for running academic benchmarks | M1 evaluation |
| **MMLU** | Massive Multitask Language Understanding | Multiple-choice knowledge benchmark across ~57 subjects | Optional task metric |
| **GSM8K** | Grade School Math 8K | Grade-school math word problems; sensitive to reasoning degradation | Optional task metric |
| **Sample size / statistical power** | — | How many test items are needed before a given difference is distinguishable from noise. 50 items cannot resolve a 1 % gap | Must be stated honestly in the spec |
| **Confound** | Confounding variable | An unintended difference that could explain the result (e.g. evaluating two tiers under different `transformers` versions) | The reason for the freeze rule |
| **Ablation** | — | A controlled experiment that changes one factor to attribute the effect to it | M2 (calibration size), M4 (mixed precision) |
| **Greedy decoding** | — | Always taking the highest-probability token (`temperature=0`) | Reproducibility requirement in M1 |
| **Seed** | — | Random-number seed; fixes sampling when greedy decoding is not used | M1 spec |
| **Thinking mode** | — | Qwen3's built-in reasoning mode that emits a long internal monologue before the answer, greatly changing output length | Must be explicitly frozen in M1 |

---

## D. Measurement methodology and project governance

| Term | Expansion | What it means | Where you meet it |
|---|---|---|---|
| **Measurement spec** | — | The frozen document defining what is measured, how, and with what tolerances | M1 deliverable |
| **Fingerprint** | Environment fingerprint | The recorded version set (torch, TRT-LLM, modelopt, transformers, driver) that identifies the exact environment behind a number | Inside every results file |
| **Reproducible** | — | Same inputs + same environment → same numbers | The project's standard |
| **Gate** | — | An explicit pass/fail criterion, evaluated automatically rather than by impression | `08_gate.py`, every milestone |
| **Deviation (D1–D3)** | — | A *known, recorded* departure from the specified environment set, with a justification | D1 = `nvidia-cuda-runtime` patch mismatch; D3 = transformers/modelopt range |
| **Drift** | — | An unintended change between two states, detected by diffing freeze files | Recovery check |
| **Regression** | — | Something that previously worked now fails | The `import torch` break after the cu12 removal |
| **Read-back** | — | Restating a mechanism in your own words, as evidence of understanding rather than mere execution | M0/M1 sign-off |
| **M0–M6** | Milestone 0 … 6 | Project phases: environment → measurement → INT8 → FP8/KV → 4-bit → serving → report | The roadmap |

---

## E. Environment, build and systems terms

| Term | Expansion | What it means | Where you meet it |
|---|---|---|---|
| **ABI** | Application Binary Interface | The binary-level contract between a compiled library and its consumers | Why a torch version change can break a compiled extension |
| **soname** | Shared object name | The ABI identity of a `.so`, e.g. `libcudnn.so.9`. Two versions sharing a soname are ABI-compatible | 13.0.48 vs 13.0.96 were both `libcudart.so.13` |
| **Pin** | — | An exact version requirement (`==`) in package metadata | Source of both dependency deadlocks |
| **Resolver** | — | The dependency solver inside pip that chooses which versions to install | It silently substituted your torch build |
| **Wheel** | — | A prebuilt, installable Python binary package (`.whl`) | TRT-LLM's wheel is 2.5 GB |
| **RPATH** | — | A library search path baked into an ELF binary; why a wheel's own CUDA libs are used instead of the system toolkit | Loader-provenance checks |
| **SASS / PTX / cubin** | — | Final GPU machine code / forward-compatible intermediate code / a compiled kernel image. Absent SASS for your architecture forces JIT | Why `sm_120` in `arch_list` matters |
| **sm_120 / compute capability** | — | The GPU architecture number. `sm_120` is consumer Blackwell (your 5090) | Kernel compatibility |
| **JIT** | Just-In-Time compilation | Compiling at first use instead of ahead of time; causes first-call stalls | Must sit outside every timed region |
| **Driver API vs Runtime API vs Toolkit** | — | Three layers: the driver's capability ceiling, the runtime a program links against, and the compile-time toolkit (which does not affect execution) | The CUDA-alignment lecture |
| **bind-mount / overlayfs / lowerdir** | — | Linux mechanisms by which WSL makes the Windows driver's GPU libraries appear inside Linux, with one directory layer shadowing another | The `590.62` vs `592.01` puzzle |
| **WSL2 mirrored vs NAT networking** | Windows Subsystem for Linux | Two networking modes: `mirrored` shares the host's loopback; `NAT` gives the VM its own stack behind a gateway | The overnight outage and its fix |
| **MPI** | Message Passing Interface | The standard for multi-process/multi-node communication; used by inference stacks for multi-GPU | The thing that hung |
| **OpenMPI / `orted` / BTL / OOB** | — | A specific MPI implementation / its daemon process / its network transport component / its out-of-band messaging component | `orted` was the process mpirun waited for |
| **mpi4py** | — | The Python binding for MPI | Hard-imported by TRT-LLM |
| **RECORD** | — | pip's per-package manifest of installed file paths; the reason uninstalling one package can delete another's files | The `libcudnn.so.9` incident |
| **Freeze file** | `pip freeze` output | The exact list of installed package versions; usable as a revert point | `pip_freeze_before_cu12_removal.txt` |
| **Orphan package** | — | Installed, but no other package declares a dependency on it | The 15 CUDA-12 packages |
| **Monorepo / editable install** | — | Irrelevant to this project; listed only because `pip install -e` appears in logs | — |

---

## F. How to use this document

You do **not** need all of this before writing code. For M1 specifically, only these are load-bearing:

`TTFT`, `TPOT`, `throughput`, `p50/p95/p99`, `warmup`, `peak VRAM`, `perplexity`, `oracle`, `fingerprint`, `gate`, `confound`, `thinking mode`.

Everything in section A matters from M2 onward, and section E is context you have already lived through once.
