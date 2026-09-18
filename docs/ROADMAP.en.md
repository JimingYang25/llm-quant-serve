# LLM Quantize + Serve · Independent Completion Roadmap

**English** | [中文](ROADMAP.md)

> Project codename: **Qwen3-8B PTQ → TRT-LLM Serving**
> Mode: **You do the work yourself; I only explain mechanisms + accept checkpoints (read-back)**. I do not write implementation code for you.
> Goal: a public repository that can be linked directly as a resume project, containing reproducible scripts, raw data, and a quantization report.

---

## 0. Project Definition

**In one sentence**: take Qwen3-8B from its BF16 baseline through three PTQ tiers — INT8 / FP8 / 4bit — land it on a TensorRT-LLM server, and quantify each tier's "accuracy loss × latency gain × VRAM gain" **under the same measuring stick**, finally delivering an actually stress-testable OpenAI-compatible service.

**Why this project is worth something** (the three things recruiters look at; all three must be present):

1. **You genuinely understand the taxonomy of quantization**: W8A8 vs W4A16 vs FP8 vs NVFP4, weight-only vs weight+activation, per-tensor vs per-channel vs per-group, KV cache quantization — not "I ran bitsandbytes".
2. **You genuinely understand the physical constraints of deployment**: TTFT/TPOT/throughput/peak VRAM/concurrency curves **pull against one another** (the KV cache is the dominant memory term at long context, and batch size degrades TPOT linearly); you can explain the trade-off with data.
3. **You dare to report half-hit conclusions**: which tier actually paid off, which tier is a trap, and which tier simply does not hold on your hardware (FP4 support on sm_120 is an **item to be measured**, not an assumption). Honest numbers are worth more than all-green numbers.

**Final deliverable** (repository structure — lay down the skeleton first; you fill in the scripts yourself):

```
llm-quant-serve/
├── README.md                  # 项目说明 + 一张 Pareto 结论表 + 复现步骤
├── docs/
│   ├── report.md              # 量化报告正文（方法论/数据/结论/局限）
│   └── measurement_spec.md    # 【M1 冻结】测量口径规范，全项目唯一尺子
├── env/
│   ├── wsl_setup.md           # 环境搭建全过程记录（含踩坑）
│   └── requirements.txt
├── configs/
│   ├── w8a8_int8.yaml         # ModelOpt 量化配置
│   ├── fp8.yaml
│   ├── w4a16_nvfp4.yaml
│   └── quantize_kv_cache.yaml
├── scripts/
│   ├── 00_env_check.sh        # GPU/驱动/CUDA/TRT-LLM 可见性自检
│   ├── 01_fetch_model.py      # 拉 Qwen3-8B（HF_ENDPOINT 可切镜像）
│   ├── 02_baseline_bf16.py    # 【oracle】BF16 基线：延迟/显存/吞吐
│   ├── 03_ptq_export.py       # ModelOpt PTQ → 量化权重（+ 可选 ONNX）
│   ├── 04_build_or_load.py    # TRT-LLM 引擎构建 / PyTorch 后端直载
│   ├── 05_serve_openai.py     # OpenAI 兼容服务端
│   ├── 06_bench.py            # 并发/上下文/批大小 sweep → TTFT/TPOT/显存
│   ├── 07_eval_accuracy.py    # lm-eval-harness 子集：精度对齐
│   ├── 08_gate.py             # 汇总 → results/gate.json（PASS/FAIL + 理由）
│   └── 09_sensitivity.py      # 【加分档】逐层敏感度探针（你 EZTrain 的老手艺）
├── results/                   # 原始 json/csv，绝不手工编辑
│   ├── baseline/  w8a8/  fp8/  w4a16/  kv_int8/
│   └── gate.json
└── tests/                     # 至少覆盖：指标计算、prompt 集合哈希、配置加载
```

---

## 1. Prior Facts (measured 2026-09-12, not speculation)

| Item | Measured value | Impact |
|---|---|---|
| GPU | RTX 5090 Laptop, 24463 MiB, **driver 592.01 / CUDA 13.1**, power limit **173W** | 24GB can hold the Qwen3-8B BF16 oracle; 173W means **thermal control is mandatory**, otherwise latency numbers are not comparable |
| Architecture | Blackwell **sm_120** | FP8 has native support; **whether NVFP4 is usable must be measured** (see M4) |
| Windows CUDA | CUDA Toolkit **v13.3** (nvcc present) | ⚠️ This serves only the Windows side; inside WSL you must **install a separate Linux CUDA toolkit** |
| `modelopt` env (Win) | Py3.11.16 · nvidia-modelopt **0.46.0** · onnxruntime-gpu 1.29.0 | Toolchain on the Windows side — leave it alone for now |
| `openvla`/`lingbot-vla` env | **torch 2.14.0+cpu / 2.7.1+cpu** | All CPU builds, unrelated to this project; do not reuse |
| WSL2 | User confirms it is available (I am blocked by the sandbox on my side; M0 requires you to self-verify) | Determines whether TRT-LLM/vLLM can be used |
| HF network | Past record: direct connection to official HF does not work → use `HF_ENDPOINT=https://hf-mirror.com` | The 8B weights are about 16GB; the download strategy must be fixed at M0 |

### 1.1 M0 Measurement Record (2026-09-12, user's actual machine output)

**WSL side (Ubuntu 24.04.4 LTS, conda base = /home/jiming/miniconda3)**

| Item | Measured | Verdict |
|---|---|---|
| `/dev/dxg` | present | ✅ passthrough normal |
| WSL `nvidia-smi` driver line | `Driver Version: 592.01` (= Windows side) | ✅ the kernel driver read through NVML is the current driver |
| WSL `nvidia-smi` tool line | `NVIDIA-SMI 590.62` | ⚠️ the tool binary is one minor version older than the driver — an N-1 skew |
| `nvidia-smi` provenance | `/usr/bin/nvidia-smi → /usr/lib/wsl/lib/nvidia-smi`, `dpkg -S` = belongs to no deb package | ✅ **only one copy**, mounted by the WSL driver, not installed by apt |
| `dpkg -l \| grep nvidia-driver/kernel/dkms` | none | ✅ **red line not crossed** |
| WSL CUDA | **full CUDA 13.3** installed via apt (cuda-nvcc-13-3 / cudart / cublas / nsight…, cuda-keyring 1.1-1) | ✅ but the **version tag must be aligned with the TRT-LLM wheel** |
| `/usr/lib/wsl/lib/` | `libcuda.so`, `libnvidia-ml.so.1` (Jan 2026), `libnvidia-gpucomp.so.590.62 -> libnvidia-gpucomp.so` (symlink date = today's WSL boot time) | see the "mount structure" row below |
| **`/usr/lib/wsl/lib` mount structure** | **overlayfs**: `lowerdir=/gpu_lib_packaged:/gpu_lib_inbox`, `upperdir=/gpu_lib/rw/upper` | ✅ **the source of 590.62 is pinned down**: the GPU user-space library stacks two layers, "shipped with the WSL distro (packaged)" + "Windows driver inbox", with the packaged layer taking precedence → the tool binary is one minor version older than the actual driver |
| `/usr/lib/wsl/drivers` | 9p read-only mount (direct passthrough of the Windows DriverStore) | lets you verify driver file provenance directly inside WSL |
| WSL kernel | `6.18.33.2-microsoft-standard-WSL2` | ✅ recorded for reference |
| `llmquant` env | Py3.12.14 · **torch 2.14.0+cu130** (runtime CUDA 13.0 ≤ driver ceiling 13.1 → legal) · triton 3.8.0 · cudnn 9.24 | ✅ |
| torch `arch_list` | `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` | ✅ **includes sm_120**, no PTX JIT fallback risk |
| GPU enumeration | `avail True` · sm_120 · **82 SMs** (82×128 = 10496 FP32 lanes, matching the RTX 5090 Laptop specs) | ✅ hardware fully recognized |
| FP32 matmul smoke test | 4096³ = **10.187 ms → 13.5 TFLOPS** | ⚠️ **must not be used as a performance baseline**: cold start without warmup removal, TF32 off, P8 low-power state. It only proves that the kernel runs |
| Distro vhdx | `D:\WSL\Ubuntu-24.04\ext4.vhdx` (D: has 544.7GB free) | ✅ the project's 70–100GB fits on D: with room to spare |
| WSL memory | **31GB** + 8GB swap (host has about 64GB) | ⚠️ on the low side; recommended to raise it to 48GB in .wslconfig |
| WSL disk | `/dev/sdd` 1007G, **732G** available | ✅ |
| Cores | 24 | ✅ |
| conda base Python | **3.14.6** | ❌ **a real blocker**: TRT-LLM / ModelOpt do not support py3.14; a new py3.12 env must be created |

**M0 key conclusion (mechanism)**: WSL's GPU user-space libraries are bind-mounted into `/usr/lib/wsl/lib` from the Windows driver directory. The "tool version line" of `nvidia-smi` is a **string compiled into the binary** (590.62), whereas "Driver Version" is **read at runtime from the kernel driver via NVML** (592.01). The mismatch = the mounted user-space library is one minor version older than the current kernel driver; **this is common in WSL and usually harmless**, but it must be judged by **functional verification** (a real kernel launch), not by self-comfort from version-number alignment. Verification method: install CUDA-build torch into the py3.12 env → `torch.cuda.is_available()` is True → run a real matrix multiply + print `torch.version.cuda` and the SM count.

### 1.2 M0 Wrap-up: Verified Items, Deviation Ledger, Open Items (2026-09-12 night)

**Achieved (measured evidence)**
- `torch 2.9.1+cu130` (runtime 13.0) + `tensorrt 10.14.1.48.post1` + `tensorrt_llm 1.2.1`: the trio imports successfully and actually runs
- The dependency deadlock has been diagnosed: `nvidia-cuda-runtime==13.0.48` (declared by torch cu130) and `==13.0.96` (declared by `tensorrt_cu13_libs → cuda-toolkit[cudart]`) are mutually exclusive; pip did not report an error but **silently swapped torch for a cu128 build**; fixed by re-pinning cu130 with `pip install --no-deps --force-reinstall`, and both import and kernel execution were measured to work normally

**Deviation Log — must be re-verified after every environment change**

| # | Deviation | Verdict | Status |
|---|---|---|---|
| D1 | `nvidia-cuda-runtime 13.0.96` ≠ the `13.0.48` declared by torch | Same soname (`libcudart.so.13`), patch level; import + kernel verified working; `pip check` reports only this one | Accepted, recorded (original text on file) |
| D2 | CUDA 12 family leftovers, **15 packages** (`nvidia-*-cu12`, ≈3GB), introduced by the cu128 incident | **Level-4 loader measurement**: in-process only `libcudart.so.13 / libcublas.so.13 / libcublasLt.so.13 / libcudnn.so.9 / libnccl.so.2` are mapped, with **no `.12` co-resident** → the leftovers are inert and did not contaminate the numerical stack | Pending cleanup (a hygiene action, not a correctness fix): the 15 packages form the "transitive closure of an orphan root" and can be removed in one `pip uninstall -y`; after removal, `pip check` + the loader measurement must be re-run |
| D3 | `transformers 4.57.3` (exact pin by TRT-LLM) ∉ the range `[4.48, 4.57)` declared by `nvidia-modelopt 0.37.0` | Both pins ship together in the same NVIDIA version combination → judged a **stale range check**; taking option A, pending verification of the ModelOpt↔HF path with a 0.6B toy-scale rerun | Pending verification |

**Open items (by priority)**
1. ~~Environment ambiguity~~ **clarified (2026-09-12)**: the directory `/home/jiming/miniforge3` does not exist (`readlink -f` echoes the original string for a nonexistent path, hence the earlier misjudgment); the canonical interpreter = `/home/jiming/miniconda3/envs/llmquant/bin/python`, and `sys.executable / sys.prefix / modelopt.__file__ / transformers.__file__` self-verify consistently at all four points; llmquant is unique in `conda env list`. **Lesson: a hand-transcribed path cannot serve as evidence; the interpreter's own report is authoritative**
2. ~~Level 4 loader measurement, `pip check`, orphan package list~~ **executed (2026-09-12)**: only the CUDA 13 family is mapped in-process; `pip check` reports only D1; all 15 `nvidia-*-cu12` packages are in the orphan transitive closure (pending cleanup). Note that `libnvinfer` showing as not loaded is normal — that process only imported torch, not TensorRT
3. The three M0 read-back questions (environment-ownership criteria)
4. The M1 prerequisite deliverable `docs/measurement_spec.md` has not yet been written

**One non-negotiable constraint**

> The `transformers` version used for accuracy evaluation must remain **consistent across all tiers** — BF16 / INT8 / FP8 / 4bit; otherwise tokenizer and chat template differences contaminate the accuracy delta and the conclusion cannot be attributed.

### 1.3 Incident Post-mortem: File-level Ownership Conflict (2026-09-12 night, triggered by the D2 cleanup)

**Symptom**: after deleting the 15 `nvidia-*-cu12` packages on the "orphan transitive closure" verdict, `import torch` died outright:

```
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
```

**Root cause (mechanism)**: `nvidia-cudnn-cu12` and `nvidia-cudnn-cu13` **unpack into the same path namespace** `site-packages/nvidia/cudnn/lib/`, and cudnn 9.x's soname is **completely identical** in both CUDA families (`libcudnn.so.9`, `libcudnn_graph.so.9`, …). Therefore:

1. The cu13 variant was installed first → files landed in `nvidia/cudnn/lib/`
2. Later, dependency resolution made pip swap torch for cu128 → the cu12 variant was installed → **files with the same names were overwritten in place**, and pip registered these paths only in cu12's RECORD
3. Running `pip uninstall nvidia-cudnn-cu12` → pip deleted the paths according to RECORD → **it deleted the files that the cu13 variant likewise needed**

**Conclusion correction (important)**:
- D2's leftovers **are not inert** — they do not merely occupy 3GB of disk; they **overwrite/replace files of the cu13 family with same-named files**. The earlier Level-4 measurement showed `libcudnn.so.9` being mapped, but at the time **it was impossible to determine which family that file belonged to** (same soname, same path); therefore the conclusion "a single family in-process" was an **over-assertion**, now corrected to: "no `.12`/`.13` dual soname co-residency in-process; but **families with the same name and the same path cannot be distinguished by soname**"
- Other packages carrying the same-class/same-path/same-name risk: `nccl` (libnccl.so.2), `curand` (libcurand.so.10), `cusparse` (libcusparse.so.12), `nvjitlink` (libnvJitLink.so.12), `nvtx`, `cusparselt`, `cufile`, `nvshmem`

**General rule (new; to be written into the acceptance script)**:

> **A metadata-level dependency graph cannot predict file-level ownership conflicts.** `Required-by: NONE` only answers "does any other distribution **declare** a dependency on it"; it does not answer "does any other distribution **share file paths** with it." Before any deletion you must first perform a **file-level conflict audit** (comparing the intersection of each distribution's `RECORD`), rather than looking only at the dependency graph.

**Rollback basis**: `docs/pip_freeze_before_cu12_removal.txt` (this is exactly the value validation of the discipline "freeze before deleting").

### 1.4 M0 Closure Record (2026-09-12 late night · agent-run recovery + acceptance)

**Recovery action**: derive the 15 cu13-family packages from `pip_freeze_before_cu12_removal.txt` → `pip install --no-deps --force-reinstall -r` → all files unpacked again.

**Acceptance measurement output**

| Check | Result |
|---|---|
| Interpreter identity | `/home/jiming/miniconda3/envs/llmquant/bin/python` (unique) |
| torch | `2.9.1+cu130` · rt `13.0` · `sm_120 True` · **NVIDIA GeForce RTX 5090 Laptop GPU** |
| TensorRT | `10.14.1.48.post1` |
| Functional smoke test | `sm_120 SMs=82  9.640 ms  14.3 TFLOPS` |
| loader full paths | CUDA13 core libraries come from `nvidia/cu13/lib/`; `libcudnn.so.9` from `nvidia/cudnn/lib/`, `libnccl.so.2` from `nvidia/nccl/lib/`, `libnvshmem_host.so.3` from `nvidia/nvshmem/lib/`, `libcusparseLt.so.0` from `nvidia/cusparselt/lib/` |
| `pip check` | only D1 |
| freeze diff | **only 15 deletions of `*-cu12` lines, zero additions, zero version changes → recovery with no drift** |
| File-level conflict audit | 2 entries: `build_backend.py` / `.pyc` (`flashinfer-python` ↔ `torch_c_dlpack_ext`), unrelated to the CUDA family; harmless but recorded |

**Key correction (the true version of the conflict's cause)**: package directory layouts are **not uniform**. The cu13 variants place CUDA core libraries in a **family-specific directory** `nvidia/cu13/lib/`, whereas `cudnn / nccl / nvshmem / cusparselt` keep using a **family-agnostic directory** `nvidia/<name>/lib/`. Therefore:

- **Will conflict** (cu12 and cu13 write the same path + same-named soname): `cudnn`, `nccl`, `nvshmem`, `cusparselt` (same-layout peers also include `curand`, `nvtx`, `cufile`)
- **Will not conflict**: `cudart`, `cublas`, `cusparse`, `nvJitLink` (cu12 in `nvidia/<name>/lib`, cu13 in `nvidia/cu13/lib`)
- The earlier D2 table's claim that "cusparse/curand/nvjitlink carry conflict risk" was an **inference that had never been observed** and has been refuted by measurement; the correct approach is to **read the paths** (RECORD / `pip show -f`), not to guess from package names

**D2 closed**: the environment is a single CUDA 13 family, with no cross-family same-name file shadowing. The recovery script has been installed as `scripts/00_env_check.sh` (the M0 acceptance test; re-run after any environment change).

**M0 verdict: PASS**

### 1.5 New Finding: `import tensorrt_llm` Blocked by OpenMPI (2026-09-12 late night, unresolved)

**Symptom**: `import tensorrt_llm` hangs. Module-by-module bisection located it at `tensorrt_llm.bindings` (all other modules: torch/tensorrt/mpi4py/flashinfer/xgrammar/transformers/modelopt all <3s).

**Mechanism (confirmed down to the line number)**:
- The blocked state is `S (sleeping)`, `wchan = anon_pipe_read`, single-threaded, only 3s of CPU within 60s → **not compiling, waiting for a child process**
- The child process = **OpenMPI's `orted` HNP daemon** (`orted --hnp --set-sid --report-uri 8 --singleton-died-pipe 9 …`)
- Trigger point: `tensorrt_llm/_utils.py` **module level** line 498 `comm = pkl5.Intracomm(MPI.COMM_WORLD)`. Accessing `MPI.COMM_WORLD` makes mpi4py initialize MPI → spawns orted → the handshake never returns
- `mpi_disabled()` at `_utils.py:537` does read `TLLM_DISABLE_MPI == "1"`, but it is used only inside functions (lines 541/551/567) and **does not govern module-level line 498** → hence `TLLM_DISABLE_MPI=1` was measured to have no effect
- `from mpi4py import MPI` (`_utils.py:37`) is a hard dependency: temporarily moving mpi4py away immediately yields `ModuleNotFoundError`, so **uninstalling mpi4py is not a viable workaround**

**Reproduction independent of TRT-LLM**: `mpirun -n 1 hostname` hangs just the same (zero output within 45s). Ruled out: environment-variable contamination (`env -i` still hangs), hostfile pointing to 127.0.0.1, conda library contamination (`ldd /usr/bin/orted` resolves entirely to `/lib/x86_64-linux-gnu`), `/dev/shm` (20G, writable), MCA overrides (`btl_tcp_if_include=lo`, `oob_tcp_if_include=lo`, `btl=self,vader`, `oob=^tcp`, `orte_keep_fqdn_hostnames=0` — all ineffective).

**Environment facts**: system OpenMPI **4.1.6** (apt); `ldd` shows mpi4py's openmpi variant links the system `libmpi.so.40`; an mpich variant exists but `libmpi.so.12` is missing. Hostname resolution: `getent hosts Jiming` returns only an **IPv6 link-local** address, while `/etc/hosts` has `127.0.1.1 Jiming.localdomain Jiming`. The WSL VM was restarted at **19:41:48** (`.wslconfig` memory=40GB has taken effect; `free -g` now shows 39GB).

**Side effect (handled)**: every hanging attempt **leaks one orted process** (15 accumulated), and they inherit the stdout pipe → probing scripts using `| tail` block forever. Cleanup: `pkill -9 -f orted`. **This holds for all subsequent debugging scripts: commands that spawn child processes must not pipe their output; redirect to a file.**

**Scope of impact**: **does not block M1** (the HF baseline + measurement spec do not need tensorrt_llm). **It blocks TRT-LLM-side loading in M2–M4**, so it must be resolved before M1 ends.

**Candidate solutions (by priority, to be verified)**:
1. Determine whether it is "slow" or "dead": a run of `mpirun -n 1 hostname` and `import tensorrt_llm` with a 30-minute bound is in progress (results to be archived)
2. Replace the apt version pairwise with conda-forge's `openmpi + mpi4py` (same build source, eliminating the build differences between OpenMPI and mpi4py)
3. Test switching back to NAT networking (`networkingMode=NAT`) — a classic difference point for WSL + OpenMPI, at the cost of one WSL restart and possible impact on the existing proxy download paths

### 1.6 Root Cause Confirmed: WSL mirrored networking's **loopback black hole** (2026-09-12 late night)

**Verdict**: MPI is not "slow", it is **permanently blocked**. Two tests with a 30-minute bound: `mpirun -n 1 hostname` 1712s with no output; `import tensorrt_llm` 1665s with no output.

**Evidence chain (narrowing level by level)**
1. Module-by-module bisection: only `tensorrt_llm.bindings` hangs; everything else <3s
2. Blocked state `S` + `wchan=anon_pipe_read`, single-threaded, 3s CPU within 60s → waiting on a child process, not compiling
3. The child process = `orted --hnp …` (OpenMPI 4.1.6)
4. `/proc/<mpirun_pid>/net/tcp` caught the culprit: **`127.0.0.1:48406 -> 127.0.0.1:6001`, state=`02` (SYN_SENT)**, socket inode `17188`
5. Loopback behavior comparison (decisive):

   | Target | Result |
   |---|---|
   | `127.0.0.1:6001` (no listener) | **TIMEOUT 3.02s** — SYN swallowed |
   | `127.0.0.1:59999` (no listener) | **TIMEOUT 3.01s** — the same for any port |
   | `127.0.1.1:6001` (no listener) | REFUSED **0.000s** (normal) |
   | `192.168.1.101:6001` (no listener) | REFUSED **0.000s** (normal) |
   | `127.0.0.1:<self-created listener>` | CONNECTED 0.000s (loopback itself works) |

6. Windows-side comparison: `TcpClient.Connect('127.0.0.1',6001)` → **actively refused** (normal)
   → the anomaly **exists only on the loopback path of WSL mirrored networking**, not in the host network stack

**Mechanism**: under `networkingMode=mirrored`, WSL loopback is shared with the host; **when the target port has no listener, the RST is not returned and the SYN is silently dropped**. OpenMPI's singleton initialization connects to a loopback port (6001 in this case) for rendezvous — it used to get ECONNREFUSED immediately and continue, but now it waits forever, so `import tensorrt_llm` hangs. **This is neither a TRT-LLM bug nor an MPI bug.**

**Scope of impact (worth remembering)**: any program in WSL that "probes an unlistened loopback port" will **hang instead of failing fast** — this affects service health checks, port probing, and some test frameworks. M1 does not need TRT-LLM and **is not blocked**; but it must be fixed before M2.

**Fix attempts and the final solution (2026-09-13 morning, closed out)**

| Attempt | Result |
|---|---|
| `Set-NetFirewallHyperVVMSetting -LoopbackEnabled True` (administrator) + VM restart | ❌ **hypothesis refuted** — loopback is still a black hole (the firewall is not the cause) |
| MCA interface overrides: `oob_tcp_if_include` = lo / eth1 / 192.168.1.101, `if_exclude lo`, `btl=self,vader` | ❌ all still block |
| `oob_tcp_static_ports 6001`, `6001-6100` | ❌ still blocks |
| `TLLM_DISABLE_MPI=1` | ❌ ineffective: `_utils.py:498` initializes MPI at module level, outside that switch's control |
| Removing `mpi4py` | ❌ not viable: `_utils.py:37` is a hard import; moving it away yields `ModuleNotFoundError` |
| **`networkingMode=NAT`** + `wsl --shutdown` | ✅ **complete fix** |

**Supplementary observation (key)**: during the block, **no `orted` process existed at all and nothing was listening on port 6001**; and mpirun's target port is **always 6001** (`0x1771`, source port random). In other words this is **two defects stacked**: ① the peer that should listen on 6001 never started up at all; ② the loopback black hole turned "the peer does not exist" from something that **should error immediately** into **an indefinite wait** — so there was neither a retry nor an error. Hence fixing the network layer (defect ① being masked by ②) is the only viable path.

**NAT-mode verification result (measured)**
```
127.0.0.1:6001 / :59999        → REFUSED 0.000s        （回环语义恢复正常）
mpirun -n 1 hostname           → rc=0, 0s, 输出 "Jiming"
from mpi4py import MPI         → rc=0, 0s, rank 0 / 1
import tensorrt_llm            → rc=0, **13.5s**（此前 30 分钟不返回）
```

**Cost and verification (NAT does not support a localhost proxy; must be accounted for)**
- WSL startup warns "localhost proxy configuration detected, but NAT mode does not support it"; the Windows proxy (127.0.0.1:7897) **binds only to loopback**, so WSL cannot use it via the gateway IP
- But **the download paths this project needs work with direct connections**: `hf-mirror.com` 200 (11s), `pypi.org` 200 (1.9s), `download.pytorch.org` 200 (1.2s), `pypi.nvidia.com` 301 normal
- `huggingface.co` direct connection fails (expected, hence the mirror); `github.com` responds slowly (did not complete within 18s) — **must be re-checked before pushing the repo at M6**; if necessary, enable Allow LAN in the proxy client and use `http_proxy=http://172.18.192.1:7897`, or switch to pushing over SSH
- Backup: `.wslconfig.bak-nat`; rolling back only requires changing `networkingMode` back to `mirrored` and running `wsl --shutdown`

**Acceptance criteria (all passed, 2026-09-13 09:4x)**
1. Connecting to `127.0.0.1:6001` gives `ConnectionRefusedError` within milliseconds ✅
2. `mpirun -n 1 hostname` prints the hostname ✅
3. `import tensorrt_llm` returns `1.2.1` (13.5s) ✅
4. Re-running `scripts/00_env_check.sh` is all green ✅

**Red lines (remember these first; they can save you two days)**:
- ❌ **Never install the Linux NVIDIA graphics driver inside WSL**. WSL2's GPU is passed through by the Windows driver; installing a Linux driver directly breaks `/dev/dxg`. Inside WSL install only the **CUDA Toolkit (not the driver)**.
- ❌ Do not let the Windows side and the WSL side share the same HF cache / conda env. Keep them physically isolated.
- ⚠️ A laptop with a 173W power limit + WDDM (Windows) means **the same number can drift by 15% across different time windows**. All cross-version comparisons must be done within the same session, after the temperature has stabilized, and the report must state the temperature.

---

## 2. Milestone Overview

| # | Milestone | Core mechanism | Acceptance gate (I judge PASS/FAIL on this basis) |
|---|---|---|---|
| **M0** | Environment and self-verification | GPU is really visible inside WSL2; CUDA/TRT-LLM versions self-consistent | `00_env_check.sh` output complete + `nvidia-smi` visible inside WSL + torch.cuda.is_available()=True |
| **M1** | **Freeze the measuring stick first** | Fix the measurement convention before discussing optimization | `docs/measurement_spec.md` written down + HF baseline server comes up + bench produces the TTFT/TPOT/VRAM trio |
| **M2** | First step: W8A8 INT8 | The physical meaning of PTQ calibration: replace the min/max extremes with the real activation distribution | Quantized weights loadable, accuracy loss quantifiable, latency/VRAM numbers in results/ |
| **M3** | Second step: FP8 + KV cache | Blackwell native FP8; the dominant VRAM term at long context is the **KV cache**, not the weights | context length sweep curves (1k/4k/16k/32k) + VRAM gain from FP8 KV |
| **M4** | Third step: 4bit (NVFP4/AWQ) | Breaking the VRAM wall + **identifying sensitive layers for mixed precision** | Whether NVFP4 holds on sm_120 (measured conclusion); the mixed-precision map is grounded, not guesswork |
| **M5** | Serving engineering | continuous batching / paged KV / request scheduling | TTFT-TPOT curves at concurrency 1/4/8/16/32 + OpenAI API compatibility + raw stress-test data |
| **M6** | Report and open source | Three-dimensional Pareto + honest conclusions | `report.md` complete + `gate.json` conclusions for all tiers + the repo reproducible by a stranger in one go |

---

## 3. Milestone-by-Milestone Details

### M0 · Environment and Self-verification (estimated 0.5–1 day)

**Mechanism**: WSL2's GPU is a three-layer structure: "Windows driver + `/dev/dxg` passthrough + CUDA user-space libraries inside WSL". Hence the driver version shown by `nvidia-smi` inside WSL **necessarily equals** the Windows side (592.01) — this is the only reliable signal for judging whether passthrough is normal. The CUDA toolkit version and the driver version are two different things; do not conflate them.

**What you need to do**:
1. Run the verification inside WSL (paste the results to me):
   ```bash
   nvidia-smi                      # 驱动版本应为 592.01
   ls /dev/dxg                     # 存在=透传正常
   lscpu | head -5; free -g        # 分配了几核几 G
   df -h ~                         # 权重+引擎要预留 ≥ 80GB
   ```
2. Check/adjust WSL resources (Windows side `C:\Users\21327\.wslconfig`):
   ```ini
   [wsl2]
   memory=48GB        # 按你实机 RAM 调整（我这边读不到 RAM，你确认）
   processors=12
   swap=16GB
   ```
   After changing, run `wsl --shutdown` and re-enter. Reason: for a 7B-class model the **host memory** peak (loading + quantization calibration) often blows up earlier than VRAM.
3. Install the CUDA Toolkit inside WSL (**toolkit only**):
   ```bash
   # 选与驱动 592.01 兼容的版本，装 runfile 时取消 driver 勾选
   sudo apt-get install -y build-essential
   # CUDA 12.x/13.x 二选一，与后面 TRT-LLM wheel 的 CUDA tag 对齐
   ```
4. Create an independent environment (**do not** reuse the Windows-side conda env):
   ```bash
   conda create -n llmquant python=3.12 -y   # 或 venv，你定
   ```
5. Install TRT-LLM: prefer the **prebuilt wheel from NVIDIA's official pip index** rather than compiling from source:
   ```bash
   pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
   ```
   That `modelopt 0.46.0` on the Windows side serves only as a **reference version number**; inside WSL install the modelopt that matches TRT-LLM.

**The three read-back questions (you must be able to answer them aloud; if you cannot, this step is not done)**:
1. Who determines the CUDA version seen inside WSL? Why does installing a Linux driver break WSL?
2. Which execution path does your TRT-LLM version take — the **engine build path** or the **PyTorch backend direct-load path**? Where do the two differ in their requirements on the quantized weight format?
3. After switching `HF_ENDPOINT` to a mirror, who is responsible for the **verification** of model downloads? Are the mirror's files and the official repository's files comparable one by one?

**The acceptance standard I give you**: paste the real output of the commands above + correct answers to the three questions. Any failure → do not proceed to M1. **Do not skip this and go install the model** — this is the source of 80% of the project's pain.

---

### M1 · Freeze the Measuring Stick First (estimated 1 day, the most important in the whole project)

**Mechanism**: the vast majority of academic failures in quantization projects are not because the quantization was done wrong, but because **the stick was changed**. You already paid for this once on the EZTrain DualTac case — after fixing the pairing bug, clean RMSE went from 0.15935 to 0.116 and all the old "hit rate / ranking" conclusions were invalidated. Repeating that lesson here costs three days.

**`docs/measurement_spec.md`, which must be written down first, must contain at least**:
- A fixed prompt set (≥20 items, mixed long and short; **write the prompt file hash into the result json**, otherwise it is not comparable)
- Sampling parameters: greedy (temperature=0) for reproducibility; a fixed max_new_tokens
- Metric definitions:
  - **TTFT**: time to first token (from request issue to first token returned)
  - **TPOT/ITL**: average per-token time excluding the first token
  - **Throughput**: tokens/s (two conventions — per single request and aggregate; do not mix them)
  - **Peak VRAM**: prefer the value reported by the engine/process itself; `nvidia-smi` only as corroboration
  - Always report **p50 / p95 / p99**, never the mean
- Measurement discipline: discard ≥ 3 warmup rounds; `torch.cuda.synchronize()` before and after timing; repeat each configuration ≥ 5 times
- Environment snapshot: GPU temperature, power, clock, current batch, software version numbers — **all written into the result json**

**What you need to do**:
1. Write `measurement_spec.md` (show it to me first; I review it)
2. Bring up a minimal OpenAI-compatible server using **native HF transformers** (no rush for TRT) and get the bench flow working end to end → produce `results/baseline/bf16_hf/`
3. This HF baseline is **not the final oracle**; it only lets the stick take readings first. The real oracle is re-measured with TRT-LLM BF16 (unquantized fp16/bf16 engine) before M2, and the **differences between the two and their causes must be explicitly recorded** (this itself is a highlight in the report: explaining "why native HF latency cannot serve as a deployment baseline").

**The three read-back questions**:
1. Which part of the computation dominates TTFT and TPOT respectively? Why can chunked prefill improve TTFT while possibly worsening TPOT?
2. Why must p99 be reported? At what concurrency does the mean lie to you?
3. In your bench script, which line guarantees that the "measurement window" does not include model loading/compilation time?

**gate**: `measurement_spec.md` exists and I approve it + bench can stably reproduce the same set of numbers (deviation between two runs < 5%; if exceeded, find the cause before continuing).

---

### M2 · W8A8 INT8 PTQ (estimated 1–2 days)

**Mechanism**: the essence of PTQ is **using the activation distribution over a batch of real inputs to decide which integer interval this tensor's dynamic range should be mapped onto**. W8A8 = both weights and activations quantized; activation quantization is much harder than weight quantization, because activations are **data-dependent at runtime** (the outlier channel is the core enemy of LLM quantization), whereas weights are static. So the keyword for W8A8 is **calibration**: what the calibration set is, how many samples, per-tensor or per-channel, and whether smoothing/scale equalization should be done first.

⚠️ ModelOpt is already on your machine; use its PTQ flow (`modelopt.torch.quantize`) rather than building your own wheel; state the quantizer type for `w8a8` explicitly in the config.

**What you need to do**:
1. Choose the calibration set (128–512 items recommended; the source must be **different** from your M1 prompt set to avoid evaluation leakage — this must be written into the report)
2. Run PTQ → export quantized weights → load with TRT-LLM (engine or PyTorch backend)
3. Run `06_bench.py` + `07_eval_accuracy.py` and compare against the BF16 oracle in the same session
4. Perform at least one **calibration-set-size ablation** (e.g. 64 / 128 / 512) to see whether the accuracy loss converges with calibration volume

**The three read-back questions**:
1. Why do activation outliers destroy per-tensor quantization? Why does per-channel not help activations?
2. If your calibration set and evaluation set overlap, what illusion appears? How do you prove they do not overlap?
3. After INT8 quantization, how much did **peak VRAM** actually drop? If the drop is far smaller than the intuition of "weights halved", what does that say is the dominant memory term?

**gate**: `results/w8a8/` has the complete trio + a clear accuracy delta + you can explain where the VRAM reduction comes from.

---

### M3 · FP8 + KV Cache Quantization (estimated 1–2 days)

**Mechanism**: the fundamental difference between FP8 and INT8 is that **FP8 retains the exponent bits** — wide dynamic range, no need to fight outliers, at the cost of low mantissa precision (E4M3 has only 3 mantissa bits). Blackwell has native tensor-core support for FP8, so on a 5090 FP8 is usually the optimal tier for "smallest accuracy loss, faster speed".

What is genuinely underrated is **KV cache quantization**: as context grows, the KV cache's VRAM footprint **grows linearly and is unaffected by the weight bit width**. You quantize weights to cut 16GB down to 8GB, but if a 32k context needs 6–10GB of KV, then your concurrency ceiling is actually determined by KV, not by the weights. **This is an interview question many people cannot answer.**

**What you need to do**:
1. FP8 W8A8 quantization (again with ModelOpt) → bench + eval
2. Quantize the KV cache separately (INT8 or FP8, depending on backend support) → measure again
3. **context length sweep**: 1k / 4k / 16k / 32k, recording peak VRAM and TTFT/TPOT, and plotting two curves:
   - weight VRAM (fixed) vs KV VRAM (growing with length)
   - context length vs maximum feasible concurrency

**The three read-back questions**:
1. FP8's E4M3 and E5M2 — which are suited to quantizing weights and which to activations? Why?
2. Which metric does KV cache quantization damage — "accuracy" or "behavioral consistency at long context"? (Hint: the difference between per-token and per-tensor K/V scaling factors is amplified here)
3. On your 24GB, when running a 32k context, how much do weights / KV / activations / framework overhead each take? Which is the bottleneck?

**gate**: the two curves have real data points + you can state clearly "at what context length my deployment hits the wall because of KV rather than weights".

---

### M4 · 4bit and Mixed Precision (estimated 2–3 days, the technical core of this project)

**Mechanism**: 4bit is the means to break the VRAM wall, but 4bit's enemy is **sensitive layers**. In an LLM, by no means are all layers equally fragile — the empirical rule is isomorphic to what you found on DualTac: **the closer to the output, and the more a layer carries fine-grained regression/probability resolution, the more fragile it is**. So the correct approach is not "uniform 4bit across the whole model", but rather:

> Probe the marginal damage of each layer/module → under a given VRAM budget, solve the **mixed-precision allocation problem** of "which layers go to 4bit and which stay at 8bit/16bit".

**This is exactly the LLM version of the `probe_prefixes` → `width_map` → worker → gate chain in EZTrain.** Bringing that old craft over is what distinguishes this project from "I ran AWQ".

**What you need to do**:
1. **First measure whether NVFP4 actually holds on sm_120** (do not assume!):
   - Check the `sm` support matrix for NVFP4 in the TRT-LLM version you installed
   - Actually build an NVFP4 engine once: if it succeeds, record it; if it fails, record the **verbatim error text** in the report — **evidence of failure is also a conclusion** ("on consumer Blackwell, NVFP4 requires condition X, which my environment does not satisfy" is itself a valuable engineering conclusion)
2. If it does not hold, fall back to W4A16 (the AWQ/GPTQ route) as the 4bit tier
3. Write `09_sensitivity.py`: layer-by-layer/block-by-block measurement of the marginal damage of "demote this segment to 4bit, keep the rest at 8bit" (**note**: the LLM layer order is a purely linear chain, far easier to handle than DualTac's branch graph; but be careful about the weight tying of embedding/lm_head — do not double count)
4. Produce a mixed-precision map (`configs/mixed_*.yaml`) and compare its accuracy against uniform 4bit under the same VRAM budget

**The three read-back questions**:
1. Is 4bit's quantization granularity (per-channel / per-group, what group size) more or less sensitive to accuracy than 8bit? Why?
2. Does your sensitivity probe measure "single-layer isolated damage" or "prefix cumulative damage" on the LLM? Would the two give the same conclusion? (Recall your DualTac lesson: single layer 1e-4 vs whole model +2.0% — damage is **not additive**)
3. Is the mixed-precision map search "greedy walking the dataflow order under a budget" or "global optimum"? What is your choice and what does it cost?

**gate**: the 4bit tier (whether NVFP4 or W4A16) has real usable numbers + the accuracy advantage of mixed precision over uniform 4bit is measurable + the sensitivity method has a reproducible script.

---

### M5 · Serving Engineering (estimated 1–2 days)

**Mechanism**: the difference between a service that "can infer" and one that "can go live" is **scheduling**. Two core concepts:
- **Continuous batching**: do not wait for the whole batch to finish; whoever finishes first yields its slot — turning GPU utilization from "waiting for the slowest request" into "always having new work"
- **Paged KV cache**: KV cache allocated in blocks, eliminating fragmentation, and directly determining how much concurrency you can serve at once

These two are **coupled** with quantization: the VRAM saved by quantization ultimately cashes out as a **larger concurrent batch**, not as a faster single request. So your report must have a "VRAM saving → concurrency gain" transfer table; that is the language of a deployment engineer.

**What you need to do**:
1. Bring up an OpenAI-compatible service (TRT-LLM's own server, or wrap a FastAPI layer yourself)
2. Concurrency sweep: 1 / 4 / 8 / 16 / 32 concurrency, recording TTFT, TPOT, total throughput, peak VRAM
3. A/B comparison (a bonus item, very valuable): run **the same quantized weights** on TRT-LLM and vLLM respectively and compare throughput/latency — proving that what you understand is the "mechanism" rather than "one framework's manual"
4. Stress test with a real client (not `time.sleep` in a script), ideally a mixed workload of 1 long-context request + multiple short requests

**The three read-back questions**:
1. When concurrency goes from 4 to 16, total throughput rises but per-request TPOT worsens — why? Which curve hits its ceiling first?
2. Does Paged KV solve fragmentation or capacity? What happens if you conflate the two?
3. Quantization saves you X GB of VRAM — how many requests of concurrency does that convert to? (Must be derived from measured data; no estimating)

**gate**: the concurrency curve has real data + the OpenAI API can actually be called successfully by an external client + the VRAM→concurrency transfer table holds.

---

### M6 · Report, Open Source, and Resume Mapping (estimated 1 day)

**What you need to do**:
1. `docs/report.md`: methodology → experimental setup → four-tier data table → Pareto chart → **honest half-hit conclusions** → limitations (laptop power limit, single card, thermal drift)
2. `scripts/08_gate.py`: write the gate verdicts into `results/gate.json`, including PASS/FAIL and the reason for each tier
3. Make the repository public (⚠️ **push only code/configs/results/report; never push weights or engine files**; hard-code `*.engine`, `*.safetensors`, `*.pt` in `.gitignore`)
4. Put a conclusion table at the top of `README.md` that an HR person/interviewer can understand in 30 seconds

**Resume mapping draft (fill in the real numbers after you are done)**:
- "Implemented the full PTQ pipeline (W8A8 INT8 / FP8 / 4bit / KV cache) on Qwen3-8B, holding accuracy loss to __% and reducing peak VRAM by __%"
- "Measured the support boundary of NVFP4 on an RTX 5090 Blackwell (sm_120), giving the conclusion __"
- "Built a mixed-precision allocation scheme based on layer-wise sensitivity probes, improving accuracy by __% over uniform 4bit under an equal VRAM budget"
- "Completed continuous batching + paged KV tuning on the TensorRT-LLM server; __ tok/s throughput and TTFT p99 __ ms at __ concurrency"
- "Ran an A/B benchmark of the same quantized weights on TRT-LLM and vLLM, locating the source of the differences"

---

## 4. Global Risks and Contingency Plans

| Risk | Trigger signal | Contingency |
|---|---|---|
| WSL CUDA/driver incompatibility | Abnormal `nvidia-smi` at M0 or torch.cuda False | Downgrade the WSL kernel/CUDA combination; in the extreme case fall back to native Windows + ONNX Runtime |
| TRT-LLM prebuilt wheel does not match the CUDA tag | pip installs it but import fails | Switch the CUDA version to align the wheel; do not compile TRT-LLM from source on Windows |
| NVFP4 does not hold on sm_120 | Engine build reports sm unsupported | Fall back to W4A16 (AWQ/GPTQ) and write "does not hold" as a report conclusion (**this too is a result**) |
| Laptop thermal drift makes data incomparable | Two results for the same configuration differ by > 10% | Fix the power mode, wait for the temperature to fall before each measurement, annotate the temperature in the report; if necessary cap clocks in exchange for stability |
| Downloading the 16GB of weights fails/is extremely slow | Direct HF connection unavailable | `HF_ENDPOINT=https://hf-mirror.com`; use `huggingface-cli download` with resume support |
| Calibration/evaluation data leakage | The accuracy delta looks suspiciously good | The calibration set and evaluation set must come from different sources, and both hashes must be listed in the report |

---

## 5. My Acceptance Protocol for You (read-back system)

At the end of each milestone, you give me **three things**:
1. **Raw evidence**: command output / the json under results / curve plots
2. **Mechanism restatement**: answer that milestone's three read-back questions in your own words (not a copy of my explanation)
3. **Next-step judgment**: whether you think you should continue, go back, or change route, and why

I judge PASS only when all three hold simultaneously: "evidence complete + mechanism explained coherently + gate met". **A half-hit is not a failure** — write down what the data says; that is worth more than all-green.

---

## 6. M0 read-back Reference Answers (agent version · once the user confirms by restating, "environment ownership" is deemed met)

**Q1 · Who determines the "CUDA version" inside WSL? Why does installing a Linux graphics driver inside WSL destroy the GPU stack?**

- Three-layer structure: **Windows kernel driver** (the only layer that touches the hardware, 592.01) → `/dev/dxg` passthrough → `/usr/lib/wsl/lib` (bind-mounted from the Windows driver directory, and itself an overlayfs: `gpu_lib_packaged` over `gpu_lib_inbox`)
- The "tool version line" of `nvidia-smi` (590.62) comes from a **compile-time string in the tool binary**; the "Driver Version" line (592.01) is read **at runtime** from the kernel driver by NVML. The mismatch = the mounted user-space library lags one minor version → N-1 skew, common and harmless
- Only two relationships are actually binding on applications: **the driver capability ceiling (CUDA 13.1) ≥ the CUDA runtime version the program links against**; **the compile-time toolkit version does not affect execution** (`/usr/local/cuda` or the apt 13.3 is only a compilation environment)
- Installing a Linux driver inside WSL creates a second driver in kernel space, competing with the Windows driver for the same GPU, and breaks `/dev/dxg` passthrough and the mount assumptions of `/usr/lib/wsl/lib` → the entire GPU stack fails immediately

**Q2 · Which execution path does TRT-LLM take? Where do their requirements on the quantized weight format differ?**

- **① engine build path**: `trtllm-build` → produces a `.engine` file. **Strongly bound to the TensorRT version and GPU architecture (sm_120); not portable across versions/architectures**; quantization metadata must go through the conversion flow together with the checkpoint
- **② PyTorch backend direct-load path**: `tensorrt_llm._torch` loads the quantized checkpoint directly, **with no engine file**, faster iteration, no recompilation tax
- Strategy for this project: M2–M4 use **②** (fast iteration across the three quantization tiers), and in the M5 serving phase the **optimal tier** uses **①** to capture the latency gain — the measured difference between the two paths is itself a comparison table in the report

**Q3 · How do you prove that weights downloaded from a mirror match the official ones? At which step in the flow does verification happen?**

- Mechanism: on the Hub side every file has an LFS SHA256 (`lfs.oid`); after downloading, the client writes metadata under `.cache/huggingface/download/**` (including commit hash / etag) and verifies locally
- Three-step verification: ① **pin the commit revision** (use the 40-character sha, not `main`) ② after the download completes, **recompute sha256 locally for each shard** and compare it one by one against the `lfs.oid` returned by the API ③ record the file list and total byte count of `model.safetensors.index.json`
- Where it happens: **at the moment the download completes**, and write `(repo_id, revision, per-file sha256)` into `docs/env_matrix.md` and every result json — otherwise M2's accuracy numbers cannot be traced back to a determinate set of weights

---

## 7. M1 Starting Point

M0 has PASSED. M1's first deliverable is **`docs/measurement_spec.md` (the measurement convention spec)**, which must be frozen before any benchmark script exists; the required fields are in the §3 M1 entry. Acceptance order: spec → I review → only then write the bench script.

---

## 8. M2 Record: INT8 Fails, FP8 Passes (2026-09-14)

### 8.1 The Pit Fixed First: ModelOpt's CUDA Extension Had Never Actually Been Loaded

**Two mutually independent defects**; without fixing either one the extension cannot be compiled:

| # | Mechanism | Fix |
|---|---|---|
| A | When `CXX` is unset, torch resolves the compiler to `c++`, whose compatibility check **throws a `TypeError` while printing the warning** (mismatched number of format arguments), crashing before any compilation happens | Point `CXX` at the real `g++` (`/usr/bin/g++` exists) |
| B | The failing file is a **`.cu`** file; `-std=c++20` added only to `extra_cflags` **never reaches nvcc**; torch 2.9.1's bundled headers (`decltype(...)::difference_type` at `ATen/core/List_inl.h:201`) fail to compile under C++17 | `-std=c++20` must be added to **both** `extra_cflags` and `extra_cuda_cflags` |

**Measured evidence per variant**: default `-std=c++17` ❌ / c++20 only in `extra_cflags` ❌ / **both flags added ✅** / `-fpermissive` ❌ / c++20+fpermissive ❌.
After the fix, `modelopt_cuda_ext`, `_fp8`, and `_mx` are all LOADED (38.5s / 37.0s / 45.9s, then cached at 0.2–0.6s). The fix is packaged in `scripts/modelopt_ext_patch.py` and **must be imported before importing modelopt**.

**Lesson**: `Unable to load extension ... falling back to CPU version` is only a **warning**, and a warning does not terminate the flow — so INT8 kept running on the CPU implementation, while FP8 directly produced `nan`. This matches the pattern that recurs throughout this project: **failures do not raise errors; they just produce a plausible-looking number.**

### 8.2 Three-Tier Quantization Results (128 calibration samples · 32752-token perplexity · BF16 oracle = 9.3190 · threshold +2%)

| Recipe | Perplexity | Relative change | Verdict |
|---|---|---|---|
| **FP8 (W8A8 float8)** | **9.3434** | **+0.26%** | ✅ **PASS** |
| INT8 + SmoothQuant | 14.0729 | +51.01% | ❌ FAIL |
| INT8 per-tensor (`INT8_DEFAULT_CFG`) | 19.7043 | +111.44% | ❌ FAIL |

**Key control**: the two INT8 values are **bit-for-bit identical** between the "CPU fallback" run and the "CUDA extension" run → the CPU fallback is numerically equivalent, so +51% / +111% is a property of **the scheme itself**, not a build artifact. FP8, by contrast, went from `nan` to +0.26% — previously it simply could not run.

**Mechanism**: INT8's activations use **static per-tensor scaling**, and the scale factor is dominated by a few extreme outliers (calibration observed `down_proj` input amax reaching 1152), flattening the rest of the distribution; FP8 retains the exponent field, whose dynamic range naturally covers outliers, hence near-lossless on the same model and the same calibration set. **This is precisely the "outlier channel" theory repeated since M0, now backed by measured numbers.**

**Conclusion (honest version)**: the "W8A8 INT8" originally planned for M2 **does not hold** on Qwen3-8B, and it is not a matter of parameters not being tuned well — SmoothQuant only recovers half of it. The viable tier is **FP8**. For the report this is good material: one refuted scheme plus one validated alternative is more persuasive than "everything passed".

### 8.3 M2 Formal verdict: **FP8 PASS** (2026-09-14, adoptable)

Full evaluation (128 calibration samples · 32752-token perplexity · 500-question MMLU · exported checkpoint):

```
perplexity  9.3190 → 9.3434   +0.26%   (limit +2.0%)    PASS
MMLU        0.6120 → 0.6180   +0.60 pp (limit −4.0 pp)  PASS
VERDICT: PASS
```

**MMLU actually rose by 0.6 pp, while its standard error is 2.17 pp** → statistically **indistinguishable** from BF16. Conclusion: **FP8 W8A8 is nearly lossless on Qwen3-8B** and can serve as the deployment baseline for subsequent tiers.

**Exported artifacts (measured)**:

| Object | Size |
|---|---|
| BF16 original checkpoint | 16 GiB |
| **FP8 quantized checkpoint** | **8.8 GiB (−42%)** |

The difference can be explained exactly: the quantizers for the embedding and `lm_head` are `disabled` and are still saved in bf16 (2.49 GB), while the remaining 6.95 B of linear-layer parameters halve in bit width → a total of 9.44 GB = 8.79 GiB, matching the measurement.

**Note (convention boundary)**: M2 measured the accuracy of the **in-memory fake-quant model**; the imported checkpoint is for the server side. **Latency and VRAM gains are not within M2's gate** — the real gains only materialize at M5 after that checkpoint is loaded with TRT-LLM (inside HF transformers the weights still reside in bf16). This must go into the report, otherwise "FP8 saves half the VRAM" will be misread as an already-measured conclusion.

**Three pits fixed along the way this round**
1. Wrong placement of the `HF_ENDPOINT` setting: `huggingface_hub` reads that variable **at import time**, and `transformers` imports it first → writing it after the import **has no effect at all**, the requests still went to the unreachable huggingface.co, and each dataset retried 5 times before falling back to the local cache. It is now set at the very top of the file, before any torch/transformers import.
2. The calibration corpus was switched to a **local cache** (`data/calibration/calib_128.json`, sha `1e752ba4…`): the corpus is part of the measurement, its hash is recorded in the results and must be consistent across tiers; re-downloading it every time is handing your identity to the network.
3. `--smoke` no longer quietly shrinks the calibration set: the number of calibration samples is the main driver of quantization quality (8 samples ⇒ +55% perplexity, independent of the scheme). At the same time, **the gate now automatically returns `PROVISIONAL` instead of PASS/FAIL when evaluation conventions are inconsistent** — it nearly recorded "MMLU −16.2 pp" as a conclusion, whereas the 95% confidence interval for a 20-question sample is ±22 pp.

**Next (M3)**: take FP8 as the baseline tier, and do KV cache quantization plus a context-length sweep (1k / 4k / 16k / 32k). Qwen3-8B's KV cache is **144 KiB/token** (2×36 layers×8 kv_heads×128×2B), so a 32k context is **4.5 GiB** — once the weights are halved, KV genuinely becomes the dominant VRAM term, and that is exactly the phenomenon M3 is meant to measure.

---

## 9. M3 Record: KV Cache Quantization Is a **Context-Dependent Trade-off** (2026-09-17)

### 9.1 Measured Results (FP8 weight checkpoint · batch 1 · 128 tokens generated · fixed 33984-token pool)

| Context | auto (FP16 KV) | fp8 KV | fp8/auto |
|---|---|---|---|
| 1 024 | 71.7 tok/s | 53.7 | **0.75** |
| 4 096 | 64.0 | 51.1 | **0.80** |
| 16 384 | 31.1 | 32.7 | **1.05** |
| 32 768 | 17.8 | 22.6 | **1.27** |

**The crossover is around 16k.** At short context FP8 KV is **25% slower** (dequantization overhead dominates; the cache is small here, so saving memory is meaningless); at long context FP8 KV is **27% faster** (attention becomes VRAM-bandwidth-bound, and halving the KV bytes starts to win).

**KV pool VRAM as reported by the engine itself** (same 33984-token pool):

| KV precision | Pool allocation | Per token |
|---|---|---|
| auto (FP16) | **4.67 GiB** | 144 B |
| fp8 | **2.33 GiB** | 72 B |

**Exactly half**, consistent with the formula predicted in `model_provenance.md` (2×36×8×128×2 B = 144 B/token).

**The context's own latency wall**: auto goes from 71.7 → 17.8 tok/s (1k→32k, **a 4.0× drop**) — prefill cost rises with context.

### 9.2 Three Methodological Findings (worth more than the numbers)

1. **peak VRAM is completely insensitive to KV precision** (all 8 cells at 21.623 GiB). The pool is fixed and reserved, and the peak is dominated by weights and workspace. **Peak device VRAM is the wrong instrument for this question**; the right instrument is the engine's own pool allocation log (`Allocated X GiB for max tokens in paged KV cache (N)`).
2. **`max_num_tokens` defaults to 8192**: 16k/32k prompts are rejected outright (`RequestError: prompt length 16384 should not exceed max_num_tokens 8192`), and that exception aborted the entire sweep. It is now set explicitly and changed to **per-cell persistence**, so a single crash no longer loses all results.
3. **`nvfp4` KV cache hard-crashes the worker process** (MPI abort + backtrace, not a catchable exception); it has been excluded from the grid and recorded separately as a finding — directly relevant to M4's four-tier set.

### 9.3 API Constraints (measured)

For an FP8 checkpoint, `KvCacheConfig(dtype=...)` accepts only **`('fp8', 'nvfp4', 'auto')`**; `int8` is rejected:
```
ValueError: Overriding KV cache quantization with an invalid type "int8".
```
That is, **FP8 weights + INT8 KV is not supported on this stack**. The KV cache is a **runtime** configuration and is not written into the checkpoint (the exported `kv_cache_quant_algo: None` is the evidence).

### 9.4 The Value of This Result for the Report and for Interviews

It refutes the intuition that "quantization is always cheaper and faster" and gives a **reproducible, crossover-containing, mechanism-explained** conclusion:

> KV cache quantization pays a 25% throughput cost at 1k context and buys back a 27% gain at 32k, with a crossover around 16k; "whether to quantize KV" should be decided by the **deployment's actual context distribution**, not enabled by default.

---

## 10. M4 Record: 4-bit, Mixed Precision, and a Wildcard Bug (2026-09-17/18)

### 10.1 Good News: Weight-side NVFP4 Works on sm_120

```
uniform NVFP4（W4A4，128 校准样本，32752 token）  perplexity 9.5799  vs oracle 9.3190 = +2.80%
```

**+2.80% is quite good for 4-bit weights + activations**, in sharp contrast to INT8 (per-tensor +111%, SmoothQuant +51%). Mechanism: NVFP4's **block-wise microscaling** accommodates outlier channels — exactly the problem that destroyed per-tensor INT8.

**Contrast with M3**: `nvfp4` as a **KV cache** precision hard-crashes the worker process; as a **weight** precision it works fine. Same name, different path, opposite conclusions — this is precisely why isolation experiments are necessary.

### 10.2 Three Methodological Traps (the first version of the sensitivity probe hit all three; it has been rewritten)

| # | Cause | Evidence | Consequence |
|---|---|---|---|
| 1 | **Budget mismatch (my mistake)** | Same corpus under BF16: 4096→9.1705; 8192→**10.6146**; 16384→8.3988; 32768→9.3190 | BF16 itself swings ±14%; comparing 8192 against the 32768 oracle yields a fake "+18.56%", whereas the true value is about +4.1% |
| 2 | **disable/enable is not idempotent** | Same config, same tokens: disable all→10.6146 (exactly equal to BF16 at that budget, showing that disable works), then enable→**18.5262** (originally 11.0482) | 36 cycles progressively damage the model, producing a monotone fake ranking |
| 3 | **Lazy extension loading** | The log line `Loading extension modelopt_cuda_ext…` appears midway through the experiment | The first half runs on the CPU fallback and the second half on CUDA; the numerical path switches silently |

**Correction method** (aligning with EZTrain's width_map): write the protections into the **quantization config**, **re-quantize each candidate from BF16**; pre-load the extensions; fix the budget at 32768. The corrected version self-verifies: uniform 9.4870, protecting block 0 = +0.0826, consistent with the uncontaminated earlier probe.

### 10.3 Incidental Output: Perplexity Is Sensitive to the Token Budget (a hard rule)

`results/accuracy/budget_sensitivity_bf16.json` — same BF16 model, same corpus, only the number of scoring tokens changed:

| Budget | Perplexity | Relative to 32k |
|---|---|---|
| 4 096 | 9.1705 | −1.59% |
| 8 192 | 10.6146 | **+13.90%** |
| 16 384 | 8.3988 | −9.87% |
| 32 768 | 9.3190 | 0.00% |

**Perplexity is comparable only within the same token budget.** This is the third error of the same kind in this project (M1 same-session, M2 sample size, M4 budget).

### 10.4 Correction: the Wildcard Bug — "protect 5 blocks" actually protected 14 blocks

**How it was found**: the export size did not add up. The mixed tier exported **9.57 GiB, larger than FP8's 8.79 GiB** — 4-bit should not be larger. A per-tensor count:

```
bf16      245 tensors  7.35 GiB (76.8%)   ← 应为 147 个
U8        154 tensors  1.98 GiB (20.6%)
F8_E4M3   154 tensors  0.25 GiB
```

Only 154 linear layers are 4-bit, **98 remained in BF16**; "protecting 5 blocks" should have produced only 35. 98 ÷ 7 = **14 blocks** = `{0} ∪ {1,10,…,19} ∪ {12} ∪ {34} ∪ {35}`.

**Root cause**: the protection was written as two patterns, and the trailing `*` in `*model.layers.{b}*` **swallows the next digit**, so b=1 also matches `layers.10…19`; the other pattern `*layers.{b}.*` (with a dot) is safe, but both were active → "protect 1 block" became "protect 11 blocks".

**Two contaminations must be corrected**: ① the sensitivity ranking (block 1 appeared most sensitive at +0.1263, but that was actually the combined effect of 11 layers); ② the mixed-tier conclusion ("+0.56%, 9.57 GiB" was in fact the result of protecting 14 blocks).

**Measured after the fix (genuinely protecting 5 blocks: 0/1/12/34/35)**:

| Tier | Perplexity | vs oracle | MMLU | Size |
|---|---|---|---|---|
| BF16 oracle | 9.3190 | — | 0.6120 | 15.26 GiB |
| FP8 W8A8 | 9.3434 | **+0.26%** | 0.6180 | 8.79 GiB |
| **Mixed NVFP4 (5 blocks, corrected)** | **9.4634** | **+1.55%** | **0.5560 (−5.6 pp)** | **7.25 GiB** |
| Mixed (14 blocks, buggy version) | 9.3716 | +0.56% | 0.5800 (−3.2 pp) | 9.57 GiB |
| uniform NVFP4 | 9.5799 | +2.80% | not measured | to be measured |

**M4 honest conclusion**:
- By decision 16 (perplexity ≤ +5%, MMLU ≥ 0.572): the corrected mixed tier **FAILs on MMLU**, and PASSes on perplexity.
- But **MMLU's resolution at N=500 is ±4.3 pp**, and the threshold 0.572 falls inside [0.513, 0.599] → **this FAIL is indistinguishable from noise**.
- The mechanism direction is consistent: more protection is better (14 blocks→0.5800; 5 blocks→0.5560).

### 10.5 M4 Final Verdict: Paired Comparison after Raising MMLU to N=2000 (2026-09-18)

**The same batch of 2000 questions** was fed to BF16 and the 4-bit mixed tier respectively (same seed ⇒ same question subset, identical `sample_sha256`):

| Model | MMLU (N=2000) | Standard error |
|---|---|---|
| BF16 | **0.6260** | ±0.0108 |
| Mixed NVFP4 (5 blocks protected) | **0.5490** | ±0.0111 |
| **Difference** | **−7.70 pp** | pooled SE 1.55 pp → **z = 4.96** |

**Verdict: conclusively significant (p ≪ 0.001); the real loss of the 4-bit tier is about 7.7 percentage points.** Decision 16's MMLU threshold (≥0.572) is clearly breached, and this is no longer a dispute over the noise boundary.

**A live demonstration of sampling fluctuation (same model, same config, only a different 500 questions)**: `0.5560` (N=500 subset A) vs `0.5080` (the first 500 of the N=2000) — **a 4.8 pp difference, purely from question sampling**. This is exactly why M1 insisted on annotating N and resolution, demonstrated live.

### 10.6 M4's Most Important Scientific Conclusion: Perplexity **underestimates** task-level damage

```
困惑度：  9.3190 → 9.4634   =  +1.55%      （看似几乎无损）
MMLU  ：  0.6260 → 0.5490   =  −7.70 pp    （配对、z=4.96，确凿受损）
```

**Perplexity rises only 1.55%, yet MMLU drops 7.7 percentage points.** This refutes the project's earlier assumption — M1/M2 had designated perplexity the "real gate" (because it is dense, low-variance, and able to resolve sub-percentage-point changes) and demoted MMLU to a "coarse screen". **In this case the dense metric severely underestimated the damage.**

**Corrected methodology (to be written into the report)**:
- **Dense metrics are precise but insufficient**: they can resolve a +0.5% change, yet cannot see the narrow, deep damage to "reasoning ability".
- **Task-level metrics have low resolution but are irreplaceable**: keeping MMLU under decision 12 paid off — it is what caught what perplexity missed.
- Conclusion: **both must be reported together, and both gates must be in force simultaneously**; passing either one alone does not constitute "lossless".

### 10.7 M4 Final Tier Table (all measured)

| Tier | Perplexity | vs oracle | MMLU | Size | Verdict |
|---|---|---|---|---|---|
| BF16 oracle | 9.3190 | — | 0.6260 (N=2000) | 15.26 GiB | baseline |
| **FP8 W8A8** | 9.3434 | **+0.26%** | 0.6180 (N=500) | 8.79 GiB | ✅ **both gates pass → deployment tier** |
| Mixed NVFP4 (5 blocks, corrected) | 9.4634 | +1.55% | **0.5490 (N=2000)** | 7.25 GiB | ❌ MMLU −7.70 pp |
| Mixed NVFP4 (14 blocks, buggy) | 9.3716 | +0.56% | 0.5800 (N=500) | 9.57 GiB | void (wrong pattern) |
| uniform NVFP4 | 9.5799 | +2.80% | not measured | to be measured | reference |
| INT8 SmoothQuant / per-tensor | 14.07 / 19.70 | +51% / +111% | — | — | ❌ unusable |

**Deployment recommendation**: **choose FP8.** 4-bit looks viable on perplexity (+1.55%), but a 7.7 pp task-level accuracy loss is unacceptable; on the standard of "both gates passing cleanly", FP8 is the only qualifying tier. This is also cognate with M2's INT8 failure: **on this model/stack, the activation side cannot afford anything below 8 bit.**

**Process lesson (the fourth of its kind)**: v2 of `09_sensitivity.py` exists only in WSL while the workspace still has v1, and a subsequent bidirectional `cp` overwrote the workspace's ROADMAP §10. **Scripts and documents must have exactly one authoritative copy, and synchronization must be one-way only.**

## 11. M6 Record: Report and Release (2026-09-18)

**Outputs**: `README.md` (the 30-second facade: a table of six tiers + the migration chain + an index of ten defects), `docs/report.md` (the full technical report: method / results / methodological findings / defect post-mortems / limitations / conclusion), `LICENSE` (MIT).

**Pre-release check**: `git diff --cached` contains no `*.safetensors / *.bin / *.pt / *.engine / *.onnx / *.gguf` → **zero weights and engines committed**; the repo's `.git` is only 1.6 MB.

**The report's core narrative (three points)**:
1. **FP8 is the deployment tier**: +0.26% perplexity, MMLU within noise, size −42%, throughput 1.5–1.7×, TPOT −40%, KV capacity ×2.28.
2. **8-bit is the floor**: INT8 fails catastrophically (+51%/+111%); 4-bit still loses 7.70 pp MMLU even with sensitivity protection.
3. **Dense metrics underestimate task-level damage**: perplexity +1.55% while MMLU −7.70 pp (paired N=2000, z=4.96) — both gates must be in force simultaneously.

**Release command (pending the user's decision on public/private)**:
```bash
gh repo create llm-quant-serve --public --source=. --remote=origin --push
```

**M0–M6 all complete.**

### 11.1 Bilingual documentation (2026-09-18)

Every document in the public repository now exists in both languages: the English primaries
`README.md`, `docs/report.md`, `docs/measurement_spec.md`, `docs/glossary.md` and
`docs/how_to_run.md` each gained a `*.zh-CN.md` counterpart, and this Chinese-primary file gained
`ROADMAP.en.md`. Each carries a language-switcher link under its H1. `model_provenance.md` is
generated by a script and stays English-only.

**Fidelity is enforced by a script, not by good intentions.** `scripts/check_docs_bilingual.py`
was added: its invariant is that every figure appearing in the source document must also appear
in its translation, and it checks heading, code-block, table-row and section counts plus the
switcher links, with `-v` printing per-section line counts. It is deliberately structural only —
prose may be reworded, but a measurement may not disappear.

**It caught a real bug on its first run.** The Chinese README's switcher line had been copied
verbatim from the English one, so the `**English**` label pointed at the Chinese page itself:
clicking "English" returned the reader to where they already were. The script reported
`NO LANGUAGE SWITCHER`, and after the fix all six document pairs pass.


**Two documentation defects surfaced during the bilingual cross-check and were fixed in the
same pass.** (a) In `docs/measurement_spec.md` the `Decision 1` justification paragraph sat
inside the §1 table; GFM ends a table at the first non-row line, so the four rows after it —
`max_new_tokens` 2, `Sampling` 3, `Seed` 4, `Thinking mode` 5 — rendered as pipe-containing
text rather than table rows. The paragraph now follows the table. (b) README and report claimed
"16 decisions" while the spec's own prose said "fifteen" and the numbers actually referenced
are 14 (1–5 and 7–15; 6 and 16 unused, now stated in the spec). The first audit of that count
matched only the literal phrase "Decision N", missed the bare index column of the two
`| Field | Decision | Value |` tables, and under-reported the total as 11 — the wrong number
reached the owner before the table columns were checked.


_Generation time: 2026-09-12 · This file is maintained by the agent; come back and update §1 and §4 when new facts appear within a milestone._
