# LLM Quantize + Serve · 独立完成路线图

> 项目代号：**Qwen3-8B PTQ → TRT-LLM Serving**
> 模式：**你自己动手，我只做机制讲解 + 检查点验收（read-back）**。我不代写实现代码。
> 目标：一份可直接作为简历项目链接的公开仓库，含可复现脚本、原始数据、量化报告。

---

## 0. 项目定义

**一句话**：把 Qwen3-8B 从 BF16 基线出发，经过 INT8 / FP8 / 4bit 三档 PTQ，落到 TensorRT-LLM 服务端，并在**同一把测量尺**下量化每档的「精度损失 × 延迟收益 × 显存收益」，最后给出一个真实可压测的 OpenAI 兼容服务。

**为什么这个项目值钱**（招聘方看的三件事，必须都出现）：

1. **你真懂量化的分类学**：W8A8 vs W4A16 vs FP8 vs NVFP4、weight-only vs weight+activation、per-tensor vs per-channel vs per-group、KV cache 量化——而不是"我跑了 bitsandbytes"。
2. **你真懂部署的物理约束**：TTFT/TPOT/吞吐/峰值显存/并发曲线之间是**互相拉扯**的（KV cache 是长上下文下的显存主项，批大小与 TPOT 线性恶化），能拿数据说清 trade-off。
3. **你敢报半命中的结论**：哪一档量化真的赚了、哪一档是坑、哪一档在你的硬件上直接不成立（sm_120 的 FP4 支持是**待实测**项，不是假设项）。诚实数字比全绿数字值钱。

**最终交付物**（仓库结构，先立骨架，脚本自己填）：

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

## 1. 前置事实（2026-09-12 实测，非推测）

| 项 | 实测值 | 影响 |
|---|---|---|
| GPU | RTX 5090 Laptop，24463 MiB，**驱动 592.01 / CUDA 13.1**，功耗墙 **173W** | 24GB 放得下 Qwen3-8B BF16 oracle；173W 意味**必须控温**，否则延迟数据不可比 |
| 架构 | Blackwell **sm_120** | FP8 有原生支持；**NVFP4 是否可用必须实测**（见 M4） |
| Windows CUDA | CUDA Toolkit **v13.3**（nvcc 在） | ⚠️ 这只服务 Windows 侧，WSL 里要**另装 Linux CUDA toolkit** |
| `modelopt` env（Win） | Py3.11.16 · nvidia-modelopt **0.46.0** · onnxruntime-gpu 1.29.0 | Windows 侧的工具链，先别动它 |
| `openvla`/`lingbot-vla` env | **torch 2.14.0+cpu / 2.7.1+cpu** | 全部是 CPU build，与本项目无关，别复用 |
| WSL2 | 用户确认可用（我这边被沙箱挡住，M0 需你自证） | 决定 TRT-LLM/vLLM 是否能上 |
| HF 网络 | 过往记录：HF 官方直连不通 → 用 `HF_ENDPOINT=https://hf-mirror.com` | 8B 权重约 16GB，下载策略要在 M0 定 |

### 1.1 M0 实测记录（2026-09-12，用户实机输出）

**WSL 侧（Ubuntu 24.04.4 LTS，conda base = /home/jiming/miniconda3）**

| 项 | 实测 | 判定 |
|---|---|---|
| `/dev/dxg` | 存在 | ✅ 透传正常 |
| WSL `nvidia-smi` 驱动行 | `Driver Version: 592.01`（= Windows 侧） | ✅ NVML 读到的内核驱动就是当前驱动 |
| WSL `nvidia-smi` 工具行 | `NVIDIA-SMI 590.62` | ⚠️ 工具二进制比驱动旧一个次版本，属 N-1 skew |
| `nvidia-smi` 血统 | `/usr/bin/nvidia-smi → /usr/lib/wsl/lib/nvidia-smi`，`dpkg -S` = 不属于任何 deb 包 | ✅ **只有一份**，是 WSL 驱动挂载的，不是 apt 装的 |
| `dpkg -l \| grep nvidia-driver/kernel/dkms` | 无 | ✅ **未踩红线** |
| WSL CUDA | apt 装的 **CUDA 13.3 全套**（cuda-nvcc-13-3 / cudart / cublas / nsight…，cuda-keyring 1.1-1） | ✅ 但**版本 tag 必须与 TRT-LLM wheel 对齐** |
| `/usr/lib/wsl/lib/` | `libcuda.so`、`libnvidia-ml.so.1`(Jan 2026)、`libnvidia-gpucomp.so.590.62 -> libnvidia-gpucomp.so`（符号链接日期 = 今日 WSL 启动时间） | 见下条「挂载结构」 |
| **`/usr/lib/wsl/lib` 挂载结构** | **overlayfs**：`lowerdir=/gpu_lib_packaged:/gpu_lib_inbox`, `upperdir=/gpu_lib/rw/upper` | ✅ **590.62 的来源锁定**：GPU 用户态库由「WSL 发行包自带(packaged)」+「Windows 驱动 inbox」两层叠加，packaged 层优先 → 工具二进制比实际驱动旧一个次版本 |
| `/usr/lib/wsl/drivers` | 9p 只读挂载（Windows DriverStore 直通） | 可在 WSL 内直接核对驱动文件血统 |
| WSL 内核 | `6.18.33.2-microsoft-standard-WSL2` | ✅ 记录备查 |
| `llmquant` env | Py3.12.14 · **torch 2.14.0+cu130**（运行时 CUDA 13.0 ≤ 驱动上限 13.1 → 合法）· triton 3.8.0 · cudnn 9.24 | ✅ |
| torch `arch_list` | `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` | ✅ **含 sm_120**，无 PTX JIT 兜底风险 |
| GPU 枚举 | `avail True` · sm_120 · **82 SMs**（82×128 = 10496 FP32 lanes，与 5090 Laptop 规格吻合） | ✅ 硬件被完整识别 |
| FP32 matmul 冒烟 | 4096³ = **10.187 ms → 13.5 TFLOPS** | ⚠️ **禁止当性能基线**：冷启动未剔除 warmup、TF32 未开、P8 低功耗态。仅证明 kernel 能跑 |
| 发行版 vhdx | `D:\WSL\Ubuntu-24.04\ext4.vhdx`（D 盘空闲 544.7GB） | ✅ 项目 70–100GB 落 D 盘无压力 |
| WSL 内存 | **31GB** + swap 8GB（主机约 64GB） | ⚠️ 偏低，建议 .wslconfig 提到 48GB |
| WSL 磁盘 | `/dev/sdd` 1007G，可用 **732G** | ✅ |
| 核数 | 24 | ✅ |
| conda base Python | **3.14.6** | ❌ **真阻塞**：TRT-LLM / ModelOpt 不支持 py3.14，必须新建 py3.12 env |

**M0 关键结论（机制）**：WSL 的 GPU 用户态库由 Windows 驱动目录 bind-mount 进 `/usr/lib/wsl/lib`。`nvidia-smi` 的「工具版本行」是**二进制里编译死的字符串**（590.62），而「Driver Version」是**运行时通过 NVML 从内核驱动读出的**（592.01）。两者不一致 = 挂载的用户态库比当前内核驱动旧一个次版本，**在 WSL 里常见且通常无害**，但必须用**功能验证**（真实 kernel launch）判定，不能靠版本号对齐自我安慰。验证方式：py3.12 env 装 CUDA 版 torch → `torch.cuda.is_available()` 为 True → 跑一次真实矩阵乘 + 打印 `torch.version.cuda` 与 SM 数。

### 1.2 M0 收尾：已验证项、偏离台账、未决项（2026-09-12 夜）

**已达成（实测证据）**
- `torch 2.9.1+cu130`（runtime 13.0）+ `tensorrt 10.14.1.48.post1` + `tensorrt_llm 1.2.1`：三件套 import 成功且实跑通过
- 依赖死锁已确诊：`nvidia-cuda-runtime==13.0.48`（torch cu130 声明）与 `==13.0.96`（`tensorrt_cu13_libs → cuda-toolkit[cudart]` 声明）互斥；pip 未报错，而是**静默把 torch 换成 cu128 构建**；用 `pip install --no-deps --force-reinstall` 重钉 cu130 修复，实测 import 与 kernel 执行均正常

**偏离台账（Deviation Log）——每次环境变更后必须复验**

| # | 偏离内容 | 判定 | 状态 |
|---|---|---|---|
| D1 | `nvidia-cuda-runtime 13.0.96` ≠ torch 声明的 `13.0.48` | 同 soname（`libcudart.so.13`），patch 级；实测 import + kernel 通过；`pip check` 只报此一条 | 接受，记录（原文已留档） |
| D2 | CUDA 12 家族残留 **15 包**（`nvidia-*-cu12`，≈3GB），由 cu128 事件引入 | **Level-4 loader 实测**：进程内只映射 `libcudart.so.13 / libcublas.so.13 / libcublasLt.so.13 / libcudnn.so.9 / libnccl.so.2`，**无 `.12` 同驻** → 残留为惰性，未污染数值栈 | 待清理（卫生动作，非正确性修复）：15 包构成「孤儿根的传递闭包」，可一次性 `pip uninstall -y`；删后必须复跑 `pip check` + loader 实测 |
| D3 | `transformers 4.57.3`（TRT-LLM 精确 pin）∉ `nvidia-modelopt 0.37.0` 声明区间 `[4.48, 4.57)` | 两 pin 均由 NVIDIA 同版本组合自带 → 判为**陈旧区间检查**；采用方案 A，待 0.6B 玩具级复演验证 ModelOpt↔HF 路径 | 待验证 |

**未决项（按优先级）**
1. ~~环境歧义~~ **已澄清（2026-09-12）**：`/home/jiming/miniforge3` 目录不存在（`readlink -f` 对不存在路径回显原串，故先前误判）；规范解释器 = `/home/jiming/miniconda3/envs/llmquant/bin/python`，`sys.executable / sys.prefix / modelopt.__file__ / transformers.__file__` 四处自证一致，`conda env list` 中 llmquant 唯一。**教训：手工转写的路径不能作为证据，以解释器自报为准**
2. ~~Level 4 loader 实测、`pip check`、孤儿包清单~~ **已执行（2026-09-12）**：进程内仅映射 CUDA 13 家族；`pip check` 仅报 D1 一条；15 个 `nvidia-*-cu12` 包全部为孤儿传递闭包（待清理）。注意 `libnvinfer` 显示未加载属正常——该进程只 import 了 torch，未 import TensorRT
3. M0 read-back 三问（环境所有权判据）
4. M1 前置交付物 `docs/measurement_spec.md` 尚未编写

**不可交易的一条约束**

> 精度评测所用的 `transformers` 版本必须在 BF16 / INT8 / FP8 / 4bit **全档位保持一致**；否则 tokenizer 与 chat template 差异会污染精度 delta，结论不可归因。

### 1.3 事故复盘：文件级所有权冲突（2026-09-12 夜，D2 清理引发）

**现象**：按「孤儿传递闭包」判定删除 15 个 `nvidia-*-cu12` 包后，`import torch` 直接死亡：

```
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
```

**根因（机制）**：`nvidia-cudnn-cu12` 与 `nvidia-cudnn-cu13` **解包到同一个路径命名空间** `site-packages/nvidia/cudnn/lib/`，且 cudnn 9.x 的 soname 在两个 CUDA 家族里**完全相同**（`libcudnn.so.9`、`libcudnn_graph.so.9`、…）。于是：

1. 先装 cu13 变体 → 文件落在 `nvidia/cudnn/lib/`
2. 后来 pip 因依赖解析把 torch 换成 cu128 → 装 cu12 变体 → **同名文件被就地覆盖**，pip 只在 cu12 的 RECORD 里登记这些路径
3. 执行 `pip uninstall nvidia-cudnn-cu12` → pip 按 RECORD 删除路径 → **把 cu13 变体同样需要的文件一起删掉了**

**结论修正（重要）**：
- D2 的残留**并非惰性**——它不只是占 3GB 磁盘，而是在**用同名文件覆盖/顶替 cu13 家族的文件**。先前 Level-4 实测显示 `libcudnn.so.9` 被映射，当时**无法判定该文件属于哪个家族**（soname 相同、路径相同），因此「进程内单一家族」这个结论属于**过度断言**，现修正为：「进程内无 `.12`/`.13` 双 soname 同驻；但**同名同路径的家族无法用 soname 区分**」
- 同类同路径同名风险的包还有：`nccl`(libnccl.so.2)、`curand`(libcurand.so.10)、`cusparse`(libcusparse.so.12)、`nvjitlink`(libnvJitLink.so.12)、`nvtx`、`cusparselt`、`cufile`、`nvshmem`

**通用规则（新增，写入验收脚本）**：

> **元数据级依赖图无法预测文件级所有权冲突。** `Required-by: NONE` 只回答「有没有别的发行版**声明**依赖它」，不回答「有没有别的发行版**与它共用文件路径**」。任何删除前，必须先做**文件级冲突审计**（比对各发行版 `RECORD` 的交集），而不是只看依赖图。

**回滚依据**：`docs/pip_freeze_before_cu12_removal.txt`（这正是「删除前先 freeze」这条纪律的价值验证）。

### 1.4 M0 关闭记录（2026-09-12 深夜 · agent 代跑恢复 + 验收）

**恢复动作**：由 `pip_freeze_before_cu12_removal.txt` 派生 cu13 家族 15 包 → `pip install --no-deps --force-reinstall -r` → 文件全部重新解包。

**验收实测输出**

| 检查 | 结果 |
|---|---|
| 解释器身份 | `/home/jiming/miniconda3/envs/llmquant/bin/python`（唯一） |
| torch | `2.9.1+cu130` · rt `13.0` · `sm_120 True` · **NVIDIA GeForce RTX 5090 Laptop GPU** |
| TensorRT | `10.14.1.48.post1` |
| 功能冒烟 | `sm_120 SMs=82  9.640 ms  14.3 TFLOPS` |
| loader 全路径 | CUDA13 核心库来自 `nvidia/cu13/lib/`；`libcudnn.so.9` 来自 `nvidia/cudnn/lib/`、`libnccl.so.2` 来自 `nvidia/nccl/lib/`、`libnvshmem_host.so.3` 来自 `nvidia/nvshmem/lib/`、`libcusparseLt.so.0` 来自 `nvidia/cusparselt/lib/` |
| `pip check` | 仅 D1 一条 |
| freeze diff | **只有 15 行 `*-cu12` 删除，零新增、零版本变动 → 恢复无漂移** |
| 文件级冲突审计 | 2 条：`build_backend.py` / `.pyc`（`flashinfer-python` ↔ `torch_c_dlpack_ext`），与 CUDA 家族无关，无害但已记账 |

**关键修正（冲突成因的真实版）**：包目录布局**不统一**。cu13 变体把 CUDA 核心库放入**家族专属目录** `nvidia/cu13/lib/`，而 `cudnn / nccl / nvshmem / cusparselt` 沿用**家族无关目录** `nvidia/<name>/lib/`。于是：

- **会冲突**（cu12 与 cu13 写同一路径 + 同名 soname）：`cudnn`、`nccl`、`nvshmem`、`cusparselt`（同布局者另有 `curand`、`nvtx`、`cufile`）
- **不会冲突**：`cudart`、`cublas`、`cusparse`、`nvJitLink`（cu12 在 `nvidia/<name>/lib`，cu13 在 `nvidia/cu13/lib`）
- 先前 D2 表格里「cusparse/curand/nvjitlink 有冲突风险」属**未经观测的推断**，实测已否定；正确姿势是**读路径**（RECORD / `pip show -f`），不是按包名猜

**D2 关闭**：环境为单一 CUDA 13 家族，无跨家族同名文件遮蔽。恢复用脚本已安装为 `scripts/00_env_check.sh`（M0 验收测试，任何环境变更后复跑）。

**M0 判定：PASS**

### 1.5 新发现：`import tensorrt_llm` 被 OpenMPI 阻塞（2026-09-12 深夜，未决）

**现象**：`import tensorrt_llm` 卡死。逐模块二分后定位到 `tensorrt_llm.bindings`（其余模块：torch/tensorrt/mpi4py/flashinfer/xgrammar/transformers/modelopt 全部 <3s）。

**机制（已确证到行号）**：
- 阻塞态为 `S (sleeping)`、`wchan = anon_pipe_read`、单线程、60s 内仅 3s CPU → **不是编译，是等待子进程**
- 子进程 = **OpenMPI 的 `orted` HNP daemon**（`orted --hnp --set-sid --report-uri 8 --singleton-died-pipe 9 …`）
- 触发点：`tensorrt_llm/_utils.py` **模块级** 第 498 行 `comm = pkl5.Intracomm(MPI.COMM_WORLD)`。访问 `MPI.COMM_WORLD` 会让 mpi4py 初始化 MPI → 拉起 orted → 握手不返回
- `_utils.py:537` 的 `mpi_disabled()` 确实读 `TLLM_DISABLE_MPI == "1"`，但它只在函数内（541/551/567 行）被使用，**管不到模块级第 498 行** → 因此 `TLLM_DISABLE_MPI=1` 实测无效
- `from mpi4py import MPI`（`_utils.py:37`）是硬依赖：临时移走 mpi4py 会直接 `ModuleNotFoundError`，**不能靠卸载 mpi4py 绕过**

**独立于 TRT-LLM 的复现**：`mpirun -n 1 hostname` 同样挂死（45s 内零输出）。已排除：环境变量污染（`env -i` 仍挂）、hostfile 指向 127.0.0.1、conda 库污染（`ldd /usr/bin/orted` 全部解析到 `/lib/x86_64-linux-gnu`）、`/dev/shm`（20G 可写）、MCA 覆盖（`btl_tcp_if_include=lo`、`oob_tcp_if_include=lo`、`btl=self,vader`、`oob=^tcp`、`orte_keep_fqdn_hostnames=0` 全部无效）。

**环境事实**：系统 OpenMPI **4.1.6**（apt）；`ldd` 显示 mpi4py 的 openmpi 变体链接系统 `libmpi.so.40`；mpich 变体存在但 `libmpi.so.12` 缺失。宿主名解析：`getent hosts Jiming` 只返回 **IPv6 link-local**，而 `/etc/hosts` 有 `127.0.1.1 Jiming.localdomain Jiming`。WSL VM 于 **19:41:48 重启**（`.wslconfig` memory=40GB 已生效，`free -g` 已显示 39GB）。

**副作用（已处理）**：每次挂死的尝试都会**泄漏一个 orted 进程**（累计 15 个），且它们会继承 stdout 管道 → 用 `| tail` 的探测脚本会永久阻塞。清理：`pkill -9 -f orted`。**这条对后续所有调试脚本都成立：派生子进程的命令不要把输出接到管道，重定向到文件。**

**影响面**：**不阻塞 M1**（HF baseline + 测量规范不需要 tensorrt_llm）。**阻塞 M2–M4 的 TRT-LLM 侧加载**，因此必须在 M1 结束前解决。

**候选解法（按优先级，待验证）**：
1. 确认是「慢」还是「死」：正在以 30 分钟界跑 `mpirun -n 1 hostname` 与 `import tensorrt_llm`（结果待归档）
2. 用 conda-forge 的 `openmpi + mpi4py` 成对替换 apt 版（同一构建来源，消除 OpenMPI 与 mpi4py 的构建差异）
3. 测试切回 NAT 网络（`networkingMode=NAT`）——WSL + OpenMPI 的经典差异点，代价是一次 WSL 重启且可能影响现有代理下载路径

### 1.6 根因确证：WSL 镜像网络的**回环黑洞**（2026-09-12 深夜）

**判决**：MPI 不是「慢」，是**永久阻塞**。30 分钟界的两次测试：`mpirun -n 1 hostname` 1712s 无输出；`import tensorrt_llm` 1665s 无输出。

**证据链（逐级收窄）**
1. 逐模块二分：只有 `tensorrt_llm.bindings` 挂，其余全部 <3s
2. 阻塞态 `S` + `wchan=anon_pipe_read`，单线程、60s 仅 3s CPU → 在等子进程，不是编译
3. 子进程 = `orted --hnp …`（OpenMPI 4.1.6）
4. `/proc/<mpirun_pid>/net/tcp` 抓到真凶：**`127.0.0.1:48406 -> 127.0.0.1:6001`，state=`02`（SYN_SENT）**，socket inode `17188`
5. 回环行为对照（决定性）：

   | 目标 | 结果 |
   |---|---|
   | `127.0.0.1:6001`（无监听） | **TIMEOUT 3.02s** — SYN 被吞 |
   | `127.0.0.1:59999`（无监听） | **TIMEOUT 3.01s** — 任意端口皆然 |
   | `127.0.1.1:6001`（无监听） | REFUSED **0.000s**（正常） |
   | `192.168.1.101:6001`（无监听） | REFUSED **0.000s**（正常） |
   | `127.0.0.1:<自建监听>` | CONNECTED 0.000s（回环本身可用） |

6. Windows 侧对照：`TcpClient.Connect('127.0.0.1',6001)` → **主动拒绝**（正常）
   → 异常**只存在于 WSL 镜像网络的回环路径**，不在宿主网络栈

**机制**：`networkingMode=mirrored` 下 WSL 回环与宿主共享；**目标端口无监听时 RST 不回传，SYN 被静默丢弃**。OpenMPI 的 singleton 初始化会连回环端口（本次 6001）做 rendezvous——过去能立刻拿到 ECONNREFUSED 并继续，现在永久等待，于是 `import tensorrt_llm` 卡死。**既不是 TRT-LLM 的 bug，也不是 MPI 的 bug。**

**影响面（要记住）**：WSL 内任何「探测未监听回环端口」的程序都会**挂住而不是快速失败**——会波及服务健康检查、端口探测、部分测试框架。M1 不需要 TRT-LLM，**不阻塞 M1**；M2 之前必须修好。

**修复尝试与最终解法（2026-09-13 上午，已闭环）**

| 尝试 | 结果 |
|---|---|
| `Set-NetFirewallHyperVVMSetting -LoopbackEnabled True`（管理员）+ VM 重启 | ❌ **假设被否定**——回环仍黑洞（防火墙不是成因） |
| MCA 接口覆盖：`oob_tcp_if_include` = lo / eth1 / 192.168.1.101、`if_exclude lo`、`btl=self,vader` | ❌ 全部仍阻塞 |
| `oob_tcp_static_ports 6001`、`6001-6100` | ❌ 仍阻塞 |
| `TLLM_DISABLE_MPI=1` | ❌ 无效：`_utils.py:498` 在模块级初始化 MPI，不受该开关管辖 |
| 移除 `mpi4py` | ❌ 不可行：`_utils.py:37` 硬 import，移走即 `ModuleNotFoundError` |
| **`networkingMode=NAT`** + `wsl --shutdown` | ✅ **完全修复** |

**补充观测（关键）**：阻塞期间 **没有任何 `orted` 进程存在、6001 端口无监听**；而 mpirun 的目标端口**恒为 6001**（`0x1771`，源端口随机）。也就是说这是**两个缺陷叠加**：① 应该监听 6001 的对端根本没起来；② 回环黑洞把「对端不存在」从**应当立即报错**变成了**永久等待**——所以既没有重试也没有报错。因此修网络层（① 由 ② 掩盖）是唯一可行路径。

**NAT 模式验证结果（实测）**
```
127.0.0.1:6001 / :59999        → REFUSED 0.000s        （回环语义恢复正常）
mpirun -n 1 hostname           → rc=0, 0s, 输出 "Jiming"
from mpi4py import MPI         → rc=0, 0s, rank 0 / 1
import tensorrt_llm            → rc=0, **13.5s**（此前 30 分钟不返回）
```

**代价与验证（NAT 不含 localhost 代理，需记账）**
- WSL 启动会提示「检测到 localhost 代理配置，但 NAT 模式不支持」；Windows 代理（127.0.0.1:7897）**只绑回环**，故 WSL 无法经网关 IP 使用它
- 但**项目所需下载路径直连即可**：`hf-mirror.com` 200（11s）、`pypi.org` 200（1.9s）、`download.pytorch.org` 200（1.2s）、`pypi.nvidia.com` 301 正常
- `huggingface.co` 直连失败（预期内，故用镜像）；`github.com` 响应慢（18s 内未完成）——**M6 推仓库前需复查**，必要时在代理客户端开 Allow LAN 并用 `http_proxy=http://172.18.192.1:7897`，或改用 SSH 推
- 备份：`.wslconfig.bak-nat`；回退只需把 `networkingMode` 改回 `mirrored` 并 `wsl --shutdown`

**验收判据（全部已通过，2026-09-13 09:4x）**
1. 连 `127.0.0.1:6001` 毫秒级 `ConnectionRefusedError` ✅
2. `mpirun -n 1 hostname` 打印主机名 ✅
3. `import tensorrt_llm` 返回 `1.2.1`（13.5s）✅
4. 复跑 `scripts/00_env_check.sh` 全绿 ✅

**红线（先记住，能省你两天）**：
- ❌ **绝不在 WSL 里装 Linux 版 NVIDIA 显卡驱动**。WSL2 的 GPU 是 Windows 驱动透传的，装 Linux 驱动会直接把 `/dev/dxg` 搞坏。WSL 里只装 **CUDA Toolkit（不装 driver）**。
- ❌ 不要让 Windows 侧和 WSL 侧共用同一个 HF cache / conda env。物理隔离。
- ⚠️ 笔记本 173W 功耗墙 + WDDM（Windows）意味着**同一个数字换个时间段能飘 15%**。所有跨版本对比必须在同一次 session 内、温度稳定后完成，并在报告里写明温度。

---

## 2. 里程碑总览

| # | 里程碑 | 核心机制 | 验收 gate（我据此判 PASS/FAIL） |
|---|---|---|---|
| **M0** | 环境与自证 | WSL2 里 GPU 真的可见、CUDA/TRT-LLM 版本自洽 | `00_env_check.sh` 输出齐全 + `nvidia-smi` 在 WSL 内可见 + torch.cuda.is_available()=True |
| **M1** | **先冻结尺子** | 测量口径先定，再谈优化 | `docs/measurement_spec.md` 写死 + HF baseline server 起得来 + bench 能出 TTFT/TPOT/显存三件套 |
| **M2** | 第一台阶：W8A8 INT8 | PTQ 校准的物理意义：用真实激活分布换掉 min/max 极值 | 量化权重可加载、精度损失可量化、延迟/显存数字进 results/ |
| **M3** | 第二台阶：FP8 + KV cache | Blackwell 原生 FP8；长上下文显存主项是 **KV cache** 不是权重 | context length sweep 曲线（1k/4k/16k/32k）+ FP8 KV 的显存收益 |
| **M4** | 第三台阶：4bit（NVFP4/AWQ） | 显存墙突破 + **敏感层识别做混合精度** | NVFP4 是否在 sm_120 成立（实测结论）；混合精度 map 有依据、不是拍脑袋 |
| **M5** | 服务工程化 | continuous batching / paged KV / 请求调度 | 并发 1/4/8/16/32 的 TTFT-TPOT 曲线 + OpenAI API 兼容 + 压测原始数据 |
| **M6** | 报告与开源 | 三维 Pareto + 诚实结论 | `report.md` 完成 + `gate.json` 全档位结论 + 仓库可被陌生人一键复现 |

---

## 3. 逐里程碑详解

### M0 · 环境与自证（预计 0.5–1 天）

**机制**：WSL2 的 GPU 是「Windows 驱动 + `/dev/dxg` 透传 + WSL 内 CUDA 用户态库」三层结构。所以 WSL 里 `nvidia-smi` 显示的驱动版本**必然等于** Windows 侧（592.01），这是判断透传是否正常的唯一可靠信号。CUDA toolkit 版本与驱动版本是两件事，不要混。

**你要做的**：
1. 在 WSL 里跑验证（结果贴给我）：
   ```bash
   nvidia-smi                      # 驱动版本应为 592.01
   ls /dev/dxg                     # 存在=透传正常
   lscpu | head -5; free -g        # 分配了几核几 G
   df -h ~                         # 权重+引擎要预留 ≥ 80GB
   ```
2. 检查/调 WSL 资源（Windows 侧 `C:\Users\21327\.wslconfig`）：
   ```ini
   [wsl2]
   memory=48GB        # 按你实机 RAM 调整（我这边读不到 RAM，你确认）
   processors=12
   swap=16GB
   ```
   改完 `wsl --shutdown` 再进。理由：7B 级模型的**主机内存**峰值（加载 + 量化校准）常常比显存更早爆。
3. WSL 内装 CUDA Toolkit（**只装 toolkit**）：
   ```bash
   # 选与驱动 592.01 兼容的版本，装 runfile 时取消 driver 勾选
   sudo apt-get install -y build-essential
   # CUDA 12.x/13.x 二选一，与后面 TRT-LLM wheel 的 CUDA tag 对齐
   ```
4. 建独立环境（**不要**复用 Windows 侧 conda env）：
   ```bash
   conda create -n llmquant python=3.12 -y   # 或 venv，你定
   ```
5. 装 TRT-LLM：优先 **NVIDIA 官方 pip index 的预编译 wheel**，而不是源码编译：
   ```bash
   pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
   ```
   Windows 侧那个 `modelopt 0.46.0` 只作为**参考版本号**，WSL 里装与 TRT-LLM 匹配的 modelopt。

**read-back 三问（你必须能口头答出来，答不出说明这步没做完）**：
1. WSL 里看到的 CUDA 版本由谁决定？为什么装 Linux 驱动会把 WSL 搞坏？
2. 你的 TRT-LLM 版本用的是哪条执行路径——**engine 构建路径** 还是 **PyTorch 后端直载路径**？两者对量化权重格式的要求差在哪？
3. `HF_ENDPOINT` 换成镜像后，模型下载的**校验**由谁负责？镜像与官方仓库文件是否逐一可比对？

**我给你的验收标准**：贴出上述命令的真实输出 + 三问答对。任一不满足 → 不进入 M1。**不要跳过这里去装模型**，这是全项目 80% 痛苦的来源。

---

### M1 · 先冻结测量尺（预计 1 天，全项目最重要）

**机制**：量化项目的绝大多数学术性失败不是量化做错了，而是**尺子换了**。你在 EZTrain DualTac 案例上已经吃过一次亏——修掉配对 bug 后 clean RMSE 从 0.15935 变成 0.116，旧的"命中率/排序"结论全部作废。那次教训在这里重演一次的成本是三天。

**必须先写死的 `docs/measurement_spec.md`，至少含**：
- 固定 prompt 集（≥20 条，长短混合；**把 prompt 文件哈希写进结果 json**，否则不可比）
- 采样参数：greedy（temperature=0）用于可复现性；固定 max_new_tokens
- 指标定义：
  - **TTFT**：首 token 延迟（从请求发出到第一个 token 返回）
  - **TPOT/ITL**：除首 token 外的平均每 token 时间
  - **吞吐**：tokens/s（分单请求与总吞吐两种口径，别混）
  - **峰值显存**：优先用引擎/进程自身报告的值；`nvidia-smi` 只作旁证
  - 一律报 **p50 / p95 / p99**，不报 mean
- 测量纪律：warmup ≥ 3 轮丢弃；计时前后 `torch.cuda.synchronize()`；每个配置重复 ≥ 5 次
- 环境快照：GPU 温度、功耗、clock、当前 batch、软件版本号，**全部落进结果 json**

**你要做的**：
1. 写 `measurement_spec.md`（先给我看，我审）
2. 用 **HF transformers 原生**（不急上 TRT）起一个最小 OpenAI 兼容 server，把 bench 流程跑通 → 产出 `results/baseline/bf16_hf/`
3. 这份 HF 基线**不是最终 oracle**，只是让尺子先能读数。真正的 oracle 在 M2 前用 TRT-LLM BF16（fp16/bf16 未量化引擎）重测一次，并**明确记录两者差异与原因**（这条本身就是报告里的一个亮点：解释"为什么 HF 原生延迟不能当部署 baseline"）。

**read-back 三问**：
1. TTFT 与 TPOT 分别由哪部分计算主导？为什么 chunked prefill 能改善 TTFT 但可能恶化 TPOT？
2. 为什么必须报 p99？在什么并发下 mean 会骗你？
3. 你的 bench 脚本里，哪一行保证「测量区间」没有把模型加载/编译时间算进去？

**gate**：`measurement_spec.md` 存在且我认可 + bench 能稳定复现出同一组数字（两次运行偏差 < 5%，若超了，找出原因再继续）。

---

### M2 · W8A8 INT8 PTQ（预计 1–2 天）

**机制**：PTQ 的本质是**用一批真实输入的激活分布，去决定「这个张量的动态范围该映射到哪一段整数区间」**。W8A8 = 权重和激活都量化；activation 量化比 weight 难得多，因为激活是**运行时数据相关**的（离群值通道 outlier channel 是 LLM 量化的核心敌人），而权重是静态的。所以 W8A8 的关键词是 **calibration**：校准集选什么、多少条、per-tensor 还是 per-channel、要不要先做 smoothing/scale 均衡。

⚠️ 你机器上 ModelOpt 已有，直接用它的 PTQ 流程（`modelopt.torch.quantize`）而不是自己造轮子；配置里显式写清 `w8a8` 的 quantizer 类型。

**你要做的**：
1. 挑校准集（建议 128–512 条，来源与你 M1 的 prompt 集**不同**，避免评估泄漏——这点必须写进报告）
2. 跑 PTQ → 导出量化权重 → 用 TRT-LLM 加载（engine 或 PyTorch 后端）
3. 跑 `06_bench.py` + `07_eval_accuracy.py`，与 BF16 oracle 同 session 对比
4. 至少做一组**校准集大小消融**（如 64 / 128 / 512），看精度损失是否随校准量收敛

**read-back 三问**：
1. 为什么 activation 的离群值会毁掉 per-tensor 量化？per-channel 为什么帮不上 activation？
2. 你的校准集与评估集如果重叠，会出现什么假象？怎么证明没重叠？
3. INT8 量化后，**峰值显存**实际降了多少？如果降幅远小于"权重砍半"的直觉，说明谁是显存主项？

**gate**：`results/w8a8/` 有完整三件套 + 精度 delta 明确 + 你能解释显存降幅的来源。

---

### M3 · FP8 + KV cache 量化（预计 1–2 天）

**机制**：FP8 与 INT8 的根本差异是**它保留了指数位**——动态范围大、不需要对抗离群值，代价是尾数精度低（E4M3 只有 3 位尾数）。Blackwell 对 FP8 有原生张量核支持，所以 FP8 在 5090 上通常是"精度损失最小、速度快"的最优档。

而真正被低估的是 **KV cache 量化**：上下文变长时，KV cache 的显存占用**线性增长且不受权重位数影响**。你量化权重把 16GB 砍到 8GB，但如果 32k 上下文的 KV 就要 6-10GB，那你的并发能力上限其实由 KV 决定，不由权重决定。**这一条是很多人答不上来的面试题。**

**你要做的**：
1. FP8 W8A8 量化（同样 ModelOpt）→ bench + eval
2. KV cache 单独量化（INT8 或 FP8，看后端支持）→ 再测
3. **context length sweep**：1k / 4k / 16k / 32k，记录峰值显存与 TTFT/TPOT，画两条曲线：
   - 权重显存（固定）vs KV 显存（随长度增长）
   - 上下文长度 vs 最大可行并发数

**read-back 三问**：
1. FP8 的 E4M3 与 E5M2 分别适合量化权重还是激活？为什么？
2. KV cache 量化损害的是哪个指标——是"精度"还是"长上下文下的行为一致性"？（提示：per-token 的 K/V 缩放因子与 per-tensor 的差别在这里放大）
3. 在你的 24GB 上，跑 32k 上下文时，权重 / KV / 激活 / 框架开销 各占多少？谁是瓶颈？

**gate**：两条曲线有真实数据点 + 你能说清"我的部署在什么上下文长度上会由于 KV 而不是权重撞墙"。

---

### M4 · 4bit 与混合精度（预计 2–3 天，本项目的技术核心）

**机制**：4bit 是显存墙突破手段，但 4bit 的敌人是**敏感层**。LLM 里绝不是所有层都一样脆弱——经验规律与你在 DualTac 上的发现同构：**越靠近输出、越承担精确回归/概率细分辨率的层越脆弱**。所以正确做法不是"全模型 uniform 4bit"，而是：

> 探针测出每层/每模块的边际损伤 → 在给定显存预算下，求解"哪些层降 4bit、哪些留 8bit/16bit"的**混合精度分配问题**。

**这正是 EZTrain 里 `probe_prefixes` → `width_map` → worker → gate 那套东西的 LLM 版**。把老手艺搬过来，是你这个项目区别于"我跑了 AWQ"的关键。

**你要做的**：
1. **先实测 NVFP4 在 sm_120 上到底成不成立**（不要假设！）：
   - 查你装的 TRT-LLM 版本对 NVFP4 的 `sm` 支持矩阵
   - 实际构建一次 NVFP4 引擎，成功就记录，失败就把**报错原文**记进报告——**失败的证据同样是结论**（"在消费级 Blackwell 上 NVFP4 需要 X 条件，我的环境不满足"本身就是有价值的工程结论）
2. 若不成立，退到 W4A16（AWQ/GPTQ 路线）作为 4bit 档
3. 写 `09_sensitivity.py`：逐层/逐块做「把这一段降到 4bit、其余保持 8bit」的边际损伤测量（**注意**：LLM 的层序是纯线性链，比 DualTac 的分支图好处理得多；但要注意 tie 住 embedding/lm_head 的 weight tying，别重复计）
4. 产出混合精度 map（`configs/mixed_*.yaml`），在同一显存预算下与 uniform 4bit 对比精度

**read-back 三问**：
1. 4bit 的量化粒度（per-channel / per-group，group size 多少）对精度的敏感度比 8bit 更高还是更低？为什么？
2. 你的敏感性探针在 LLM 上测的是「单层隔离损伤」还是「前缀累积损伤」？两者结论会一样吗？（回想你在 DualTac 上的教训：单层 1e-4 vs 全模型 +2.0%，损伤**不可加**）
3. 混合精度 map 的搜索是"贪心按预算走数据流序"还是"全局最优"？你的选择与代价？

**gate**：4bit 档（无论 NVFP4 还是 W4A16）有真实可用数字 + 混合精度相对 uniform 4bit 的精度优势可测 + 敏感性方法有可复现脚本。

---

### M5 · 服务工程化（预计 1–2 天）

**机制**：一个"能推理"的服务和一个"能上线"的服务差在**调度**。两个核心概念：
- **Continuous batching**：不等整批做完，谁先结束谁让位——把 GPU 利用率从"等最慢的请求"变成"永远有新活"
- **Paged KV cache**：KV cache 按块分配，消除碎片，直接决定你能同时服务多少并发

这两者与量化是**耦合**的：量化省下的显存，最终兑现的形式是**更大的并发批**，而不是单请求更快。所以你的报告必须有一张「显存节省 → 并发能力提升」的传递表，这才是部署工程师的语言。

**你要做的**：
1. 起 OpenAI 兼容服务（TRT-LLM 自带 server，或自己包一层 FastAPI）
2. 并发 sweep：1 / 4 / 8 / 16 / 32 并发，记录 TTFT、TPOT、总吞吐、峰值显存
3. A/B 对比（加分项，很值钱）：**同一份量化权重**分别跑 TRT-LLM 与 vLLM，比吞吐/延迟——证明你理解的是"机制"而不是"某个框架的操作手册"
4. 用真实客户端压测（不是脚本里 `time.sleep`），最好有 1 个长上下文 + 多个短请求的混合负载

**read-back 三问**：
1. 并发从 4 涨到 16 时，总吞吐上升但单请求 TPOT 恶化，为什么？哪条曲线会先"触顶"？
2. Paged KV 解决的是碎片还是容量？两者混了会怎样？
3. 量化把你的显存省了 X GB，换算成并发数是几个请求？（必须用实测数据推，不许估）

**gate**：并发曲线有真实数据 + OpenAI API 能被外部客户端真实调用成功 + 显存→并发传递表成立。

---

### M6 · 报告、开源与简历映射（预计 1 天）

**你要做的**：
1. `docs/report.md`：方法论 → 实验设置 → 四档数据表 → Pareto 图 → **诚实的半命中结论** → 局限（笔记本功耗墙、单卡、温度漂移）
2. `scripts/08_gate.py`：把 gate 判定写进 `results/gate.json`，含每档 PASS/FAIL 与理由
3. 仓库公开（⚠️ **只推代码/配置/结果/报告，绝不推权重与引擎文件**；`.gitignore` 里写死 `*.engine`、`*.safetensors`、`*.pt`）
4. `README.md` 顶部放一张能让 HR/面试官 30 秒看懂的结论表

**简历映射草稿（做完再回填真实数字）**：
- 「对 Qwen3-8B 实施 PTQ 全流程（W8A8 INT8 / FP8 / 4bit / KV cache），精度损失控制在 __%，峰值显存降低 __%」
- 「在 RTX 5090 Blackwell (sm_120) 上实测 NVFP4 的支持边界，给出 __ 结论」
- 「基于逐层敏感度探针构建混合精度分配方案，在同等显存预算下相对 uniform 4bit 提升 __% 精度」
- 「TensorRT-LLM 服务端完成 continuous batching + paged KV 调优，__ 并发下吞吐 __ tok/s、TTFT p99 __ ms」
- 「同一量化权重在 TRT-LLM 与 vLLM 上完成 A/B 基准，定位差异来源」

---

## 4. 全局风险与预案

| 风险 | 触发信号 | 预案 |
|---|---|---|
| WSL CUDA/驱动不兼容 | M0 `nvidia-smi` 异常或 torch.cuda False | 降 WSL 内核/CUDA 组合；极端情况下退回 Windows 原生 + ONNX Runtime 路线 |
| TRT-LLM 预编译 wheel 与 CUDA tag 不匹配 | pip 装上但 import 失败 | 换 CUDA 版本对齐 wheel；不要在 Windows 上源码编译 TRT-LLM |
| NVFP4 在 sm_120 不成立 | 引擎构建报 sm 不支持 | 退 W4A16（AWQ/GPTQ），并把「不成立」写成报告结论（**这也是成果**） |
| 笔记本热漂移导致数据不可比 | 同配置两次结果差 > 10% | 固定功耗模式、每次测前等温度回落、报告中标注温度；必要时限频换稳定性 |
| 下载 16GB 权重失败/极慢 | HF 直连不通 | `HF_ENDPOINT=https://hf-mirror.com`；用 `huggingface-cli download` 断点续传 |
| 校准/评估数据泄漏 | 精度 delta 异常好看 | 校准集与评估集必须来自不同来源，并在报告中列出两者的哈希 |

---

## 5. 我对你的验收协议（read-back 制）

每个里程碑结束时，你给我**三样东西**：
1. **原始证据**：命令输出 / results 下的 json / 曲线图
2. **机制复述**：用你自己的话回答该里程碑的 read-back 三问（不是抄我的解释）
3. **下一步判断**：你认为该继续、该回头、还是该换路线，以及理由

我只在「证据齐全 + 机制讲通 + gate 达标」三者同时满足时判 PASS。**半命中不算失败**——数据说了什么就写什么，这比全绿更值钱。

---

## 6. M0 read-back 参考答案（agent 版 · 用户复述确认后即视为「环境所有权」达标）

**Q1 · WSL 里的「CUDA 版本」由谁决定？为何在 WSL 里装 Linux 显卡驱动会毁掉 GPU 栈？**

- 三层结构：**Windows 内核驱动**（唯一接触硬件的一层，592.01）→ `/dev/dxg` 透传 → `/usr/lib/wsl/lib`（由 Windows 驱动目录 bind-mount，且本身是 overlayfs：`gpu_lib_packaged` 压 `gpu_lib_inbox`）
- `nvidia-smi` 的「工具版本行」（590.62）来自**工具二进制里的编译期字符串**；「Driver Version」行（592.01）由 NVML **运行时**从内核驱动读出。两者不一致 = 挂载的用户态库落后一个次版本 → N-1 skew，常见且无害
- 对应用真正有约束力的关系只有两条：**驱动能力上限（CUDA 13.1）≥ 程序链接的 CUDA runtime 版本**；**编译期 toolkit 版本不影响运行**（`/usr/local/cuda` 或 apt 的 13.3 只是编译环境）
- 在 WSL 内装 Linux 驱动会让内核态出现第二套驱动、与 Windows 驱动争夺同一 GPU，并破坏 `/dev/dxg` 透传与 `/usr/lib/wsl/lib` 的挂载假设 → 整个 GPU 栈立刻失效

**Q2 · TRT-LLM 走哪条执行路径？各自对量化权重格式的要求差在哪？**

- **① engine 构建路径**：`trtllm-build` → 产出 `.engine` 文件。**与 TensorRT 版本和 GPU 架构（sm_120）强绑定，不可跨版本/跨架构移植**；量化元数据需随 checkpoint 一起经过转换流程
- **② PyTorch 后端直载路径**：`tensorrt_llm._torch` 直接加载量化 checkpoint，**无 engine 文件**，迭代快、无重编译税
- 本项目策略：M2–M4 走 **②**（快速迭代三档量化），M5 服务化阶段对**最优档位**走 **①** 取延迟收益——两条路径的实测差异本身就是报告里的一张对比表

**Q3 · 如何证明镜像下载的权重与官方一致？验证发生在流程哪一步？**

- 机制：Hub 侧每个文件有 LFS 的 SHA256（`lfs.oid`），客户端下载后会写 `.cache/huggingface/download/**` 元数据（含 commit hash / etag）并本地校验
- 三步验证：① **锁定 commit revision**（用 40 位 sha，不用 `main`）② 下载完成后**对每个分片本地重算 sha256**，与 API 返回的 `lfs.oid` 逐一比对 ③ 记录 `model.safetensors.index.json` 的文件清单与总字节数
- 发生位置：**下载完成的那一刻**，并把 `(repo_id, revision, 每文件 sha256)` 写进 `docs/env_matrix.md` 与每次结果 json——否则 M2 的精度数字无法回溯到确定权重

---

## 7. M1 起点

M0 已 PASS。M1 的第一个交付物是 **`docs/measurement_spec.md`（测量口径规范）**，它必须在任何 benchmark 脚本存在之前冻结；必填字段见 §3 M1 条目。验收顺序：规范 → 我审 → 才写 bench 脚本。

---

## 8. M2 记录：INT8 失败、FP8 通过（2026-09-14）

### 8.1 先修的坑：ModelOpt 的 CUDA 扩展从未真正加载过

**两个互相独立的缺陷**，缺任一个都编不出扩展：

| # | 机制 | 修复 |
|---|---|---|
| A | `CXX` 未设时 torch 把编译器解析为 `c++`，其兼容性检查在**打印警告时抛 `TypeError`**（格式化参数个数不符），在任何编译发生之前就崩掉 | 令 `CXX` 指向真正的 `g++`（`/usr/bin/g++` 存在） |
| B | 失败的是 **`.cu`** 文件，`-std=c++20` 只加进 `extra_cflags` **到不了 nvcc**；torch 2.9.1 自带头文件（`ATen/core/List_inl.h:201` 的 `decltype(...)::difference_type`）在 C++17 下编译失败 | `-std=c++20` 必须**同时**加进 `extra_cflags` 与 `extra_cuda_cflags` |

**逐变体实测证据**：默认 `-std=c++17` ❌ / 仅 `extra_cflags` 加 c++20 ❌ / **两个 flag 都加 ✅** / `-fpermissive` ❌ / c++20+fpermissive ❌。
修复后 `modelopt_cuda_ext`、`_fp8`、`_mx` 全部 LOADED（38.5s / 37.0s / 45.9s，此后缓存 0.2–0.6s）。修复封装在 `scripts/modelopt_ext_patch.py`，**必须在 import modelopt 之前导入**。

**教训**：`Unable to load extension ... falling back to CPU version` 只是一句**警告**，警告不终止流程——于是 INT8 一直跑在 CPU 实现上，而 FP8 直接产出 `nan`。与本项目反复出现的同一模式一致：**失败不报错，只给出一个看起来合理的数字。**

### 8.2 三档量化结果（128 校准样本 · 32752 token 困惑度 · BF16 oracle = 9.3190 · 门槛 +2%）

| 配方 | 困惑度 | 相对变化 | 判定 |
|---|---|---|---|
| **FP8（W8A8 float8）** | **9.3434** | **+0.26%** | ✅ **PASS** |
| INT8 + SmoothQuant | 14.0729 | +51.01% | ❌ FAIL |
| INT8 per-tensor（`INT8_DEFAULT_CFG`） | 19.7043 | +111.44% | ❌ FAIL |

**关键对照**：INT8 两个数值在「CPU 回退」与「CUDA 扩展」两次运行中**逐位相同** → CPU 回退在数值上等价，因此 +51% / +111% 是**方案本身**的性质，不是构建伪影。FP8 则由 `nan` 变为 +0.26%——它此前根本无法运行。

**机理解释**：INT8 的激活采用**静态 per-tensor 缩放**，缩放因子被少数极端离群值支配（校准中观测到 `down_proj` 输入 amax 达 1152），其余分布被压扁；FP8 保留指数字段，动态范围天然覆盖离群值，故同模型同校准集下几乎无损。**这正是从 M0 起反复讲的「离群值通道」理论，现在有了实测数字支撑。**

**结论（诚实版）**：M2 原计划的「W8A8 INT8」在 Qwen3-8B 上**不成立**，且不是参数没调好——SmoothQuant 也只救回一半。可行档位是 **FP8**。对报告而言这是好素材：一个被否定的方案加一个被验证的替代方案，比「全部通过」更有说服力。

### 8.3 M2 正式 verdict：**FP8 PASS**（2026-09-14，可采纳）

完整评测（128 校准样本 · 32752 token 困惑度 · 500 题 MMLU · 导出检查点）：

```
perplexity  9.3190 → 9.3434   +0.26%   (limit +2.0%)    PASS
MMLU        0.6120 → 0.6180   +0.60 pp (limit −4.0 pp)  PASS
VERDICT: PASS
```

**MMLU 反而上升 0.6 pp，而其标准误为 2.17 pp** → 统计上与 BF16 **不可区分**。结论：**FP8 W8A8 在 Qwen3-8B 上几乎无损**，可作为后续档位的部署基准。

**导出产物（实测）**：

| 对象 | 大小 |
|---|---|
| BF16 原始检查点 | 16 GiB |
| **FP8 量化检查点** | **8.8 GiB（−42%）** |

差额可精确解释：embedding 与 `lm_head` 的量化器为 `disabled`，仍以 bf16 保存（2.49 GB），其余 6.95 B 线性层参数位宽减半 → 合计 9.44 GB = 8.79 GiB，与实测一致。

**注意（口径边界）**：M2 测的是**内存中 fake-quant 模型**的精度，导入的检查点是给服务端用的。**延迟与显存收益不在 M2 门槛内**——真实收益要到 M5 用 TRT-LLM 加载该检查点后才体现（HF transformers 里权重仍以 bf16 驻留）。这条必须写进报告，否则「FP8 省了一半显存」会被误读为已测结论。

**本轮顺手修掉的三个坑**
1. `HF_ENDPOINT` 设置位置错误：`huggingface_hub` 在 **import 时**读取该变量，而 `transformers` 会先 import 它 → 写在 import 之后**完全无效**，请求仍打向不可达的 huggingface.co，每个数据集重试 5 次后才回落本地缓存。现已在文件最顶部、任何 torch/transformers import 之前设置。
2. 校准语料改为**本地缓存**（`data/calibration/calib_128.json`，sha `1e752ba4…`）：语料是测量的一部分，其 hash 记录在结果里，必须跨档位一致；每次重新下载等于把身份交给网络。
3. `--smoke` 不再偷偷缩小校准集：校准样本数是量化质量的主因（8 个样本 ⇒ +55% 困惑度，与方案无关）。同时**门禁在评测口径不一致时自动判 `PROVISIONAL` 而非 PASS/FAIL**——曾几乎把「MMLU −16.2 pp」当成结论，而 20 题样本的 95% 置信区间是 ±22 pp。

**下一步（M3）**：以 FP8 为基准档位，做 KV cache 量化与上下文长度扫描（1k / 4k / 16k / 32k）。Qwen3-8B 的 KV cache 为 **144 KiB/token**（2×36 层×8 kv_heads×128×2B），32k 上下文即 **4.5 GiB**——权重减半之后，KV 才真正成为显存主项，这正是 M3 要测量的现象。

---

## 9. M3 记录：KV cache 量化是**上下文相关的权衡**（2026-09-17）

### 9.1 实测结果（FP8 权重检查点 · batch 1 · 生成 128 token · 固定 33984 token 池）

| 上下文 | auto（FP16 KV） | fp8 KV | fp8/auto |
|---|---|---|---|
| 1 024 | 71.7 tok/s | 53.7 | **0.75** |
| 4 096 | 64.0 | 51.1 | **0.80** |
| 16 384 | 31.1 | 32.7 | **1.05** |
| 32 768 | 17.8 | 22.6 | **1.27** |

**交叉点约在 16k。** 短上下文下 FP8 KV **慢 25%**（反量化开销主导，此时 cache 很小，省内存无意义）；长上下文下 FP8 KV **快 27%**（attention 转为显存带宽受限，KV 字节数减半开始取胜）。

**引擎自报的 KV 池显存**（同一 33984 token 池）：

| KV 精度 | 池分配 | 每 token |
|---|---|---|
| auto（FP16） | **4.67 GiB** | 144 B |
| fp8 | **2.33 GiB** | 72 B |

**恰好一半**，与 `model_provenance.md` 的公式预测一致（2×36×8×128×2 B = 144 B/token）。

**上下文自身的延迟墙**：auto 从 71.7 → 17.8 tok/s（1k→32k，**4.0× 下降**）——prefill 成本随上下文上升。

### 9.2 三个方法论发现（比数字更值钱）

1. **peak VRAM 对 KV 精度完全不敏感**（8 个 cell 全部 21.623 GiB）。池被固定并预留，峰值由权重与工作区主导。**peak device VRAM 是回答此问题的错误仪器**；正确仪器是引擎自报的池分配日志（`Allocated X GiB for max tokens in paged KV cache (N)`）。
2. **`max_num_tokens` 默认 8192**：16k/32k 的 prompt 直接被拒（`RequestError: prompt length 16384 should not exceed max_num_tokens 8192`），且该异常终止整轮扫描。现已显式设置并改为**每 cell 落盘**，避免一次崩溃丢掉全部结果。
3. **`nvfp4` KV cache 令 worker 进程硬崩溃**（MPI abort + 回溯，非可捕获异常），已排除出网格、单独记为发现——与 M4 的四位档位直接相关。

### 9.3 API 约束（实测）

对 FP8 检查点，`KvCacheConfig(dtype=...)` 只接受 **`('fp8', 'nvfp4', 'auto')`**；`int8` 被拒：
```
ValueError: Overriding KV cache quantization with an invalid type "int8".
```
即 **FP8 权重 + INT8 KV 在本栈上不被支持**。KV cache 是**运行时**配置，不写进检查点（导出的 `kv_cache_quant_algo: None` 即为证据）。

### 9.4 这条结果对报告与面试的价值

它推翻「量化总是更省更快」的直觉，给出**可复现、有交叉点、有机制解释**的结论：

> KV cache 量化在 1k 上下文付出 25% 吞吐代价，在 32k 换回 27% 收益，交叉点约 16k；「是否量化 KV」应由**部署的实际上下文分布**决定，而非默认开启。

---

## 10. M4 记录：4-bit、混合精度，以及一个通配符 bug（2026-09-17/18）

### 10.1 好消息：weight-side NVFP4 在 sm_120 上可用

```
uniform NVFP4（W4A4，128 校准样本，32752 token）  perplexity 9.5799  vs oracle 9.3190 = +2.80%
```

**+2.80% 对 4-bit 权重+激活而言相当好**，与 INT8 形成鲜明对照（per-tensor +111%、SmoothQuant +51%）。机制：NVFP4 的**分块 microscaling** 容纳离群值通道——正是摧毁 per-tensor INT8 的那个问题。

**与 M3 的对照**：`nvfp4` 作为 **KV cache** 精度会让 worker 进程硬崩溃；作为**权重**精度却工作正常。同名、不同路径、结论相反——这就是必须做隔离实验的理由。

### 10.2 三个方法论陷阱（第一版敏感性探针全中，已重写）

| # | 成因 | 证据 | 后果 |
|---|---|---|---|
| 1 | **预算不匹配（我的错）** | BF16 同一语料：4096→9.1705；8192→**10.6146**；16384→8.3988；32768→9.3190 | BF16 自身摆 ±14%；拿 8192 比 32768 的 oracle 得出假「+18.56%」，真实约 +4.1% |
| 2 | **disable/enable 不幂等** | 同配置同 token：disable 全部→10.6146（恰等于该预算下 BF16，说明 disable 正确），再 enable→**18.5262**（原 11.0482） | 36 次循环逐步损坏模型，产生单调假排序 |
| 3 | **扩展懒加载** | 日志中 `Loading extension modelopt_cuda_ext…` 出现在实验中途 | 前半程 CPU 回退、后半程 CUDA，数值路径静默切换 |

**修正方法**（对齐 EZTrain 的 width_map）：保护写进**量化配置**，每个候选**从 BF16 重新量化**；扩展预加载；预算固定 32768。修正版自证：uniform 9.4870、保护 block 0 = +0.0826，与未受污染的早期探针一致。

### 10.3 附带产出：困惑度对 token 预算敏感（硬规则）

`results/accuracy/budget_sensitivity_bf16.json` —— 同一 BF16 模型、同一语料、只改评分 token 数：

| 预算 | 困惑度 | 相对 32k |
|---|---|---|
| 4 096 | 9.1705 | −1.59% |
| 8 192 | 10.6146 | **+13.90%** |
| 16 384 | 8.3988 | −9.87% |
| 32 768 | 9.3190 | 0.00% |

**困惑度只在同一 token 预算内可比较。** 本项目第三次同类错误（M1 同会话、M2 样本量、M4 预算）。

### 10.4 更正：wildcard bug——「保护 5 块」实际保护了 14 块

**发现方式**：导出体积对不上。混合档导出 **9.57 GiB，比 FP8 的 8.79 GiB 还大**——4-bit 不该更大。逐 tensor 清点：

```
bf16      245 tensors  7.35 GiB (76.8%)   ← 应为 147 个
U8        154 tensors  1.98 GiB (20.6%)
F8_E4M3   154 tensors  0.25 GiB
```

只有 154 个线性层是 4-bit，**98 个留在 BF16**；「保护 5 块」应只产生 35 个。98 ÷ 7 = **14 块** = `{0} ∪ {1,10,…,19} ∪ {12} ∪ {34} ∪ {35}`。

**根因**：保护写成两个 pattern，其中 `*model.layers.{b}*` 的尾部 `*` **吞掉下一位数字**，b=1 时同时匹配 `layers.10…19`；另一个 `*layers.{b}.*`（带点）安全，但两个同时生效 → 「保护 1 块」变成「保护 11 块」。

**两处污染必须更正**：① 敏感性排序（block 1 看似最敏感 +0.1263，实为 11 层合力）；② 混合档结论（「+0.56%、9.57 GiB」实为 14 块保护的结果）。

**修正后实测（真正保护 5 块：0/1/12/34/35）**：

| 档位 | 困惑度 | 相对 oracle | MMLU | 体积 |
|---|---|---|---|---|
| BF16 oracle | 9.3190 | — | 0.6120 | 15.26 GiB |
| FP8 W8A8 | 9.3434 | **+0.26%** | 0.6180 | 8.79 GiB |
| **混合 NVFP4（5 块，修正后）** | **9.4634** | **+1.55%** | **0.5560 (−5.6 pp)** | **7.25 GiB** |
| 混合（14 块，bug 版） | 9.3716 | +0.56% | 0.5800 (−3.2 pp) | 9.57 GiB |
| uniform NVFP4 | 9.5799 | +2.80% | 未测 | 待测 |

**M4 诚实结论**：
- 按决策 16（困惑度 ≤ +5%、MMLU ≥ 0.572）：修正后混合档 **MMLU 判 FAIL**、困惑度 PASS。
- 但 **MMLU 在 N=500 的分辨率是 ±4.3 pp**，门槛 0.572 落在 [0.513, 0.599] 内 → **这个 FAIL 与噪声不可区分**。
- 机制方向一致：保护越多越好（14 块→0.5800；5 块→0.5560）。
- **工程结论**：在「两个门禁都干净通过」上，**FP8 优于 4-bit 混合档**（+0.26%、MMLU +0.6 pp、8.79 GiB）。该模型/栈上 4-bit 的激活侧仍承受不起；多花的 1.5 GiB 是必要代价。
- **收官动作**：MMLU 提到 **N=2000**（分辨率 ±2.1 pp，约 20 分钟）才能真正判定修正后混合档是否掉出 −4 pp。

**流程教训（第四条同类）**：`09_sensitivity.py` 的 v2 只存在于 WSL、workspace 里仍是 v1，随后的双向 `cp` 又把 workspace 的 ROADMAP §10 覆盖掉。**脚本与文档必须只有一个权威副本，同步只能是单向的。**

_生成时间：2026-09-12 · 本文件由 agent 维护，里程碑内出现新事实时回来更新 §1 与 §4。_
