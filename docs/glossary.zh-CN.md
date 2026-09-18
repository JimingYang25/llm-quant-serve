# 术语表 — LLM 量化与推理服务

**中文** | [English](glossary.md)

本项目中用到的每一个缩写，附其全称、通俗含义，以及你实际会在哪里遇到它。条目按主题分组，而非按字母顺序，以便相关概念聚在一起。

---

## A. 量化：核心词汇

| 术语 | 全称 | 含义 | 你会在哪里遇到它 |
|---|---|---|---|
| **PTQ** | Post-Training Quantization（训练后量化） | 用一小批文本样本，把一个已经训练好的模型转换成更低精度，不做任何重新训练 | M2 — 你的第一个量化档位 |
| **QAT** | Quantization-Aware Training（量化感知训练） | 在训练*过程中*模拟低精度，让模型学会容忍它 | 本项目不使用；与 PTQ 对照 |
| **Calibration** | — | 让几百条文本样本跑过模型，记录每个张量实际会取到的数值范围 | M2 — 决定每一个量化尺度 |
| **Calibration set** | — | 用于上述过程的样本语料。它不得与评估集重叠，否则报告的精度会被虚高 | M2 |
| **Scale / zero-point** | — | 仿射映射（affine mapping）里的两个数字：`real ≈ scale × integer + zero_point` | 解释了为什么粒度（granularity）重要 |
| **per-tensor / per-channel / per-group** | — | 一个张量分到多少个尺度：整个张量共用一个、每个输出通道一个，还是每一小组权重一个 | M4 — 分组大小（group size）是关键旋钮 |
| **Weight-only vs weight+activation** | — | 是只量化权重（激活保持 16 位），还是两者都量化 | 整个项目中最重要的一处区分 |
| **W8A8 / W4A16** | Weight 8-bit, Activation 8-bit / Weight 4-bit, Activation 16-bit（权重 8 位、激活 8 位 / 权重 4 位、激活 16 位） | 量化配方的紧凑记法 | M2（W8A8）、M4（W4A16） |
| **INT8** | 8-bit integer（8 位整数） | 均匀整数格式；需要一个标定过的范围 | M2 |
| **FP8 E4M3 / E5M2** | 8-bit float, 4 exponent + 3 mantissa bits / 5 + 2（8 位浮点，4 位指数 + 3 位尾数 / 5 + 2） | 浮点格式：动态范围宽、尾数位少。E4M3 用于权重/激活，E5M2 用于梯度 | M3 — Blackwell 原生具备 FP8 |
| **FP4 / NVFP4 / MXFP4** | 4-bit float（NVIDIA 方案 / OCP 微缩放（microscaling）方案） | 4 位浮点格式；NVFP4 是 NVIDIA 在 Blackwell 时代的变体 | M4 — sm_120 上的可用性必须被*实测* |
| **Outlier channel** | — | 少数激活通道的数值幅度远大于其余通道，会毁掉一个统一的共享尺度 | 这就是激活量化比权重量化更难的原因 |
| **AWQ** | Activation-aware Weight Quantization（激活感知权重量化） | 在量化到 4 位之前，先保护那些对激活最重要的权重 | 若 NVFP4 失败，作为 M4 的退路 |
| **GPTQ** | Generative Pre-trained Transformer Quantization（一种 PTQ 方法） | 逐层做权重量化，使输出误差最小化 | M4 中 AWQ 的替代方案 |
| **SmoothQuant** | — | 通过重缩放（rescaling）把激活上的困难迁移进权重，使 W8A8 更容易 | M2 的可选改进 |
| **Fake quantization / QDQ** | Quantize-DeQuantize（量化-反量化） | 插入量化 + 反量化算子，让浮点模型表现得*如同*已量化——用于在不部署的情况下测量损伤 | 敏感性探测（M4） |
| **KV cache quantization** | Key/Value cache | 对缓存下来的注意力 key 和 value 做量化，也就是随上下文长度增长的那部分显存 | M3 — 常常是真正的显存瓶颈 |
| **Perplexity** | — | `exp(mean negative log-likelihood)`：模型对一段文本语料预测得好不好的连续度量。越低越好 | M1 — 精度 oracle |
| **Accuracy delta** | — | 某个量化档位与未量化参考之间在指标上的差值 | 你的报告就建立在这个数字之上 |

---

## B. 服务与性能指标

| 术语 | 全称 | 含义 | 你会在哪里遇到它 |
|---|---|---|---|
| **Oracle / baseline** | — | 未量化的参考测量，其余一切都以它为比较基准 | M1 — 你要产出两个（精度、延迟） |
| **TTFT** | Time To First Token（首个 token 的时间） | 从提交请求到第一个输出 token 之间的延迟；主要取决于 prompt 处理 | M1 规格、M5 扫描 |
| **TPOT** | Time Per Output Token（每个输出 token 的时间） | 第一个 token *之后*每个 token 的平均耗时；决定答案「流」给用户的速度 | M1 规格、M5 |
| **ITL** | Inter-Token Latency（token 间延迟） | 与 TPOT 是同一个量，只是从流式（streaming）视角命名 | 出现在 TRT-LLM 自己的报告里 |
| **Throughput** | — | 每秒 token 数。必须始终注明限定条件：单流，还是全部请求合计 | M5 |
| **Prefill / decode** | — | 两个阶段：读取 prompt（计算受限、可并行）与生成 token（显存带宽受限、必须串行） | 解释了 TTFT 与 TPOT 的区别 |
| **Chunked prefill** | — | 把长 prompt 拆成若干段，避免一次巨大的 prefill 卡住其他请求的 decode | M1 的 read-back 问题 |
| **Continuous batching** | — | 一旦有空位就立刻接纳新请求，而不是等整批结束 | M5 |
| **KV cache** | Key/Value cache | 为已经处理过的 token 保存下来的注意力状态；显存随上下文长度线性增长 | M3 — 上下文长度扫描 |
| **Paged KV cache / PagedAttention** | — | 把 KV cache 存进固定大小的块（block）里，以消除碎片 | M5 |
| **CUDA Graph** | — | 把固定的 kernel 序列录制一次然后重复回放，以去掉每一步的启动开销 | M5 的可选收益 |
| **Concurrency / batch size** | — | 在途请求数 / 一起被处理的序列数 | M5 扫描：1、4、8、16、32 |
| **Context length** | — | prompt + 生成输出中的 token 总数 | M3 扫描：1k、4k、16k、32k |
| **p50 / p95 / p99** | 50th / 95th / 99th percentile（第 50 / 95 / 99 百分位） | 有该比例的测量值落在其下的那个数值。p99 暴露尾部延迟 | 每一张结果表 |
| **Warmup** | — | 测量前丢弃的若干次迭代，从而排除冷缓存、JIT 编译与时钟升频 | M1 的纪律 |
| **Peak VRAM** | Video RAM（显存） | 运行期间分配的最大 GPU 显存 | M1 指标、M3 瓶颈分析 |
| **Engine** | TensorRT engine | 由模型生成的一份已编译、与 GPU 架构绑定的产物；不可跨 TRT 版本或架构移植 | M5 |
| **TRT-LLM** | TensorRT-LLM | NVIDIA 的 LLM 推理栈（两种模式：构建 engine，或直接跑 checkpoint） | 本项目的运行时 |
| **ONNX** | Open Neural Network Exchange（开放神经网络交换格式） | 一种模型交换格式 | 可选的导出路径 |
| **ORT** | ONNX Runtime | 执行 ONNX 图的运行时；它自己也能施加 INT8 量化 | TRT 的替代方案 |
| **OpenAI-compatible API** | — | 与 OpenAI 的 schema 相匹配的 HTTP 接口，使现有的客户端无需改动即可使用 | M5 服务层 |

---

## C. 精度评估

| 术语 | 全称 | 含义 | 你会在哪里遇到它 |
|---|---|---|---|
| **lm-eval-harness** | Language Model Evaluation Harness（语言模型评估框架） | 用来运行学术基准测试的标准开源框架 | M1 评估 |
| **MMLU** | Massive Multitask Language Understanding（大规模多任务语言理解） | 覆盖约 57 个学科的选择题知识基准 | 可选的任务指标 |
| **GSM8K** | Grade School Math 8K（小学数学 8K） | 小学数学应用题；对推理能力的退化很敏感 | 可选的任务指标 |
| **Sample size / statistical power** | — | 要把某个差异与噪声区分开，需要多少条测试样本。50 条样本无法分辨 1 % 的差距 | 必须在规格中如实写明 |
| **Confound** | Confounding variable（混杂变量） | 一个非预期的差异，它才可能是结果的解释（例如在不同 `transformers` 版本下评估两个档位） | 冻结规则（freeze rule）存在的理由 |
| **Ablation** | — | 一种受控实验：只改变一个因素，从而把效应归因于它 | M2（校准集大小）、M4（混合精度） |
| **Greedy decoding** | — | 总是取概率最高的那个 token（`temperature=0`） | M1 中的可复现性要求 |
| **Seed** | — | 随机数种子；在不使用贪心解码时用来固定采样 | M1 规格 |
| **Thinking mode** | — | Qwen3 内置的推理模式，会在给出答案之前输出一长段内部独白，极大地改变输出长度 | 必须在 M1 中显式冻结 |

---

## D. 测量方法论与项目治理

| 术语 | 全称 | 含义 | 你会在哪里遇到它 |
|---|---|---|---|
| **Measurement spec** | — | 冻结下来的文档，定义测量什么、怎么测、容差是多少 | M1 交付物 |
| **Fingerprint** | Environment fingerprint（环境指纹） | 记录下来的版本集合（torch、TRT-LLM、modelopt、transformers、驱动），用来标识某个数字背后的确切环境 | 每个结果文件内部 |
| **Reproducible** | — | 相同输入 + 相同环境 → 相同数字 | 本项目的标准 |
| **Gate** | — | 一条明确的通过/失败判据，由程序自动评估，而不是靠印象 | `08_gate.py`，每个里程碑 |
| **Deviation (D1–D3)** | — | 对规定环境集合的一次*已知且记录在案*的偏离，并附有理由 | D1 = `nvidia-cuda-runtime` 补丁版本不匹配；D3 = transformers/modelopt 版本范围 |
| **Drift** | — | 两个状态之间非预期的变化，通过比对冻结文件发现 | 恢复检查 |
| **Regression** | — | 以前能用的东西现在坏了 | 移除 cu12 之后 `import torch` 挂掉 |
| **Read-back** | — | 用自己的话复述某个机制，作为理解的证据，而不只是执行过 | M0/M1 签收 |
| **M0–M6** | Milestone 0 … 6（里程碑 0 … 6） | 项目阶段：环境 → 测量 → INT8 → FP8/KV → 4-bit → 服务 → 报告 | 路线图 |

---

## E. 环境、构建与系统术语

| 术语 | 全称 | 含义 | 你会在哪里遇到它 |
|---|---|---|---|
| **ABI** | Application Binary Interface（应用二进制接口） | 一个已编译库与其调用方之间的二进制层面契约 | 为什么换一个 torch 版本就可能弄坏已编译的扩展 |
| **soname** | Shared object name（共享对象名） | 一个 `.so` 的 ABI 身份，例如 `libcudnn.so.9`。共享同一 soname 的两个版本是 ABI 兼容的 | 13.0.48 与 13.0.96 都叫 `libcudart.so.13` |
| **Pin** | — | 包元数据里的精确版本要求（`==`） | 两次依赖死锁的共同根源 |
| **Resolver** | — | pip 内部选择安装哪些版本的依赖求解器 | 它悄悄替换了你的 torch 构建 |
| **Wheel** | — | 预构建好的、可直接安装的 Python 二进制包（`.whl`） | TRT-LLM 的 wheel 有 2.5 GB |
| **RPATH** | — | 被写入 ELF 二进制内部的库搜索路径；这就是为什么用的是 wheel 自带的 CUDA 库，而不是系统 toolkit | 加载来源（loader-provenance）检查 |
| **SASS / PTX / cubin** | — | 最终的 GPU 机器码 / 向前兼容的中间代码 / 一份已编译的 kernel 镜像。缺少面向你所处架构的 SASS 时，就会被迫走 JIT | 为什么 `arch_list` 里的 `sm_120` 很重要 |
| **sm_120 / compute capability** | — | GPU 架构编号。`sm_120` 是消费级 Blackwell（你的 5090） | kernel 兼容性 |
| **JIT** | Just-In-Time compilation（即时编译） | 在首次使用时才编译，而不是提前编译；会造成首次调用的停顿 | 必须被排除在每一个计时区间之外 |
| **Driver API vs Runtime API vs Toolkit** | — | 三个层次：驱动所设的能力上限、程序所链接的运行时、以及编译期 toolkit（它不影响执行） | CUDA 对齐讲座 |
| **bind-mount / overlayfs / lowerdir** | — | WSL 用来把 Windows 驱动的 GPU 库呈现到 Linux 内部的 Linux 机制，其中一个目录层会遮蔽另一个 | `590.62` 与 `592.01` 之谜 |
| **WSL2 mirrored vs NAT networking** | Windows Subsystem for Linux（适用于 Linux 的 Windows 子系统） | 两种网络模式：`mirrored` 共享宿主机的 loopback；`NAT` 让虚拟机在网关之后拥有自己的网络栈 | 那一夜的断网及其修复 |
| **MPI** | Message Passing Interface（消息传递接口） | 多进程/多节点通信的标准；推理栈用它来做多 GPU | 那个卡住的东西 |
| **OpenMPI / `orted` / BTL / OOB** | — | 一个具体的 MPI 实现 / 它的守护进程 / 它的网络传输组件 / 它的带外（out-of-band）消息组件 | `orted` 就是 mpirun 一直在等的那个进程 |
| **mpi4py** | — | MPI 的 Python 绑定 | 被 TRT-LLM 硬导入 |
| **RECORD** | — | pip 为每个包记录的已安装文件路径清单；这就是卸载一个包可能删掉另一个包文件的原因 | `libcudnn.so.9` 事件 |
| **Freeze file** | `pip freeze` 输出 | 已安装包版本的确切清单；可用作回滚点 | `pip_freeze_before_cu12_removal.txt` |
| **Orphan package** | — | 已安装，但没有任何其他包声明依赖它 | 那 15 个 CUDA-12 包 |
| **Monorepo / editable install** | — | 与本项目无关；仅因为日志里出现了 `pip install -e` 才列出 | — |

---

## F. 如何使用本文档

在写代码之前，你**并不**需要掌握这里的全部内容。具体到 M1，只有下面这些是承重的：

`TTFT`, `TPOT`, `throughput`, `p50/p95/p99`, `warmup`, `peak VRAM`, `perplexity`, `oracle`, `fingerprint`, `gate`, `confound`, `thinking mode`.

A 节的全部内容从 M2 起才开始起作用；而 E 节是你已经亲身经历过一遍的背景。
