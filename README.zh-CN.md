# LLM 量化与部署（LLM Quantize + Serve）

[English](README.md) | **中文** ・ 报告：[English](docs/report.md) | [中文](docs/report.zh-CN.md)

在单张 **RTX 5090 Laptop（24 GB，Blackwell sm_120）** 上、WSL2 环境下对 **Qwen3-8B** 做的量化与部署研究——**先冻结测量口径再动手**，基线经过验证，**负面结果作为一等发现如实上报**。

下面每个数字都能追溯到 `results/` 里的结果文件与仓库中的脚本。凡是被后来发现无效的测量，都**记录更正**而非悄悄替换。

---

## 核心结果

| 档位 | 困惑度（32752 token） | 相对 BF16 | MMLU | 产物体积 | 判定 |
|---|---|---|---|---|---|
| **BF16（基准）** | 9.3190 | — | 0.6260（N=2000） | 15.26 GiB | 基准 |
| **FP8 W8A8** | **9.3434** | **+0.26 %** | 0.6180 | **8.79 GiB** | ✅ **部署档** |
| 混合 NVFP4 + 保护 5 块 | 9.4634 | +1.55 % | 0.5490（N=2000） | 7.25 GiB | ❌ MMLU −7.70 pp |
| uniform NVFP4（W4A4） | 9.5799 | +2.80 % | — | ≈5.6 GiB | 参考 |
| INT8 + SmoothQuant | 14.0729 | **+51 %** | — | — | ❌ 不可用 |
| INT8 per-tensor | 19.7043 | **+111 %** | — | — | ❌ 不可用 |

**结论：在这个模型与这套栈上，8 bit 就是地板。** 低于 8 bit 的**激活侧**量化救不回来——即使做了敏感性引导的逐层保护。INT8 的 per-tensor 激活缩放灾难性失败，而 FP8 只贵 0.26 %，差别在于 **FP8 保留指数字段**。机制是**激活离群值通道**：实测 `down_proj` 输入的 `amax` 达 1152–4016，单个共享整数缩放因子被少数离群值吃光；NVFP4 之所以比 INT8 好，是因为它有 16 元素分块的 microscaling。

## 服务化：量化到底买到了什么

用 `trtllm-serve`（OpenAI 兼容接口）做并发扫描：

| 并发 | BF16 tok/s | FP8 tok/s | 加速 | BF16 TPOT p50 | FP8 TPOT p50 |
|---|---|---|---|---|---|
| 1 | 44.2 | 75.0 | **1.70×** | 23.04 ms | **13.47 ms** |
| 4 | 177.0 | 298.3 | 1.69× | 22.26 | 12.87 |
| 8 | 342.9 | 514.4 | 1.50× | 23.03 | 15.19 |
| 16 | 642.9 | 982.4 | 1.53× | 24.53 | 15.71 |
| 32 | 1131.3 | 1668.0 | 1.47× | 27.57 | 18.26 |

**迁移链（端到端实测）：**

```
权重        15.26 GiB → 8.79 GiB      （小 42 %，省下 6.47 GiB）
KV cache    34,333   → 78,170 tokens  （同一显存比例下 ×2.28）
并发能力         8   → 19 条序列       （4096 token 上下文）
吞吐                    32 并发时 +47 %
单 token 延迟                  −40 %
```

**先说清楚一条：显存峰值几乎没变**（21.97 vs 21.58 GiB），因为服务端会把空闲显存全用上。**省下来的东西变成了缓存容量与吞吐，不是"占用变少"。**

## KV cache：一个有交叉点的权衡

FP8 权重、batch 1，fp8 KV cache 对 fp16 KV cache：

| 上下文 | fp16 KV | fp8 KV | 比值 |
|---|---|---|---|
| 1 024 | 71.7 tok/s | 53.7 | **0.75** |
| 4 096 | 64.0 | 51.1 | 0.80 |
| 16 384 | 31.1 | 32.7 | **1.05** |
| 32 768 | 17.8 | 22.6 | **1.27** |

量化 KV cache **在 ~16k 上下文以下要付 25 % 吞吐代价，在 32k 换回 27 % 收益**——cache 小的时候反量化开销主导，cache 大的时候显存带宽主导。池本身减半（33984 token 对应 4.67 → 2.33 GiB）。**"要不要量化 KV"是部署决策，取决于你真实的上下文分布，不是默认打开的开关。**

## 最重要的一条方法论发现

```
困惑度 : 9.3190 → 9.4634  = +1.55 %      "几乎无损"
MMLU   : 0.6260 → 0.5490  = −7.70 pp     严重受损（配对 N=2000, z = 4.96）
```

**稠密指标严重低估了任务级损伤。** 本项目原本把困惑度定为「真正的门禁」（因为它稠密、低方差、能分辨亚百分点），把 MMLU 降级为粗筛。这个测量把结论反过来了：困惑度看不见**窄而深**的损伤，比如推理能力退化。**两个指标缺一不可，两个门禁必须同时生效。**

---

## 出过什么问题（以及为什么留在仓库里）

这里几乎每一个硬结论背后都有一个**不报错、只给出合理数字**的 bug。每一条都在 `docs/ROADMAP.md` 里附了证据：

| # | 故障 | 表现 | 根因与修复 |
|---|---|---|---|
| 1 | WSL 镜像网络吞掉回环 SYN | `import tensorrt_llm` 永久挂死；`mpirun` 卡在 `SYN_SENT` → `127.0.0.1:6001` | 切 NAT 网络 |
| 2 | ModelOpt 的 CUDA 扩展从未编成 | FP8 直接 `NaN`；INT8 静默跑在 CPU 回退实现上 | `CXX` 必须指向真的 `g++`；`-std=c++20` 必须加在 **`extra_cuda_cflags`**（失败的是 `.cu`） |
| 3 | Hugging Face 端点设置在 import 之后 | 每个数据集重试 5 次才回落本地缓存 | `HF_ENDPOINT` 必须在 `transformers` 之前设置 |
| 4 | **通配符越界匹配** | 「保护 5 块」实际保护了 **14 块**（`*layers.1*` 同时匹配 `layers.10–19`）；4-bit 导出体积**比 FP8 还大** | 只用一个 pattern，结尾停在组件边界 |
| 5 | `disable_quantizer` → `enable_quantizer` 不幂等 | 一次开关循环让困惑度从 11.05 变 18.53 | 每个配置都从 BF16 重新量化，绝不在活模型上开关 |
| 6 | 跨 token 预算比较困惑度 | BF16 自身在 4k/8k/16k/32k 上是 9.17 / 10.61 / 8.40 / 9.32 | 预算属于测量口径的一部分 |
| 7 | 扩展懒加载 | 一半探针跑在另一条数值路径上 | 测量前强制加载扩展 |
| 8 | 用 token 一致率当等价性指标 | 「27 % 失败」而两份输出都通顺——贪心解码会放大一次翻转 | 改为 teacher-forced 逐位置比对 |
| 9 | 思考模式泄漏进测量 | 13 token 的提问 → 400 token 的回复 | `enable_thinking=False`，且通过 token id 传递 |
| 10 | WSL2 无进程级 NVML | 显存报 `NaN` | 全程设备峰值减空闲基线 |

**反复出现的教训：这套栈里的错误不会报错，它只会报出一个可信的数字。**

---

## 仓库结构

```
prompts/          冻结的 30 条 prompt（v1.1）+ 实测 token 统计 + SHA-256
docs/
  measurement_spec.md  协议：14 条决策、门禁、修正记录
  report.md            完整技术报告（英文）
  report.zh-CN.md      完整技术报告（中文）
  model_provenance.md  权重摘要、revision、推导出的显存常数
  glossary.md          本文用到的全部缩写
  how_to_run.md        速查表：每个里程碑跑什么、每个数字在哪一行生成
  ROADMAP.md           M0–M6 全过程，含所有更正
  ROADMAP.en.md        ROADMAP 的英文版
  *.zh-CN.md           以上文档的中文版
scripts/
  00_env_check.sh       环境验收测试
  02_baseline_bf16.py   延迟基线（加载 → 预热 → 计时循环）
  03_ptq_w8a8.py        PTQ：校准 → 量化 → 评测 → 导出 → 过门禁
  04_kv_cache_sweep.py  KV cache 精度 × 上下文扫描
  06_bench.py           对服务端点的并发扫描
  07_eval_accuracy.py   困惑度 + MMLU
  09_sensitivity.py     哪些块承担了 4-bit 的损伤
  10_mixed_precision.py 混合精度档（保护集直接写进量化配置）
  11_transfer_table.py  量化 → 缓存 → 并发 的迁移链
  modelopt_ext_patch.py 让 ModelOpt 的 CUDA 扩展能被编译出来
  compare_runs.py       5 % 可重复性门禁
  analyze_baseline.py   分桶拆解与 prefill 拟合
results/           上述每个测量的原始 JSON
```

## 复现

```bash
conda create -n llmquant python=3.12 && conda activate llmquant

# 环境（完整审计与偏离台账见 docs/ROADMAP.md §1）
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com

bash scripts/00_env_check.sh                    # 环境验收测试
python scripts/check_prompt_set.py              # 工作负载 hash 必须与协议一致
python scripts/02_baseline_bf16.py --tag run1   # 延迟基准（约 10 分钟）
python scripts/07_eval_accuracy.py --tier bf16  # 精度基准（约 5 分钟）
python scripts/03_ptq_w8a8.py --method fp8 --tag fp8   # 量化 + 门禁
python scripts/04_kv_cache_sweep.py             # KV cache 扫描
python scripts/06_bench.py                      # 服务并发扫描（需先起 trtllm-serve）
```

**硬件前提**：所有数字基于单张 24 GB Blackwell 显卡；未标注并发的延迟数字均为单请求、batch 1。

## 局限

- **单卡单机**，无张量并行、无多节点。
- **笔记本功耗墙（175 W）+ 24 GB 上限**：两次基线运行里最差的那项指标落在 5 % 门禁的 **+4.89 %**，只差 0.11 个百分点——热裕度很薄，如实记录而非掩盖。
- **MMLU 在 N=500 时分辨率是 ±4.3 pp**；4-bit 的结论只由 N=2000 的配对比较（−7.70 pp, z=4.96）支撑。
- **敏感性排序不完整**：通配符 bug 被发现前只有效探测了 36 块中的 10 块，保护集是「这十块里最好的」，未被证明全局最优。
- **NVFP4 权重可用，但 NVFP4 KV cache 会让 worker 进程硬崩溃**（本栈版本）。

## 许可

MIT（见 `LICENSE`）。模型与第三方二进制各有其许可；**本仓库不包含任何模型权重或引擎文件。**
