# 速查表 —— 如何运行本项目

**中文** | [English](how_to_run.md)

为「细节已经多到压垮人」的情形而写。如果你在动手之前只读一个文件，就读这一个。完整论证在
`docs/measurement_spec.md`；你不需要把它记在脑子里。

---

## 1. 用四个数字概括整个项目

| 数字 | BF16 取值 | 由谁产出 |
|---|---|---|
| 困惑度（perplexity，wikitext-103 切片） | **9.3190** | `scripts/07_eval_accuracy.py` |
| MMLU 准确率（N=500） | **0.6120 ± 0.0218** | `scripts/07_eval_accuracy.py` |
| TTFT p50 / TPOT p50 | **35 ms / 34 ms** | `scripts/02_baseline_bf16.py` |
| 峰值显存 | **15.98 GiB** | `scripts/02_baseline_bf16.py` |

项目中的其他一切，要么是产出这些数字的手段，要么是证明这些数字可信的手段。量化（quantization，从
M2 开始）会改变这些数字；我们的工作是**只**在能够为其辩护的幅度内改变它们。

---

## 2. 唯一需要记住的规则

> **如果环境、提示集（prompt set）、权重或测量脚本发生变化，
> 旧的数字就不再具有可比性。**

这就是为什么每个结果文件都携带哈希值。你不需要背下版本矩阵 —— 文件自己带着它。

---

## 3. 该运行什么，该看什么

### 任何测量之前（30 秒）

```bash
bash scripts/00_env_check.sh          # environment sane: torch, TRT, MPI, CUDA family
python scripts/check_prompt_set.py    # prompt set unchanged; prints its hash
```

检查：没有报错，且打印出的哈希与 `docs/measurement_spec.md` 中的一致。

### M1 —— BF16 基准真值（oracle，已跑过一次）

```bash
python scripts/02_baseline_bf16.py --tag run1      # ~10 min
python scripts/02_baseline_bf16.py --tag run2      # ~10 min, same session
python scripts/compare_runs.py results/baseline_bf16/*run1.json \
                               results/baseline_bf16/*run2.json
python scripts/analyze_baseline.py results/baseline_bf16/*run1.json
```

只需检查两件事：
1. `compare_runs.py` 输出 **PASS**（所有受门禁（gate）约束的指标都在 5 % 以内）。
2. `analyze_baseline.py` 显示 TTFT 随提示长度上升，而 TPOT 大致持平。

### 准确率，适用于任何档位（tier）

```bash
python scripts/07_eval_accuracy.py --tier bf16 --smoke    # seconds: shape check
python scripts/07_eval_accuracy.py --tier bf16            # ~5 min: real numbers
```

检查：MMLU 远高于 0.25（随机猜测水平）。如果接近 0.25，说明提示或思考模式（thinking mode）有问
题 —— 不要记录该次运行。

### 查看任意结果

```bash
python scripts/show_result.py                          # newest file
python scripts/show_result.py results/accuracy/<file>.json
```

---

## 4. 每个数字在代码中的生成位置

### `scripts/02_baseline_bf16.py` —— 延迟与显存

| 行号 | 它做什么 | 它产出的数字 |
|---|---|---|
| 68–73 | 协议常量：`MAX_NEW_TOKENS=128`、`WARMUP=3`、`REPS=5`、贪心解码（greedy）、百分位（percentile）方法 | 每一个数字背后的设置 |
| 110 | `build_fingerprint()` | 每个结果文件中的版本信息块 |
| 143 | `load_prompt_set()` | 把 `doc_refs` + `paragraph_slice` 解析为提示文本 |
| 165 | `build_inputs()` | 应用聊天模板，且思考模式**关闭** |
| 186 | `class StepTimer` | 每个解码步骤的时间戳 → **TTFT** 与 **TPOT** |
| 202 | `class VramSampler` | **峰值显存**、温度、功耗 |
| 277 | `percentiles()` | **p50 / p95 / p99** |
| 290 | `warmup()` | 在计时之前运行；按设计不产出任何数字 |
| 302 | `measure_one()` | 单次请求的 TTFT/TPOT/墙钟时间；断言思考内容没有泄漏 |
| 345 | `main()` | 执行顺序：加载 → 预热 → 开始计时 → 循环 → 写出 JSON |

### `scripts/07_eval_accuracy.py` —— 准确率

| 行号 | 它做什么 | 它产出的数字 |
|---|---|---|
| 50–54 | `PPL_WINDOW=2048`、`PPL_TOKENS=32768`、`MMLU_N=500`、随机种子 | 两项指标背后的设置 |
| 114 | `build_ppl_windows()` | 固定的文本切片及其哈希 |
| 154 | `perplexity()` | **困惑度**（按 tokens 加权，因此在数学上是正确的） |
| 173 | `load_mmlu()` | 抽样的 500 道题及其哈希 |
| 184 | `make_prompt()` | 题干 + 选项，思考模式关闭 —— **就是修掉 0.25 的那一行** |
| 211 | `score_choices()` | " A"/" B"/" C"/" D" 的对数概率（log-probability） |
| 228 | `mmlu()` | **准确率**、标准误、随机猜测水平 |

### `scripts/compare_runs.py` —— 门禁（gate）

| 行号 | 它做什么 |
|---|---|
| 26 | `TOLERANCE_PCT = 5.0` —— 决策 10 的阈值，你唯一必须满足的数字 |
| 29–35 | 两次运行要具备可比性所必须*完全相同*的东西 |
| 37 | 哪些指标受门禁约束（TTFT、TPOT），哪些仅作为上下文信息 |
| 60 | `main()` —— 打印 PASS/FAIL，失败时给出诊断顺序 |

### `scripts/analyze_baseline.py` —— 解释，而非测量

| 行号 | 它做什么 |
|---|---|
| 32 | `analyze()` —— 分桶表格、预填充（prefill）拟合、逐次重复漂移 |

这一个不产出任何受门禁约束的数字。它的存在是为了解释这些数字*为什么*长成这样 —— 也就是
“TTFT ≈ 29 ms + 66 µs × prompt_tokens” 这行结论的出处。

---

## 5. 如果看起来不对劲

| 症状 | 首先检查 |
|---|---|
| MMLU ≈ 0.25 | `07_eval_accuracy.py:184` —— 思考模式必须关闭 |
| TTFT 以秒为单位而非毫秒 | `02_baseline_bf16.py:345` —— `t0` 是在加载与预热之后吗？ |
| 两次运行相差 >5 % | 对比两个 JSON 中的 `thermal`；关闭占用 GPU 的应用；重跑 |
| 启动时哈希不匹配 | 提示集变了 —— 测量之前先重新冻结（re-freeze） |
| `import tensorrt_llm` 卡住 | MPI/回环（loopback）问题；见 ROADMAP §1.5–1.6 |

## 6. 保持两种语言同步

每份文档都有另一语言的对应版本，两侧标题下都带语言切换链接。改动任意一侧之后，跑一遍校验脚本：

```bash
python scripts/check_docs_bilingual.py      # 某对文档脱节则退出码为 1
python scripts/check_docs_bilingual.py -v   # 同时打印逐节行数
```

它比对数值——原文里出现的每一个数都必须在译文里出现——以及标题、代码块、表格行、章节数量，
还有切换链接。它刻意只做结构性检查：措辞可以改，但测量结果不许消失。
