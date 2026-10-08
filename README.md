# mini-sglang

**从零实现的轻量级 LLM 推理引擎** —— 参考 SGLang 架构，以 TDD 方式亲手构建推理系统的五层核心路径：解码循环 → KV Cache → Continuous Batching → Radix 前缀缓存 → 分页 KV / PagedAttention。

不是为了调用推理框架，而是为了回答一个问题：**vLLM/SGLang 到底在优化什么，为什么，以及优化在什么场景下不划算。**

> ⚠️ **与官方同名项目的区分**：本仓库是**教学向**实现，与 SGLang 官方的
> [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)（生产级精简框架，~5000 行 + CUDA kernel，H200 级 benchmark）**无代码或血缘关系**。
> 二者的交集只有 Radix Cache 一个概念。模块级对照与缺口分析见 [`docs/official-vs-mine.md`](docs/official-vs-mine.md)。

- 硬件：RTX 4070 Laptop (8GB) ｜ 模型：GPT-2 124M / Qwen2.5-0.5B
- 全程测试驱动：129 项测试**全部通过**，每条优化路径都与朴素实现做逐 token 一致性验证

## 快速入门

```bash
git clone https://github.com/tangqwert/mini-sglang && cd mini-sglang
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers pytest

# 可选：Triton（M5）首次启动 kernel 时会现场编译一个 CUDA 启动器，需要 C 编译器 + Python 头文件
#   sudo apt-get install -y build-essential python3-dev
# 没装也能跑 —— tests/test_triton_*.py 会自动跳过

# 1. 全部测试（单元 + 真模型集成）
pytest tests/

# 2. 看两条引擎的对比：同一长 prompt，朴素 vs KV Cache
python -m benchmark.bench --engine naive --model gpt2 --max-new-tokens 128
python -m benchmark.bench --engine kv    --model gpt2 --max-new-tokens 128

# 3. 端到端验证脚本：批处理 / 前缀缓存 / 分页调度 / kernel 的正确性与收益
python -m benchmark.verify_m2
python -m benchmark.verify_m3
python -m benchmark.verify_m4b   # 分页调度 vs M2.5 连续准入（量化"填充浪费"）
python -m benchmark.verify_m6    # 三方对照：M2.5 / 分页 / Chunked Prefill
python -m benchmark.verify_m5    # PyTorch 分页注意力 vs Triton kernel
```

预期输出（KV Cache 路径）：

```
prompt tokens      : 5
new tokens         : 128
avg time/token     : 6.1 ms
decode throughput  : 163.5 tokens/s
```

## 架构

```
┌──────────────────────────────────────────────┐
│ benchmark/  bench.py(--engine naive|kv)      │  测量与验证
│             verify_m2/m3/m4b/m5/m6.py        │
├──────────────────────────────────────────────┤
│ engine/paged_schedule.py  分页/合流调度      │  页表 + varlen（M4b/M6）
│ engine/model_forward.py   自研 GPT-2 前向    │  接管 attention（M4b）
│ engine/paged_kv.py       分页 KV 池 + 注意力 │  按页表 gather（M4a）
│ engine/triton_paged.py    分页注意力 kernel  │  Triton（M5）
│ engine/radix_cache.py     Radix 树前缀缓存   │  跨请求复用 KV
│ engine/batching.py        动态退出批调度     │  多请求共享前向
│ engine/kv_cache.py       增量解码(prefill+1) │  免重复计算
│ engine/naive_decode.py    朴素解码循环       │  基线（对照组）
├──────────────────────────────────────────────┤
│ transformers （只取权重；M4b 起前向自研）    │
└──────────────────────────────────────────────┘
```

核心设计：解码循环与模型**解耦**——循环只认识 `logits_fn` / `kv_forward` 这两个可调用对象，
因此 M1/M2/M3 三次优化都没有改动循环本体，只替换了注入的前向实现。

## 基准结果（RTX 4070 Laptop，fp32）

| 优化 | 场景 | 结果 | 结论 |
|---|---|---|---|
| **KV Cache** | gpt2，696-token prompt | 29.2 → 6.5 ms/token（**4.5x**） | 消除序列长度维度的重复计算 |
| KV Cache | Qwen2.5-0.5B，601-token prompt | 28.9 → 14.8 ms/token（**1.95x**） | 收益被权重搬运地板压缩 |
| KV Cache | gpt2，5-token prompt | 5.8 → 6.1 ms/token（0.94x，略负） | 短上下文无浪费可省，机制开销反超 |
| **Batching** | 4 请求批解码 vs 逐条 | 516 vs 606 ms（**1.18x**） | 收益 ∝ 请求数 × 权重搬运占比 |
| **连续准入** | 4 请求（预算 1/5/1/5）｜槽位 2 | 静态分批 10 次前向 → **8 次**（6 步批量 + 2 次补入） | 理想 2.00x，实测 **1.25x** —— 差距即"补入需独立前向"的代价 |
| **Radix 前缀缓存** | 3 请求共享 118-token 前缀 | **0.82x**（输出逐 token 一致） | 小模型 prefill 被权重搬运主导 |
| **分页调度（Step 4）** | 1 长请求(153 tok) + 8 短请求｜槽位 2 | 喂入 token **1463 → 222（6.6x）**；填充位置 1220 → **0**；前向次数 37 → 37（不变） | 分页只消掉了"填充计算" |
| **Chunked Prefill（M6）** | 同上 | 总前向 37 → **30**；补入独立前向 7 → **0**；prefill/decode 同批 **7 步** | prefill 与 decode 合流，额外前向消失 |
| **Triton kernel（M5）** | M6 负载，端到端 | PyTorch 分页 **565 → 301 ms（1.88x）**，输出与调度统计完全一致 | 把每层十来个 kernel + 一圈 Python 循环折成 **1 个** kernel |
| Triton kernel 微基准 | 单层 decode，seq_len 128–2048 | 1.0–2.0x，且**不随 seq_len 单调** | PyTorch 版是**启动受限**（耗时几乎不随长度变），Triton 版是**带宽受限** |
| ⚠️ 同一负载的墙钟 | 同上 | M2.5 306ms → Step4 670ms → M6 624ms → **M6+Triton 301ms** | 分页路径先慢 2 倍（Python 循环），M5 追平 |

> 注：gpt2 与 Qwen 两次长 prompt 实验的 prompt 长度不同（696 / 601 token，出处见 `benchmark/results/*.json`）。
> 由于两者的权重规模差 4x、地板效应本就主导，长度差异不影响结论方向；若要严格 apples-to-apples，需在相同长度下重跑。

## 核心洞察：七个"理论收益 ≠ 实测收益"的对照实验

每个里程碑都做了理论推导与实测的对照，七次实验共同指向同一条成本模型：

```
每步耗时 = 权重搬运（固定，∝模型大小） + 序列计算（∝上下文长度）
```

- **KV Cache** 省的是"序列计算"——模型越小、prompt 越长，收益越大（4.5x ↔ 1.95x 的差异由此而来）
- **Batching** 省的是"N 条请求重复的权重搬运"——并发数上不去时收益有限（1.18x）
- **连续准入** 省的是"已完成请求占着槽位空转"——但没有 PagedAttention 时，补入一条新请求需要
  一次**独立的前向**，把理想收益从 2.00x 压到 1.25x
- **Radix Cache** 省的是"跨请求重复的 prefill 计算"——小模型 prefill 本身 ≈5ms，被 clone 与 Python 开销反超（0.82x）；真实收益场景是大模型 × 长前缀 × 高命中率（SGLang 用 PagedAttention 的零拷贝页引用消除 clone 开销）
- **分页调度（M4b Step 4）** 省的是"补入新请求时的填充计算"——左填充彻底消失，喂入 token 数降为 **1/6.6**
- **Chunked Prefill（M6）** 省的是"补入要多开一次前向"——prefill 与 decode 合进同一步
- **Triton kernel（M5）** 省的是"注意力实现里的 kernel 启动与 Python 开销"——把每层十来个 kernel 折成 1 个（端到端 565 → 301 ms）

**推论**：推理优化的收益 = 被省成分的成本 − 新增机制的开销。选型前先算清被省的部分在成本结构中占多少——这也是每个推理引擎的性能调优起点。

**两次出现“负收益 / 收益被吃光”（0.82x、1.25x vs 理想 2.00x），共同指向同一个东西：没有分页的 KV 管理开销。** 这就是 M4a 的动机。

那个 1.25x 的坑，被 M4b/M6 拆成两半、分两步修完：

| 代价 | M2.5 | Step 4（分页） | M6（合流） |
|---|---|---|---|
| 补入时的**填充计算** | 每个补入前向要算 S 个位置（S≈当前批长） | ✅ 只算该请求自己的 L | ✅ 只算 L |
| 补入需**一次独立前向** | 7 次 | ❌ 仍是 7 次 | ✅ **0 次**（与 decode 同批） |
| 喂入 token | 1463 | 222（6.6x ↓） | 222 |
| 总前向次数 | 37 | 37 | **30** |

> 这个“分两步才修完”的过程比单一加速比更有价值：它把“1.25x vs 理想 2.00x”的模糊猜测，
> 变成了一张明确的待办清单。

### 然后是：墙钟反而更慢了 → M5 把它扳回来

M4b/M6 之后同一负载 **M2.5 306ms → Step4 670ms → M6 624ms**：喂入 token 降了 6.6 倍，墙钟却**慢了 2 倍**。

| 原因 | 说明 |
|---|---|
| 逐序列注意力是 **Python 循环** | 每（序列 × query 位置 × 层）都要 gather → einsum → softmax → einsum，约 10 次 kernel 启动 + 一段 Python。12 层 × 30 步 ≈ 7000 次启动，全是 CPU 时间 |
| 小模型地板效应 | gpt2 每步要搬 0.5GB 权重，算力省下来也快不了多少 |

**M5 就是冲着这个 2 倍差距做的，而它确实兼现了：**

| 后端 | 墙钟 | 说明 |
|---|---|---|
| PyTorch 分页 | 565 ms | 查页表 gather + 逐 query 循环 |
| **Triton kernel** | **301 ms（1.88x）** | 查页表 + 因果掩码 + online softmax 全在 **1 个** kernel 里 |

至此分页路径的墙钟追平了 M2.5（301 vs 306 ms），**而喂入 token 数低 6.6 倍** ——
代价结构完全不同：M2.5 靠 HF 里融合好的注意力白拿速度，我们靠自己的 kernel。

> ⚠️ 但微基准揭示了一个**尚未解决的短板**：单序列只有 H=12 个 program（1 序列 × 12 head），
> 128 个 SM 大量空转；而且 PyTorch 版是"启动受限"（耗时几乎不随 seq_len 变）、
> Triton 版是"带宽受限"，所以微基准加速比只有 1.0–2.0x 且**不随 seq_len 单调**。
> 真 vLLM 用 **split-K（flash-decoding）** 把 KV 维也切开并行再规约 —— 这就是 M5.5 的待办。

## 与真 SGLang 的差距（诚实清单）

| 能力 | mini-sglang | 真 SGLang |
|---|---|---|
| Radix 树节点分裂 | 部分重叠直接放弃插入（宁缺毋错） | 分裂节点，重叠段共享 |
| 槽位补位（refill） | 有；补入已做到**零填充 + 与 decode 合流**（M6），`admission_forwards == 0` | Continuous admission，且 prefill 与 decode 在同一个 CUDA kernel 内完成 |
| KV 显存管理 | 分页池 + 页表 + 按需增长 + 完成即归还，调度器全程走页表（M4b Step 4） | 同上，但 gather 在 CUDA kernel 内完成 |
| 服务层 | 无（库形态） | HTTP Server + tokenizer manager |
| Kernel | **自研 Triton kernel**（M5）：分页版 flash attention，每层 2 个 kernel 启动；但无 split-K，单序列并行度只有 H | FlashInfer / 自研 CUDA（含 split-K / decode 专用 kernel） |

> 上表是精简版。**模块级对照（官方 15 个模块）、三层覆盖度评估、缺口清单与补充计划**见 [`docs/official-vs-mine.md`](docs/official-vs-mine.md)。

## 开发方式

TDD / 规格先行：`tests/` 定义行为契约（含一个**上下文依赖的假模型**——它让"丢缓存"类 bug 无法蒙混过关），`engine/` 中的实现逐里程碑完成；每个里程碑在 `benchmark/results/` 留档数据。

测试金字塔：77 项假模型 / 纯张量单元测试（毫秒级，精确断言内部行为）+ 52 项真模型测试（含「自研前向 vs HF」逐元素对比、「分页增量解码 vs 朴素解码」逐 token 对比、「Triton kernel vs PyTorch 参考」逐元素对比）。

## 路线图

- [x] **M0** Naive 解码循环 + baseline（179 tokens/s @ gpt2）
- [x] **M1** KV Cache 增量解码（4.5x @ 696-token prompt）
- [x] **M2** Continuous Batching（动态退出批调度）
- [x] **M2.5** 调度器 + 槽位连续准入（冻结行腾位、pending 立刻补入；实测 1.25x vs 理想 2.00x）
- [x] **M3** Radix 前缀缓存复用
- [ ] **M3.5** Radix 节点分裂（部分重叠序列的完整缓存）
- [x] **M4a** 分页 KV 池 + PagedAttention（按页表 gather + 多头缩放点积，与连续存储逐元素一致 < 1e-5）
- [x] **M4b** 分页接入真实前向：自研 GPT-2 前向 ✅ / attention 换分页版 ✅ / 增量解码 ✅ / 接调度器 ✅（零填充 varlen 调度，喂入 token 降为 1/6.6）
- [x] **M5** 自研 Triton kernel（分页版 flash attention：查页表 + 因果掩码 + online softmax 封进 **1 个** kernel；端到端 **1.88x**）★
- [ ] **M5.5** split-K / flash-decoding（把 KV 维也切开并行，解决"单序列只有 H 个 program、SM 大量空转"）
- [x] **M6** Chunked Prefill（prefill 与 decode 合流；`max_prefill_tokens` 给单步成本设上限）
- [ ] **M7** CUDA Graph（消除 decode 阶段的 kernel launch 开销）
- [ ] **M8** 采样策略（temperature / top_p / top_k）
- [ ] **M9** 最小 HTTP 服务（`/v1/chat/completions`）

> 与 SGLang 官方 [mini-sglang](https://github.com/sgl-project/mini-sglang) 的模块级对照、覆盖度与缺口分析见 [`docs/official-vs-mine.md`](docs/official-vs-mine.md)。
> **不计划实作**：Tensor Parallelism、Overlap Scheduling、ZMQ 多进程架构 —— 以读懂并讲清原理为目标。

## 环境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers pytest
# 若报 "NVIDIA driver too old"，改装匹配驱动的版本：
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**M5 的 Triton kernel 额外需要**：`triton` 包 + **C 编译器与 Python 头文件**
（Triton 首次启动 kernel 时会现场编译一个 CUDA 启动器）：

```bash
pip install triton
sudo apt-get install -y build-essential python3-dev
```

没装这些也能跑全量测试 —— `tests/test_triton_*.py` 会自动跳过（`importorskip` / 无 CUDA 时 `skipif`）。

集成测试需要 GPU 与已下载的 `gpt2` 权重（自动缓存于 `~/.cache/huggingface`）。
