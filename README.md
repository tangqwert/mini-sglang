# mini-sglang

**从零实现的轻量级 LLM 推理引擎** —— 参考 SGLang 架构，以 TDD 方式亲手构建推理系统的完整链路：

```
朴素解码 → KV Cache → Continuous Batching → Radix 前缀缓存 → 分页 KV / PagedAttention
        → 零填充 varlen 调度 → Chunked Prefill → 自研 GPT-2 前向 → 自研 Triton 分页注意力 kernel
        → CUDA Graph 整步捕获 → Qwen3 架构支持（GQA / RoPE / RMSNorm / SwiGLU）
        → 采样（temperature / top_k / top_p）→ OpenAI 兼容 HTTP 服务
```

不是为了调用推理框架，而是为了回答一个问题：**vLLM/SGLang 到底在优化什么，为什么，以及优化在什么场景下不划算。**

- 硬件：RTX 4070 Laptop (8GB) ｜ 模型：GPT-2 124M / **Qwen3-0.6B（GQA）** / Qwen2.5-0.5B ｜ Python / PyTorch / Triton
- **193 项测试全部通过**，每条优化路径都与朴素实现做**逐 token 一致性**验证
- **8 组优化对照实验**（含 3 个诚实的负结果/反常现象）+ **M10 serving 并发压测** + 自研 Triton kernel + 自研 CUDA Graph 解码 + **架构无关前向**

> ⚠️ **与官方同名项目的区分**：本仓库是**教学向**实现，与 SGLang 官方的
> [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)（生产级精简框架，~5000 行 + CUDA kernel，H200 级 benchmark）**无代码或血缘关系**。
> 二者的交集只有 Radix Cache 一个概念。模块级对照与缺口分析见 [`docs/official-vs-mine.md`](docs/official-vs-mine.md)。

## 快速入门

```bash
git clone https://github.com/tangqwert/mini-sglang && cd mini-sglang
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers pytest fastapi uvicorn httpx

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
python -m benchmark.verify_m7    # eager decode 步 vs CUDA Graph（三路对照拆变量）

python -m benchmark.verify_m7    # eager decode 步 vs CUDA Graph（三路对照拆变量）

# 4. 启动 OpenAI 兼容 HTTP 服务（非流式）
python -m engine.server --model gpt2 --port 8000
# 另开终端：
python -m benchmark.bench_serving --num-prompts 32 --concurrency 1,2,4 --max-tokens 16
# 或单条请求：
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
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
│             verify_m2/m3/m4b/m5/m6/m7.py     │
│             bench_serving.py (M10 HTTP)      │  并发延迟/吞吐
├──────────────────────────────────────────────┤
│ engine/paged_schedule.py  分页/合流调度      │  页表 + varlen（M4b/M6）
│ engine/llm.py             LLMEngine 入口    │  模型 + 池 + 调度（M10）
│ engine/server.py          OpenAI 兼容服务    │  FastAPI /v1/chat/completions（M10）
│ engine/graph_decode.py     CUDA Graph 解码   │  整步折成 1 次 launch（M7）
│ engine/qwen3_forward.py    Qwen3 架构前向    │  GQA/RoPE/RMSNorm（M8）
│ engine/model_forward.py   自研 GPT-2 前向    │  接管 attention（M4b）
│ engine/paged_kv.py       分页 KV 池 + 注意力 │  按页表 gather（M4a）
│ engine/triton_paged.py    分页注意力 kernel  │  Triton（M5）
│ engine/radix_cache.py     Radix 树前缀缓存   │  跨请求复用 KV（M3 教学版）
│ engine/radix_cache_paged.py 分页 Radix 缓存 │  block 粒度 + refcount（M3.5b）
│ engine/batching.py        动态退出批调度     │  多请求共享前向
│ engine/sampling.py        采样 + 停止条件    │  temperature/top_k/top_p（M9）
│ engine/kv_cache.py       增量解码(prefill+1) │  免重复计算
│ engine/naive_decode.py    朴素解码循环       │  基线（对照组）
├──────────────────────────────────────────────┤
│ transformers （只取权重；M4b 起前向自研）    │
└──────────────────────────────────────────────┘
```

核心设计：解码循环与模型**解耦**——循环只认识 `logits_fn` / `kv_forward` 这两个可调用对象，
因此 M1/M2/M3 三次优化都没有改动循环本体，只替换了注入的前向实现。
M10 再把模型 / 分页池 / 调度 / 采样收进 `LLMEngine`，HTTP 层只依赖 `chat()` / `generate()`。

## 基准结果（RTX 4070 Laptop，fp32）

| 优化 | 场景 | 结果 | 结论 |
|---|---|---|---|
| **KV Cache** | gpt2，696-token prompt | **TPOT 27.5 → 5.9 ms（4.6x，不含 prefill）**；TTFT 23.7 → 19.4 ms；整段生成 3.52 → 0.78 s（4.5x） | 消除序列长度维度的重复计算 |
| KV Cache | Qwen2.5-0.5B，601-token prompt | 整段生成 28.9 → 14.8 ms/token（**1.95x**） | 收益被权重搬运地板压缩 |
| KV Cache | **Qwen3-0.6B**，704-token prompt | 整段生成 64.7 → 31.1 ms/token（**2.08x**） | 现代模型（GQA + QK-Norm + SwiGLU）同结论 |
| KV Cache | gpt2，5-token prompt | 整段生成 5.8 → 6.1 ms/token（0.94x，略负） | 短上下文无浪费可省，机制开销反超 |
| **Batching** | 4 请求批解码 vs 逐条（各 24 token） | 553 → 178 ms（**3.0x**，理想 4.0x） | 权重搬运被 4 条分摊；缺口 = 填充行 + mask 拼接 + Python 记账 |
| **连续准入** | 4 请求（预算 1/5/1/5）｜槽位 2 | 静态分批 10 次前向 → **8 次**（6 步批量 + 2 次补入） | 理想 2.00x，实测 **1.25x** —— 差距即"补入需独立前向"的代价 |
| **Radix 前缀缓存** | 3 请求共享 118-token 前缀 | **0.82x**（输出逐 token 一致） | 小模型 prefill 被权重搬运主导 |
| **分页调度（Step 4）** | 1 长请求(153 tok) + 8 短请求｜槽位 2 | 喂入 token **1463 → 222（6.6x）**；填充位置 1220 → **0**；前向次数 37 → 37（不变） | 分页只消掉了"填充计算" |
| **Chunked Prefill（M6）** | 同上 | 总前向 37 → **30**；补入独立前向 7 → **0**；prefill/decode 同批 **7 步** | prefill 与 decode 合流，额外前向消失 |
| **Triton kernel（M5）** | M6 负载，端到端 | PyTorch 分页 **565 → 301 ms（1.88x）**，输出与调度统计完全一致 | 把每层十来个 kernel + 一圈 Python 循环折成 **1 个** kernel |
| Triton kernel 微基准 | 单层 decode，seq_len 128–2048 | 1.0–2.0x，且**不随 seq_len 单调** | PyTorch 版是**启动受限**（耗时几乎不随长度变），Triton 版是**带宽受限** |
| **CUDA Graph（M7）** | decode 步（每行 1 token）| A 逐行 hook → C 图路径（含 KV 写入向量化）：B=8 时 **22.4 → 4.2 ms（合计 5.3x）**；纯 Graph C/B **2.1–2.4x**；B=1 时 10.1 → 3.3 ms | 图消掉的 launch 次数**与 B 无关**，所以 C 列几乎不随 B 变 |
| **HTTP 服务（M10）** | 32 请求 × 16 token，并发 1/2/4 | 吞吐稳定 **~8.2 req/s（~131 out tok/s）**；p50 **123 → 243 → 487 ms** | 单进程锁串行：并发只增加排队，不增加吞吐 |
| ⚠️ 同一负载的墙钟 | 同上 | M2.5 306ms → Step4 670ms → M6 624ms → **M6+Triton 301ms** | 分页路径先慢 2 倍（Python 循环），M5 追平 |

> 注：三次长 prompt 实验的 prompt 长度不同（gpt2 696 / Qwen2.5 601 / Qwen3 704 token，出处见 `benchmark/results/*.json`）。
> 各 tokenizer 切分粒度不同，无法凑完全一致的 token 数。**跨模型只比加速比，不比绝对耗时。**
>
> **口径说明**：KV Cache 的 **TPOT** 是 decode 段平均每 token 耗时（不含 prefill）；
> **TTFT** 是首 token 时间；`整段生成 ms/token` 是 prefill + decode 的平均，
> **不是 decode TPOT**。gpt2 696-token 的 TPOT 加速 4.6x，与整段生成 4.5x 接近，
> 因为长 prompt 的 prefill 被 128 个 decode token 摊薄。
> M7 的 `5.3x` 是 A（逐行 hook）→ C（CUDA Graph）的**合计**加速，含 KV 写入向量化；
> 纯 Graph（C/B）约 **2.1–2.4x**。2026-10-18 复跑快照 C/A≈**3.8x**，差异来自笔记本降频，
> 见 `benchmark/results/m7_cuda_graph.json`。
>
> **数字出处**：M0–M7 的测量快照都在 [`benchmark/results/`](benchmark/results/)（含目录说明），
> 每条也可用 `benchmark/verify_m*.py` 复跑。笔记本 GPU 降频会让绝对值有约 ±30% 的波动，
> **不要跨次比较绝对毫秒**；同一脚本同一次运行内的相对关系才是有效结论。

### 系统级矩阵（`benchmark/results/matrix_gpt2.json`）

同一 prompt / 同一输出长度，单请求：

| 模式 | TTFT | TPOT | 端到端 |
|---|---:|---:|---:|
| naive | 42.9 ms | 30.54 ms | 3922 ms |
| KV（M1） | 22.6 ms | 6.45 ms | 841 ms |
| paged-PyTorch | 815.4 ms | 11.34 ms | 2255 ms |
| paged-Triton | 27.5 ms | 8.84 ms | 1150 ms |
| paged-Triton-Graph | 28.0 ms | **4.22 ms** | **563 ms** |

- **naive → full graph**：TPOT **7.2x**、端到端 **7.0x** —— 这是**整引擎单请求加速**，
  不是 KV Cache 单独的功劳。
- **paged-PyTorch → paged-Triton**：端到端 **1.96x**，这是 kernel 的功劳。
- **paged-Triton → Graph**：端到端 **2.04x**；Graph 在计时前预捕获，不把捕获成本算进 TTFT。

多请求（1 长 + 8 短，预算 64/16，B=4，所有请求同时到达）：

| 模式 | Makespan | 输出吞吐 | p50 E2E | p99 E2E |
|---|---:|---:|---:|---:|
| 顺序 KV | 1317 ms | 146 tok/s | 916.3 ms | 1317 ms |
| 静态 batching | 575 ms | 334 tok/s | 176.2 ms | 574 ms |
| full（无图） | 748 ms | 257 tok/s | 426.5 ms | 747 ms |
| **full-graph** | **309 ms** | **622 tok/s** | **166.7 ms** | **308 ms** |

- 静态 batching 是强基线（HF 融合批式注意力）；
- full 无图反而比静态 batch 慢，说明 **Triton/分页在单序列小模型上不占优**；
- full-graph 把 decode launch 开销消掉后，多请求比顺序 KV **4.3x**、比静态 batching **1.9x**。
- 该 workload 是所有请求同时到达；真实错峰到达时，静态 batching 还需要额外等待策略，full 的优势会更明显。

Qwen3-0.6B bf16（706-token / 128 new，256 blocks）：

- 单请求：naive TPOT 95.0 → full-graph **17.5 ms（5.4x）**，端到端 12.12 → 2.45 s（5.0x）；
  KV 单独为 36.0 ms，full-graph 相对 KV 再快 **2.1x**。
- 多请求（同 1 长 + 8 短 / B=4）：顺序 KV 6444 → full-graph **1126 ms（5.7x）**、30 → 171 tok/s；
  静态 batching 2588 ms，full-graph 为其 **2.3x**。

## 核心洞察：八个"理论收益 ≠ 实测收益"的对照实验

每个里程碑都做了理论推导与实测的对照，八次实验共同指向同一条成本模型：

```
每步耗时 = 权重搬运（固定，∝模型大小） + 序列计算（∝上下文长度）
```

- **KV Cache** 省的是"序列计算"——收益 ∝ 被省成分在总耗时中的占比：
  gpt2(124M) 整段 **4.51x**（TPOT **4.6x**）→ Qwen2.5-0.5B **1.95x** → Qwen3-0.6B **2.08x**。
  ⚠️ Qwen3-0.6B 参数**更多**、收益却**更高**——它 head_dim 128 × 16 头，每 token 注意力开销
  是 Qwen2.5 的 ~2.7 倍，序列项占比反而更大。**成本模型看的是"占比"，不是"绝对参数量"。**
- **Batching** 省的是"N 条请求重复的权重搬运"——4 请求批实测 **3.0x**（理想 4.0x），缺口来自填充行的无效计算与 Python 记账
  > ⚠️ 这个数字最初被写成 **1.18x**：当时的 benchmark 让批量路径**跑第一个**，
  > 于是把 CUDA 上下文 / cuBLAS 的冷启动开销（首次 581ms vs 预热后 183ms）全算在它头上。
  > 补齐预热 + best-of-3 后是 3.0x。**发现并修正自己 benchmark 的系统性偏差，比多做一个优化更值钱。**
- **连续准入** 省的是"已完成请求占着槽位空转"——但没有 PagedAttention 时，补入一条新请求需要
  一次**独立的前向**，把理想收益从 2.00x 压到 1.25x
- **Radix Cache** 省的是"跨请求重复的 prefill 计算"——小模型 prefill 本身 ≈5ms，被 clone 与 Python 开销反超（0.82x）；真实收益场景是大模型 × 长前缀 × 高命中率（SGLang 用 PagedAttention 的零拷贝页引用消除 clone 开销）
- **分页调度（M4b Step 4）** 省的是"补入新请求时的填充计算"——左填充彻底消失，喂入 token 数降为 **1/6.6**
- **Chunked Prefill（M6）** 省的是"补入要多开一次前向"——prefill 与 decode 合进同一步
- **Triton kernel（M5）** 省的是"注意力实现里的 kernel 启动与 Python 开销"——把每层十来个 kernel 折成 1 个（端到端 565 → 301 ms）
- **CUDA Graph（M7）** 省的是"**整步**几十个 kernel 的 launch 开销"——它消掉的次数与 batch 无关，所以图路径耗时几乎不随 B 变（3.3 → 4.2 ms），而 eager 从 10.1 涨到 22.4。**注意：5.3x 是向量化写入 + 图合计，纯图约 2.1–2.4x。**

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
| Radix 树节点分裂 | M3 教学版直接放弃插入；**M3.5b 分页版已实现**（block 粒度分裂，共享块只读） | 分裂节点，重叠段共享 |
| 槽位补位（refill） | 有；补入已做到**零填充 + 与 decode 合流**（M6），`admission_forwards == 0` | Continuous admission，且 prefill 与 decode 在同一个 CUDA kernel 内完成 |
| KV 显存管理 | 分页池 + 页表 + 按需增长 + 完成即归还；**M3.5b** 加了 block refcount + LRU 驱逐（池子耗尽自动回收未共享叶子） | 同上 + 抢占；gather 在 CUDA kernel 内完成 |
| 服务层 | **最小 HTTP 服务（M10）**：FastAPI `/v1/chat/completions`，非流式、单进程锁串行；无 tokenizer worker | HTTP Server + tokenizer manager + 流式输出 |
| Kernel | **自研 Triton kernel**（M5）：分页版 flash attention，每层 2 个 kernel 启动；但无 split-K，单序列并行度只有 H | FlashInfer / 自研 CUDA（含 split-K / decode 专用 kernel） |

> 上表是精简版。**模块级对照（官方 15 个模块）、三层覆盖度评估、缺口清单与补充计划**见 [`docs/official-vs-mine.md`](docs/official-vs-mine.md)。

## 开发方式

TDD / 规格先行：`tests/` 定义行为契约（含一个**上下文依赖的假模型**——它让"丢缓存"类 bug 无法蒙混过关），`engine/` 中的实现逐里程碑完成；M0–M7 的测量快照入库在 `benchmark/results/`，可用 `verify_m*.py` 复跑。

测试金字塔：114 项假模型 / 纯张量 / HTTP 契约测试（毫秒级，精确断言内部行为）+ 79 项真模型 / GPU 集成测试（含「自研前向 vs HF」逐元素对比、「分页增量解码 vs 朴素解码」逐 token 对比、「Triton kernel vs PyTorch 参考」逐元素对比、「Qwen3 走完整引擎 vs 朴素解码」逐 token 对比）。

## 路线图

- [x] **M0** Naive 解码循环 + baseline（179 tokens/s @ gpt2）
- [x] **M1** KV Cache 增量解码（gpt2 696-token：TPOT 4.6x / 整段生成 4.5x）
- [x] **M2** Continuous Batching（动态退出批调度）
- [x] **M2.5** 调度器 + 槽位连续准入（冻结行腾位、pending 立刻补入；实测 1.25x vs 理想 2.00x）
- [x] **M3** Radix 前缀缓存复用
- [x] **M3.5b** 分页 Radix 缓存：block 粒度建树 + 节点分裂 + block refcount + LRU 驱逐（只共享完整 block，共享块只读；池子耗尽自动回收未共享叶子）
- [x] **M4a** 分页 KV 池 + PagedAttention（按页表 gather + 多头缩放点积，与连续存储逐元素一致 < 1e-5）
- [x] **M4b** 分页接入真实前向：自研 GPT-2 前向 ✅ / attention 换分页版 ✅ / 增量解码 ✅ / 接调度器 ✅（零填充 varlen 调度，喂入 token 降为 1/6.6）
- [x] **M5** 自研 Triton kernel（分页版 flash attention：查页表 + 因果掩码 + online softmax 封进 **1 个** kernel；端到端 **1.88x**）★
- [ ] **M5.5** split-K / flash-decoding（把 KV 维也切开并行，解决"单序列只有 H 个 program、SM 大量空转"）
- [x] **M6** Chunked Prefill（prefill 与 decode 合流；`max_prefill_tokens` 给单步成本设上限）
- [x] **M7** CUDA Graph（decode 步 B=8 时 A→C 合计 **22.4 → 4.2 ms（5.3x，含向量化）**；纯 Graph C/B **2.1–2.4x**）
- [x] **M7.5** 把图路径接进调度器（纯 decode 步走图，`stats.graph_steps` 记账；输出与调度统计不变）
- [x] **M8** **Qwen3 架构支持**（GQA + RoPE + RMSNorm + SwiGLU + QK-Norm；与 HF 逐元素一致 max|diff| ~2e-5，且接入完整引擎后与朴素解码逐 token 一致；CUDA Graph 路径通过 `forward_fn` 同样支持）
- [x] **M9** 采样策略（temperature / top_p / top_k + seed / stop_token_ids / ignore_eos；默认贪心零回归，逐请求 RNG 可复现）
- [x] **M10** 最小 HTTP 服务（FastAPI `/v1/chat/completions`；Pydantic 校验、采样参数透传、usage 统计；非流式、单进程锁串行）

> 与 SGLang 官方 [mini-sglang](https://github.com/sgl-project/mini-sglang) 的模块级对照、覆盖度与缺口分析见 [`docs/official-vs-mine.md`](docs/official-vs-mine.md)。
> **不计划实作**：Tensor Parallelism、Overlap Scheduling、ZMQ 多进程架构 —— 以读懂并讲清原理为目标。
> 注：本文与 `docs/official-vs-mine.md` 使用同一套里程碑编号（Qwen3 = M8，采样 = M9，HTTP = M10）。

## 文档

- [`docs/official-vs-mine.md`](docs/official-vs-mine.md) — 与官方项目的模块级对照、三层覆盖度评估、缺口清单

> 面试准备手册与简历数据档案为本地文件，不随公开仓库发布。

## 环境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers pytest fastapi uvicorn httpx
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

## 许可

MIT —— 见 [`LICENSE`](LICENSE)。本项目是**个人学习/作品集项目**，不追求生产可用性；
与 SGLang 官方及其任何衍生项目无关联。
