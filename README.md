# mini-sglang

**从零实现的轻量级 LLM 推理引擎** —— 参考 SGLang 架构，以 TDD 方式亲手构建推理系统的五层核心路径：解码循环 → KV Cache → Continuous Batching → Radix 前缀缓存 → 分页 KV / PagedAttention。

不是为了调用推理框架，而是为了回答一个问题：**vLLM/SGLang 到底在优化什么，为什么，以及优化在什么场景下不划算。**

> ⚠️ **与官方同名项目的区分**：本仓库是**教学向**实现，与 SGLang 官方的
> [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)（生产级精简框架，~5000 行 + CUDA kernel，H200 级 benchmark）**无代码或血缘关系**。
> 二者的交集只有 Radix Cache 一个概念。模块级对照与缺口分析见 [`docs/official-vs-mine.md`](docs/official-vs-mine.md)。

- 硬件：RTX 4070 Laptop (8GB) ｜ 模型：GPT-2 124M / Qwen2.5-0.5B
- 全程测试驱动：47 项测试**全部通过**，每条优化路径都与朴素实现做逐 token 一致性验证

## 快速入门

```bash
git clone https://github.com/tangqwert/mini-sglang && cd mini-sglang
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers pytest

# 1. 全部测试（单元 + 真模型集成）
pytest tests/

# 2. 看两条引擎的对比：同一长 prompt，朴素 vs KV Cache
python -m benchmark.bench --engine naive --model gpt2 --max-new-tokens 128
python -m benchmark.bench --engine kv    --model gpt2 --max-new-tokens 128

# 3. 端到端验证脚本：批处理 / 前缀缓存 的正确性与收益
python -m benchmark.verify_m2
python -m benchmark.verify_m3
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
┌─────────────────────────────────────────────┐
│ benchmark/  bench.py(--engine naive|kv)     │  测量与验证
│             verify_m2.py / verify_m3.py     │
├─────────────────────────────────────────────┤
│ engine/paged_kv.py      分页 KV 池 + 注意力 │  按页表 gather（M4a）
│ engine/radix_cache.py   Radix 树前缀缓存     │  跨请求复用 KV
│ engine/batching.py      动态退出批调度       │  多请求共享前向
│ engine/kv_cache.py      增量解码(prefill+1)  │  免重复计算
│ engine/naive_decode.py  朴素解码循环         │  基线（对照组）
├─────────────────────────────────────────────┤
│ transformers  (仅提供模型前向，禁用 generate) │
└─────────────────────────────────────────────┘
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

> 注：gpt2 与 Qwen 两次长 prompt 实验的 prompt 长度不同（696 / 601 token，出处见 `benchmark/results/*.json`）。
> 由于两者的权重规模差 4x、地板效应本就主导，长度差异不影响结论方向；若要严格 apples-to-apples，需在相同长度下重跑。

## 核心洞察：四个"理论收益 ≠ 实测收益"的对照实验

每个里程碑都做了理论推导与实测的对照，四次实验共同指向同一条成本模型：

```
每步耗时 = 权重搬运（固定，∝模型大小） + 序列计算（∝上下文长度）
```

- **KV Cache** 省的是"序列计算"——模型越小、prompt 越长，收益越大（4.5x ↔ 1.95x 的差异由此而来）
- **Batching** 省的是"N 条请求重复的权重搬运"——并发数上不去时收益有限（1.18x）
- **连续准入** 省的是"已完成请求占着槽位空转"——但没有 PagedAttention 时，补入一条新请求需要
  一次**独立的前向**，把理想收益从 2.00x 压到 1.25x
- **Radix Cache** 省的是"跨请求重复的 prefill 计算"——小模型 prefill 本身 ≈5ms，被 clone 与 Python 开销反超（0.82x）；真实收益场景是大模型 × 长前缀 × 高命中率（SGLang 用 PagedAttention 的零拷贝页引用消除 clone 开销）

**推论**：推理优化的收益 = 被省成分的成本 − 新增机制的开销。选型前先算清被省的部分在成本结构中占多少——这也是每个推理引擎的性能调优起点。

**两次出现"负收益 / 收益被吃光"（0.82x、1.25x vs 理想 2.00x），共同指向同一个东西：没有分页的 KV 管理开销。** 这就是 M4a 的动机。

## 与真 SGLang 的差距（诚实清单）

| 能力 | mini-sglang | 真 SGLang |
|---|---|---|
| Radix 树节点分裂 | 部分重叠直接放弃插入（宁缺毋错） | 分裂节点，重叠段共享 |
| 槽位补位（refill） | 有，但补入需一次独立前向 + 左填充碎片 | Continuous admission（与 decode 合并进同一 kernel，零填充） |
| KV 显存管理 | 有分页池 + gather（M4a），但调度器仍用整块 `DynamicCache`，clone 即拷贝 | PagedAttention 分页，零拷贝引用贯穿调度 |
| 服务层 | 无（库形态） | HTTP Server + tokenizer manager |
| Kernel | PyTorch 算子 | FlashInfer / 自研 CUDA |

> 上表是精简版。**模块级对照（官方 15 个模块）、三层覆盖度评估、缺口清单与补充计划**见 [`docs/official-vs-mine.md`](docs/official-vs-mine.md)。

## 开发方式

TDD / 规格先行：`tests/` 定义行为契约（含一个**上下文依赖的假模型**——它让"丢缓存"类 bug 无法蒙混过关），`engine/` 中的实现逐里程碑完成；每个里程碑在 `benchmark/results/` 留档数据。

测试金字塔：37 项单元测试（毫秒级，假模型精确断言内部行为）+ 2 项真模型集成测试（GPT-2 前向，无 GPU 自动跳过）+ 8 项分页 KV / PagedAttention 测试。

## 路线图

- [x] **M0** Naive 解码循环 + baseline（179 tokens/s @ gpt2）
- [x] **M1** KV Cache 增量解码（4.5x @ 696-token prompt）
- [x] **M2** Continuous Batching（动态退出批调度）
- [x] **M2.5** 调度器 + 槽位连续准入（冻结行腾位、pending 立刻补入；实测 1.25x vs 理想 2.00x）
- [x] **M3** Radix 前缀缓存复用
- [ ] **M3.5** Radix 节点分裂（部分重叠序列的完整缓存）
- [x] **M4a** 分页 KV 池 + PagedAttention（按页表 gather + 多头缩放点积，与连续存储逐元素一致 < 1e-5）
- [ ] **M4b** 分页显存管理（页表搬运 + 驱逐 / 抢占）
- [ ] **M5** 自研 Triton kernel（把 M4a 的 gather + 注意力翻译成 kernel）★ 差异化重点
- [ ] **M6** Chunked Prefill（长 prompt 切块前向，压显存峰值）
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

集成测试需要 GPU 与已下载的 `gpt2` 权重（自动缓存于 `~/.cache/huggingface`）。
