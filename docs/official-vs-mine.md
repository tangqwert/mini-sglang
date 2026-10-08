# 与官方 mini-sglang 的对照

本文对照 **[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)** —— SGLang 官方（LMSYS）出品的精简单机/多卡推理框架（约 5000 行 Python + C/CUDA，H200 级 benchmark，39 位贡献者）。

**定位区分（重要）**：本项目与它**没有代码或血缘关系**。它是「精简但仍要高性能」的**生产级引擎**；本项目是「每一步都要讲清动机与代价」的**教学向实现**。二者的共同点是都叫 `mini-sglang`，交集只有 Radix Cache 一个概念。

> 本文的用途：① 学习笔记（读完官方代码后逐行记录差异）；② 面试素材（证明不只写了代码，还横向读过工业实现）；③ 缺口清单（决定下一个里程碑做什么）。

---

## 一、官方能力一览（来自其 README / `docs/features.md`）

| 类别 | 能力 |
|---|---|
| 单机优化 | Radix Cache、Chunked Prefill、Overlap Scheduling、CUDA Graph |
| 并行 | Tensor Parallelism（`--tp n`）、多进程 + ZMQ + NCCL |
| Kernel | FlashAttention / FlashInfer / TensorRT-LLM fmha 后端可切；自研 CUDA kernel（tvm-ffi JIT） |
| 服务 | OpenAI 兼容 API（FastAPI `/v1/chat/completions`）、交互式 shell、Docker |
| 显存 | PagedAttention，`--page-size` 可配 |
| 模型 | Llama-3 / Qwen-3（含 MoE）/ Qwen-2.5 |

---

## 二、模块级对照表

> 状态图例：✅ 已覆盖 ｜ ⚠️ 部分覆盖 ｜ 🚧 进行中 ｜ ⬜ 已计划 ｜ ❌ 未覆盖

### 算法与访存层（本项目的主战场）

| 官方模块 | 我的对应 | 状态 | 差异 |
|---|---|---|---|
| `kvcache` — `NaiveCacheManager` | `engine/kv_cache.py` | ✅ | 我直接用 HF `past_key_values`，未抽管理器接口 |
| `kvcache` — `RadixCacheManager` | `engine/radix_cache.py` | ⚠️ | **无节点分裂、无驱逐、无 refcount**（mini 版放弃部分重叠插入） |
| `kvcache` — `MHAKVCache`（分页池） | `engine/paged_kv.py` | ✅ M4a | 手写 PyTorch 版（gather 三步 + 多头点积）；官方在 CUDA kernel 内完成 gather |
| `attention` — 后端抽象（fa / fi / trtllm） | `engine/paged_kv.py:paged_attention` | ⚠️ M4a | **无后端抽象层**；无 FlashAttention / FlashInfer 集成 |
| `kernel` — 自研 CUDA（tvm-ffi + JIT） | — | ⬜ M5 | 我计划用 **Triton**，不引入 tvm-ffi |
| `benchmark` | `benchmark/bench.py`、`verify_m2.py`、`verify_m3.py` | ✅ | 官方测吞吐/延迟；我额外做**逐 token 一致性**与 **`tokens_fed` 精确账本** |

### 单机系统层（缺口最大）

| 官方模块 | 我的对应 | 状态 | 差异 |
|---|---|---|---|
| `core` — `Req` / `Batch` / `Context` / `SamplingParams` | `engine/batching.py:Request` | ⚠️ | 只有简化 `Request`；**无全局 `Context`**；**无采样参数**（仅贪心） |
| `engine` — `Engine` 类（model + ctx + kv + attn + cudagraph） | `engine/*.py`（函数式） | ⚠️ | 无 `Engine` 对象、无语境管理；**无 CUDA Graph** |
| `scheduler` — 每 TP rank 一个 `Scheduler` | `engine/batching.py`（`batched_generate` + `continuous_generate`） | ✅ M2 / M2.5 | **有连续准入**（冻结行腾位、pending 补入）；但补入需一次独立前向 + 左填充碎片。**无 Chunked Prefill、无抢占** |
| `llm` — `LLM` python 接口 | `benchmark/bench.py`（脚本） | ⚠️ | 无统一入口类 |
| 显存管理（驱逐 / 抢占 / refcount / LRU） | — | ❌ | M4b 计划中 |
| `tokenizer` — tokenize/detokenize worker | —（调用侧直接用 HF） | ❌ | 无独立 worker |
| `server` — FastAPI + `launch_server` | — | ❌ | 无服务层，项目不可"演示" |
| `utils` — logger / zmq 包装 | — | ❌ | 无 |

### 并行与工程层（本项目不做）

| 官方模块 | 我的对应 | 状态 | 说明 |
|---|---|---|---|
| `distributed` — all-reduce / all-gather / `DistributedInfo` | — | ❌ | 单卡。**理解原理即可**，不实作 |
| `layers` — TP-aware Linear / RMSNorm / RoPE | —（借 HF） | ❌ | 模型层完全交给 `transformers` |
| `models` — Llama / Qwen3 + 权重分片 | —（借 HF） | ❌ | 同上 |
| `message` — ZMQ 消息（自动序列化） | — | ❌ | 单进程，函数直接返回 |
| Overlap Scheduling | — | ❌ | 需多 CUDA stream + 异步调度。**理解动机即可** |

---

## 三、覆盖度小结

| 层次 | 覆盖度 | 判断依据 |
|---|---|---|
| **算法 / 数据结构**（算得对） | **~70%** | M0–M4a 正好覆盖这条线，且有测试与实测分析 |
| **单机系统**（服务得起来） | **~30%** | 有 batching + 连续准入，缺显存管理、服务层 |
| **并行 / 工程**（多卡、多进程） | **0%** | 单卡单进程，无 TP / ZMQ / CUDA Graph |

**三条缺口规律**：

1. 覆盖了「**算得对**」，缺「**服务得起来**」——请求从哪来、结果往哪去、怎么并发接入。
2. 覆盖了「**单次计算的效率**」，缺「**系统级效率**」——CPU 侧开销（Overlap / CUDA Graph）与显存峰值（Chunked Prefill / 驱逐）。
3. 走了「算法 → 访存 → kernel」这条线，没碰「分布式 → 服务化」那条线。

> **根本原因**：当 kernel 足够快时，瓶颈会从「算得慢」转移到「CPU 调度」与「显存管理」。本项目仍停留在 PyTorch 算子层，计算本身是瓶颈，所以系统级优化还看不到收益 —— 这也解释了 bench 中反复出现的"地板效应"。

---

## 四、补充计划

### P0 — 必补（与自研 kernel 的差异化路线直接相关）

| 里程碑 | 内容 | 为什么 |
|---|---|---|
| **M5** | Triton kernel：把 M4a 的 `gather_layer` + `paged_attention` 翻译成 kernel | 本项目最具面试价值的一块：能讲清 PagedAttention 的访存模式，而非"我调了 FlashInfer" |
| **M6** | Chunked Prefill（长 prompt 切块 prefill） | 算法简单，但引出**显存峰值管理**这一系统级概念；与 M4b 天然衔接 |
| **M7** | CUDA Graph（捕获 / 重放 decode 步） | 代码量小，概念极重要：decode 阶段 kernel launch 开销可占大头 |

### P1 — 该补（让项目"完整、可演示"）

| 里程碑 | 内容 | 为什么 |
|---|---|---|
| **M2.5** ✅ | 调度器 + 槽位连续准入 | 已完成。**发现**：没有 PagedAttention 时，补入一条新请求需要一次独立前向 + 左填充碎片，把理想收益从 2.00x 压到 1.25x —— 这是 M4a 的直接动机 |
| **M4b** | 分页显存管理（页表搬运 + 驱逐 / 抢占） | M4a 只是"能分页"，M4b 才"会管理" |
| **M8** | 采样策略（temperature / top_p / top_k） | 几十行，但让"生成"完整；面试高频话题 |
| **M9** | 最小 HTTP 服务（`/v1/chat/completions`） | 让项目可演示（录屏 / 简历链接） |

### 理解即可，不实作

| 能力 | 建议做法 |
|---|---|
| Tensor Parallelism | 读官方 `minisgl.distributed` + `layers`，讲清 all-reduce 切在哪、权重如何分片。想动手可"模拟 TP"（按 head 维切两半，单卡跑两个半模型） |
| Overlap Scheduling | 理解「CPU 调度与 GPU 计算重叠」的动机（NanoFlow 论文）即可 |
| 多进程 + ZMQ | 理解 API Server / Tokenizer / Detokenizer / Scheduler 四类进程的分工即可 |
| 模型层自研（RoPE / RMSNorm / 权重分片） | 属另一条路线（"从零实现 Transformer"），**建议另起项目** |

---

## 五、路线图总览

```
M0   解码循环                              ✅
M1   KV Cache                              ✅
M2   Continuous Batching（动态退出）        ✅
M2.5 调度器 + 槽位连续准入                  ✅
M3   Radix 前缀缓存（mini 版）              ✅
M3.5 Radix 节点分裂 + 驱逐                  ⬜
M4a  分页 KV 池 + PagedAttention           ✅
M4b  分页调度器（页表 + 抢占）              ⬜ P1
M5   Triton kernel（gather + attention）   ⬜ P0 ★
M6   Chunked Prefill                       ⬜ P0
M7   CUDA Graph                            ⬜ P0
M8   采样策略（temperature / top_p）        ⬜ P1
M9   最小 HTTP 服务                         ⬜ P1
```

---

## 六、如何维护本文

- 每完成一个里程碑 → 把对应行的状态从 ⬜ 改成 ✅，并在「差异」列补一句实测结论。
- 每次读官方某个模块的源码 → 在「差异」列补一行"官方怎么做的、我为什么不同"。
- **不要照抄官方实现**。本项目的价值在"能讲清"与"诚实分析"，照抄会同时失去两者。

**参考入口**：`docs/features.md`（功能清单）、`docs/structures.md`（系统架构 + 模块划分 + 数据流）。
