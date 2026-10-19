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
| `kvcache` — `RadixCacheManager` | `engine/radix_cache.py` + `engine/radix_cache_paged.py` | ✅ M3.5b | 分页版：block 粒度建树、**节点分裂**、**每块 refcount**、**LRU 驱逐**。与官方差异：官方用节点 `lock_ref` + token 级 indices，我把计数放在 block 上（节点分裂后仍然精确，不会漏保护/泄漏）。`radix_cache.py` 保留为 M3 教学版（DynamicCache 克隆） |
| `kvcache` — `MHAKVCache`（分页池） | `engine/paged_kv.py` | ✅ M4a | 手写 PyTorch 版（gather 三步 + 多头点积）；官方在 CUDA kernel 内完成 gather |
| `attention` — 后端抽象（fa / fi / trtllm） | `engine/paged_kv.py:paged_attention` | ⚠️ M4a | **无后端抽象层**；无 FlashAttention / FlashInfer 集成，但自研 kernel 已就位（M5） |
| `kernel` — 自研 CUDA（tvm-ffi + JIT） | `engine/triton_paged.py` | ✅ M5 | 用 **Triton** 而非 tvm-ffi+CUDA；实现了分页版 flash attention（查页表 + 因果掩码 + online softmax），**无 split-K** |
| `benchmark` | `benchmark/bench.py`、`verify_m2/m3/m4b/m5/m6/m7.py`、`bench_serving.py`、`results/*.json` | ✅ | 官方测吞吐/延迟；我额外做**逐 token 一致性**与 **`tokens_fed` 精确账本**；M0–M7 与 M10 serving 快照入库 |

### 单机系统层（缺口最大）

| 官方模块 | 我的对应 | 状态 | 差异 |
|---|---|---|---|
| `core` — `Req` / `Batch` / `Context` / `SamplingParams` | `engine/batching.py:Request` + `engine/sampling.py` | ⚠️ M9 | **采样参数已补**（temperature / top_k / top_p / seed / stop_token_ids / ignore_eos，逐请求 RNG）；`Request` 仍偏简化，**无全局 `Context` / `Batch` 对象**（函数式 + 并行列表） |
| `engine` — `Engine` 类（model + ctx + kv + attn + cudagraph） | `engine/llm.py:LLMEngine` | ⚠️ M10 | 有统一入口类（model + pool + radix + backend + graph）与 `LLMConfig`；仍无全局 `Context`（配置显式传递）；已有 CUDA Graph |
| `scheduler` — 每 TP rank 一个 `Scheduler` | `engine/batching.py` + `engine/paged_schedule.py` | ✅ M2–M6 | 有连续准入、分页 varlen（零填充）、Chunked Prefill 合流、纯 decode 步 CUDA Graph；**无抢占** |
| `llm` — `LLM` python 接口 | `engine/llm.py:LLMEngine` | ✅ M10 | `LLMEngine.chat()` / `generate()`；模型 / 池 / 调度 / 采样收进一个对象 |
| 显存管理（驱逐 / 抢占 / refcount / LRU） | `engine/radix_cache_paged.py` | ⚠️ M3.5b | 有 **LRU 驱逐 + block 级 refcount**（池子耗尽自动回收未被活跃请求共享的叶子）；仍**无抢占**（不会打断运行中的请求） |
| `tokenizer` — tokenize/detokenize worker | —（调用侧直接用 HF） | ❌ | 无独立 worker |
| `server` — FastAPI + `launch_server` | `engine/server.py` | ✅ M10 | OpenAI 兼容 `/v1/chat/completions` + `/health` + `/v1/models`；非流式、单进程锁串行 |
| `utils` — logger / zmq 包装 | — | ❌ | 无 |

### 并行与工程层（TP / 多进程不做）

| 官方模块 | 我的对应 | 状态 | 说明 |
|---|---|---|---|
| `distributed` — all-reduce / all-gather / `DistributedInfo` | — | ❌ | 单卡。**理解原理即可**，不实作 |
| `layers` — TP-aware Linear / RMSNorm / RoPE | `engine/qwen3_forward.py`（非 TP） | ⚠️ M8 | 非 TP 的 RoPE / RMSNorm / SwiGLU / QK-Norm 已实现；TP-aware 切分不做 |
| `models` — Llama / Qwen3 + 权重分片 | `engine/qwen3_forward.py`（无分片） | ⚠️ M8 | Qwen3 单卡前向已实现；Llama 与权重分片不做 |
| `message` — ZMQ 消息（自动序列化） | — | ❌ | 单进程，函数直接返回 |
| Overlap Scheduling | — | ❌ | 需多 CUDA stream + 异步调度。**理解动机即可** |

---

## 三、覆盖度小结

| 层次 | 覆盖度 | 判断依据 |
|---|---|---|
| **算法 / 数据结构**（算得对） | **~85%** | M0–M9 覆盖 KV 管理、分页、Radix（block 粒度分裂/驱逐）、kernel、采样；缺 token 级 Radix 细节与 split-K |
| **单机系统**（服务得起来） | **~65%** | 有 batching + 连续准入 + 分页调度 + Chunked Prefill + CUDA Graph + 驱逐 + LLMEngine + HTTP 服务；缺全局 Context、抢占、tokenizer worker |
| **并行 / 工程**（多卡、多进程） | **0%** | 单卡单进程，无 TP / ZMQ / Overlap |

> 覆盖度是“模块 + 测试 + 实测”三项的主观估计，不是代码行数比例：有对应实现且通过测试记主要分；有实测结论再加分；仅有概念理解记 0。

**三条缺口规律**：

1. 覆盖了「**算得对**」，补了最小 HTTP 服务（M10）；仍缺 tokenizer worker 与真正的并发接入（当前单进程锁串行）。
2. 覆盖了「**单次计算的效率**」，补了 CUDA Graph / Chunked Prefill / 驱逐；仍缺 Overlap Scheduling 与抢占。
3. 走了「算法 → 访存 → kernel」这条线，没碰「分布式 → 服务化」那条线。

> **根本原因**：当 kernel 足够快时，瓶颈会从「算得慢」转移到「CPU 调度」与「显存管理」。本项目在 M4b 之前停留在 PyTorch 算子层，计算本身是瓶颈，所以系统级优化看不到收益 —— 这也解释了 bench 中反复出现的"地板效应"。
>
> **M4b/M5/M6 把这个判断逐条验证了**：分页 + varlen 把喂入 token 降了 6.6 倍，墙钟却慢了 2 倍
> （瓶颈真的转到了 CPU 侧）；换成 Triton kernel 后才追平（565 → 301 ms）。
> M7 已经把 CPU 调度这一段兑现（B=8 时**合计 5.3x**，含 KV 写入向量化；纯 Graph 约 2.1–2.4x）；现在下一个量化出的瓶颈是 **kernel 并行度（M5.5 split-K）**。

---

## 四、补充计划

### P0 — 必补（与自研 kernel 的差异化路线直接相关）

| 里程碑 | 内容 | 为什么 |
|---|---|---|
| **M5** ✅ | Triton kernel：把 M4a 的 `gather_layer` + `paged_attention` 翻译成 kernel | 已完成（`engine/triton_paged.py`）。端到端 1.88x，输出与 PyTorch 后端逐 token 一致 |
| **M5.5** | split-K / flash-decoding | 我的 kernel 单序列只有 H 个 program，SM 大量空转 —— 这是 M5 量化出的下一个瓶颈 |
| **M6** ✅ | Chunked Prefill（长 prompt 切块 + prefill/decode 合流） | 已完成。总前向 37 → 30，补入独立前向 7 → 0 |
| **M7** ✅ | CUDA Graph（捕获 / 重放 decode 步） | 已完成（`engine/graph_decode.py`）。只覆盖纯 decode 步；批大小按 {1,2,4,8} 预捕获，不足用 scratch 页补齐。实测 decode B=8：A 逐行 hook → C 图路径**合计** 22.4 → 4.2 ms（**5.3x**，含 KV 写入向量化；纯 Graph C/B 2.1–2.4x），B=1：10.1 → 3.3 ms |
| **M7.5** ✅ | 图路径接入调度器 | 已完成：纯 decode 步走图，含 prefill 的步走 eager；`stats.graph_steps` 记账。M9 起非贪心采样请求自动禁用图 |

### P1 — 该补（让项目"完整、可演示"）

| 里程碑 | 内容 | 为什么 |
|---|---|---|
| **M2.5** ✅ | 调度器 + 槽位连续准入 | 已完成。**发现**：没有 PagedAttention 时，补入一条新请求需要一次独立前向 + 左填充碎片，把理想收益从 2.00x 压到 1.25x —— 这是 M4a 的直接动机 |
| **M3.5b** ✅ | Radix 节点分裂 + refcount + 驱逐（分页版） | 已完成。只共享**完整 block**（共享块从此只读，无 copy-on-write）；节点分裂 / 驱逐都不拷贝 KV；池子耗尽时按 LRU 驱逐未被活跃请求引用的叶子 |
| **M4b** ✅ | 分页调度器（页表 + varlen + 零填充） | 已完成（含连续准入与 M6 合流）。驱逐已由 M3.5b 补上；**仍缺**：抢占（不会打断运行中的请求） |
| **M8** ✅ | Qwen3 架构支持（GQA / RoPE / RMSNorm / SwiGLU + QK-Norm） | 已完成（`engine/qwen3_forward.py`）。与 HF 逐元素一致 max|diff| ~2e-5；接入完整引擎后与朴素解码逐 token 一致 |
| **M9** ✅ | 采样策略（temperature / top_p / top_k + seed / stop） | 已完成（`engine/sampling.py`）。默认贪心零回归；top_k/top_p 截断顺序与 vLLM 一致；逐请求 RNG 可复现；分块 prefill 不在中间块白采样 |
| **M10** ✅ | 最小 HTTP 服务（`/v1/chat/completions`） | 已完成（`engine/server.py` + `engine/llm.py`）。Pydantic 校验、采样参数 / stop 透传、usage 统计；非流式、单进程锁串行 |

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
M3.5b Radix 节点分裂 + 驱逐（分页版）       ✅ block 粒度 + refcount + LRU
M4a  分页 KV 池 + PagedAttention           ✅
M4b  分页调度器（页表 + varlen + 零填充）   ✅
M5   Triton kernel（分页 flash attention）✅ 端到端 1.88x
M5.5 split-K / flash-decoding              ⬜ P0 ★
M6   Chunked Prefill（prefill/decode 合流）✅
M7   CUDA Graph                            ✅ 纯 decode 步；B=8 合计 5.3x（纯图 2.1–2.4x）
M7.5 图路径接入调度器                       ✅ 纯 decode 步走图
M8   Qwen3 架构支持（GQA/RoPE/RMSNorm）     ✅ 与 HF max|diff| ~2e-5
M9   采样策略（temperature / top_k / top_p）  ✅ + seed / stop / ignore_eos
M10  最小 HTTP 服务                         ✅ OpenAI 兼容 /v1/chat/completions
```

> 注：本文与 README 使用同一套里程碑编号（Qwen3 = M8，采样 = M9，HTTP = M10）。

---

## 六、如何维护本文

- 每完成一个里程碑 → 把对应行的状态从 ⬜ 改成 ✅，并在「差异」列补一句实测结论。
- 每次读官方某个模块的源码 → 在「差异」列补一行"官方怎么做的、我为什么不同"。
- **不要照抄官方实现**。本项目的价值在"能讲清"与"诚实分析"，照抄会同时失去两者。

**参考入口**：`docs/features.md`（功能清单）、`docs/structures.md`（系统架构 + 模块划分 + 数据流）。
