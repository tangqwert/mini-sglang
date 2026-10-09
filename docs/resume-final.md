# 简历投递版（小红书风格）

> 格式对齐参考简历：`项目技术栈:` + `项目概述:` + 「技术名 + 优化/重构」式 bullet。
> **所有数字均出自本仓库 `benchmark/results/*.json` 与 `sgemm-cuda/bench.csv`，无编造。**
>
> **篇幅：正文 36 行**（markdown 里的空行是为了可读性，粘到 Word 时删掉），
> 按 10.5pt / 2cm 边距 / 段间距 6pt 折算 ≈ **43 行**，单页可容纳（约 48 行）。
> 若 Word 里仍溢出，按此顺序砍：① 整块删 MIPS（-5 行）② mini-sglang 概述缩到 1 行
> （-1）③ SGEMM 第一条 bullet 删掉 ncu 细节（-1）。

---

## 求职方向

AI Infra / 推理优化 / 算子开发（可立即到岗，实习期 6 个月以上：2026.09 – 2027.07）

## 教育背景

**2022.09 – 2026.07　西交利物浦大学 – 通信工程　本科**　|　英语能力：雅思 6.5

---

## 项目经历

### 基于 Mini-SGLang 的轻量级推理引擎优化

**项目技术栈**：Python、PyTorch、Triton、GPT-2 / Qwen2.5-0.5B / Qwen3-0.6B、KV Cache、Continuous Batching、PagedAttention、Radix 前缀缓存、Chunked Prefill、CUDA Graph、GQA

**项目概述**：基于 PyTorch 从零实现轻量级 LLM 推理引擎，围绕显存管理、请求调度与算子实现三个维度优化；解码循环与模型解耦（仅依赖注入的 logits_fn / kv_forward），八次优化均未改动循环本体；每条优化与朴素实现逐 token 全等，150 项测试全绿。

- **显存管理（KV Cache + PagedAttention）**：基于注意力位置不变性手写 prefill + 单 token 增量前向，gpt2 696-token 下 TPOT 29.2 → 6.5 ms（**4.51×**），并在 Qwen2.5-0.5B / Qwen3-0.6B 复现（1.95× / 2.08×），归纳出「每步耗时 = 权重搬运 + 序列计算」成本模型；进一步实现分页 KV 池与页表寻址，varlen 零填充调度使喂入 token 1463 → 222（**6.6×**）、填充位置 1220 → **0**。

- **请求调度（Continuous Batching + Chunked Prefill）**：实现左填充 + attention_mask + 显式 position_ids 的批解码与 per-request 动态退出（4 请求 **3.0×**，理想 4.0×），并实现固定槽位调度器与 Decode 严格优先、Token Budget 驱动的 Chunked Prefill；在 1 长 + 8 短请求 / 2 槽位下，总前向 37 → **30 次**、补入独立前向 7 → **0 次**、单步峰值 token 157 → **16**。

- **算子与执行（Triton Kernel + CUDA Graph）**：手写页表访存 + Online Softmax + GQA 的分页 attention kernel，把每层数十个 PyTorch 小 kernel 折成**单 kernel**，端到端 565 → 301 ms（**1.88×**）；将 decode step 整步捕获为 CUDA Graph（padding 行路由至 scratch block），decode 加速 **3.1× ~ 5.3×**（B=1 → 8）。

### CUDA SGEMM 算子优化与性能分析

**项目技术栈**：CUDA C++、cuBLAS、Nsight Compute、Shared Memory、Register Tiling

**项目概述**：基于 CUDA C++ 在 RTX 4070 Laptop 上实现严格 FP32 SGEMM；M=N=K=4096 下性能由 1.11 TFLOPS 提升至 8.95 TFLOPS（**8.0×**），达同精度 cuBLAS 的 **73.4%**。

- **基准框架与基线归因**：自建 harness（CPU 参考验语义 + cuBLAS 对照 + ncu 采证）；用 ncu 证明 naive 版瓶颈在 L1TEX 访存管线（l1tex 96.5% / DRAM 3.4%），访存模式本身已最优（A 广播 1 sector、B 连续 4 sectors/128B），瓶颈在「访存条数」而非「模式」。

- **二维寄存器分块（v3）**：设计 BM×BN×BK=128×128×8 Block Tile 与 TM×TN=8×8 Thread Tile，每线程 64 个累加结果驻留寄存器，用外积计算提高 smem 复用率与指令级并行度，较 Naive 加速 **8.0×**。

### 基于 Verilog 的 32 位 MIPS 单周期处理器 RTL 设计

**项目技术栈**：Verilog HDL、波形仿真、MIPS 指令集

- 实现 Control Unit、ALU、Register File、Memory Interface 及 Datapath，扩展 BEQ / Jump / ANDI 指令路径；将各级总线位宽参数化，使顶层 MIPS1CYCLE 切换至 28 位后同步适配并波形验证。

---

**可立即到岗，实习期 6 个月以上**：坚持「预热 + best-of-3 + 逐轮重置」的测量纪律，两次发现并修正自己 benchmark 的系统性偏差（冷启动污染、图捕获计入计时）。

---

## 技术能力

- **编程语言**：Python、C++、Triton 算子开发、CUDA C/C++、Verilog
- **深度学习框架**：PyTorch；手写 GPT-2 / Qwen2.5 / Qwen3 前向并与 HuggingFace 对齐 <5e-5
- **推理优化核心技术**：KV Cache 管理与压缩、Continuous Batching、PagedAttention、Radix 前缀缓存、Chunked Prefill、CUDA Graph、算子融合、Online Softmax、SGEMM 与访存优化
- **性能分析**：Nsight Compute / Systems（DRAM 吞吐、sectors-per-request、bank conflict）　|　**工程工具**：Git、Linux/WSL、Docker（了解）

---

## 格式对照（为什么这样写）

| 参考简历的做法 | 本文件的对应 |
|---|---|
| `项目技术栈:` 一行顿号长列表 | 同 |
| `项目概述:` 一句话含**最终数字** | 同（mini-sglang 概述给解耦设计 + 测试数；SGEMM 概述给 8.0× / 73.4%） |
| 一个优化点 = 一条 bullet | mini-sglang 6 条（M1/M2/M4b/M6/M5/M7），SGEMM 2 条 |
| bullet 命名 = 「技术名 + 优化/重构」 | 同 |
| bullet 内：**背景问题 → 手段 → 数字** | 同 |
| 数字带「相对上一版 X×」 | 「较 Naive Kernel 加速 8.0×」「总前向由 37 次降至 30 次」 |
| 术语中英混排 | Kernel / Pass / Token Budget / Online Softmax / Context 重放 |
| **单页控制** | 参考简历：nano-vLLM 3 条 + SGEMM 4 条 = 7 条 bullet | 本文件：mini-sglang 3 条 + SGEMM 2 条 + MIPS 1 条 = **6 条** |

> ⚠️ **与参考简历的差异是刻意的**：
> ① 参考简历的 `58.24×` 来自极低的 naive 基线（113.55 GFLOPS），本仓库 naive 为 1112.3 GFLOPS
> （快 9.8×），故总加速比只有 8.0× 但**最终性能 8.95 TFLOPS 高于其 6.61 TFLOPS**；
> ② 参考简历未含「负结果」，本文件主动保留（0.94× / 1.25× / 0.82×）。
> **不要为了数字好看去改基线或删负结果** —— 面试官一追问就穿帮。
