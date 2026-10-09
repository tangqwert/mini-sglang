# 简历投递版（小红书风格）

> 格式对齐参考简历：`项目技术栈:` + `项目概述:` + 「技术名 + 优化/重构」式 bullet。
> **所有数字均出自本仓库 `benchmark/results/*.json` 与 `sgemm-cuda/bench.csv`，无编造。**

---

## 求职方向

AI Infra / 推理优化 / 算子开发（可立即到岗，实习期 6 个月以上：2026.09 – 2027.07）

## 教育背景

**2022.09 – 2026.07　西交利物浦大学 – 通信工程　本科**　|　英语能力：雅思 6.5

---

## 项目经历

### 基于 Mini-SGLang 的轻量级推理引擎优化

**项目技术栈**：Python、PyTorch、Triton、GPT-2、Qwen2.5-0.5B、Qwen3-0.6B、KV Cache、Continuous Batching、PagedAttention、Radix 前缀缓存、Chunked Prefill、CUDA Graph、GQA、Online Softmax、RoPE/RMSNorm/SwiGLU/QK-Norm、Nsight

**项目概述**：基于 PyTorch 从零实现轻量级 LLM 推理引擎，围绕显存管理、请求调度与算子实现三个维度逐层优化；解码循环与模型完全解耦（仅依赖注入的 `logits_fn` / `kv_forward` / `attention_fn`），M1–M8 八次优化均未改动循环本体；每条优化与朴素实现逐 token 全等，150 项测试全绿。

- **KV Cache 增量解码与跨模型收益边界**：基于注意力位置不变性手写 prefill + 单 token 增量前向，消除序列长度维度的重复计算，gpt2 696-token prompt 下 TPOT 由 29.2 ms 降至 6.5 ms（**4.51×**）；并在 Qwen2.5-0.5B / Qwen3-0.6B 上复现（1.95× / 2.08×），归纳出「每步耗时 = 权重搬运（固定）+ 序列计算（∝ 上下文长度）」成本模型 —— Qwen3-0.6B 参数更多但 head_dim 128×16 头使序列项占比反而更高，收益略高于 Qwen2.5-0.5B；短 prompt（5 token）下为 **0.94×**，机制开销反超所省计算。

- **Continuous Batching 与槽位连续准入**：实现左填充 + attention_mask + 显式 position_ids 的批解码与 per-request 动态退出，4 请求实测 **3.0×**（理想 4.0×）；进一步实现固定槽位调度器，冻结行立刻让位、pending 请求补入（左填充到当前批长 + 单独 prefill + 拷贝 cache 行，与原生批内数值等价，logits 差 < 4e-5），4 请求 / 2 槽位下总前向由 10 次降至 8 次（**1.25×**），并定位出差距来源是「补入需独立前向」。

- **PagedAttention 分页显存管理**：实现分页 KV 池与页表寻址，支持 varlen 变长序列的零填充调度；在 1 长请求（153 token）+ 8 短请求 / 2 槽位的负载下，喂入 token 由 1463 降至 222（**6.6×**）、填充位置由 1220 降至 **0**。

- **Chunked Prefill 混合调度**：针对原 Step 级 Prefill/Decode 互斥导致长 prompt 阻塞在线 decode 的问题，重构调度主循环，实现 Decode 严格优先、Token Budget 驱动的 Chunked Prefill —— 同一调度 Step 内优先以 CUDA Graph 执行 Decode，再用剩余预算以 eager 分片处理 Prefill，并跨 Step 维护增量 KV 状态；总前向由 37 次降至 **30 次**，补入独立前向由 7 次降至 **0 次**，prefill/decode 同批 **7 步**，单步峰值 token 由 157 降至 **16**。

- **Triton 分页 Flash Attention Kernel**：手写页表访存 + Online Softmax + GQA 的分页 attention kernel（`kh = pid_h // G`，`input_precision="ieee"` 严格 fp32，无效行位置钳制避免 NaN），将每层数十个 PyTorch 小 kernel 与 Python 循环折叠为**单 kernel**；端到端由 565 ms 降至 301 ms（**1.88×**），输出与调度统计逐项一致。

- **CUDA Graph 整步捕获**：将整个 decode step 捕获为静态图（预置 ids / position / 页表 / 输出缓冲，侧流捕获 + 动态 Context 重放），padding 行路由至 scratch block 避免污染真实序列；decode 步在 B=1 → B=8 下，逐行 hook 为 10.1 → 22.4 ms，图路径为 3.3 → 4.2 ms，加速 **3.1× ~ 5.3×**，且图路径几乎不随 B 增长（图消掉的 launch 次数与 B 无关）。

### CUDA SGEMM 算子优化与性能分析

**项目技术栈**：CUDA C++、cuBLAS、Nsight Compute、Shared Memory、Register Tiling、Vectorized Load

**项目概述**：基于 CUDA C++ 在 NVIDIA RTX 4070 Laptop GPU 上实现严格 FP32 SGEMM，围绕共享内存分块与寄存器分片逐层优化；在 M=N=K=4096 场景下将性能由 1.11 TFLOPS 提升至 8.95 TFLOPS，实现约 **8.0× 加速**，达到同精度 NVIDIA cuBLAS 的 **73.4%**。

- **统一基准框架与基线归因**：自建 harness 三件套（CPU 参考验语义 + cuBLAS 对照测性能 + ncu 采证），每版本输出 CSV；先用 ncu 证明 naive 版瓶颈在 L1TEX 访存管线（`l1tex__throughput` 96.5% / `dram__throughput` 3.4% / L1 sector 命中率 95%），访存模式本身已最优（A 广播 1 sector、B 连续 4 sectors/128B），瓶颈是「访存条数」而非「模式」，据此确定后续优化路线（1.11 TFLOPS，cuBLAS 10.0%）。

- **二维寄存器分块（v3）**：设计 BM×BN×BK=128×128×8 Block Tile 和 TM×TN=8×8 Thread Tile，每线程使用寄存器数组缓存 A/B 片段及 **64 个累加结果**，通过外积计算提高 Shared Memory 数据复用率与指令级并行度，N=4096 下性能达到 **8.95 TFLOPS**，较 Naive Kernel 加速 **8.0×**。

### 基于 Verilog 的 32 位 MIPS 单周期处理器 RTL 设计

**项目技术栈**：Verilog HDL、ModelSim / 波形仿真、MIPS 指令集、单周期数据通路

**项目概述**：独立设计并验证单周期 32 位 MIPS 处理器，覆盖控制单元、ALU、寄存器堆、存储器接口与数据通路；在基础指令链路上扩展分支与跳转，并把各级总线位宽参数化。

- **核心 RTL 模块设计**：基于 Verilog HDL 实现 Control Unit、ALU/ALU Control、Register File、Memory Interface 及 Datapath 等核心模块。

- **指令集扩展**：在基础 lw / sw / add 指令链路上扩展 BEQ、Jump 与 ANDI：补充控制信号、PC 分支与跳转选择，以及立即数执行路径，并通过时钟沿、opcode、PC 与 ALUResultOut 波形核对执行结果。

- **总线参数化**：将 Memory、Register File、ALU、Datapath 的数据总线抽象为外部参数，使顶层 MIPS1CYCLE 参数切换到 28 位后各级总线同步适配，并完成波形验证。

---

**可立即到岗，实习期 6 个月以上（2026.09 – 2027.07）**：具备较强的工程调试与问题定位能力，坚持「预热 + best-of-3 + 逐轮重置」的测量纪律 —— 两次发现并修正自己 benchmark 的系统性偏差（CUDA 冷启动污染使 3.0× 被错记为 1.18×；图捕获计入计时造出假的 0.81×）。

---

## 技术能力

- **编程语言**：Python、C++，熟悉 Triton 算子开发、CUDA C/C++、Verilog
- **深度学习框架**：PyTorch，熟悉 HuggingFace Transformers 模型加载与前向逐元素比对；手写 GPT-2 / Qwen2.5 / Qwen3 前向
- **推理优化核心技术**：KV Cache 管理与压缩、Continuous Batching、PagedAttention、Radix 前缀缓存、Chunked Prefill、CUDA Graph、算子融合、Online Softmax、GQA、RoPE/RMSNorm/SwiGLU/QK-Norm、SGEMM 优化、寄存器分块与访存优化
- **性能分析工具**：Nsight Compute（DRAM 吞吐 / sectors-per-request / bank conflict / long-scoreboard stall）、Nsight Systems
- **工程工具**：Git、Linux/WSL、Docker（了解）

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

> ⚠️ **与参考简历的差异是刻意的**：
> ① 参考简历的 `58.24×` 来自极低的 naive 基线（113.55 GFLOPS），本仓库 naive 为 1112.3 GFLOPS
> （快 9.8×），故总加速比只有 8.0× 但**最终性能 8.95 TFLOPS 高于其 6.61 TFLOPS**；
> ② 参考简历未含「负结果」，本文件主动保留（0.94× / 1.25× / 0.82×）。
> **不要为了数字好看去改基线或删负结果** —— 面试官一追问就穿帮。
