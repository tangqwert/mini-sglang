# 简历投递包 — mini-sglang

> 用途：① 简历「项目经历」栏的直接素材；② 面试深挖的弹药库；③ 数字复核底账。
> 纪律：**每个数字都能在 `benchmark/results/` 指认出处；每条 bullet 都要能扛住 10 分钟深挖。**

## 0. 一句话定位

从零手写轻量级 LLM 推理引擎，覆盖 **解码循环 → KV Cache → Continuous Batching →
Radix 前缀缓存 → 分页 KV / PagedAttention → 零填充 varlen 调度 → Chunked Prefill →
自研 Triton kernel**；TDD 驱动，每一步都与朴素实现做逐 token 一致性验证，
并用实测数据反推各优化的**收益边界**（含 3 个诚实的负结果 / 反常现象）。

⚠️ **与官方同名项目的区分**：本仓库与 SGLang 官方的
[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)（生产级精简框架，~5000 行 + CUDA kernel，H200 级）
**无代码或血缘关系**，交集只有 Radix Cache 一个概念。模块级对照见 [`official-vs-mine.md`](./official-vs-mine.md)。

## 1. 简历条目

### §1A 投递版（**只放这 4 条**）

> ⚠️ **简历不是文档。** 早先的版本每条 5–6 行、共 20 行，一页简历根本放不下。
> 重写原则：
> ① 全条 **10 行以内**（4 条 × 2 行 + 2 行标头）② 数字**前置** ③ 每条只讲**一个能力信号**。
>
> | 条目 | 想传递的信号 |
> |---|---|
> | 引擎架构 | 系统设计能力（解耦、后端可替换） |
> | 显存与调度 | 核心算法 + 两个最硬的数字（**4.51x / 6.6x**） |
> | 自研 kernel | **差异化**：真的写了 kernel，不只调框架 |
> | 架构无关前向 | 跨架构能力 + 会验证 + 敢报负结果 |

**纯文本粘贴版**（直接复制到 Word / LaTeX / 超级简历，不含任何 markdown 标记）：

```
Mini-SGLang：从零实现的轻量级 LLM 推理引擎                  2026.09 – 至今
个人项目 | Python / PyTorch / Triton | github.com/tangqwert/mini-sglang

· 引擎架构：从零实现推理全链路（KV Cache → Continuous Batching → Radix 前缀缓存 →
  PagedAttention → Chunked Prefill → CUDA Graph），解码循环与模型解耦、后端可整体替换。
· 显存与调度：手写 prefill + 单 token 增量前向，长 prompt 下 TPOT 29.2 → 6.5 ms（4.51x）；
  并在 Qwen2.5-0.5B / Qwen3-0.6B 复现（1.95x / 2.08x），定位出收益受权重搬运约束；实现
  分页 KV 池 + varlen 零填充调度，喂入 token 1463 → 222（6.6x）、填充位置 1220 → 0。
· 自研 kernel：以 Triton 手写分页版 flash attention（页表访存 + online softmax + GQA），
  端到端 565 → 301 ms（1.88x）；CUDA Graph 整步捕获，decode 步加速 5.3x（B=8）。
· 架构无关前向：手写 GPT-2 / Qwen3 前向（GQA / RoPE / RMSNorm / SwiGLU / QK-Norm）与 HF
  逐元素一致（<5e-5）；150 项测试，每条优化与朴素实现逐 token 全等；报告 3 个负结果。
```

**10 行**（4 条 × 2 行 + 2 行标头）—— 一页简历里给单个项目留 10–12 行是合理配额。

> **排版提示**：
> - 「Mini-SGLang」这一行是条目标题：左侧项目名、右侧时间、下一行放链接与关键词
> - 数字加粗（Word 选中 → Ctrl+B）：**4.51x / 6.6x / 1.88x / 5.3x / 150**
> - ⚠️ 上面的硬换行按「10.5pt + 2cm 页边距」排，每行 ≤88 半宽字符。字号/边距不同的话，
>   直接删掉行尾硬换行让 Word 自动折行即可，内容不受影响。
> - 若版面富余想加回细节，**优先加第 2 条**（补 Chunked Prefill 的 37 → 30、7 → 0）；
>   若版面不够，**先删第 1 条**的括号内容，只留「从零实现推理全链路」。

> **被砍掉的内容去哪了**：为了压到 10 行，丢掉了「Continuous Batching 3.0x」「Chunked Prefill
> 37 → 30 / 7 → 0」「单步峰值 157 → 16」「B=1 时图也有 3.1x」「成本模型」「跨模型对照
> 1.95x / 0.94x」「90 假模型 + 60 真模型」等细节。它们**全部在 §1B**，是面试深挖的弹药 ——
> **简历负责拿面试，§1B 负责过面试。**

### §1B 面试弹药（**不上简历**，但每条都要能讲）

- **KV Cache 增量解码**：基于注意力位置不变性手写 prefill + 单 token 增量前向，输出与朴素解码逐 token 一致；
  gpt2 696-token prompt 下 29.2 → 6.5 ms/token（**4.51x**）
- **跨模型瓶颈分析**：同一优化在 3 个模型上复现 —— gpt2(124M/696tok) **4.51x**、
  Qwen2.5-0.5B(0.5B/601tok) **1.95x**、Qwen3-0.6B(0.6B/704tok) **2.08x**；归纳出
  「每步耗时 = 权重搬运（固定，∝参数量）+ 序列计算（∝长度 × 每 token 注意力开销）」成本模型，
  并解释两个反直觉现象：① 短 prompt 下 **0.94x 微负**（省的计算抵不过 cache 读写开销）；
  ② **Qwen3-0.6B 参数更多，收益反而略高于 Qwen2.5-0.5B** —— 因其 head_dim 128 × 16 头，
  每 token 注意力开销是 Qwen2.5 的 ~2.7 倍，序列项占比更高（收益 ∝ 被省成分的成本占比）
- **Continuous Batching**：实现左填充 + attention_mask + 显式 position_ids 的多请求批解码与
  per-request 动态退出，4 请求实测 **3.0x**（理想 4.0x），且与逐条输出逐 token 一致
- **槽位连续准入**：实现调度器 + 固定槽位，冻结行立刻让位、pending 队列补入新请求
  （补入靠"左填充到当前批长 + 单独 prefill + 拷 cache 行"，与原生批内数值等价，误差 < 4e-5）；
  4 请求 / 2 槽位下总前向 10 → 8 次（**1.25x**，理想 2.00x），**并定位出差距来源是"补入需独立前向"**
- **Radix 前缀缓存**：trie 前缀匹配 + KV 继承 + 收工回写，共享前缀请求的 prefill 计算量从 O(T) 降到 O(后缀)；
  实测 **0.82x** —— 诚实报告负收益，并给出真实收益场景（大模型 × 长前缀 × 高命中率）
- **分页 KV + PagedAttention**：实现分页 KV 池 + 按块表 gather + 多头缩放点积注意力；
  与连续存储的参考实现**逐元素一致**（误差 < 1e-5）——验证了「物理布局改变、数学结果不变」
- **自研 GPT-2 前向 + 分页接入**：脱离 `transformers` 的 forward，直接从权重手写
  embed → 12 × (LayerNorm → 多头注意力 → 残差 → MLP → 残差) → lm_head，与 HF 输出
  **逐元素一致**（max|diff| ~5e-5）；再把 attention 换成「写进分页池 → 按页表 gather →
  paged_attention」，logits 不变（block_size ∈ {1,2,4,7,64} 均一致）
- **分页增量解码**：prefill（整条 prompt 一次前向）与 decode（每步 1 token、显式绝对
  `position_ids`）共用同一个分页钩子；KV 全程住在分页池、每步按页表 gather，输出与 M0
  朴素解码**逐 token 一致**，且不受 block_size（1/2/4/8）影响
- **分页调度（连续准入 + 零填充）**：调度器全程走页表——每条序列按需分页增长、
  完成即归还显存；多请求 prompt 拍平成 **varlen** 一次前向（对应 flash-attn 的
  `cu_seqlens` 语义），左填充彻底消失。同负载下喂入 token 数 **1463 → 222（6.6x）**、
  填充位置 1220 → **0**，输出仍与朴素解码逐 token 一致
- **Chunked Prefill（M6）**：把 prefill 与 decode 合进**同一次** varlen 前向 ——
  varlen 批次本来就不要求各序列等长，于是一个批里可以“活跃序列各 1 个 token（decode）
  + 新序列一大块 prompt（prefill）”。总前向次数 37 → **30**、补入独立前向 7 → **0**、
  prefill/decode 同批 7 步；再用 `max_prefill_tokens`（对应 vLLM 的
  `max_num_batched_tokens`）给单步成本设上限（单步 token 峰值 157 → 16），把 ITL 摊平
- **自研 Triton 分页注意力 kernel（M5）**：把「查页表 gather + 因果掩码 + online softmax
  + 加权求和」封进**一个** kernel（即分页版 flash attention）——PyTorch 版每层要起十来个
  kernel 加一圈 Python 循环，Triton 版每层只剩 2 个（1 写 + 1 算）。端到端
  **565 → 301 ms（1.88x）**，且输出与调度统计与 PyTorch 后端**完全一致**
  （严格 fp32、不开 TF32，实测 max|diff| ~3e-7）
- **测试驱动开发**：**129 项测试全部通过**（77 项假模型/纯张量单元 + 52 项真模型），
  每项优化均与朴素实现做逐 token 一致性验证，并用「喂入 token 数」精确账本断言开销

## 2. Bullet ↔ 面试深挖对照表（每条都要能扛 10 分钟）

| 追问 | 我的答案要点 |
|---|---|
| **为什么 decode 是 memory-bound？** | 每步只算 1 个 token，但要把全部权重从 HBM 搬进 SM；计算量小、搬运量大 → 瓶颈在显存带宽。gpt2 fp32 权重 0.5GB，每步搬一遍 |
| **4.5x 怎么测的？** | 同一条 696-token prompt、`--engine naive` / `--engine kv` 两条路径同口径；测 128 个新 token 的稳态耗时；**同一条 prompt 保证可比**（出处 `benchmark/results/*.json`） |
| **⚠️ 你考虑 kernel launch 开销了吗？** | 一开始的模型漏了。后来把墙钟拆成 CUDA event（GPU 侧）与差值（CPU 侧：launch + Python + 框架），并注意到 M0 侧 attention 是 **O(T²)**，所以真实模型应是「固定项 + O(T) + O(T²)」三项。**待补**：用 T = 128/512/1024/2048 做 log-log 拟合斜率判定（计划中） |
| **为什么 Qwen 收益小？** | 成本公式：每步 = 固定开销（权重搬运 + launch + Python）+ 序列计算（∝ 长度）。Qwen 权重 4x → 固定项大 → 收益被压缩（1.95x vs 4.51x），实测 14.8 ms/token 与推算吻合 |
| **为什么短 prompt 反而变慢？** | 5-token prompt 时 0.94x。KV Cache 省下的序列计算 < 新增机制开销（cache 更新 + Python 循环 + 额外的前向调用）。**优化的收益 = 被省成分的成本 − 新增机制的开销**，这是我在三个里程碑里反复验证的结论 |
| **为什么需要 attention_mask / position_ids？** | 左填充的 pad 若不被 mask 会污染注意力；pad 占位又打乱了位置编号。两者都是**踩过真 RuntimeError 后**才补上的 |
| **连续准入怎么实现的？为什么要"拷 cache 行"？** | HF 的批 cache 要求**所有行等长**，所以一条长度不同的新请求没法直接插进去。技巧是：把它左填充到当前批长 → 单独 prefill 一次拿到它自己的 K/V → 再把那一行拷进空槽。因果注意力逐行独立 ⇒ 与"它一开始就在批里"数值等价（实测 logits 差 < 4e-5） |
| **那为什么收益只有 1.25x（理想 2.00x）？** | 两笔代价：① 补入要算 S 个填充位置（S ≈ 当前批长）；② 补入要多开**一次独立前向**。**分两步修完**：M4b 用分页 + varlen 消掉①（喂入 token 1463 → 222），M6 把 prefill 与 decode 合流消掉②（总前向 37 → 30，补入独立前向 7 → **0**）。**能把这个坑拆成两半分别验证，比报一个加速比更有说服力** |
| **Chunked Prefill 是什么？为什么要 chunk？** | 把 prompt 切块、与 decode 合进同一步。不 chunk 的话一条 4096-token prompt 独占一步，这一步算力是别人的 4096 倍 → 其他请求的 ITL 被打爆。用 `max_prefill_tokens` 设上限后单步峰值 **157 → 16 token**，代价是前向次数变多（总计算量不变）。分块的正确性靠两点：**绝对位置跨块连续** + 后续块能看到前面块（已在池里，gather 天然满足）；我用 **chunk=1** 做了最狠的压力测试 |
| **⚠️ 那分页路径墙钟反而更慢？** | 确实曾经更慢：喂入 token 降 6.6 倍，墙钟 **306 → 624ms**。因为逐序列注意力是 Python 循环（每序列 × 每 query 位置 × 每层十来个 kernel），而 M2.5 用的是 HF 融合好的批式注意力。**于是我写了 M5**：Triton kernel 把这一段折成 1 个 kernel，端到端 **565 → 301ms（1.88x）**，墙钟追平 M2.5 而 token 数低 6.6 倍 |
| **Triton kernel 具体怎么写？** | 每个 program 负责「一条序列 × 一个 query 块 × 一个 head」；沿 KV 方向以 BLOCK_N 为步长循环，每次按页表查出 block id 与槽位、载入 K/V 块；online softmax（m_i / l_i / acc 三元组滚动更新）避免物化整个 scores。踩过的坑：① `tl.dot` 要求 K ≥ 16（我测试用的 head_dim=8 直接编译失败）；② BLOCK_M > n_q 时无效行被整行掩掉 → -inf → NaN，得把无效行的位置钳到最后一个有效位置 |
| **⚠️ 为什么微基准加速比不随 seq_len 单调？** | 因为两个实现的瓶颈不同：PyTorch 版**启动受限**（耗时几乎不随 seq_len 变，0.36 → 0.50 ms），Triton 版**带宽受限**（要真去搬更多 KV），所以 seq_len 小时 Triton 赢在"少十几次启动"，seq_len 大时差距收窄。更根本的短板是我**没做 split-K**——单序列只有 H=12 个 program，128 个 SM 大量空转。真 vLLM 用 flash-decoding 把 KV 维也切开并行再规约 |
| **「命中缓存」的严谨定义？** | token 序列的**精确公共前缀**（非语义相似）。依据是注意力的位置不变性 ⇒ 前缀的 K/V 可无损复用 |
| **真 SGLang 和你的差距？** | **不分裂**（部分重叠直接放弃插入）、**无 refill**、**无驱逐**、**kernel 是 PyTorch**。完整清单见 `official-vs-mine.md` |

## 3. 王牌素材：那个差 1 个 token 的 bug（面试用）

**故事梗概**：M3 的 `cached_generate` 用 `radix.insert(ids + generated, cache)` 记账，但 cache 实际只覆盖到
`generated[:-1]`（最后一个 token 的 K/V 要等下一次 forward 才进 cache）。树节点因此**多认领了 1 个 token**；
后续请求「完整命中」时 `clone_cache_prefix` 会**静默截断**（张量切片不报错），丢掉最后一个 token 的 K/V → 输出错误。

**为什么 30 项测试全绿仍漏掉**：现有测试的共享前缀都是 **partial hit**（如 `[1,2,3,4,5]` vs `[1,2,3,4,9]` 只命中 4 个），
`use` 远小于 cache 长度 → 截断不触发。**没有任何测试让新请求「完整命中并继续延长」。**

**修复**：`insert(ids + generated[:-1], cache)`（只声明 cache 真正覆盖的部分）+ 补 `test_full_prefix_match_extends`。

**这段为什么值钱**：它同时证明三件事 —— ① 能发现深层 bug；② 理解测试盲区（**partial hit 测了，full hit 没测**）；
③ 会诚实复盘。**比「我很努力」有说服力得多。**

> 同类的第二个例子（M1）：`kv_generate` 的 prefill 步曾缺 EOS 判定，若首个 token 就是 EOS 会比 M0 多生成一个 token。
> 也是边界盲区（**测了 EOS 在第 2 个 token，没测第 1 个**）。教训：**「与 X 逐 token 一致」的实现，盲区总在边界那一步。**

## 4. 数据档案（防面试官复查）

| 实验 | 数据 | 出处 |
|---|---|---|
| M1 KV cache（gpt2, **696** tok） | 3.739s → 0.828s；29.2 → 6.5 ms/token，**4.51x** | `benchmark/results/m0_naive_long.json` / `m1_kv_long.json` |
| M1 KV cache（Qwen2.5-0.5B, **601** tok） | 3.705s → 1.900s；28.9 → 14.8 ms/token，**1.95x** | `benchmark/results/m0_naive_qwen_long_en.json` / `m1_kv_qwen_long.json` |
| M1 KV cache（**Qwen3-0.6B**, 704 tok） | 8.282s → 3.980s；64.7 → 31.1 ms/token，**2.08x** | `benchmark/results/m0_naive_qwen3_long.json` / `m1_kv_qwen3_long.json` |
| M1 短 prompt（gpt2, **5** tok） | 0.738s → 0.783s；5.8 → 6.1 ms/token，**0.94x（微负）** | `benchmark/results/m0_naive_short.json` / `m1_kv_short.json` |
| M2 batching（4 请求，各 24 token） | 553 → 178 ms，**3.0x**（理想 4.0x）；批内每 forward 5.8 → 7.4 ms（填充行 + 记账） | `benchmark/verify_m2.py` |
| M2 batching 修正 | ⚠️ 旧值 **1.18x** 是**测量 bug**：benchmark 让批量路径跑第一个，独占了 CUDA/cuBLAS 冷启动（首次 581ms vs 预热后 183ms） | `benchmark/verify_m2.py` 注释 |
| M2.5 连续准入（4 请求 / 2 槽位） | 静态分批 10 次前向 → **8 次**（6 步批量 + 2 次补入）；理想 2.00x，实测 **1.25x** | `tests/test_batching.py::TestContinuous` |
| M3 radix 正确性 | 3 请求共享 118-token 前缀，cached vs 逐条输出**逐 token 一致** | `benchmark/verify_m3.py` |
| M3 radix 耗时 | **0.82x** —— 小模型 prefill ≈5ms 被权重搬运主导，省下的计算 < clone + Python 开销 | `benchmark/verify_m3.py` |
| M4b Step 4 分页调度 | 1 长请求(153 tok) + 8 短请求｜槽位 2：喂入 token **1463 → 222（6.6x）**；填充位置 1220 → **0**；前向次数 **37 → 37（不变）** | `benchmark/verify_m4b.py` |
| M4b Step 4 KV 显存 | 2.2 MB（批宽锁死 S=182 × 2 行，含填充）→ **1.2 MB**（峰值实际占用，完成即归还） | `benchmark/verify_m4b.py` |
| M6 Chunked Prefill | 同负载：总前向 37 → **30**；补入独立前向 7 → **0**；prefill/decode 同批 7 步 | `benchmark/verify_m6.py` |
| M6 单步成本上限 | `max_prefill_tokens` 不限 / 64 / 16 → 单步峰值 157 / 64 / **16** token（总前向 30 / 32 / 39） | `benchmark/verify_m6.py` |
| ⚠️ 同一负载的墙钟 | M2.5 **306ms** → Step4 670ms → M6 624ms → **M6+Triton 301ms**（重复 3 次取最短） | `benchmark/verify_m6.py` / `verify_m5.py` |
| M5 Triton 端到端 | PyTorch 分页 **565 → 301 ms（1.88x）**；输出与调度统计完全一致 | `benchmark/verify_m5.py` |
| M7 CUDA Graph（三路对照） | decode 步：A 逐行 hook / B eager+向量化 / C 图。B=8：**22.4 → 4.2 ms（5.3x）**；B=1：10.1 → 3.3 ms | `benchmark/verify_m7.py` |
| M7 变量拆分 | 向量化写入 B/A = 0.78x(B=1) → **0.38x(B=8)**；CUDA Graph C/B 稳定在 **0.41–0.50** | `benchmark/verify_m7.py` |
| M8 Qwen3 前向 | 与 HF 逐元素一致 max\|diff\| **~2e-5**（fp32, 28 层）；接入完整引擎后与朴素解码逐 token 一致 | `tests/test_qwen3_forward.py` |
| M5 微基准（单层 decode） | seq_len 128–2048、block_size 8/16/32：1.0–2.0x，且**不随 seq_len 单调** | `benchmark/verify_m5.py` |

> ⚠️ 三个模型的长 prompt 实验**长度不同**（gpt2 696 / Qwen2.5 601 / Qwen3 704），因为各 tokenizer 切分粒度不同，无法凑完全一致。报告中必须写清，否则被追问会措手不及。**跨模型只比加速比，不比绝对耗时。**
> ⚠️ **Step 4 / M6 的墙钟曾【比 M2.5 慢 2 倍】**（306ms → 624ms），而喂入 token 降了 6.6 倍。
> 原因：① 逐序列注意力是 Python 循环（每层十来个 kernel + 一圈 Python）；② 小模型地板效应。
> **M5 的 Triton kernel 把这段差距补上了（565 → 301 ms，1.88x）**，墙钟追平 M2.5 而 token 数低 6.6 倍。
> 这条“先量出问题 → 再针对性写 kernel”的弧线，比直接说“我写了 Triton kernel”有说服力得多。
> ⚠️ 同时要**主动交代尚未解决的部分**：我的 kernel 没有 split-K，单序列只有 H=12 个 program，
> 微基准加速比只有 1.0–2.0x 且不随 seq_len 单调。知道自己的 kernel 瓶颈在哪，比报一个好看的数字更像工程师。

**测试统计**（`pytest tests/ --collect-only`）：

| 范围 | 数量 |
|---|---|
| M0–M2.5 | **39**（37 假模型单元 + 2 真模型集成） |
| M4a 分页 KV + PagedAttention | 19 |
| M4b Step 1/2 自研前向 + 分页接入 | 12 |
| M4b Step 3 分页增量解码 | 6 |
| M4b Step 4 分页调度（零填充 varlen） | 12 |
| M6 Chunked Prefill | 10 |
| M5 Triton kernel（对齐参考实现） | 21 |
| M5 Triton 端到端（换后端，输出与统计不变） | 10 |
| M7 CUDA Graph（含 padding 不污染真实序列的回归测试） | 9 |
| M7.5 图接进调度器（输出与调度统计不变） | 4 |
| M8 Qwen3 前向（GQA / RoPE / RMSNorm / SwiGLU） | 8 |
| 合计 | **150（全部通过，无 xfail）** |

## 5. 投递前 Checklist

### 代码 / 仓库
- [x] `pytest tests/` 全绿（150 passed，无 xfail）
- [x] M1/M3 两处边界 bug 已修 + 回归测试已补
- [x] README 数字与 `benchmark/results/` 一致（696 / 601 / 704 已核对）
- [x] `docs/official-vs-mine.md` 已就位（用于回答"与官方差距"）
- [x] GitHub 定位声明（README 顶部，解决同名混淆）
- [x] `benchmark/results/*.json` **已入库** —— 之前被 `.gitignore` 漏掉，会让"数字可现场核对"变成空话
- [x] `LICENSE`（MIT）已补
- [ ] GitHub About 栏 + topics ← **见 §7，5 分钟**

### 简历（投递前必做）
- [ ] 把 **§1A 的「纯文本粘贴版」**直接粘进简历项目经历栏（不要粘 §1B 的 12 条）
- [ ] 五个关键数字加粗：**4.51x / 6.6x / 1.88x / 5.3x / 150**
- [ ] 按 **§8** 清理现有简历的问题（Vibe Coding / 实习时长 / "熟悉结构" / `xxx` 占位 / 技能栏措辞）
- [ ] 导出 PDF，检查：① 一页 ② 链接可点 ③ 无错别字
- [ ] **用手机打开仓库链接**，确认 README 排版没崩（架构图、表格、嵌套代码块）

### 投递节奏
- [ ] 先投 3–5 家「不那么想去」的练手，24h 内复盘被问到什么
- [ ] 再投目标公司（字节 / 阿里 / 百度 / NVIDIA 中国 / 国产 GPU 厂商）
- [ ] Boss 直聊开场带数字：「做过从零实现的 LLM 推理引擎，含自研 PagedAttention Triton kernel；可立即到岗，实习 6 个月以上」

## 6. 数字纪律（写简历前的自检）

1. **不写没有出处的数**：每个 GFLOPS / ms / 倍数都要能指到 `benchmark/results/` 或 `verify_*.py`
2. **不写没做完的功能**：M4a 未完成前不写 PagedAttention；SGEMM 没有数据前不写
3. **不夸大技能**：技能栏写「正在学习 CUDA 并做算子练习」，不写「熟练」
4. **不留占位符**：`xxx` / `待定` 一律不能出现在投递版
5. **不写 AI 辅助工具**：`Cursor` / `Codex` / "Vibe Coding" 对 AI Infra 岗位是减分项
6. **必写**：GitHub 链接 + 可实习时长（**可立即到岗，实习期 6 个月以上** —— 这是你的稀缺优势）
7. **两个被测对象必须等价预热**：冷启动会**系统性地惩罚"先跑的那条路径"**
   （实证：同一份代码首次 581ms vs 预热后 183ms，导致 M2 的收益被低估成 1.18x）。
   所有 `verify_m*.py` / `bench.py` 统一口径：**预热一次 → best-of-3 → 取最短**
8. **数字奇怪就先怀疑测量，再怀疑代码**：M2 报出 1.18x 时先做反向校验
   （直接测单次前向在 B=1..8 的耗时，发现硬件支持近线性加速）→ 说明是测量问题。
   **这条比多做一个优化更能证明工程素养。**

## 7. GitHub 展示（5 分钟，性价比最高的一步）

仓库：https://github.com/tangqwert/mini-sglang

**① About 栏**（仓库首页右上角齿轮 ⚙ → Description）：

```
从零实现 LLM 推理引擎：KV Cache / Continuous Batching / PagedAttention / Chunked Prefill / 自研 Triton kernel｜129 测试全绿
```

**② Topics**（同一面板，逐个填）：

```
llm-inference   kv-cache   paged-attention   continuous-batching
chunked-prefill   triton   cuda   pytorch   from-scratch   inference-engine
```

**③ 固定到主页**：Profile → Customize your pins → 勾上 `mini-sglang`（让 HR 一眼看到）

**④ Social preview**（可选）：Settings → Social preview 传一张 README 截图

> ⚠️ **About 栏不要写**「熟悉 Transformer 原理」这类无信息量的话。
> 写**你造了什么**（from-scratch inference engine）+ **最硬的证据**（自研 Triton kernel / 129 测试）。

## 8. 你现有简历需要改的地方

> 下面几条是在早前沟通里提到的点。把简历原文发我，我可以直接逐句改。

| 位置 | 现在是 | 改成 | 为什么 |
|---|---|---|---|
| 技能/其他栏 | 「具有 Vibe Coding 基础，会使用 Codex 等 agent 协助助手」 | **整行删掉** | 对 AI Infra 是**减分项**：它暗示「代码不是你写的」。而你最强的资产恰恰是「从零手写、逐 token 一致性验证」 |
| 实习时间 | 「可实习三个月以上」 | 「**可立即到岗，实习期 6 个月以上**（2026.09–2027.07）」 | 你是 gap year，能连续实习 6–11 个月 —— 相对在校生这是**最大稀缺优势**，必须明确写出来 |
| 项目描述 | 「熟悉 mini-sglang 的整体结构」 | 「**从零实现** mini-sglang：…」（用 §1A 的 5 条） | 「熟悉结构」读起来像在读别人的仓库；这个项目是**你自己写的** |
| 占位符 | 文中的 `xxx` | 填掉或整条删掉 | 投递版留占位符 = 不认真（§6 第 4 条） |
| 技能栏（若写了 CUDA） | 「熟悉 CUDA」 | 「CUDA：掌握并行编程模型与 profiling；正在做算子练习」 | 诚实且不露怯；被追问 coalescing / tiling 细节时不会翻车（§6 第 3 条） |
| 技能栏 | 「熟悉 Triton」（若只列在技能栏） | 「Triton：手写过分页注意力 kernel（online softmax + 页表访存）」 | 写在**项目**栏里、且在技能栏标出具体做到什么程度，比单写一个词可信得多 |

