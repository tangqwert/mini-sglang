# 简历投递包 — mini-sglang

> 用途：① 简历「项目经历」栏的直接素材；② 面试深挖的弹药库；③ 数字复核底账。
> 纪律：**每个数字都能在 `benchmark/results/` 指认出处；每条 bullet 都要能扛住 10 分钟深挖。**

## 0. 一句话定位

从零手写轻量级 LLM 推理引擎，覆盖 **解码循环 → KV Cache → Continuous Batching → Radix 前缀缓存**；
TDD 驱动，每一步都与朴素实现做逐 token 一致性验证，并用实测数据反推各优化的**收益边界**。

⚠️ **与官方同名项目的区分**：本仓库与 SGLang 官方的
[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)（生产级精简框架，~5000 行 + CUDA kernel，H200 级）
**无代码或血缘关系**，交集只有 Radix Cache 一个概念。模块级对照见 [`official-vs-mine.md`](./official-vs-mine.md)。

## 1. 简历条目（投递版 · v1）

> **Mini-SGLang：从零实现轻量级 LLM 推理引擎**（个人项目｜Python / PyTorch）
> 2026.09 - 至今　github.com/tangqwert/mini-sglang

- **KV Cache 增量解码**：基于注意力位置不变性手写 prefill + 单 token 增量前向，输出与朴素解码逐 token 一致；
  gpt2 696-token prompt 下 29.2 → 6.5 ms/token（**4.51x**）
- **跨模型瓶颈分析**：对照实验发现收益被「权重搬运地板效应」压缩（Qwen2.5-0.5B 仅 **1.95x**，
  短 prompt 反而 **0.94x 微负**），归纳出「每步耗时 = 固定开销 + 序列计算（∝ 上下文长度）」成本模型
- **Continuous Batching**：实现左填充 + attention_mask + 显式 position_ids 的多请求批解码与
  per-request 动态退出，4 请求批 516ms vs 逐条 606ms（**1.18x**），且与逐条输出逐 token 一致
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
- **测试驱动开发**：**87 项测试全部通过**（56 项假模型单元 + 31 项真模型），
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
| **那为什么收益只有 1.25x（理想 2.00x）？** | 补入需要**一次独立前向**（没有分页就没法和 decode 合并进同一 kernel），还留下左填充碎片。**这就是 PagedAttention 存在的理由** —— 分页后每行页表独立，既不用填充，补入也能并进同一个 kernel |
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
| M1 KV cache（gpt2, **696** tok） | 3.739s → 0.828s；29.2 → 6.5 ms/token，**4.51x** | `m0_naive_long.json` / `m1_kv_long.json` |
| M1 KV cache（Qwen2.5-0.5B, **601** tok） | 3.705s → 1.900s；28.9 → 14.8 ms/token，**1.95x** | `m0_naive_qwen_long_en.json` / `m1_kv_qwen_long.json` |
| M1 短 prompt（gpt2, **5** tok） | 0.738s → 0.783s；5.8 → 6.1 ms/token，**0.94x（微负）** | `m0_naive_short.json` / `m1_kv_short.json` |
| M2 batching（4 请求） | 批 516ms vs 逐条 606ms，**1.18x** | `benchmark/verify_m2.py` |
| M2.5 连续准入（4 请求 / 2 槽位） | 静态分批 10 次前向 → **8 次**（6 步批量 + 2 次补入）；理想 2.00x，实测 **1.25x** | `tests/test_batching.py::TestContinuous` |
| M3 radix 正确性 | 3 请求共享 118-token 前缀，cached vs 逐条输出**逐 token 一致** | `benchmark/verify_m3.py` |
| M3 radix 耗时 | **0.82x** —— 小模型 prefill ≈5ms 被权重搬运主导，省下的计算 < clone + Python 开销 | `benchmark/verify_m3.py` |
| M4b Step 4 分页调度 | 1 长请求(153 tok) + 8 短请求｜槽位 2：喂入 token **1463 → 222（6.6x）**；填充位置 1220 → **0**；前向次数 **37 → 37（不变）**；墙钟 818 → 650ms（1.26x） | `benchmark/verify_m4b.py` |
| M4b Step 4 KV 显存 | 2.2 MB（批宽锁死 S=182 × 2 行，含填充）→ **1.2 MB**（峰值实际占用，完成即归还） | `benchmark/verify_m4b.py` |

> ⚠️ gpt2 与 Qwen 的长 prompt 实验**长度不同**（696 / 601）。报告中必须写清，否则被追问会措手不及。
> ⚠️ Step 4 的墙钟只快 1.26x 而 token 数降了 6.6x —— 因为 ① 前向次数没变（补入仍需独立前向）、
> ② 逐序列注意力是 Python 循环（真引擎用 varlen kernel）、③ 小模型地板效应。**主动讲这条，比只报 6.6x 更可信。**

**测试统计**（`pytest tests/ --collect-only`）：

| 范围 | 数量 |
|---|---|
| M0–M2.5 | **39**（37 假模型单元 + 2 真模型集成） |
| M4a 分页 KV + PagedAttention | 19 |
| M4b Step 1/2 自研前向 + 分页接入 | 12 |
| M4b Step 3 分页增量解码 | 6 |
| M4b Step 4 分页调度（零填充 varlen） | 11 |
| 合计 | **87（全部通过，无 xfail）** |

## 5. 上简历前 Checklist

- [x] `pytest tests/` 全绿（87 passed，无 xfail）
- [x] M1/M3 两处边界 bug 已修 + 回归测试已补
- [x] README 数字与 `benchmark/results/` 一致（696 / 601 已核对）
- [x] `docs/official-vs-mine.md` 已就位（用于回答"与官方差距"）
- [x] GitHub 定位声明（README 顶部，解决同名混淆）
- [ ] GitHub About 栏补描述 + topics（`llm-inference` / `kv-cache` / `paged-attention` / `tutorial`）
- [ ] （可选）完成 M4a → 追加 PagedAttention bullet（v2）
- [ ] （投递期）性能建模修正版 → 追加建模 bullet（v3）

## 6. 数字纪律（写简历前的自检）

1. **不写没有出处的数**：每个 GFLOPS / ms / 倍数都要能指到 `benchmark/results/` 或 `verify_*.py`
2. **不写没做完的功能**：M4a 未完成前不写 PagedAttention；SGEMM 没有数据前不写
3. **不夸大技能**：技能栏写「正在学习 CUDA 并做算子练习」，不写「熟练」
4. **不留占位符**：`xxx` / `待定` 一律不能出现在投递版
5. **不写 AI 辅助工具**：`Cursor` / `Codex` / "Vibe Coding" 对 AI Infra 岗位是减分项
6. **必写**：GitHub 链接 + 可实习时长（**可立即到岗，实习期 6 个月以上** —— 这是你的稀缺优势）

