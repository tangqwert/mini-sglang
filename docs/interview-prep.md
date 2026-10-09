# 面试准备手册 — mini-sglang

> 用途：把 1791 行代码变成**能讲 20 分钟、能被追问 3 层还站得住**的东西。
> 纪律：**每个数字都要能立刻说出出处**；每句话都要能指到文件与行号。

---

## 0. 先搞清楚面试官在考什么

不是考"你记得什么"，而是考三件事：

| 考的 | 表现 | 对应准备方式 |
|---|---|---|
| **你真的做过吗** | 能被追问到函数名、参数、边界条件 | 闭卷默画执行流程 |
| **你理解为什么** | 能说出每个优化的**收益边界**（什么时候不划算） | 背 §3 的负结果 |
| **你能迁移吗** | 把这套思路用到新问题上 | 准备"如果让你做 X"的回答 |

> ⚠️ **最常见的翻车方式**：能答"是什么"，答不了"为什么这么选"，更答不了"什么情况下不划算"。
> 你这个项目最大的优势恰恰在第三个——因为你有 3 个负结果。

---

## 1. 自我介绍三档（背下来，按场合切换）

### 60 秒版（HR 面 / 电话初筛）

> 我做了一个从零实现的 LLM 推理引擎，参考 SGLang 的架构，用 TDD 写了 1791 行核心代码。
> 主线是把推理引擎的优化逐层做一遍：KV Cache、Continuous Batching、Radix 前缀缓存、
> PagedAttention、Chunked Prefill，最后自己写了一个 Triton 的分页注意力 kernel。
> 全程 129 项测试，每做一层优化都要和朴素实现做逐 token 一致性验证——
> 也就是性能可以变，输出一个字都不能变。
> 我印象最深的不是加速比，是有几次优化**实测比不优化还慢**，我把它拆开找到了原因。

### 5 分钟版（技术面开场）

在 60 秒版基础上加三块：
1. **成本模型**：每步耗时 = 权重搬运（固定，∝模型大小）+ 序列计算（∝上下文长度）
   → 这条公式能解释我所有实验的结果差异
2. **一段技术弧线**（挑最有代表性的）：分页之后我量出"算力降了 6.6 倍，墙钟反而慢 2 倍"，
   定位到是 Python 循环里 7000 次 kernel 启动，于是写了 Triton kernel 把它追平
3. **诚实短板**：我的 kernel 没做 split-K，单序列并行度只有 12 个 program，
   下一个瓶颈在哪我已经量化了

### 20 分钟版
= 5 分钟版 + 现场对着 GitHub 仓库**讲代码**（见 §7 的阅读路线）

---

## 2. 必答 8 题（骨架题，必须闭卷）

### Q1. 走一遍完整流程：从 prompt 到第一个 token

```
Request(prompt, max_new_tokens, eos)
  ↓ 调度器组批（chunked_prefill_generate, engine/paged_schedule.py:160）
  ↓ 拍平成 varlen 一条 1D 序列 + 每序列一张页表
  ↓ gpt2_forward（engine/model_forward.py:113）
      embed(token + position) → 12 × (LN → attention → 残差 → LN → MLP → 残差) → LM head
  ↓ attention_fn = BatchedPagedAttentionHook（或 Triton 版）
      ① 把本步 K/V 写进分页池（scatter_kv）
      ② 按页表 gather 出该序列全部历史 K/V
      ③ 逐 query 算因果缩放点积注意力
  ↓ 取最后位置 logits → argmax → 第一个 token
  ↓ 之后每步只喂 1 个 token，position = 已写入池的 token 数
```

**追问陷阱**：「decode 步只喂 1 个 token，为什么还要传 position_ids？」
→ 因为 S=1 但它是**第 T 个**位置。漏了就退化成"永远按位置 0 算"，输出全错
（我在 M1 踩过同类坑）。代码里 `gpt2_forward` 的 docstring 专门警告了这点。

### Q2. 为什么解码循环和模型要解耦？

循环只认识 `logits_fn` / `kv_forward` / `attention_fn` 这类**可调用对象**。

价值（用事实说）：
- M1（KV Cache）、M2（batching）、M3（radix）三次优化**没有改动循环本体**
- M5 换 Triton kernel 时，只给调度器加了一个 `hook_cls` 参数，输出与调度统计完全一致

### Q3. KV Cache 为什么能加速？

prefill 算了整条 prompt，但下一步其实只需要**新 token 对历史 KV 的注意力**——
历史那部分的 K/V 只依赖它们自己和更早的 token，不会变。

所以缓存它们，每步的序列计算从 O(T) 降到 O(1)。

**关键数字**：
- gpt2 696-token：29.2 → 6.5 ms/token（**4.51x**）
- Qwen2.5-0.5B 601-token：28.9 → 14.8（**1.95x**）
- gpt2 5-token：5.8 → 6.1（**0.94x，微负**）
- 出处：`benchmark/results/m0_naive_long.json` 等

**追问陷阱**：「为什么 Qwen 收益小 4 倍？」→ 成本公式里 Qwen 权重大 4 倍，固定项托底；
「为什么短 prompt 更慢？」→ 省下的序列计算 < 新增机制开销。见 §3。

### Q4. Continuous Batching 解决什么？

**Static batching 的问题**：一批请求跑到底，短的生成完了还占着槽位空转。

`batched_generate`（`engine/batching.py:100`）做了动态退出：每行有独立的
预算/EOS，完成就冻结（喂 pad 占位保持形状），其余继续。

**实测 3.0x**（4 请求、各生成 24 token：553 → 178 ms），理想 4.0x。
批内每次 forward 比单条贵 ~1.3x（5.8 → 7.4 ms：填充行 + mask 拼接 + Python 记账），
但它一次服务 4 条请求 → 净 3.0x。

**追问陷阱 1**：「为什么不是 4.0x？」→ 批内每 forward 贵 ~1.3x，乘上 4 条分摊，净得 3.0x。

**追问陷阱 2（这条更重要）**：「这个数字怎么测的？」
→ **必须主动讲预热**：我这个数字最初是 **1.18x**，错在 benchmark 让批量路径跑第一个，
独自承担了 CUDA/cuBLAS 冷启动（首次 581 ms、预热后 183 ms）。
补齐预热 + best-of-3 后是 3.0x。**冷启动偏差会系统性地惩罚「先跑的那条路径」**。

### Q5. PagedAttention 是什么？页表怎么工作？

**要解决的问题**：
1. 左填充碎片（M2.5 实测 1220 个填充位置 = 83% 的算力白算）
2. 补入新请求得整块拷 cache

**做法**：KV 不再连续存，而是切成固定大小的 block 存进预分配的池子；
每条序列只持有**页表**（block id 列表）。

```
逻辑位置 p  →  block_table[p // block_size] 的第 p % block_size 个槽位
```

- 写：`scatter_kv`（一次高级索引赋值，不是逐 token 循环）
- 读：`gather_layer` —— `pool.keys[layer][block_table]` 拿到
  `[n_blocks, BS, H, D]`，reshape 成 `[n_blocks*BS, H, D]`，截取前 seq_len 行
  （最后一个 block 通常没写满）
- 数值不变性：与连续存储的参考实现**逐元素一致**（< 1e-5）

### Q6. 为什么要自己写前向？不能用 HF 吗？

**因为 HF 挡住了路**：HF 的 attention 内部会先把 `past_k / past_v` 拼成
`[B, H, S, D]` 再算注意力 —— 分页存储"不必连续"的优势在那一刻就被抹掉了。

所以要用上 M4a 的 gather，**必须接管 attention 本身**；最干净的做法就是自己写前向。

`engine/model_forward.py:113`，整个前向 6 步：
`embed → 12 × (LN₁ → 多头注意力 → 残差 → LN₂ → MLP → 残差) → LN_f → lm_head`

验收：与 HF 输出逐元素一致（max|diff| ~5e-5）。
**约定**：HF 的 Conv1D 权重是 `[in, out]`，前向写成 `x @ W + b`，我保持同样约定才能对上。

### Q7. Chunked Prefill 解决什么？

M4b Step 4 消掉了"填充计算"，但**前向次数一个没少**——补入一条新请求仍要多开一次 prefill。

M6 把 prefill 与 decode 合进**同一次** varlen 前向：
varlen 批次本来就不要求各序列等长，所以一个批里可以混着
「活跃序列各 1 个 token（decode）+ 新序列一大块 prompt（prefill）」。

结果：总前向 37 → **30**，补入独立前向 7 → **0**，`mixed_steps = 7`。

**为什么还要 chunk**：一条 4096-token 的 prompt 独占一步，这一步算力是别人的 4096 倍，
其他请求的 ITL（逐 token 延迟）会被打爆。用 `max_prefill_tokens`
（= vLLM 的 `max_num_batched_tokens`）切块摊平：单步峰值 157 → 16 token。

**分块正确性靠两点**（易错，一定要主动说）：
1. 位置编号必须是**绝对位置**（跨块连续，第 2 块从 `take` 开始）
2. 后续块的 query 要能看到前面块 —— 前面块已在池里，gather 天然满足

### Q8. Triton kernel 怎么写？为什么快？

`engine/triton_paged.py`，每个 program 负责 **(一条序列, 一个 query 块, 一个 head)**：

```python
载入 query 块 [BLOCK_M, D]
for start_n in range(0, total_k, BLOCK_N):
    n_idx = start_n + offs_n
    blk  = load(BT + seq*stride + n_idx // BLOCK_SIZE)   # ← 查页表，分页的全部魔法
    slot = n_idx % BLOCK_SIZE
    k = load(K + blk*stride_kb + slot*stride_ks + head*stride_kh + offs_d)
    s = dot(q, trans(k), input_precision="ieee") * scale
    s = where(n_idx <= m_pos, s, -inf)                   # 绝对位置因果掩码
    m_new = maximum(m_i, max(s, 1))                      # online softmax
    alpha = exp(m_i - m_new)
    l_i = l_i * alpha + sum(exp(s - m_new), 1)
    acc = acc * alpha + dot(exp(s - m_new), v)
    m_i = m_new
```

**为什么快**：PyTorch 版每层要起十来个 kernel + 一圈 Python 循环
（gather → einsum → softmax → einsum，× 序列 × query 位置）；
Triton 版每层只剩 **2 个**（1 写 + 1 算），中间量留在寄存器里。

端到端 565 → **301 ms（1.88x）**。

**两个踩坑（讲出来加分）**：
1. `tl.dot` 要求 K ≥ 16 —— 我测试用的 `head_dim=8` 直接编译失败
2. `BLOCK_M > n_q` 时无效行会被因果掩码整行掩掉 → max = -inf → **NaN**；
   得把无效行的位置钳到最后一个有效位置

---

## 3. 四个王牌故事（面试官最容易记住的部分）

### 故事 A：**差 1 个 token 的 bug**（证明你能发现深层 bug）

M3 的 `cached_generate` 用 `radix.insert(ids + generated, cache)` 记账，
但 cache 实际只覆盖到 `generated[:-1]`（最后一个 token 的 K/V 要等下一次 forward 才进 cache）。

树节点因此**多认领了 1 个 token**；后续请求"完整命中"时 `clone_cache_prefix`
会**静默截断**（张量切片不报错），丢掉最后一个 token 的 K/V → 输出错。

**证据**：`cached [1,2,3,7,4,7,1,2]` vs `naive [1,2,3,7,4,7,5,0]`
**修复**：`insert(ids + generated[:-1], cache)` + 补 `test_full_prefix_match_extends`

**为什么值钱**：同时证明三件事
1. 能发现深层 bug
2. **理解测试盲区**：现有测试的共享前缀都是 *partial hit*（`[1,2,3,4,5]` vs `[1,2,3,4,9]`
   只命中 4 个），`use` 远小于 cache 长度 → 截断不触发。**没有任何测试让新请求"完整命中并继续延长"**
3. 会诚实复盘

> 同类盲区还出现过两次：M1 的 `kv_generate` prefill 步缺 EOS 判定（测了 EOS 在第 2 个
> token，没测第 1 个）；Step 4 的 `bases` 传了整张表而非活跃子集（槽位 0 先完成时静默错位）。
> **教训：「与 X 逐 token 一致」的实现，盲区总在边界那一步。**

### 故事 B：**把 1.25x 拆成两半，分两步修完**（证明你会定位问题）

M2.5 的连续准入：理想 2.00x，实测 **1.25x**。
我没有停在"可能是有开销吧"，而是把差距拆成两笔可验证的账：

| 代价 | M2.5 | Step 4（分页） | M6（合流） |
|---|---|---|---|
| 补入时的**填充计算** | 每个补入前向要算 S 个位置 | ✅ 只算该请求自己的 L | ✅ 只算 L |
| 补入需**一次独立前向** | 7 次 | ❌ 仍是 7 次 | ✅ **0 次** |
| 喂入 token | 1463 | **222（6.6x↓）** | 222 |
| 总前向次数 | 37 | 37 | **30** |

> 面试时这句是加分点：**"这个坑我没有一次修完，而是拆成两半分别验证"** ——
> 比"我做了一个优化拿到 2 倍加速"可信得多。

### 故事 C：**墙钟反而慢了 2 倍 → 写了 kernel**（证明你能闭环）

分页 + 合流之后：喂入 token 降 **6.6 倍**，墙钟却从 306ms 涨到 624ms。

拆开看：逐序列注意力是 Python 循环，每（序列 × query 位置 × 层）
约 10 次 kernel 启动 + 一段 Python。12 层 × 30 步 ≈ **7000 次启动**，全是 CPU 时间。

→ 于是写了 Triton kernel，端到端 565 → **301 ms**，追平 M2.5 而 token 数低 6.6 倍。

> **这条弧线（先量出问题 → 再针对性写 kernel）比你直接说"我写了 Triton kernel"
> 强 10 倍。** 因为它证明 kernel 不是"我想学 Triton"，而是"我量化出了瓶颈"。

### 故事 D：**发现并修正了自己 benchmark 的系统性偏差**（证明测量严谨）

**现象**：M2 batching 只有 **1.18x** —— 看起来"批量几乎没用"。
但我先做了个反向校验：直接测单次前向，B=1 到 B=8 耗时几乎不变
（~6 ms，且 GPU 时间 ≈ 墙钟）—— 硬件完全支持近线性加速，**1.18x 说不通**。

**排查**：我把 `batched_generate` 的单步拆成 6 个阶段分别计时
（建 frozen 张量 / `torch.where` / `cat mask` / `position_ids` / forward / `synchronize`），
排除了"逐 token Python 开销"的嫌疑。最后发现问题**不在代码，在测量方法**：

`verify_m2.py` 让**批量路径跑第一个**，于是 CUDA 上下文创建、cuBLAS handle、
kernel 编译的冷启动开销**全部落在它头上**。

**物证**：

| | 首次（冷） | 预热后（best of 3） |
|---|---|---|
| batched | **581 ms** | **183 ms** |

**修复**：预热 + best-of-3 → 收益 **3.0x**（553 vs 178 ms，三次重复 2.94–3.23x）。

> 对照：`bench.py` 里的 M0/M1 实验**有**预热，所以 4.51x 是可信的；
> `verify_m3/m4b/m5/m6` 也都有预热。**只有最早写的那一个脚本漏了**——
> 这也说明为什么"建立测量纪律"比"多做一个优化"重要。

**为什么值钱**：它证明**我会怀疑自己的数据**。
大多数候选人只会说"我测出 1.18x"，而我会追问"这个 1.18x 合理吗"，
然后定位到是自己的 benchmark 有偏差。**AI Infra 岗每天在做的事就是这个。**

---

## 4. 你的薄弱点会被打哪里（**必须提前想好**）

> 校准：以下几条是你自己确认过的基础薄弱项。**不要试图装懂**——
> 面试官追问两层就会露馅，然后**整份简历的可信度都受影响**。
> 正确做法是：**承认边界 + 展示你知道边界在哪 + 说清你打算怎么补**。

| 可能被问 | ❌ 别这样答 | ✅ 这样答 |
|---|---|---|
| **C++ 基础**（虚函数、指针、内存） | 硬答 | 「我的项目是 Python/Triton；C++ 我目前在补，能写基本的数据结构与 RAII，但还没写过生产级 C++ 项目」 |
| **CUDA 细节**（warp shuffle、bank conflict、occupancy） | 编 | 「我手写过的是 Triton，它把这些封装掉了。我知道 warp/共享内存/coalescing 的概念，但 CUDA C++ 我只到能读懂的程度」 |
| **为什么不用 CUDA C++ 写 kernel** | 尴尬 | 「Triton 让我把精力放在**访存模式和并行划分**上，这两点才是 PagedAttention 的核心。等我需要做 split-K 的细粒度同步时，会考虑下到 CUDA」 |
| **数据结构**（trie / 哈希表原理） | 含糊 | 「Radix 树我是**边学边写**的——这是我刻意补的短板，因为推理引擎里到处是数据结构（页表、前缀树、LRU）。我的实现做了简化：**不做节点分裂**，部分重叠直接放弃插入」 |
| **Attention 数学细节** | 跳过 | 你能讲：`softmax(QKᵀ/√d)V`、为什么除 √d（量级补偿）、因果掩码、多头是切最后一维。这些你**代码里都写过**，指着代码讲 |
| **模型结构**（RoPE / GQA / MoE） | 没听过 | 「GPT-2 是 learned positional embedding，所以我没实现 RoPE——我的 `position_ids` 接口正好是 RoPE 需要的那个位置，换模型时这块要改。GQA/MoE 我读过概念，没动手」 |
| **分布式 / TP / PP** | 装懂 | 「这是我**明确不打算做**的部分，我在 `official-vs-mine.md` 里写了：读懂原理为目标，不实作。因为单卡 + 小模型看不到收益」 |

**万能收尾句**（承认边界后必须接这句）：
> 「这个我不熟。不过我的项目做到过类似的处境——**先把它量化，再决定要不要动手**。
> 比如我发现分页后墙钟慢了 2 倍，量化出是 kernel 启动开销，才去写的 Triton。」

---

## 5. 必须能立刻说出的数字（闭卷默写）

| 数字 | 含义 | 出处 |
|---|---|---|
| **4.51x** | gpt2 696-token 的 KV Cache 加速 | `benchmark/results/m1_kv_long.json` |
| **1.95x** | Qwen2.5-0.5B 601-token（地板效应压缩） | `m1_kv_qwen_long.json` |
| **0.94x** | gpt2 5-token（**微负**） | `m1_kv_short.json` |
| **3.0x** | 4 请求 batching（理想 4.0x） | `benchmark/verify_m2.py` |
| **581 → 183 ms** | 同一份代码的冷启动 vs 预热后（**测量偏差的物证**） | `benchmark/verify_m2.py` 注释 |
| **1.25x**（理想 2.00x） | M2.5 连续准入 | `tests/test_batching.py::TestContinuous` |
| **0.82x** | M3 Radix 缓存（**负收益**） | `benchmark/verify_m3.py` |
| **1463 → 222（6.6x）** | 分页后喂入 token 数 | `benchmark/verify_m4b.py` |
| **1220 → 0** | 填充位置数 | `benchmark/verify_m4b.py` |
| **37 → 30** | M6 总前向次数；补入独立前向 **7 → 0** | `benchmark/verify_m6.py` |
| **565 → 301 ms（1.88x）** | M5 Triton kernel 端到端 | `benchmark/verify_m5.py` |
| **129** | 测试数（77 假模型/纯张量 + 52 真模型） | `pytest tests/ --collect-only` |
| **1791** | `engine/` 行数 | `wc -l engine/*.py` |

**成本模型（必须背）**：
```
每步耗时 = 权重搬运（固定，∝模型大小） + 序列计算（∝上下文长度）
优化收益 = 被省成分的成本 − 新增机制的开销
```

---

## 6. 5 天准备计划

每天 2–3 小时。**D1–D2 读代码，D3–D4 自问自答，D5 模拟面试。**

### D1：主干（不看细节，先把流程走通）
- [ ] 读 `engine/naive_decode.py`（100 行）—— 这是基线，一切对比的参照
- [ ] 读 `engine/kv_cache.py`（105 行）—— 注意 prefill 与 decode 共用一个循环
- [ ] 读 `engine/batching.py:64-100`（`pad_left` / `build_position_ids`）
- [ ] **闭卷默画** Q1 的流程图

### D2：分页与 kernel（难点）
- [ ] 读 `engine/paged_kv.py` —— 重点 `gather_layer`（L71）和 `scatter_kv`（L205）
- [ ] 读 `engine/model_forward.py:113` —— 前向只有 6 步，别怕
- [ ] 读 `engine/triton_paged.py` 的 kernel（L45–110）—— **逐行搞懂 online softmax**
- [ ] 读 `engine/paged_schedule.py:160`（`chunked_prefill_generate`）—— 混合批怎么组
- [ ] **跑一遍** `python -m benchmark.verify_m5`，对着输出讲一遍每个数字

### D3：数字与故事
- [ ] 背 §5 的数字表（**默写**，错一个补一遍）
- [ ] 把 §3 的三个故事**各讲一遍**（对着手机录音，听自己讲得顺不顺）
- [ ] 读 `docs/official-vs-mine.md` —— 面试问"和官方差距"直接照答

### D4：薄弱点演练
- [ ] 把 §4 表格**遮住答案**，自己说一遍
- [ ] 找 3 个你答不上的问题，去查清楚（只查 30 分钟，别陷进去）
- [ ] 准备**反问面试官**的问题（见 §8）

### D5：模拟面试
- [ ] 让我（或同学）从 §2 + §7 随机抽 10 题问你
- [ ] 完整跑一次 5 分钟自我介绍（掐表）
- [ ] 挑一个模块，对着 GitHub 仓库**边滚代码边讲**

---

## 7. 32 道自测题（遮住答案自问自答）

> 概念层 6 · 实现层 10 · **why 层 9**（最容易翻车）· 设计层 7

**概念层**
1. 为什么 decode 是 memory-bound？
2. `√head_dim` 为什么要除？
3. 左填充为什么是"左"不是"右"？
4. 为什么 pad 必须 mask 掉？不 mask 会怎样？
5. 显式 `position_ids` 解决什么问题？
6. prefill 和 decode 的本质区别是什么？

**实现层**
7. `gather_layer` 那三步分别是什么？为什么 reshape 就"逻辑连续"了？
8. 页表里最后一个 block 通常没写满，怎么处理？
9. `ensure_blocks` 为什么要"按需增长"而不是一次开够？
10. `scatter_kv` 为什么比逐 token 写快？
11. `BatchedPagedAttentionHook` 的 `cu` 数组是什么？（提示：`cu_seqlens`）
12. 钩子被 12 层各调用一次，为什么"写入位置"只能推进一次？
13. Triton kernel 的 grid 是几维？每个 program 负责什么？
14. online softmax 的 `m_i` / `l_i` / `acc` 各是什么？不这么做会怎样？
15. 为什么用 `input_precision="ieee"` 而不是默认的 TF32？
16. chunked prefill 里，第 2 个块的位置编号从几开始？

**why 层（最容易翻车）**
17. 为什么 KV Cache 在短 prompt 上是负收益？
18. 为什么 Qwen 的收益比 gpt2 小？
19. Radix 缓存为什么实测 0.82x？
20. 连续准入为什么只有 1.25x 而不是理想的 2.00x？
21. 分页后喂入 token 降 6.6 倍，为什么墙钟反而慢 2 倍？
22. 为什么 Triton 微基准的加速比不随 seq_len 单调？
23. 你的 kernel 下一个瓶颈在哪？为什么？
24. **你怎么保证你的 benchmark 数字可信？** （预热 / best-of-N / 单变量对照 / 反向校验）
25. **如果某个优化收益比你预期低很多，你怎么判断是代码问题还是测量问题？**

**设计层**
26. 为什么解码循环要和模型解耦？举一个它带来好处的具体例子。
27. 为什么非要把 `attention_fn` 做成可注入的钩子？
28. 为什么测试要用"上下文依赖的假模型"？
29. 你怎么保证优化后输出不变？"逐 token 一致"和"误差 < 1e-5"哪个更强？
30. 你的 Radix 缓存不做节点分裂，代价是什么？
31. 池子耗尽时你的实现怎么处理？真引擎呢？
32. 如果给你 4096-token 的 prompt 和 8GB 显存，你的实现会先在哪崩？

> 32 题答案都在代码与本次对话里。**答不上来的写在一张纸上**，那是你 D4 的复习清单。

---

## 8. 反问面试官（准备 3 个，展示你在想什么）

1. 「你们的推理引擎现在最大的性能瓶颈是在 GPU 侧还是 CPU 侧？在做什么优化？」
2. 「团队里有做 kernel 的吗？我这个项目下一步打算做 split-K，想听听实际工程里这块怎么做」
3. 「如果我进来，第一个月大概会从哪块入手？」（展示想干活，不是想学习）

⚠️ **不要问**：「贵公司的主要业务是什么？」（没做功课）、「加班多吗？」（第一轮别问）

---

## 9. 最后三条纪律

1. **不知道就说不知道**，然后接万能收尾句（§4）。装懂的代价是整份简历失信。
2. **每个数字都要能说出处**。说不出来就别写进简历（见 `resume-project.md` §6）。
3. **主动讲负结果 / 反常现象**。你有 4 个（0.94x / 0.82x / 墙钟反慢 2 倍 / 自己 benchmark 的冷启动偏差）——
   这是你**最强的差异化**，因为绝大多数候选人的项目全是"我优化了 X 倍"。
