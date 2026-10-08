"""engine.triton_paged — M5：把分页注意力写成一个 Triton kernel。

═══════════════════════════════════════════════════════════════════
 为什么需要 M5（动机不是"我想学 Triton"，是被量出来的）：
   M4a/M4b 的分页注意力正确，但**慢**。`BatchedPagedAttentionHook` 里那三层
   Python 循环，对每个 (序列 × query 位置 × 层) 都要：
       gather → einsum(打分) → softmax → einsum(加权) → 写回
   即约 10 次 kernel 启动 + 一段 Python 代码。12 层 × 2 序列 × 30 步 ≈ 7000 次
   启动，全是 CPU 时间。实测结果：喂入 token 数比 M2.5 降了 6.6 倍，**墙钟反而慢
   2 倍**（306 → 624 ms）。这就 M5 要省的东西。

 本文件做的事：把"查页表 gather + 因果掩码 + online softmax + 加权求和"
 全部封进【一个】kernel。配合已向量化的 KV 写入（scatter_kv），
 分页路径每层只剩两个 kernel 启动：1 个写 + 1 个算。

 数学与 `paged_attention` 完全一致（严格 fp32，用 input_precision="ieee"），
 差异只在访存与调度方式 —— 所以输出必须逐 token 一致。
═══════════════════════════════════════════════════════════════════

kernel 结构（＝ 分页版 flash attention / 真 vLLM 的 paged_attention）：

    每个 program 负责 (一条序列, 一个 query 块, 一个 head)
      ├─ 载入 query 块 [BLOCK_M, D]
      ├─ 沿 KV 方向以 BLOCK_N 为步长循环：
      │    n_idx → 页表查到 block id 与槽位 → 载入 K/V 块（跨块自动拼接）
      │    online softmax（m_i / l_i / acc 三元组滚动更新，无需物化整个 scores）
      │   因果掩码：key 的绝对位置 ≤ query 的绝对位置
      └─ 归一化后写回
"""
import torch

from engine.paged_kv import PagedKVPool, scatter_kv

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:                                   # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _paged_attn_kernel(
        Q, K, V, Out,                       # Q/Out: [H, S_total, D] 视图；K/V: [NB, BS, H, D]
        BT, CuQ, Base, TotalK, scale,
        stride_qh, stride_qs,
        stride_kb, stride_ks, stride_kh,
        stride_oh, stride_os,
        stride_bt,
        H: tl.constexpr, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        pid_s = tl.program_id(0)            # 第几条序列
        pid_m = tl.program_id(1)            # 第几个 query 块
        pid_h = tl.program_id(2)            # 第几个 head

        cu = tl.load(CuQ + pid_s)                       # 该序列在拍平维度的起点
        n_q = tl.load(CuQ + pid_s + 1) - cu             # 本次要算几个 query
        base = tl.load(Base + pid_s)                    # 首个 query 的绝对位置
        total_k = tl.load(TotalK + pid_s)               # 可用 key 总数（= base + n_q）

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        m_mask = offs_m < n_q
        # 无效行（offs_m >= n_q）的位置钳到最后一个有效位置：
        # 否则这些行会被因果掩码整行掩掉 → max 为 -inf → exp 出 NaN。
        # （这些行的结果本来就不写回，但 NaN 会污染 tl.max 的规约路径。）
        m_pos = base + tl.minimum(offs_m, n_q - 1)

        q = tl.load(Q + pid_h * stride_qh + (cu + offs_m)[:, None] * stride_qs
                      + offs_d[None, :],
                    mask=m_mask[:, None], other=0.0)

        m_i = tl.full([BLOCK_M], float('-inf'), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)

        offs_n = tl.arange(0, BLOCK_N)
        for start_n in range(0, total_k, BLOCK_N):
            n_idx = start_n + offs_n
            n_ok = n_idx < total_k
            # ① 查页表：逻辑位置 → (block id, 槽位) —— 分页的全部魔法就这一行
            blk = tl.load(BT + pid_s * stride_bt + n_idx // BLOCK_SIZE,
                          mask=n_ok, other=0)
            slot = n_idx % BLOCK_SIZE
            k = tl.load(K + blk[:, None] * stride_kb + slot[:, None] * stride_ks
                          + pid_h * stride_kh + offs_d[None, :],
                        mask=n_ok[:, None], other=0.0)
            v = tl.load(V + blk[:, None] * stride_kb + slot[:, None] * stride_ks
                          + pid_h * stride_kh + offs_d[None, :],
                        mask=n_ok[:, None], other=0.0)

            # ② 打分（严格 fp32，不开 TF32 —— 我们要与参考实现逐元素对齐）
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            causal = n_idx[None, :] <= m_pos[:, None]
            s = tl.where(causal & n_ok[None, :], s, float('-inf'))

            # ③ online softmax：不物化整个 scores，边扫边纠正历史统计量
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)                 # 旧统计量的重新缩放因子
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(Out + pid_h * stride_oh + (cu + offs_m)[:, None] * stride_os
                     + offs_d[None, :],
                 acc, mask=m_mask[:, None])


class TritonPagedAttentionHook:
    """与 `BatchedPagedAttentionHook` 同接口、同数学，但每层只启动 2 个 kernel。

    用法与多序列钩子完全一致（`gpt2_forward(..., attention_fn=hook)`），
    所以可以当作 step 4 / M6 调度器的可替换零件注进去。

    Args:
        pool, block_tables, bases, new_lens: 同 BatchedPagedAttentionHook。
        BLOCK_M: query 方向分块（decode 时 n_q=1，只有首行有效）。
        BLOCK_N: KV 方向分块（online softmax 的步长）。
    """

    def __init__(self, pool: PagedKVPool, block_tables: list,
                 bases: list, new_lens: list,
                 block_m: int = 16, block_n: int = 64):
        assert HAS_TRITON, "未安装 triton，无法使用 TritonPagedAttentionHook"
        self.pool = pool
        self.block_tables = list(block_tables)
        self.base = list(bases)
        self.new_lens = list(new_lens)
        self.n_seq = len(self.new_lens)
        assert len(self.block_tables) == self.n_seq == len(self.base)
        assert pool.head_dim >= 16, \
            f"head_dim={pool.head_dim} 太小：tl.dot 要求 K 维至少 16"
        self.block_m, self.block_n = block_m, block_n

        dev = pool.keys.device
        # 这些元数据在一次 forward 内不变，构造时整理好，避免每层重复搭建
        self.cu = [0]
        for n in self.new_lens:
            self.cu.append(self.cu[-1] + n)
        self.cu_t = torch.tensor(self.cu, dtype=torch.int32, device=dev)
        self.base_t = torch.tensor(self.base, dtype=torch.int32, device=dev)
        self.totalk_t = torch.tensor(
            [b + n for b, n in zip(self.base, self.new_lens)],
            dtype=torch.int32, device=dev)
        width = max(len(t) for t in self.block_tables)
        bt = torch.zeros(self.n_seq, width, dtype=torch.int32, device=dev)
        for i, t in enumerate(self.block_tables):
            bt[i, :len(t)] = torch.as_tensor(t, dtype=torch.int32, device=dev)
        self.bt_t = bt
        self.kv_writes = 0

    def __call__(self, q, k, v, layer_idx: int):
        B, H, S_total, Dh = k.shape
        assert B == 1 and S_total == self.cu[-1], \
            f"拍平长度对不上：前向给了 {S_total}，钩子声明 {self.cu[-1]}"

        # ① 写（向量化 scatter，每个序列一个 index_put_ kernel）
        for s in range(self.n_seq):
            b, n = self.base[s], self.new_lens[s]
            a = self.cu[s]
            scatter_kv(self.pool, layer_idx, torch.as_tensor(
                self.block_tables[s], dtype=torch.long, device=k.device), b,
                k[0, :, a:a + n, :].transpose(0, 1),
                v[0, :, a:a + n, :].transpose(0, 1))
            self.kv_writes += n

        # ② 一次 kernel 算完全部序列 × 全部 query × 全部 head
        out = torch.empty_like(q)
        keys = self.pool.keys[layer_idx]          # [NB, BS, H, D]
        vals = self.pool.values[layer_idx]
        qv, ov = q[0], out[0]                     # [H, S_total, D]
        max_q = max(self.new_lens)
        grid = (self.n_seq, triton.cdiv(max_q, self.block_m), H)
        _paged_attn_kernel[grid](
            qv, keys, vals, ov,
            self.bt_t, self.cu_t, self.base_t, self.totalk_t,
            Dh ** -0.5,
            qv.stride(0), qv.stride(1),
            keys.stride(0), keys.stride(1), keys.stride(2),
            ov.stride(0), ov.stride(1),
            self.bt_t.stride(0),
            H=H, D=Dh,
            BLOCK_M=self.block_m, BLOCK_N=self.block_n,
            BLOCK_SIZE=self.pool.block_size,
        )
        return out
