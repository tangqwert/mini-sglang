"""engine.paged_kv — M4a：分页 KV 池 + 分页注意力（PagedAttention 的地基）。

═══════════════════════════════════════════════════════════════════
 你的任务：实现 gather_layer 与 paged_attention，使 tests/test_paged_kv.py 全绿。
 核心契约：分页存储 + 按页表 gather 后做注意力，结果与连续存储的
 参考实现【逐元素一致】（误差 < 1e-5）。
═══════════════════════════════════════════════════════════════════

背景知识（M3 的 0.82x → M4 的解药）：
  M3 的 KV 继承要 clone 整块 DynamicCache（拷贝 + 24 次内核启动），吃掉收益。
  分页方案：KV 写进预分配的"页池"，每条请求只持有【页表】（block id 列表）：
    - 继承 = 复制页表（几个整数）→ 零拷贝
    - 注意力读 K/V 时按页表 gather（访存模式变了，数学没变）

  逻辑位置 → 物理位置的映射：
    seq_len=35, block_size=16, block_table=[7, 2, 5]
      位置 0-15  → block 7 的槽 0-15
      位置 16-31 → block 2 的槽 0-15
      位置 32-34 → block 5 的槽 0-2   （最后一个块通常没写满）

  为什么这值得手写：vLLM 的 PagedAttention CUDA kernel 做的就是
  "按页表 gather + 本文件这个注意力数学"——M4a 用 PyTorch 写正确版，
  M5 把它翻译成 Triton kernel。正确性现在验证，性能那时兑现。

设计约束：
  - 池子预分配、一次成型（教学版不做动态扩容）。
  - num_kv_heads == num_q_heads（GPT-2 没有 GQA，简化）。
  - 写满一个 block 才领下一个（allocate/free 由调用方——M4b 的调度器——管理）。
"""
import math

import torch
import torch.nn.functional as F


class PagedKVPool:
    """分页 KV 池：所有序列的 K/V 都住在这一块预分配的大张量里。

    keys/values 形状: [num_layers, num_blocks, block_size, num_heads, head_dim]
    一条序列占用若干 block，由它的 block_table（block id 列表）描述。
    """

    def __init__(self, num_blocks: int, block_size: int,
                 num_layers: int, num_heads: int, head_dim: int,
                 device="cpu"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.keys = torch.zeros(num_layers, num_blocks, block_size,
                                num_heads, head_dim, device=device)
        self.values = torch.zeros_like(self.keys)
        self.free_blocks = list(range(num_blocks))

    def allocate(self) -> int:
        """领一个空 block，返回 block id。池子耗尽抛异常（真引擎此时会抢占/驱逐）。"""
        if not self.free_blocks:
            raise RuntimeError("KV 池已满——真引擎会在这里做抢占/驱逐")
        return self.free_blocks.pop()

    def free(self, block_id: int):
        """归还 block（序列结束/被驱逐时）。"""
        self.free_blocks.append(block_id)

    def write(self, layer: int, block_id: int, slot: int, k: torch.Tensor, v: torch.Tensor):
        """把一个 token 的 K/V（[num_heads, head_dim]）写进指定槽位。"""
        self.keys[layer, block_id, slot] = k
        self.values[layer, block_id, slot] = v

    def gather_layer(self, layer: int, block_table: list, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按页表把散落各 block 的 K/V 拼成"逻辑连续"的两段。

        这是分页注意力的关键动作（真 vLLM 里这步在 CUDA kernel 内部完成）：
          pool[layer, block_table]          → [n_blocks, block_size, H, D]
          展平成 [n_blocks * block_size, H, D]
          再截取前 seq_len 行                → [seq_len, H, D]
        （最后一个 block 可能没写满，截断即可——block_table 里全是该序列的块，
          顺序即逻辑顺序。）

        Returns:
            (k_seq, v_seq)：各 [seq_len, num_heads, head_dim]
        """
        # ① 按页表索引：从池子里挑出该序列占用的那些 block
        k_blocks = self.keys[layer][block_table]        # [n_blocks, block_size, H, D]
        v_blocks = self.values[layer][block_table]
        # ② 展平：block_table 的顺序就是逻辑顺序、块内槽位也是顺序的，
        #    所以直接 reshape 就拼成了"逻辑连续"的一长条
        k_seq = k_blocks.reshape(-1, self.num_heads, self.head_dim)
        v_seq = v_blocks.reshape(-1, self.num_heads, self.head_dim)
        # ③ 截断：最后一个 block 通常没写满，只取前 seq_len 行
        return k_seq[:seq_len], v_seq[:seq_len]


def paged_attention(q: torch.Tensor, k_seq: torch.Tensor, v_seq: torch.Tensor) -> torch.Tensor:
    """单 query 对一段 K/V 的注意力（缩放点积，多头各自独立）。

    Args:
        q:     [num_heads, head_dim]    当前 token 的 query
        k_seq: [seq_len, num_heads, head_dim]   gather 出来的历史 K
        v_seq: [seq_len, num_heads, head_dim]   gather 出来的历史 V

    Returns:
        [num_heads, head_dim]  注意力输出（每个 head 独立算，互不掺和）

    提示（数学 = M0 老三样，只是多了 head 维）:
        1. scores[h, t] = q[h] · k_seq[t, h] / sqrt(head_dim)
           ——einsum 或逐 head 循环都行；einsum 写法：
           torch.einsum('hd,thd->ht', q, k_seq)
        2. 每个 head 独立 softmax（dim=-1）
        3. out[h] = Σ_t w[h,t] * v_seq[t, h]    —— einsum('ht,thd->hd', w, v_seq)
        缩放因子 sqrt(head_dim)：量级补偿，GPT-2 原生注意力也这么做。
    """
    # ① 每个 head 独立打分：scores[h, t] = q[h] · k_seq[t, h] / sqrt(D)
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("hd,thd->ht", q, k_seq) * scale      # [H, T]
    # ② 每个 head 独立 softmax（dim=-1 是序列维 t）
    w = torch.softmax(scores, dim=-1)                          # [H, T]
    # ③ 加权求和
    return torch.einsum("ht,thd->hd", w, v_seq)                # [H, D]


# ─────────────────── M4b：把分页池接进自研前向 ───────────────────

def allocate_for(pool: PagedKVPool, seq_len: int) -> list:
    """给一条长度为 seq_len 的序列领足够的 block，返回它的页表（block id 列表）。

    每 ⌈seq_len / block_size⌉ 个 token 占一个 block。
    同一张页表对【所有层】通用 —— 因为池子的第一维就是层，block_id 在层维度之后索引。
    """
    n_blocks = -(-seq_len // pool.block_size)          # 向上取整
    return [pool.allocate() for _ in range(n_blocks)]


class PagedAttentionHook:
    """分页版 attention_fn —— 它是 M4a 与真实前向之间的桥。

    用法（prefill + 增量 decode）::

        pool = PagedKVPool(num_blocks=32, block_size=4, num_layers=12,
                           num_heads=12, head_dim=64)
        hook = PagedAttentionHook(pool, allocate_for(pool, capacity))

        logits = gpt2_forward(prompt, weights, attention_fn=hook)          # prefill
        logits = gpt2_forward(tok, weights, attention_fn=hook,
                              position_ids=torch.tensor([[T]]))            # decode

    数学与连续版**完全一样**，只是 K/V 中途绕了一圈池子::

        写进池（按页表定位） → 按页表 gather → 逐 query 的 paged_attention

    写入位置的管理：一次 forward 会逐层调用本钩子，所以位置只在
    `layer_idx == 0` 时推进一次（每层都推进就会写乱）。
    钩子不关心"这是 prefill 还是 decode"，只看这次喂进来几个 token。

    约定：q/k/v 为 [B, H, S, Dh]；本实现只支持 B=1（批处理留到 M4b Step 4）。
    """

    def __init__(self, pool: PagedKVPool, block_table: list, start: int = 0):
        self.pool = pool
        self.block_table = block_table
        self.written = start      # 已写入池的 token 数 = 下一个待写位置
        self._base = start        # 本次 forward 的写入起点
        self.kv_writes = 0        # 统计：写进池的 K/V 条数（每层每 token 各 1 条）

    def __call__(self, q, k, v, layer_idx: int):
        B, _, S, _ = k.shape
        assert B == 1, "PagedAttentionHook 目前只支持单序列（批处理见 M4b Step 4）"
        if layer_idx == 0:                     # 只在第 0 层推进写入窗口
            self._base = self.written
            self.written += S
        base, total, bs = self._base, self.written, self.pool.block_size
        assert len(self.block_table) * bs >= total, \
            f"页表只给了 {len(self.block_table)} 个 block（容量 {len(self.block_table) * bs}），装不下 {total} 个位置"

        # ① 把本层的 K/V 写进分页池：绝对位置 p → block_table[p // bs] 的第 p % bs 个槽
        for t in range(S):
            p = base + t
            self.pool.write(layer_idx, self.block_table[p // bs], p % bs,
                            k[0, :, t, :], v[0, :, t, :])
            self.kv_writes += 1

        # ② 按页表 gather 出【全部历史】的 K/V（[total, H, Dh]）
        k_seq, v_seq = self.pool.gather_layer(layer_idx, self.block_table, total)

        # ③ 逐 query 算注意力：query 的绝对位置是 base+i，只能看 key 的 0..base+i
        outs = [paged_attention(q[0, :, i, :],
                                k_seq[: base + i + 1], v_seq[: base + i + 1])
                for i in range(S)]
        return torch.stack(outs, dim=1).unsqueeze(0)       # [1, H, S, Dh]


# ─────────────── M4b Step 4：多序列（varlen）分页注意力 ───────────────

def ensure_blocks(pool: PagedKVPool, block_table: list, total_len: int) -> None:
    """确保页表装得下 total_len 个 token；不够就向池子领新块（页表原地增长）。

    真引擎也是这么做的：显存按需分页增长，而不是一次性预留"最长可能长度"。
    所以"分页省显存"的另一半兑现方式 = 分页表随序列变长而变长。
    """
    need = -(-total_len // pool.block_size)              # 向上取整
    while len(block_table) < need:
        block_table.append(pool.allocate())

def scatter_kv(pool: PagedKVPool, layer: int, table_t: torch.Tensor,
               base: int, k: torch.Tensor, v: torch.Tensor) -> None:
    """把一段连续的 K/V 一次性写进分页池（向量化，M5 第①级）。

    逐 token 的 Python 循环在 prefill 时是灾难：一条 153-token 的 prompt
    每层要转 153 次 × 12 层 —— 纯粹的 CPU 开销，跟算力无关。
    这里换成一次高级索引赋值（底层就是一个 index_put_ kernel）：

        绝对位置 p → 块 table_t[p // bs] 的第 p % bs 个槽位

    Args:
        table_t: 该序列的页表（int64 张量，已在 device 上）
        base: 这段 K/V 的起始【绝对位置】
        k, v: [n, num_heads, head_dim] —— 只含该序列这一段，且已转成
              (token, head, dim) 布局
    """
    n = k.shape[0]
    if n == 0:
        return
    pos = torch.arange(base, base + n, device=k.device)
    blk = table_t[pos // pool.block_size]
    slot = pos % pool.block_size
    pool.keys[layer, blk, slot] = k
    pool.values[layer, blk, slot] = v

class BatchedPagedAttentionHook:
    """多序列（varlen / ragged batch）分页注意力钩子。

    输入是【拍平】的一维批次：N 条序列本次要算的 token 首尾相接成
    ``[1, H, S_total, Dh]``（真引擎里就是 flash-attn 的 varlen 布局）。
    每条序列有自己的页表与写入起点，注意力逐序列独立计算。

    ⇒ **完全不需要左填充**：既没有 pad 占显存，也没有 pad 白算。
      这正是抹掉 M2.5 那笔账（填充碎片 + "补入得填充到当前批长 S"）的地方。

    真引擎对应物是 ``flash_attn_varlen_func`` 的 ``cu_seqlens``；本类用
    Python 循环 + 分页 gather 实现同一语义（M5 再翻译成 kernel）。

    Args:
        pool: PagedKVPool。
        block_tables: 每条序列的页表（调用方需保证容量，见 ensure_blocks）。
        bases: 每条序列【本次 forward 之前】已写入池的 token 数。
        new_lens: 每条序列【本次 forward】要算的 token 数（decode 步全为 1）。
    """

    def __init__(self, pool: PagedKVPool, block_tables: list,
                 bases: list, new_lens: list):
        self.pool = pool
        self.block_tables = list(block_tables)
        self.base = list(bases)
        self.new_lens = list(new_lens)
        self.n_seq = len(self.new_lens)
        # 三个列表必须按【同一个序列顺序】一一对应 —— 少一个就会把 A 的起点
        # 配到 B 的页表上（写入位置静默错位，结果全错）。
        assert len(self.block_tables) == self.n_seq == len(self.base), (
            f"参数不对齐：block_tables={len(self.block_tables)}, "
            f"bases={len(self.base)}, new_lens={self.n_seq}")
        self.written = [b + n for b, n in zip(self.base, self.new_lens)]
        # 页表转成张量（S× 只做一次，而不是每层每 token 再转）
        self.bt = [torch.as_tensor(t, dtype=torch.long, device=pool.keys.device)
                   for t in self.block_tables]
        # 每条序列在拍平维度上的起点（cu_seqlens）
        self.cu = [0]
        for n in self.new_lens:
            self.cu.append(self.cu[-1] + n)
        self.kv_writes = 0            # 统计：写进池的 K/V 条数（每层每 token 各 1）

    def __call__(self, q, k, v, layer_idx: int):
        B, _, S_total, _ = k.shape
        assert B == 1 and S_total == self.cu[-1], \
            f"拍平长度对不上：前向给了 {S_total}，页表声明 {self.cu[-1]}"
        bs = self.pool.block_size

        # ① 写：各序列的新 token 落到各自页表的对应槽位（向量化，一次索引赋值/序列）
        #    （绝对位置 p → 第 p // bs 块的第 p % bs 槽；每层都写）
        for s in range(self.n_seq):
            b, n, tbl = self.base[s], self.new_lens[s], self.block_tables[s]
            assert len(tbl) * bs >= b + n, \
                f"序列 {s} 页表容量不足（{len(tbl) * bs} < {b + n}）：请先 ensure_blocks"
            a = self.cu[s]
            scatter_kv(self.pool, layer_idx, self.bt[s], b,
                       k[0, :, a:a + n, :].transpose(0, 1),
                       v[0, :, a:a + n, :].transpose(0, 1))
            self.kv_writes += n

        # ② 逐序列 gather，再逐 query 算注意力
        #    （query 的绝对位置 base+t 只能看到 key 的 0..base+t —— 因果性）
        outs = []
        for s in range(self.n_seq):
            b, n, tbl = self.base[s], self.new_lens[s], self.block_tables[s]
            k_seq, v_seq = self.pool.gather_layer(layer_idx, tbl, b + n)
            for t in range(n):
                outs.append(paged_attention(q[0, :, self.cu[s] + t, :],
                                            k_seq[: b + t + 1],
                                            v_seq[: b + t + 1]))
        # 拍平顺序 = 序列顺序 × 序列内顺序，与输入对齐
        return torch.stack(outs, dim=1).unsqueeze(0)       # [1, H, S_total, Dh]
