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

    用法（B=1 的 prefill 形态）::

        pool = PagedKVPool(num_blocks=32, block_size=4, num_layers=12,
                           num_heads=12, head_dim=64)
        bt = allocate_for(pool, seq_len)
        logits = gpt2_forward(ids, weights,
                              attention_fn=PagedAttentionHook(pool, bt, seq_len))

    数学与连续版**完全一样**，只是 K/V 中途绕了一圈池子::

        写进池（按页表定位） → 按页表 gather → 逐 query 的 paged_attention

    所以结果必须逐元素一致 —— 这就是 PagedAttention 的"无损"声明。

    约定：q/k/v 为 [B, H, S, Dh]；本实现只支持 B=1（批处理留到 M4b Step 4）。
    """

    def __init__(self, pool: PagedKVPool, block_table: list, seq_len: int):
        self.pool = pool
        self.block_table = block_table
        self.seq_len = seq_len
        self.kv_writes = 0        # 统计：写进池的 K/V 条数（每层每 token 各 1 条）

    def __call__(self, q, k, v, layer_idx: int):
        B, _, S, _ = k.shape
        assert B == 1, "PagedAttentionHook 目前只支持单序列（批处理见 M4b Step 4）"
        assert S == self.seq_len, \
            "本步骤只覆盖 prefill（S == seq_len）；增量解码见 M4b Step 3"
        bs = self.pool.block_size

        # ① 把本层的 K/V 写进分页池：逻辑位置 t → block_table[t // bs] 的第 t % bs 个槽
        for t in range(S):
            self.pool.write(layer_idx, self.block_table[t // bs], t % bs,
                            k[0, :, t, :], v[0, :, t, :])
            self.kv_writes += 1

        # ② 按页表 gather 成"逻辑连续"的 K/V（[S, H, Dh]）
        k_seq, v_seq = self.pool.gather_layer(layer_idx, self.block_table, S)

        # ③ 逐 query 算注意力；query i 只能看前 i+1 个 key（因果）
        outs = [paged_attention(q[0, :, i, :], k_seq[: i + 1], v_seq[: i + 1])
                for i in range(S)]
        return torch.stack(outs, dim=1).unsqueeze(0)       # [1, H, S, Dh]
