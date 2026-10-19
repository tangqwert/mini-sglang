"""M4a 验收测试：分页 KV 池 + 分页注意力。

跑法：pytest tests/test_paged_kv.py -x
核心契约：分页存储 + 按页表 gather 的注意力 == 连续存储的参考实现（逐元素一致）。
这条契约就是 PagedAttention 的"无损"声明——物理布局变了，数学结果不能变。
"""
import math

import pytest
import torch

from engine.paged_kv import (BatchedPagedAttentionHook, PagedAttentionHook,
                             PagedKVPool, allocate_for, ensure_blocks,
                             paged_attention)


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """参考实现：最朴素的逐 head 循环注意力（正确但慢，仅测试用）。

    q [H, D]，k/v [T, H, D] → [H, D]
    """
    H, D = q.shape
    out = torch.empty(H, D)
    for h in range(H):
        scores = (k[:, h] @ q[h]) / math.sqrt(D)      # [T]
        w = torch.softmax(scores, dim=-1)             # [T]
        out[h] = w @ v[:, h]                          # [D]
    return out


class TestPool:
    def test_alloc_free_bookkeeping(self):
        pool = PagedKVPool(num_blocks=4, block_size=8, num_layers=2,
                           num_heads=2, head_dim=4)
        b0, b1, b2 = pool.allocate(), pool.allocate(), pool.allocate()
        assert len(pool.free_blocks) == 1
        pool.free(b1)
        assert len(pool.free_blocks) == 2 and b1 in pool.free_blocks

    def test_pool_exhaustion_raises(self):
        pool = PagedKVPool(num_blocks=1, block_size=8, num_layers=2,
                           num_heads=2, head_dim=4)
        pool.allocate()
        try:
            pool.allocate()
            assert False, "池子耗尽应抛 RuntimeError"
        except RuntimeError:
            pass


class TestGather:
    def test_single_block_roundtrip(self):
        """单块：写 5 个 token，gather 出来应与写入值一致。"""
        pool = PagedKVPool(num_blocks=2, block_size=8, num_layers=2,
                           num_heads=2, head_dim=4)
        b = pool.allocate()
        k_in = torch.randn(5, 2, 4)
        v_in = torch.randn(5, 2, 4)
        for t in range(5):
            pool.write(layer=0, block_id=b, slot=t, k=k_in[t], v=v_in[t])
        k_out, v_out = pool.gather_layer(layer=0, block_table=[b], seq_len=5)
        assert torch.allclose(k_out, k_in, atol=1e-6)
        assert torch.allclose(v_out, v_in, atol=1e-6)

    def test_multi_block_gather(self):
        """跨块（分页的灵魂）：10 个 token 写进 3 个 block，逻辑上必须连续。

        位置 0-3 → block b0，4-7 → b1，8-9 → b2（最后一块只写 2 槽）。
        """
        pool = PagedKVPool(num_blocks=4, block_size=4, num_layers=1,
                           num_heads=2, head_dim=4)
        blocks = [pool.allocate() for _ in range(3)]
        k_in = torch.randn(10, 2, 4)
        v_in = torch.randn(10, 2, 4)
        for t in range(10):
            pool.write(layer=0, block_id=blocks[t // 4], slot=t % 4,
                       k=k_in[t], v=v_in[t])

        k_out, v_out = pool.gather_layer(layer=0, block_table=blocks, seq_len=10)
        assert k_out.shape == (10, 2, 4)
        assert torch.allclose(k_out, k_in, atol=1e-6), "gather 顺序必须与逻辑位置一致"

        # 部分截断：只要前 9 个位置（最后一块少读 1 槽）
        k9, _ = pool.gather_layer(layer=0, block_table=blocks, seq_len=9)
        assert k9.shape == (9, 2, 4) and torch.allclose(k9, k_in[:9], atol=1e-6)

    def test_layers_are_independent(self):
        pool = PagedKVPool(num_blocks=2, block_size=4, num_layers=3,
                           num_heads=1, head_dim=2)
        b = pool.allocate()
        for layer in range(3):
            pool.write(layer=layer, block_id=b, slot=0,
                       k=torch.full((1, 2), float(layer)), v=torch.zeros(1, 2))
        k0, _ = pool.gather_layer(layer=0, block_table=[b], seq_len=1)
        k2, _ = pool.gather_layer(layer=2, block_table=[b], seq_len=1)
        assert k0[0, 0, 0] == 0 and k2[0, 0, 0] == 2, "各层的 K/V 互不串"


class TestPagedAttention:
    def test_matches_reference_single_head(self):
        torch.manual_seed(0)
        q = torch.randn(1, 8)
        k = torch.randn(6, 1, 8)
        v = torch.randn(6, 1, 8)
        assert torch.allclose(paged_attention(q, k, v),
                              reference_attention(q, k, v), atol=1e-5)

    def test_matches_reference_multi_head(self):
        torch.manual_seed(1)
        q = torch.randn(4, 16)
        k = torch.randn(9, 4, 16)
        v = torch.randn(9, 4, 16)
        assert torch.allclose(paged_attention(q, k, v),
                              reference_attention(q, k, v), atol=1e-5)

    def test_end_to_end_paged_vs_contiguous(self):
        """终极契约：同一个 token 的 K/V 走分页池 or 连续张量，注意力输出一致。

        这就是 PagedAttention 的无损声明——把 K/V 拆进 3 个 block，
        经 gather 再注意力，结果必须与连续存储时完全相同。
        """
        torch.manual_seed(2)
        T, H, D, BS = 10, 4, 16, 4
        q = torch.randn(H, D)
        k_in = torch.randn(T, H, D)
        v_in = torch.randn(T, H, D)

        # 分页路径：写入 3 个 block → gather → 注意力
        pool = PagedKVPool(num_blocks=4, block_size=BS, num_layers=1,
                           num_heads=H, head_dim=D)
        blocks = [pool.allocate() for _ in range(3)]
        for t in range(T):
            pool.write(layer=0, block_id=blocks[t // BS], slot=t % BS,
                       k=k_in[t], v=v_in[t])
        k_page, v_page = pool.gather_layer(layer=0, block_table=blocks, seq_len=T)
        out_paged = paged_attention(q, k_page, v_page)

        # 连续路径：参考实现
        out_contig = reference_attention(q, k_in, v_in)

        assert torch.allclose(out_paged, out_contig, atol=1e-5), \
            "分页布局改变了注意力结果——PagedAttention 的无损性被破坏"


class TestPagedAttentionHook:
    """M4b Step 2 的零件：领页表 + 把 K/V 路由过分页池。"""

    def test_allocate_for_rounds_up(self):
        pool = PagedKVPool(num_blocks=8, block_size=4, num_layers=1,
                           num_heads=2, head_dim=4)
        assert len(allocate_for(pool, 1)) == 1
        assert len(allocate_for(pool, 4)) == 1
        assert len(allocate_for(pool, 5)) == 2
        assert len(pool.free_blocks) == 8 - 4, "前三次共领走 4 个 block"

    def test_hook_writes_then_gathers_back(self):
        """钩子写进池的 K/V，按页表 gather 回来应与写进去的一致（只差转置）。"""
        torch.manual_seed(0)
        H, D, BS, S = 3, 8, 4, 10
        pool = PagedKVPool(num_blocks=8, block_size=BS, num_layers=2,
                           num_heads=H, head_dim=D)
        bt = allocate_for(pool, S)
        hook = PagedAttentionHook(pool, bt)
        q, k, v = (torch.randn(1, H, S, D) for _ in range(3))

        out = hook(q, k, v, layer_idx=1)          # 只写第 1 层
        assert out.shape == (1, H, S, D)
        assert hook.kv_writes == S

        k_seq, v_seq = pool.gather_layer(1, bt, S)
        assert torch.allclose(k_seq, k[0].transpose(0, 1), atol=1e-6)
        assert torch.allclose(v_seq, v[0].transpose(0, 1), atol=1e-6)

        k0, _ = pool.gather_layer(0, bt, S)
        assert torch.all(k0 == 0), "不该写到别的层"

    def test_causal_truncation_is_applied(self):
        """query 0 只能看 1 个 key —— 若因果截断丢了，结果会不同。

        只有 1 个 key 时 softmax 权重恒为 1，所以 query 0 的输出应等于 v[0]。
        """
        torch.manual_seed(1)
        H, D, S = 2, 4, 5
        pool = PagedKVPool(num_blocks=8, block_size=4, num_layers=1,
                           num_heads=H, head_dim=D)
        hook = PagedAttentionHook(pool, allocate_for(pool, S))
        q, k, v = (torch.randn(1, H, S, D) for _ in range(3))
        out = hook(q, k, v, layer_idx=0)
        assert torch.allclose(out[0, :, 0, :], v[0, :, 0, :], atol=1e-5)


class TestBatchedHook:
    """M4b Step 4：多序列（varlen）分页注意力 —— 批式索引的等价性。

    核心契约：把 N 条序列拍平成一次前向，结果必须与"每条各自单独跑"
    **逐元素相同**。这条契约成立，才敢说"批处理只是把多条塞进一次调用"。
    """

    def _make(self, lens, block_size, H=2, D=4, layers=2):
        pool = PagedKVPool(num_blocks=64, block_size=block_size,
                           num_layers=layers, num_heads=H, head_dim=D)
        tables = []
        for L in lens:
            t = []
            ensure_blocks(pool, t, L)
            tables.append(t)
        return pool, tables

    @pytest.mark.parametrize("lens", [[5, 3], [1, 1, 1], [7, 2, 4, 1], [4, 4]])
    def test_matches_per_sequence_hooks(self, lens):
        """多序列钩子 == 多条单序列钩子（逐元素一致）。"""
        torch.manual_seed(0)
        H, D, BS = 2, 4, 4
        pool, tables = self._make(lens, BS, H, D)
        S_total = sum(lens)
        q, k, v = (torch.randn(1, H, S_total, D) for _ in range(3))

        got = BatchedPagedAttentionHook(pool, tables, [0] * len(lens), lens)(
            q, k, v, layer_idx=0)

        # 参考：完全相同的张量切片，但每条序列走一遍单序列钩子（用另一个池子，
        # 否则两条路径会写进同一块显存，互相污染）
        pool2, tables2 = self._make(lens, BS, H, D)
        refs = []
        off = 0
        for s, L in enumerate(lens):
            sl = slice(off, off + L)
            refs.append(PagedAttentionHook(pool2, tables2[s])(
                q[:, :, sl, :], k[:, :, sl, :], v[:, :, sl, :], 0))
            off += L
        ref = torch.cat(refs, dim=2)

        assert got.shape == (1, H, S_total, D)
        assert torch.allclose(got, ref, atol=1e-6), \
            f"max|diff| = {(got - ref).abs().max().item()}"

    def test_causality_across_sequences_is_isolated(self):
        """序列之间不得串味：改掉后一条序列的 K/V，前一条的输出必须纹丝不动。"""
        torch.manual_seed(2)
        H, D, BS, lens = 2, 4, 4, [5, 3]
        S_total = sum(lens)

        pool, tables = self._make(lens, BS, H, D)
        q, k, v = (torch.randn(1, H, S_total, D) for _ in range(3))
        base = BatchedPagedAttentionHook(pool, tables, [0, 0], lens)(q, k, v, 0)

        pool2, tables2 = self._make(lens, BS, H, D)
        k2, v2 = k.clone(), v.clone()
        k2[:, :, lens[0]:, :] = torch.randn_like(k2[:, :, lens[0]:, :])
        v2[:, :, lens[0]:, :] = torch.randn_like(v2[:, :, lens[0]:, :])
        changed = BatchedPagedAttentionHook(pool2, tables2, [0, 0], lens)(q, k2, v2, 0)

        assert torch.allclose(base[:, :, :lens[0], :], changed[:, :, :lens[0], :],
                              atol=1e-6), "前一条序列不该被后一条影响"
        assert not torch.allclose(base[:, :, lens[0]:, :], changed[:, :, lens[0]:, :],
                                  atol=1e-6), "后一条改了 K/V，它自己必须变"

    def test_varlen_with_nonzero_bases(self):
        """decode 形态：base > 0、每条只算 1 个 token —— 只看自己那段历史。"""
        torch.manual_seed(3)
        H, D, BS = 2, 4, 4
        bases = [6, 2]                       # 两条序列池里已有 6 / 2 个 token
        pool = PagedKVPool(num_blocks=64, block_size=BS, num_layers=1,
                           num_heads=H, head_dim=D)
        tables = []
        for b in bases:
            t = []
            ensure_blocks(pool, t, b + 1)
            tables.append(t)

        # ① 先用单序列钩子把历史灌进去（含各自最后一步的 K/V）
        hist_k, hist_v = {}, {}
        for s, b in enumerate(bases):
            kk, vv = torch.randn(1, H, b, D), torch.randn(1, H, b, D)
            hist_k[s], hist_v[s] = kk, vv
            PagedAttentionHook(pool, tables[s])(torch.zeros(1, H, b, D), kk, vv, 0)

        # ② 多序列钩子各算 1 个新 token
        q = torch.randn(1, H, 2, D)
        k = torch.randn(1, H, 2, D)
        v = torch.randn(1, H, 2, D)
        got = BatchedPagedAttentionHook(pool, tables, bases, [1, 1])(q, k, v, 0)

        # ③ 参考：各取自己那段历史 + 自己的新 token，用 paged_attention 直接算
        for s, b in enumerate(bases):
            k_all = torch.cat([hist_k[s][0].transpose(0, 1), k[0, :, s, :].unsqueeze(0)])
            v_all = torch.cat([hist_v[s][0].transpose(0, 1), v[0, :, s, :].unsqueeze(0)])
            want = paged_attention(q[0, :, s, :], k_all, v_all)
            assert torch.allclose(got[0, :, s, :], want, atol=1e-6), f"序列 {s}"

    def test_length_mismatch_is_rejected(self):
        """拍平长度对不上时应立刻报错，而不是悄悄算错。"""
        pool, tables = self._make([4, 3], block_size=4)
        hook = BatchedPagedAttentionHook(pool, tables, [0, 0], [4, 3])
        q, k, v = (torch.randn(1, 2, 6, 4) for _ in range(3))    # 6 != 7
        try:
            hook(q, k, v, layer_idx=0)
            assert False, "长度不符应 AssertionError"
        except AssertionError:
            pass

    def test_ensure_blocks_grows_lazily(self):
        """页表按需增长：容量不够才领新块（分页省显存的另一半）。"""
        pool = PagedKVPool(num_blocks=8, block_size=4, num_layers=1,
                           num_heads=2, head_dim=4)
        bt = []
        ensure_blocks(pool, bt, 1)
        assert len(bt) == 1
        ensure_blocks(pool, bt, 4)
        assert len(bt) == 1, "刚好装满，不该多领"
        ensure_blocks(pool, bt, 5)
        assert len(bt) == 2
        ensure_blocks(pool, bt, 32)          # 刚好把 8 个 block 领完（32 / 4）
        assert len(bt) == 8 and len(pool.free_blocks) == 0
        try:                                 # 再要就真没了 —— 真引擎在此抢占/驱逐
            ensure_blocks(pool, bt, 33)
            assert False, "池子耗尽应抛 RuntimeError"
        except RuntimeError:
            pass
