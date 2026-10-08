"""M4a 验收测试：分页 KV 池 + 分页注意力。

跑法：pytest tests/test_paged_kv.py -x
核心契约：分页存储 + 按页表 gather 的注意力 == 连续存储的参考实现（逐元素一致）。
这条契约就是 PagedAttention 的"无损"声明——物理布局变了，数学结果不能变。
"""
import math

import torch

from engine.paged_kv import PagedKVPool, paged_attention


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
