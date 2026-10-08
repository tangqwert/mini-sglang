"""M5 验收：Triton 分页注意力 kernel 与 PyTorch 参考实现逐元素一致。

跑法：pytest tests/test_triton_paged.py -x
（需要 CUDA —— Triton kernel 跑在 GPU 上；无 GPU 时整个文件跳过）

核心契约（与 M4a 同样的声明，只是换成了 kernel 实现）：
  **物理布局 + 调度方式变了，数学结果不能变。**
所以 Triton 版必须与 `BatchedPagedAttentionHook` 逐元素一致（严格 fp32，不开 TF32）。
"""
import pytest
import torch

from engine.paged_kv import (BatchedPagedAttentionHook, PagedAttentionHook,
                             PagedKVPool, ensure_blocks)

triton_paged = pytest.importorskip("engine.triton_paged")
if not triton_paged.HAS_TRITON:                       # pragma: no cover
    pytest.skip("未安装 triton", allow_module_level=True)

TritonPagedAttentionHook = triton_paged.TritonPagedAttentionHook

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="Triton kernel 需要 CUDA")


def make_pool(lens, block_size, H, D, layers=1, device="cuda"):
    """建一个池子并把每条序列的页表备好（容量够 base+lens）。"""
    pool = PagedKVPool(num_blocks=256, block_size=block_size, num_layers=layers,
                       num_heads=H, head_dim=D, device=device)
    tables = []
    for L in lens:
        t = []
        ensure_blocks(pool, t, L)
        tables.append(t)
    return pool, tables


@requires_cuda
class TestTritonVsTorch:
    """同一个输入、两个实现、两个池子 —— 结果必须逐元素一致。"""

    @pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16])
    @pytest.mark.parametrize("lens", [[5, 3], [1, 1, 1], [9, 2, 6]])
    def test_prefill_shaped_matches(self, block_size, lens):
        """prefill 形态：base=0，每条序列若干 query 连续算。"""
        torch.manual_seed(0)
        H, D = 2, 16          # D 必须 >= 16：tl.dot 要求 K 维至少 16（gpt2 真实值 64）
        S_total = sum(lens)
        q = torch.randn(1, H, S_total, D, device="cuda")
        k = torch.randn(1, H, S_total, D, device="cuda")
        v = torch.randn(1, H, S_total, D, device="cuda")

        pool_a, tac = make_pool(lens, block_size, H, D)
        pool_b, tbt = make_pool(lens, block_size, H, D)

        want = BatchedPagedAttentionHook(pool_a, tac, [0] * len(lens), lens)(q, k, v, 0)
        got = TritonPagedAttentionHook(pool_b, tbt, [0] * len(lens), lens)(q, k, v, 0)

        assert torch.allclose(got, want, atol=1e-5), \
            f"max|diff| = {(got - want).abs().max().item()}"

    def test_decode_shaped_matches(self):
        """decode 形态：base>0（池里已有历史）、每条只算 1 个 query。"""
        torch.manual_seed(1)
        H, D, BS = 2, 16, 4
        hist = [6, 2]                       # 池里已有的 token 数
        pool_a, tac = make_pool(hist, BS, H, D)
        pool_b, tbt = make_pool(hist, BS, H, D)

        # ① 先用【单序列】钩子把历史 K/V 灌进两个池子（同一份随机数据）
        torch.manual_seed(7)
        for s, L in enumerate(hist):
            hk = torch.randn(1, H, L, D, device="cuda")
            hv = torch.randn(1, H, L, D, device="cuda")
            PagedAttentionHook(pool_a, tac[s])(torch.zeros(1, H, L, D, device="cuda"),
                                               hk, hv, 0)
            PagedAttentionHook(pool_b, tbt[s])(torch.zeros(1, H, L, D, device="cuda"),
                                               hk, hv, 0)

        # ② 两条序列各算 1 个新 token
        torch.manual_seed(8)
        q = torch.randn(1, H, 2, D, device="cuda")
        k = torch.randn(1, H, 2, D, device="cuda")
        v = torch.randn(1, H, 2, D, device="cuda")

        want = BatchedPagedAttentionHook(pool_a, tac, hist, [1, 1])(q, k, v, 0)
        got = TritonPagedAttentionHook(pool_b, tbt, hist, [1, 1])(q, k, v, 0)

        assert torch.allclose(got, want, atol=1e-5), \
            f"max|diff| = {(got - want).abs().max().item()}"

    def test_mixed_prefill_and_decode_matches(self):
        """M6 的混合批形态：一条序列 prefill 多 token、另一条 decode 1 token。"""
        torch.manual_seed(2)
        H, D, BS = 2, 16, 4
        base = [4, 0]
        new = [7, 1]
        pool_a, tac = make_pool([b + n for b, n in zip(base, new)], BS, H, D)
        pool_b, tbt = make_pool([b + n for b, n in zip(base, new)], BS, H, D)

        torch.manual_seed(9)
        hist_k = torch.randn(1, H, base[0], D, device="cuda")
        hist_v = torch.randn(1, H, base[0], D, device="cuda")
        for pool, tbl in ((pool_a, tac), (pool_b, tbt)):
            PagedAttentionHook(pool, tbl[0])(
                torch.zeros(1, H, base[0], D, device="cuda"), hist_k, hist_v, 0)

        q = torch.randn(1, H, sum(new), D, device="cuda")
        k = torch.randn(1, H, sum(new), D, device="cuda")
        v = torch.randn(1, H, sum(new), D, device="cuda")

        want = BatchedPagedAttentionHook(pool_a, tac, base, new)(q, k, v, 0)
        got = TritonPagedAttentionHook(pool_b, tbt, base, new)(q, k, v, 0)
        assert torch.allclose(got, want, atol=1e-5), \
            f"max|diff| = {(got - want).abs().max().item()}"

    def test_causal_mask_is_applied(self):
        """因果性：query 0 只能看 1 个 key → 输出应等于 v[0]。"""
        torch.manual_seed(3)
        H, D, BS, S = 2, 16, 4, 6
        q = torch.randn(1, H, S, D, device="cuda")
        k = torch.randn(1, H, S, D, device="cuda")
        v = torch.randn(1, H, S, D, device="cuda")

        pool, tbl = make_pool([S], BS, H, D)
        got = TritonPagedAttentionHook(pool, tbl, [0], [S])(q, k, v, 0)
        assert torch.allclose(got[0, :, 0, :], v[0, :, 0, :], atol=1e-5), \
            "第一个 query 只该看到第一个 key，softmax 权重为 1 → 输出 = v[0]"

    def test_all_masked_keys_give_no_nan(self):
        """BLOCK_M > n_q 时，无效行不得产生 NaN 并污染有效行。"""
        torch.manual_seed(4)
        H, D, BS = 2, 16, 4
        q = torch.randn(1, H, 3, D, device="cuda")
        k = torch.randn(1, H, 3, D, device="cuda")
        v = torch.randn(1, H, 3, D, device="cuda")

        pool, tbl = make_pool([3], BS, H, D)
        # block_m=16 >> n_q=3：大部分行是无效行
        got = TritonPagedAttentionHook(pool, tbl, [0], [3], block_m=16)(q, k, v, 0)
        assert not torch.isnan(got).any()
        assert not torch.isinf(got).any()

    def test_write_accounting(self):
        """kv_writes 记账应与参考实现口径一致（每层每 token 各 1 条）。"""
        torch.manual_seed(5)
        H, D, BS, lens = 2, 16, 4, [5, 3]
        q, k, v = (torch.randn(1, H, sum(lens), D, device="cuda") for _ in range(3))

        pool, tbl = make_pool(lens, BS, H, D)
        hook = TritonPagedAttentionHook(pool, tbl, [0, 0], lens)
        hook(q, k, v, 0)
        assert hook.kv_writes == sum(lens)

    def test_parameter_mismatch_is_rejected(self):
        pool, tbl = make_pool([2, 2], 4, 2, 16)
        with pytest.raises(AssertionError):
            TritonPagedAttentionHook(pool, tbl, [0], [2, 2])   # bases 少一个
