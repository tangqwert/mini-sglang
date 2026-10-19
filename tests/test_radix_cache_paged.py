"""M3.5b 验收：分页版 Radix 前缀缓存（block 粒度 + refcount + LRU 驱逐）。

跑法：pytest tests/test_radix_cache_paged.py -x

三层契约：
  ① 树本体：命中长度按 block 对齐；分叉时节点分裂且共享前缀的物理块；
  ② 引用计数：活跃请求共享的块不会被驱逐；释放后按 LRU 归还池子；
  ③ 调度器集成：相同/共享前缀的请求真的少做 prefill，输出仍逐 token 等于朴素解码。
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.batching import Request
from engine.model_forward import gpt2_forward, load_gpt2_weights
from engine.naive_decode import DecodingConfig, autoregressive_generate
from engine.paged_kv import PagedKVPool
from engine.paged_schedule import chunked_prefill_generate
from engine.radix_cache_paged import PagedRadixCache


def make_pool(num_blocks=64, block_size=4) -> PagedKVPool:
    """树单元测试用的最小池子（只关心 block 记账，不写 KV 数值）。"""
    return PagedKVPool(num_blocks=num_blocks, block_size=block_size,
                       num_layers=1, num_heads=1, head_dim=1)


def insert_owned(radix: PagedRadixCache, pool: PagedKVPool, tokens: list) -> list:
    """给 tokens 分配新块并插入树；返回树最终使用的物理页表。

    自动把 insert 返回的"重复块"还给池子 —— 这是调用方的责任。
    """
    bs = pool.block_size
    n = len(tokens) // bs
    blocks = [pool.allocate() for _ in range(n)]
    unused = radix.insert(tokens, blocks)
    for b in unused:
        pool.free(b)
    return radix.match_prefix(tokens).blocks


# ─────────────────────────── 树本体 ───────────────────────────

class TestPagedRadixTree:
    def test_match_is_block_aligned(self):
        """只有完整 block 可命中：查询停在块中间 → 该块不算。"""
        pool = make_pool(block_size=4)
        radix = PagedRadixCache(pool)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # 10 个 token = 2 个完整块

        blocks = insert_owned(radix, pool, tokens)
        assert radix.cached_blocks == 2, "尾部不满一块的部分不该进树"

        m = radix.match_prefix(tokens)
        assert m.hit_tokens == 8 and m.blocks == blocks
        assert radix.match_prefix([1, 2, 3]).hit_tokens == 0
        assert radix.match_prefix([1, 2, 3, 4, 5, 6]).hit_tokens == 4
        assert radix.match_prefix([9, 9, 9, 9]).hit_tokens == 0

    def test_split_shares_prefix_block(self):
        """两个序列在第二个 block 分叉 → 第一个 block 共享，节点一分为二。"""
        pool = make_pool(num_blocks=16, block_size=4)
        radix = PagedRadixCache(pool)
        a = [1, 2, 3, 4, 5, 6, 7, 8]
        b = [1, 2, 3, 4, 9, 10, 11, 12]

        a_blocks = insert_owned(radix, pool, a)
        b_blocks = insert_owned(radix, pool, b)

        assert a_blocks[0] == b_blocks[0], "相同前缀必须指向同一个物理 block"
        assert a_blocks[1] != b_blocks[1], "分叉部分必须各存一份"
        assert radix.cached_blocks == 3, "2（A）+ 1（B 的新增块）"
        assert radix.match_prefix(a).blocks == a_blocks
        assert radix.match_prefix(b).blocks == b_blocks

    def test_duplicate_insert_returns_unused(self):
        """同一序列重复插入：物理块不同 → 返回给调用方释放，树不重复记账。"""
        pool = make_pool(num_blocks=16, block_size=4)
        radix = PagedRadixCache(pool)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        first = insert_owned(radix, pool, tokens)
        dup = [pool.allocate() for _ in range(2)]
        unused = radix.insert(tokens, dup)
        assert set(unused) == set(dup)
        for b in unused:
            pool.free(b)

        assert radix.cached_blocks == 2
        assert radix.match_prefix(tokens).blocks == first

    def test_eviction_hook_is_installed(self):
        pool = make_pool()
        radix = PagedRadixCache(pool)
        assert pool.eviction_hook == radix.evict


# ─────────────────────────── 引用计数 / 驱逐 ───────────────────────────

class TestRefcountAndEviction:
    def test_refcount_protects_shared_blocks(self):
        """活跃请求共享的块不能被驱逐；释放后才按 LRU 归还。"""
        pool = make_pool(num_blocks=4, block_size=2)
        radix = PagedRadixCache(pool)
        tokens = [1, 2, 3, 4]                     # 2 个块

        insert_owned(radix, pool, tokens)
        m = radix.match_prefix(tokens)
        assert len(pool.free_blocks) == 2

        radix.acquire_blocks(m.blocks)
        assert all(pool.ref_count[b] == 2 for b in m.blocks)

        pool.allocate()
        pool.allocate()                           # 池子耗尽
        with pytest.raises(RuntimeError):
            pool.allocate()                       # 有锁 → 驱逐不动

        radix.release_blocks(m.blocks)
        assert all(pool.ref_count[b] == 1 for b in m.blocks)
        pool.allocate()                           # 解锁后 → 驱逐成功
        assert radix.cached_blocks == 0
        assert radix.evicted_blocks == 2

    def test_lru_evicts_oldest_leaf(self):
        pool = make_pool(num_blocks=4, block_size=2)
        radix = PagedRadixCache(pool)

        old_tokens, new_tokens = [1, 2], [3, 4]
        insert_owned(radix, pool, old_tokens)
        insert_owned(radix, pool, new_tokens)

        # 手动把访问时间拉开，模拟"旧序列很久没被用到"
        radix.match_prefix(old_tokens).node.last_access = 1.0
        radix.match_prefix(new_tokens).node.last_access = 2.0

        pool.allocate()
        pool.allocate()                           # 空闲块用完
        pool.allocate()                           # 触发 LRU 驱逐

        assert radix.match_prefix(old_tokens).hit_tokens == 0, "旧叶子应被驱逐"
        assert radix.match_prefix(new_tokens).hit_tokens == 2, "新叶子应保留"
        assert radix.evicted_blocks == 1


# ─────────────────────────── 调度器集成 ───────────────────────────

@pytest.fixture(scope="module")
def gpt2():
    return (AutoModelForCausalLM.from_pretrained("gpt2").eval(),
            AutoTokenizer.from_pretrained("gpt2"))


@pytest.fixture(scope="module")
def weights(gpt2):
    return load_gpt2_weights(gpt2[0])


def make_model_pool(cfg, block_size=4, num_blocks=128) -> PagedKVPool:
    return PagedKVPool(num_blocks=num_blocks, block_size=block_size,
                       num_layers=cfg.n_layer, num_heads=cfg.n_head,
                       head_dim=cfg.n_embd // cfg.n_head)


def naive(prompt, weights, budget):
    return autoregressive_generate(lambda ids: gpt2_forward(ids, weights),
                                   prompt,
                                   DecodingConfig(max_new_tokens=budget))


class TestSchedulerWithRadix:
    def test_second_identical_request_skips_prefill(self, gpt2, weights):
        """★ 相同 prompt 的第二条请求复用前缀：prefill token 数下降，输出不变。"""
        model, _ = gpt2
        pool = make_model_pool(model.config)
        radix = PagedRadixCache(pool)
        prompt = torch.arange(100, 112).unsqueeze(0)     # 12 个 token，3 个整块

        out0, st0 = chunked_prefill_generate(
            weights, [Request(0, prompt.clone(), 2)], pool,
            max_batch_size=1, radix=radix)
        out1, st1 = chunked_prefill_generate(
            weights, [Request(1, prompt.clone(), 2)], pool,
            max_batch_size=1, radix=radix)

        assert torch.equal(out0[0], naive(prompt, weights, 2))
        assert torch.equal(out1[0], naive(prompt, weights, 2))
        assert st0.radix_hit_requests == 0, "第一次没有缓存可命中"
        assert st1.radix_hit_requests == 1
        assert st1.radix_hit_tokens > 0
        assert st1.radix_hit_tokens + st1.prefill_tokens == prompt.numel(), \
            "命中 + 现算 = prompt 全长"
        assert len(pool.free_blocks) + radix.cached_blocks == pool.num_blocks, \
            "显存守恒：树持有的块 + 空闲块 = 全部块"

    def test_shared_prefix_branches_split(self, gpt2, weights):
        """★ 共享 16 token 后分叉：命中 16，树分裂成两支，输出逐 token 正确。"""
        model, _ = gpt2
        pool = make_model_pool(model.config)
        radix = PagedRadixCache(pool)

        base = list(range(100, 124))                     # 24 token = 6 个整块
        a = torch.tensor([base])
        b = torch.tensor([base[:16] + [900, 901, 902, 903] + base[20:]])

        out_a, _ = chunked_prefill_generate(
            weights, [Request(0, a.clone(), 2)], pool,
            max_batch_size=1, radix=radix)
        out_b, st_b = chunked_prefill_generate(
            weights, [Request(1, b.clone(), 2)], pool,
            max_batch_size=1, radix=radix)

        assert torch.equal(out_a[0], naive(a, weights, 2))
        assert torch.equal(out_b[0], naive(b, weights, 2))
        assert st_b.radix_hit_tokens == 16
        assert st_b.prefill_tokens == b.numel() - 16
        # A 的 6 块 + B 分叉后的 2 块（前 4 块共享）
        assert radix.cached_blocks == 8
        assert len(pool.free_blocks) + radix.cached_blocks == pool.num_blocks

        # ★ 回归：B 分裂了 A 的节点并写了自己的分支，A 的共享前缀必须完好
        out_a2, st_a2 = chunked_prefill_generate(
            weights, [Request(2, a.clone(), 2)], pool,
            max_batch_size=1, radix=radix)
        assert torch.equal(out_a2[0], naive(a, weights, 2)), \
            "B 写入分叉块时污染了 A 的共享前缀"
        assert st_a2.radix_hit_tokens == 20, "A 的 6 块里后 2 块分叉，前 5 块应全命中"
        assert len(pool.free_blocks) + radix.cached_blocks == pool.num_blocks
