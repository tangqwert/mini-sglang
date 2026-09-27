"""M3 验收测试：Radix 前缀缓存的行为契约。

跑法：pytest tests/test_radix_cache.py -x
核心验收：cached_generate 输出与不带缓存完全一致（逐 token），
且共享前缀的请求真的少算了 token（tokens_fed 精确断言）。
"""
import torch
from types import SimpleNamespace
from transformers import DynamicCache

from engine.naive_decode import DecodingConfig, autoregressive_generate
from engine.batching import Request
from engine.radix_cache import RadixCache, cached_generate, clone_cache_prefix
from engine.kv_cache import build_kv_forward
from test_kv_cache import IncrementModel, naive_logits_fn, VOCAB, EOS


def dummy_cache(length: int):
    """造一个有 length 个位置的假 cache（树单元测试用，内容不重要）。"""
    c = DynamicCache()
    c.update(torch.zeros(1, 1, length, 1), torch.zeros(1, 1, length, 1), 0)
    return c


class CacheSumModel:
    """M3 专用假模型：下一 token = (所有已见 token 之和 + 1) % VOCAB。

    与 M1 假模型的本质区别：它的"已见"来自【真 DynamicCache】——
    把已见 token 塞进 cache 的 keys 张量（keys[b,0,t,0] = token id）。
    这意味着：树查错 / 裁剪长度错 → 读到的上下文就错 → 输出错。
    缓存复用的正确性第一次有了真正的哨兵。
    """

    def __init__(self):
        self.tokens_fed = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=False, **kw):
        self.tokens_fed += input_ids.numel()
        if past_key_values is None:
            full = input_ids
            cache = DynamicCache()
        else:
            cached_ids = past_key_values.layers[0].keys[:, 0, :, 0].long()  # [B, T]
            full = torch.cat([cached_ids, input_ids], dim=1)
            cache = past_key_values
        next_tok = (full.sum(dim=1) + 1) % VOCAB
        logits = torch.full((full.shape[0], input_ids.shape[1], VOCAB), -10.0)
        rows = torch.arange(full.shape[0])
        logits[rows, -1, next_tok] = 10.0
        # 把新喂入的 token 存进 cache（K=自己的 id，V=零）
        cache.update(input_ids.reshape(1, 1, -1, 1).float(),
                     torch.zeros(1, 1, input_ids.shape[1], 1), 0)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def per_request_naive(model, prompt: torch.LongTensor, n: int) -> torch.LongTensor:
    cfg = DecodingConfig(max_new_tokens=n, eos_token_id=None)
    return autoregressive_generate(naive_logits_fn(model), prompt.reshape(1, -1), cfg)


class TestRadixTree:
    """树本体的单元契约（match / insert 的机械行为）。"""

    def test_insert_then_match_exact(self):
        tree = RadixCache()
        tree.insert([1, 2, 3], dummy_cache(3))
        hit, cache = tree.match_prefix([1, 2, 3])
        assert hit == 3 and cache is not None and cache.get_seq_length() >= 3

    def test_match_partial_of_node(self):
        """请求只命中已存序列的前一段 → hit 为公共长度（裁剪是调用方的事）。"""
        tree = RadixCache()
        tree.insert([1, 2, 3], dummy_cache(3))
        hit, cache = tree.match_prefix([1, 2])
        assert hit == 2 and cache is not None

    def test_match_longer_than_cached(self):
        tree = RadixCache()
        tree.insert([1, 2, 3], dummy_cache(3))
        hit, _ = tree.match_prefix([1, 2, 3, 4])
        assert hit == 3, "命中的不能超过树里实际存过的"

    def test_match_miss(self):
        tree = RadixCache()
        tree.insert([1, 2, 3], dummy_cache(3))
        hit, cache = tree.match_prefix([9, 9])
        assert hit == 0 and cache is None

    def test_divergent_overlap_skipped(self):
        """mini 版限制：与已存节点部分重叠但分叉的新序列，放弃插入（宁缺毋错）。

        已存 [1,2,3]；插入 [1,2,9] 与节点 [2,3] 部分重叠 → 不分裂、不插入。
        之后 match [1,2,9] 仍应返回 hit=2（前缀匹配依然正确）。
        """
        tree = RadixCache()
        tree.insert([1, 2, 3], dummy_cache(3))
        tree.insert([1, 2, 9], dummy_cache(5))   # 应被跳过
        hit, cache = tree.match_prefix([1, 2, 9])
        assert hit == 2                          # 匹配依然正确
        assert cache.get_seq_length() >= 2

    def test_clone_prefix(self):
        src = dummy_cache(4)
        piece = clone_cache_prefix(src, 2)
        assert piece.get_seq_length() == 2 and src.get_seq_length() == 4
        piece.update(torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, 1), 0)
        assert piece.get_seq_length() == 3 and src.get_seq_length() == 4  # 互不影响


class TestCachedGenerate:
    def test_outputs_equal_naive(self):
        """终极契约：有前缀缓存 = 无缓存，输出逐 token 一致。

        req0 与 req1 共享前缀 [1,2,3,4]；req2 完全无关。
        CacheSumModel 依赖完整上下文——裁剪长度错一个，这里立刻红。
        """
        m_cache = CacheSumModel()
        m1, m2, m3 = IncrementModel(), IncrementModel(), IncrementModel()
        reqs = [
            Request(req_id=0, prompt=torch.tensor([[1, 2, 3, 4, 5]]), max_new_tokens=3),
            Request(req_id=1, prompt=torch.tensor([[1, 2, 3, 4, 9]]), max_new_tokens=3),
            Request(req_id=2, prompt=torch.tensor([[7, 8]]), max_new_tokens=2),
        ]
        kv_forward = build_kv_forward(m_cache, "cpu")
        outs = cached_generate(kv_forward, reqs, RadixCache())

        for got, m, r in zip(outs, [m1, m2, m3], reqs):
            want = per_request_naive(m, r.prompt, r.max_new_tokens)
            assert torch.equal(got, want), (r.req_id, got.tolist(), want.tolist())

    def test_prefix_hit_saves_tokens(self):
        """收益契约：req1 全额计算；req2 命中前缀 4 个 → 现场 prefill 只喂 1 个。

        tokens_fed 精确账本（T=5,n=3 与 T=5,n=3，共享前缀 4）：
          req1: prefill 5 + decode 2 = 7
          req2: 命中后 suffix 只有 [9] → prefill 1 + decode 2 = 3
        """
        m = CacheSumModel()
        reqs = [
            Request(req_id=0, prompt=torch.tensor([[1, 2, 3, 4, 5]]), max_new_tokens=3),
            Request(req_id=1, prompt=torch.tensor([[1, 2, 3, 4, 9]]), max_new_tokens=3),
        ]
        cached_generate(build_kv_forward(m, "cpu"), reqs, RadixCache())
        assert m.tokens_fed == 7 + 3, f"实际 {m.tokens_fed}——共享前缀没有被省掉？"

    def test_tree_grows_across_requests(self):
        """每条请求收工后回写：下一条能查到它的前缀。"""
        m = CacheSumModel()
        radix = RadixCache()
        reqs = [
            Request(req_id=0, prompt=torch.tensor([[1, 2, 3, 4, 5]]), max_new_tokens=2),
            Request(req_id=1, prompt=torch.tensor([[1, 2, 3, 4, 5]]), max_new_tokens=2),
        ]
        cached_generate(build_kv_forward(m, "cpu"), reqs, radix)
        hit, _ = radix.match_prefix(reqs[1].prompt[0].tolist())
        assert hit == 5, "第二条请求应命中第一条存进树的完整 prompt"

    def test_full_prefix_match_extends(self):
        """被后续请求【完整命中并延长】时必须逐 token 一致。

        曾漏：cache 比 generated 少 1 个 token，insert 却按整条序列记账，
        clone_cache_prefix 静默截断 → 丢最后一个 token 的 K/V。
        （partial hit 场景由 test_outputs_equal_naive 覆盖，本用例补 full hit。）
        """
        radix = RadixCache()
        kf = build_kv_forward(CacheSumModel(), "cpu")
        seq0 = cached_generate(kf, [Request(req_id=0, prompt=torch.tensor([[1, 2, 3]]),
                                        max_new_tokens=2)], radix)[0][0].tolist()
        prompt1 = seq0 + [7]          # 完整命中 req0 存进树的前缀，再多一个 token
        got = cached_generate(kf, [Request(req_id=1, prompt=torch.tensor([prompt1]),
                                        max_new_tokens=2)], radix)[0]
        want = per_request_naive(CacheSumModel(), torch.tensor([prompt1]), 2)
        assert torch.equal(got, want), (got.tolist(), want.tolist())
