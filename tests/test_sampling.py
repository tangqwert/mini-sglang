"""M8 验收：采样策略（temperature / top_k / top_p）+ 停止条件 + 调度器集成。

跑法：pytest tests/test_sampling.py -x

契约：
  ① 默认 temperature=0 必须与 argmax 逐 token 一致（旧行为零回归）；
  ② top_k / top_p 的截断边界正确（用固定分布做统计断言）；
  ③ 同一 seed 可复现；不同 seed 应产生不同序列；
  ④ stop_token_ids / ignore_eos 生效；
  ⑤ 分块 prefill 不在中间块上“白采样” —— 同一 seed 在不同 chunk 大小下输出一致。
"""
from dataclasses import replace

import pytest
import torch

from engine.batching import Request
from engine.naive_decode import DecodingConfig, autoregressive_generate
from engine.paged_kv import PagedKVPool
from engine.paged_schedule import chunked_prefill_generate
from engine.sampling import (SamplingParams, is_finished, sample_token,
                             sample_tokens, token_stops)


# ─────────────────────────── 参数与工具 ───────────────────────────

class TestSamplingParams:
    def test_invalid_values(self):
        with pytest.raises(ValueError):
            SamplingParams(temperature=-1)
        with pytest.raises(ValueError):
            SamplingParams(top_k=-1)
        with pytest.raises(ValueError):
            SamplingParams(top_p=0.0)
        with pytest.raises(ValueError):
            SamplingParams(top_p=1.5)

    def test_stop_ids_are_frozen_to_tuple(self):
        p = SamplingParams(stop_token_ids=[1, 2])
        assert p.stop_token_ids == (1, 2)

    def test_greedy_flag(self):
        assert SamplingParams().is_greedy
        assert SamplingParams(temperature=0.0).is_greedy
        assert not SamplingParams(temperature=0.1).is_greedy

    def test_generator_seed(self):
        p = SamplingParams(seed=7)
        g1, g2 = p.make_generator("cpu"), p.make_generator("cpu")
        a = torch.multinomial(torch.ones(4), 4, replacement=True, generator=g1)
        b = torch.multinomial(torch.ones(4), 4, replacement=True, generator=g2)
        assert torch.equal(a, b)
        assert SamplingParams(seed=None).make_generator("cpu") is None


# ─────────────────────────── 采样核心 ───────────────────────────

class TestSampleToken:
    def test_greedy_matches_argmax(self):
        torch.manual_seed(0)
        logits = torch.randn(16, 32)
        out = sample_token(logits, SamplingParams())
        assert out.shape == (16, 1)
        assert torch.equal(out[:, 0], logits.argmax(dim=-1))

    def test_single_row_shape(self):
        logits = torch.tensor([0.0, 1.0, 2.0])
        out = sample_token(logits, SamplingParams())
        assert out.shape == (1,) and int(out[0]) == 2

    def test_top_k_one_equals_argmax(self):
        torch.manual_seed(1)
        logits = torch.randn(64, 20)
        p = SamplingParams(temperature=1.0, top_k=1, seed=1)
        out = sample_token(logits, p, p.make_generator(logits.device))
        assert torch.equal(out[:, 0], logits.argmax(dim=-1))

    def test_top_k_filters_tail(self):
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        p = SamplingParams(temperature=1.0, top_k=2, seed=0)
        out = sample_token(logits.repeat(500, 1), p, p.make_generator("cpu"))
        assert set(out[:, 0].tolist()) <= {2, 3}, "top_k=2 只应采到最大的两个 token"
        assert set(out[:, 0].tolist()) == {2, 3}, "两个 token 都应有非零概率"

    def test_top_p_filters_tail(self):
        """分布 [0.6, 0.3, 0.1]，top_p=0.7 → 只保留前两个。"""
        probs = torch.tensor([0.6, 0.3, 0.1])
        logits = probs.log().unsqueeze(0).repeat(1000, 1)
        p = SamplingParams(temperature=1.0, top_p=0.7, seed=0)
        out = sample_token(logits, p, p.make_generator("cpu"))
        assert (out[:, 0] == 2).sum() == 0, "top_p 截断后第三个 token 不应出现"
        assert (out[:, 0] == 0).sum() > 0 and (out[:, 0] == 1).sum() > 0

    def test_greedy_ignores_top_k_and_top_p(self):
        logits = torch.tensor([[0.0, 10.0, 0.0]])
        p = SamplingParams(top_k=1, top_p=0.01)
        out = sample_token(logits, p)
        assert int(out[0, 0]) == 1

    def test_same_seed_reproducible(self):
        torch.manual_seed(2)
        logits = torch.randn(128, 32)
        p = SamplingParams(temperature=1.0, top_p=0.9, seed=123)
        a = sample_token(logits, p, p.make_generator("cpu"))
        b = sample_token(logits, p, p.make_generator("cpu"))
        assert torch.equal(a, b)
        p2 = replace(p, seed=124)
        c = sample_token(logits, p2, p2.make_generator("cpu"))
        assert not torch.equal(a, c), "不同 seed 应产生不同序列（128 次抽样几乎必然）"

    def test_per_row_params(self):
        torch.manual_seed(3)
        logits = torch.randn(2, 10)
        params = [SamplingParams(temperature=1.0, top_k=1, seed=0),
                  SamplingParams(temperature=1.0, seed=1)]
        gens = [p.make_generator("cpu") for p in params]
        draws = [sample_tokens(logits, params, gens) for _ in range(50)]
        assert draws[0].shape == (2, 1), "非贪心批采样必须返回 [B,1]"
        out = torch.cat(draws, dim=1)                 # [2, 50]
        assert (out[0] == logits[0].argmax()).all(), "第一行 top_k=1 → 恒等于 argmax"
        assert out[1].unique().numel() > 1, "第二行应真的在采样"


# ─────────────────────────── 停止条件 ───────────────────────────

class TestStopConditions:
    def test_token_stops(self):
        p = SamplingParams(stop_token_ids=(7,))
        assert token_stops(p, eos_token_id=2, token=2)
        assert token_stops(p, eos_token_id=2, token=7)
        assert not token_stops(p, eos_token_id=2, token=3)

    def test_ignore_eos(self):
        p = SamplingParams(ignore_eos=True)
        assert not token_stops(p, eos_token_id=2, token=2)
        assert is_finished(p, max_new_tokens=10, eos_token_id=2,
                           token=2, n_generated=1) is False

    def test_request_helpers(self):
        r = Request(req_id=0, prompt=torch.tensor([[1, 2]]), max_new_tokens=3,
                    eos_token_id=2, sampling=SamplingParams(stop_token_ids=(9,)))
        assert r.stops_on(2) and r.stops_on(9) and not r.stops_on(3)
        assert r.finished(3, n_generated=3)
        assert not r.finished(3, n_generated=2)

    def test_request_default_is_greedy(self):
        r = Request(req_id=0, prompt=torch.tensor([[1, 2]]), max_new_tokens=3)
        assert r.sampling.is_greedy


# ─────────────────────────── M0 循环 ───────────────────────────

VOCAB = 5


def fixed_logits_fn(token_id: int):
    def logits_fn(ids):
        logits = torch.zeros(ids.shape[0], ids.shape[1], VOCAB)
        logits[..., token_id] = 10.0
        return logits
    return logits_fn


def uniform_logits_fn(ids):
    return torch.zeros(ids.shape[0], ids.shape[1], VOCAB)


class TestAutoregressiveWithSampling:
    def test_greedy_default_unchanged(self):
        prompt = torch.tensor([[0, 1]])
        cfg = DecodingConfig(max_new_tokens=4)
        out = autoregressive_generate(fixed_logits_fn(3), prompt, cfg)
        assert out.tolist() == [[0, 1, 3, 3, 3, 3]]

    def test_same_seed_reproducible(self):
        prompt = torch.tensor([[0]])
        cfg = DecodingConfig(max_new_tokens=10,
                             sampling=SamplingParams(temperature=1.0, seed=7))
        a = autoregressive_generate(uniform_logits_fn, prompt, cfg)
        b = autoregressive_generate(uniform_logits_fn, prompt, cfg)
        assert torch.equal(a, b)
        c = autoregressive_generate(
            uniform_logits_fn, prompt,
            replace(cfg, sampling=SamplingParams(temperature=1.0, seed=8)))
        assert not torch.equal(a, c)

    def test_ignore_eos(self):
        prompt = torch.tensor([[0]])
        cfg = DecodingConfig(max_new_tokens=4, eos_token_id=3,
                             sampling=SamplingParams(ignore_eos=True))
        out = autoregressive_generate(fixed_logits_fn(3), prompt, cfg)
        assert out.shape == (1, 5), "ignore_eos 时应跑满预算，而不是在 EOS 停"

    def test_stop_token_ids(self):
        prompt = torch.tensor([[0]])
        cfg = DecodingConfig(max_new_tokens=4,
                             sampling=SamplingParams(stop_token_ids=(3,)))
        out = autoregressive_generate(fixed_logits_fn(3), prompt, cfg)
        assert out.shape == (1, 2), "命中 stop token 后应立即停止"


# ─────────────────────────── 调度器集成（假模型）───────────────────────────

class _NoopHook:
    """只满足调度器的构造/调用形状，不真正写 KV。"""
    def __init__(self, pool, block_tables, bases, new_lens):
        self.n = len(new_lens)
        self.total = sum(new_lens)

    def __call__(self, q, k, v, layer_idx: int):
        return torch.zeros_like(q)


def _fake_forward(ids, weights, attention_fn=None, position_ids=None):
    # 扁平批：每个位置都给同样的 uniform logits
    return torch.zeros(1, ids.shape[1], VOCAB)


def _tiny_pool(block_size: int = 4) -> PagedKVPool:
    return PagedKVPool(num_blocks=64, block_size=block_size,
                       num_layers=1, num_heads=1, head_dim=1)


class TestChunkedPrefillSampling:
    def test_chunk_size_does_not_change_rng_stream(self):
        """★ 关键回归：中间 prefill 块不采样，所以 chunk 大小不影响随机序列。"""
        prompt = torch.arange(10).unsqueeze(0) + 1
        def run(chunk: int):
            req = Request(req_id=0, prompt=prompt.clone(), max_new_tokens=3,
                          sampling=SamplingParams(temperature=1.0, seed=42))
            outs, stats = chunked_prefill_generate(
                None, [req], _tiny_pool(), max_batch_size=1,
                max_prefill_tokens=chunk, hook_cls=_NoopHook,
                forward_fn=_fake_forward)
            return outs[0], stats

        a, st_a = run(2)
        b, st_b = run(1000)
        assert torch.equal(a, b), "同一 seed 下，chunk 大小不应改变采样结果"
        assert st_a.prefill_chunks > st_b.prefill_chunks
        # 两次运行的采样次数相同：每步只有“真正产出 token”的行才采样
        assert st_a.prefill_tokens == st_b.prefill_tokens == prompt.numel()

    def test_greedy_still_works_in_scheduler(self):
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        req = Request(req_id=0, prompt=prompt, max_new_tokens=3)
        outs, _ = chunked_prefill_generate(
            None, [req], _tiny_pool(), max_batch_size=1,
            hook_cls=_NoopHook, forward_fn=_fake_forward)
        assert outs[0].shape == (1, 8)
        assert (outs[0][0, prompt.numel():] == 0).all(), "全零 logits 的 argmax 是 0"
