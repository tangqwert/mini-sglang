"""M1 验收测试：kv_generate 的行为契约。

跑法：source .venv/bin/activate && pytest tests/test_kv_cache.py -x
核心验收：输出与 M0 朴素版【逐 token 一致】，但喂入的 token 总数大幅下降。
"""
import torch
from types import SimpleNamespace

from engine.naive_decode import DecodingConfig, autoregressive_generate
from engine.kv_cache import kv_generate, build_kv_forward

VOCAB = 10
EOS = 0


class FakeSeqCache:
    """假缓存：对这个玩具模型来说，"缓存"就是已见过的序列本身。

    真 HF 模型的缓存是每层的 K/V 张量（DynamicCache），但接口形态一样：
    传进去、更新后传出来。这里用最简单的东西演示同一份契约。
    """

    def __init__(self, ids=None):
        self.ids = ids


class IncrementModel:
    """确定性假模型：下一 token = (所有已见 token 之和) % VOCAB。

    故意让输出依赖【完整上下文】而不只是最后一个 token——
    否则"丢缓存"的实现也能蒙混过关（下一 token 只看最后 token 的模型，
    有没有 cache 结果都一样）。上下文依赖是对 KV cache 的最小要求。

    eos_after 不为 None 时：最后一个 token == eos_after 则强制输出 EOS。
    tokens_fed 统计被喂进前向的 token 总数——这是 M1 性能契约的计量器。
    接口模仿 HF：model(input_ids=..., past_key_values=..., use_cache=True)
    """

    def __init__(self, eos_after=None):
        self.eos_after = eos_after
        self.tokens_fed = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        self.tokens_fed += input_ids.numel()
        if past_key_values is None:
            full = input_ids
        else:
            full = torch.cat([past_key_values.ids, input_ids], dim=1)
        next_tok = (full.sum(dim=1) + 1) % VOCAB  # 依赖完整上下文！
        if self.eos_after is not None:
            hit = full[:, -1] == self.eos_after
            next_tok = torch.where(hit, torch.full_like(next_tok, EOS), next_tok)
        # 模仿 HF：返回每个输入位置的 logits，目标 token 编码成 one-hot
        logits = torch.full((full.shape[0], input_ids.shape[1], VOCAB), -10.0)
        rows = torch.arange(full.shape[0])
        logits[rows, -1, next_tok] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=FakeSeqCache(full))


def naive_logits_fn(model):
    def logits_fn(ids):
        return model(input_ids=ids).logits
    return logits_fn


class TestCorrectness:
    """核心契约：有缓存 = 无缓存，输出逐 token 一致。"""

    def test_matches_naive_no_eos(self):
        m_naive, m_kv = IncrementModel(), IncrementModel()
        prompt = torch.tensor([[5]])
        cfg = DecodingConfig(max_new_tokens=4, eos_token_id=None)
        out_naive = autoregressive_generate(naive_logits_fn(m_naive), prompt, cfg)
        out_kv = kv_generate(build_kv_forward(m_kv, "cpu"), prompt, cfg)
        assert torch.equal(out_naive, out_kv)
        # 5 → (和=5, +1) → 6 → (和=11, →2) → 2 → 4 → 8
        assert out_kv[0].tolist() == [5, 6, 2, 4, 8]

    def test_matches_naive_with_eos(self):
        m_naive, m_kv = IncrementModel(eos_after=6), IncrementModel(eos_after=6)
        prompt = torch.tensor([[5]])
        cfg = DecodingConfig(max_new_tokens=100, eos_token_id=EOS)
        out_naive = autoregressive_generate(naive_logits_fn(m_naive), prompt, cfg)
        out_kv = kv_generate(build_kv_forward(m_kv, "cpu"), prompt, cfg)
        assert torch.equal(out_naive, out_kv)
        assert out_kv[0].tolist() == [5, 6, 0]  # EOS 本身保留：6 之后强制出 EOS

    def test_batch_matches_naive(self):
        m_naive, m_kv = IncrementModel(), IncrementModel()
        prompt = torch.tensor([[5], [8]])
        cfg = DecodingConfig(max_new_tokens=3, eos_token_id=None)
        out_naive = autoregressive_generate(naive_logits_fn(m_naive), prompt, cfg)
        out_kv = kv_generate(build_kv_forward(m_kv, "cpu"), prompt, cfg)
        assert torch.equal(out_naive, out_kv)
        assert out_kv.shape == (2, 4)
        # 行1: 5→6→2→4；行2: 8→9→8→6
        assert out_kv[0].tolist() == [5, 6, 2, 4]
        assert out_kv[1].tolist() == [8, 9, 8, 6]

    def test_input_not_mutated(self):
        m_kv = IncrementModel()
        prompt = torch.tensor([[5]])
        before = prompt.clone()
        cfg = DecodingConfig(max_new_tokens=3, eos_token_id=None)
        kv_generate(build_kv_forward(m_kv, "cpu"), prompt, cfg)
        assert torch.equal(prompt, before)


class TestEfficiency:
    """性能契约：这就是 M1 存在的意义。"""

    def test_feed_schedule_exact(self):
        """喂入时间表：prefill 喂 T 个，之后每步只喂 1 个。

        T=1, n=8：M0 喂 1+2+...+8=36 个；M1 应只喂 1(T) + 7(增量) = 8 个。
        """
        m_naive = IncrementModel()
        m_kv = IncrementModel()
        prompt = torch.tensor([[5]])
        cfg = DecodingConfig(max_new_tokens=8, eos_token_id=None)
        autoregressive_generate(naive_logits_fn(m_naive), prompt, cfg)
        kv_generate(build_kv_forward(m_kv, "cpu"), prompt, cfg)
        assert m_naive.tokens_fed == 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8  # M0 的浪费
        assert m_kv.tokens_fed == 1 + 7  # prefill + 7 次增量

    def test_long_prompt_savings_grow(self):
        """prompt 越长，M1 省得越多（对应你 bench 里 5.6ms→28.7ms 的那条曲线）。

        T=20, n=5：M0 喂 20+21+22+23+24=110；M1 只喂 20+4=24。
        """
        m_naive = IncrementModel()
        m_kv = IncrementModel()
        prompt = torch.arange(20).reshape(1, 20) % VOCAB
        cfg = DecodingConfig(max_new_tokens=5, eos_token_id=None)
        autoregressive_generate(naive_logits_fn(m_naive), prompt, cfg)
        kv_generate(build_kv_forward(m_kv, "cpu"), prompt, cfg)
        assert m_kv.tokens_fed == 20 + 4
        assert m_naive.tokens_fed == 20 + 21 + 22 + 23 + 24
