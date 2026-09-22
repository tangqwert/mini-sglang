"""M2 验收测试：batched_generate 的行为契约。

跑法：pytest tests/test_batching.py -x
核心验收：batch 里每条请求的结果 == 它单独跑时的结果（逐 token 一致），
且各请求独立完成（冻结），互不拖累。
"""
import torch

from engine.naive_decode import DecodingConfig, autoregressive_generate
from engine.kv_cache import build_kv_forward
from engine.batching import Request, batched_generate, pad_left, build_position_ids

# 复用 M1 的假模型（已升级为 mask 感知：上下文和 = (full * mask).sum）
from test_kv_cache import IncrementModel, naive_logits_fn, VOCAB, EOS


def per_request_naive(model, req: Request) -> torch.LongTensor:
    """基线：把请求单独跑一遍 M0 朴素解码。"""
    cfg = DecodingConfig(max_new_tokens=req.max_new_tokens,
                         eos_token_id=req.eos_token_id)
    return autoregressive_generate(naive_logits_fn(model), req.prompt.reshape(1, -1), cfg)


class TestCorrectness:
    def test_matches_per_request_no_eos(self):
        """不同长度 prompt、不同预算：每行结果 == 单独跑的结果。"""
        m_batch = IncrementModel()
        m1, m2, m3 = IncrementModel(), IncrementModel(), IncrementModel()
        reqs = [
            Request(req_id=0, prompt=torch.tensor([[5]]), max_new_tokens=4),
            Request(req_id=1, prompt=torch.tensor([[3, 4]]), max_new_tokens=3),
            Request(req_id=2, prompt=torch.tensor([[9]]), max_new_tokens=2),
        ]
        outs = batched_generate(build_kv_forward(m_batch, "cpu"), reqs)
        expected = [
            per_request_naive(m1, reqs[0]),
            per_request_naive(m2, reqs[1]),
            per_request_naive(m3, reqs[2]),
        ]
        for got, want, req in zip(outs, expected, reqs):
            assert torch.equal(got, want), (req.req_id, got.tolist(), want.tolist())

    def test_per_request_freeze_on_budget(self):
        """预算独立：预算小的先冻结，不能拖累别人，也不能自己超跑。"""
        m_batch = IncrementModel()
        m1, m2 = IncrementModel(), IncrementModel()
        reqs = [
            Request(req_id=0, prompt=torch.tensor([[5]]), max_new_tokens=2),
            Request(req_id=1, prompt=torch.tensor([[5]]), max_new_tokens=6),
        ]
        outs = batched_generate(build_kv_forward(m_batch, "cpu"), reqs)
        assert outs[0].shape[1] == 1 + 2, "预算 2 的请求应恰好生成 2 个"
        assert outs[1].shape[1] == 1 + 6, "预算 6 的请求应完整跑满"
        assert torch.equal(outs[0], per_request_naive(m1, reqs[0]))
        assert torch.equal(outs[1], per_request_naive(m2, reqs[1]))

    def test_per_request_freeze_on_eos(self):
        """EOS 独立：一条命中 EOS 冻结后，其余继续（M0 做不到的事！）。

        eos_after=7 的假模型下，prompt=[9] 的行第一步就命中 EOS=0；
        prompt=[5] 的行要走到自然出现 0 才停。
        """
        m_batch = IncrementModel(eos_after=7)
        m1, m2 = IncrementModel(eos_after=7), IncrementModel(eos_after=7)
        reqs = [
            Request(req_id=0, prompt=torch.tensor([[5]]), max_new_tokens=50, eos_token_id=EOS),
            Request(req_id=1, prompt=torch.tensor([[9]]), max_new_tokens=50, eos_token_id=EOS),
        ]
        outs = batched_generate(build_kv_forward(m_batch, "cpu"), reqs)
        assert torch.equal(outs[0], per_request_naive(m1, reqs[0]))
        assert torch.equal(outs[1], per_request_naive(m2, reqs[1]))
        assert outs[1].shape[1] < outs[0].shape[1], "短命的请求应该真的先停"

    def test_prompts_not_mutated(self):
        m_batch = IncrementModel()
        reqs = [
            Request(req_id=0, prompt=torch.tensor([[5]]), max_new_tokens=2),
            Request(req_id=1, prompt=torch.tensor([[3, 4]]), max_new_tokens=3),
        ]
        before = [r.prompt.clone() for r in reqs]
        batched_generate(build_kv_forward(m_batch, "cpu"), reqs)
        for r, b in zip(reqs, before):
            assert torch.equal(r.prompt, b)


class TestHelpers:
    """pad_left / build_position_ids 的机械契约（已提供，测一下以防手滑）。"""

    def test_pad_left_aligns_right(self):
        reqs = [
            Request(0, torch.tensor([[1, 2, 3]]), 1),
            Request(1, torch.tensor([[7]]), 1),
        ]
        padded, mask = pad_left(reqs, pad_token_id=0)
        assert padded.tolist() == [[1, 2, 3], [0, 0, 7]]  # 左填充：真 token 挤右边
        assert mask.tolist() == [[1, 1, 1], [0, 0, 1]]

    def test_position_ids_skip_pads(self):
        _, mask = pad_left([Request(0, torch.tensor([[1, 2, 3]]), 1),
                            Request(1, torch.tensor([[7]]), 1)], 0)
        pos = build_position_ids(mask)
        # 行 2 的真实 token 7 的位置应是 0（不是 2）——pad 不占位置
        assert pos[0].tolist() == [0, 1, 2]
        assert pos[1].tolist() == [0, 0, 0]
