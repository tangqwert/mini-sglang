"""M2 验收测试：batched_generate 的行为契约。

跑法：pytest tests/test_batching.py -x
核心验收：batch 里每条请求的结果 == 它单独跑时的结果（逐 token 一致），
且各请求独立完成（冻结），互不拖累。
"""
import torch

from engine.naive_decode import DecodingConfig, autoregressive_generate
from engine.kv_cache import build_kv_forward
from engine.batching import (Request, batched_generate, continuous_generate,
                             pad_left, build_position_ids)

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


class TestContinuous:
    """M2.5：槽位连续准入 —— 空出的槽位立刻补入下一条请求。

    关键契约有两条：
      ① 正确性不变：每条请求的输出仍与单独跑朴素解码逐 token 一致
        （补入是通过"左填充到当前批长 + 单独 prefill + 拷行"实现的，
         因果注意力逐行独立 ⇒ 与"一开始就在批里"数值等价）
      ② 收益有代价：省下的是"冻结行空转"，付出的是"补入需要独立前向"
    """

    def _assert_all_match(self, reqs, batch_size, model=None):
        model = model or IncrementModel()
        outs, stats = continuous_generate(build_kv_forward(model, "cpu"), reqs,
                                          max_batch_size=batch_size)
        for got, r in zip(outs, reqs):
            want = per_request_naive(IncrementModel(), r)
            assert torch.equal(got, want), (r.req_id, got.tolist(), want.tolist())
        return stats

    def test_no_refill_needed_when_batch_fits(self):
        """请求数 <= 槽位数：不需要补入，行为应与 M2 一致。"""
        reqs = [
            Request(req_id=0, prompt=torch.tensor([[5]]), max_new_tokens=4),
            Request(req_id=1, prompt=torch.tensor([[3, 4]]), max_new_tokens=3),
            Request(req_id=2, prompt=torch.tensor([[9]]), max_new_tokens=2),
        ]
        stats = self._assert_all_match(reqs, batch_size=4)
        assert stats.admissions == 0
        assert stats.steps == 3          # 最长预算 4 → prefill + 3 步

    def test_refill_when_queue_exceeds_slots(self):
        """请求数 > 槽位数：必须发生补入，且结果仍逐 token 一致。"""
        reqs = [
            Request(req_id=0, prompt=torch.tensor([[5]]), max_new_tokens=4),
            Request(req_id=1, prompt=torch.tensor([[3, 4]]), max_new_tokens=3),
            Request(req_id=2, prompt=torch.tensor([[9]]), max_new_tokens=2),
            Request(req_id=3, prompt=torch.tensor([[1, 2, 3]]), max_new_tokens=5),
        ]
        stats = self._assert_all_match(reqs, batch_size=2)
        assert stats.admissions == 2, "4 条请求 / 2 个槽位 → 应补入 2 次"
        assert stats.padded_positions > 0, "补入靠左填充，必然产生填充位置"

    def test_batch_size_one(self):
        """退化情形：单槽位也必须正确（等价于逐条串行）。"""
        reqs = [
            Request(req_id=0, prompt=torch.tensor([[5]]), max_new_tokens=2),
            Request(req_id=1, prompt=torch.tensor([[7]]), max_new_tokens=3),
        ]
        stats = self._assert_all_match(reqs, batch_size=1)
        assert stats.admissions == 1
        assert stats.idle_slot_steps == 0, "单槽位时不该有空转"

    def test_eos_freezes_slot_and_frees_it(self):
        """命中 EOS 的行立刻释放槽位，把机会让给后面的请求。"""
        reqs = [
            Request(req_id=0, prompt=torch.tensor([[9]]), max_new_tokens=50, eos_token_id=EOS),
            Request(req_id=1, prompt=torch.tensor([[5]]), max_new_tokens=3),
        ]
        stats = self._assert_all_match(reqs, batch_size=1, model=IncrementModel(eos_after=7))
        # prompt=[9] 首步即命中 EOS=0 → 释放槽位 → 补入 req1
        assert stats.admissions == 1

    def test_beats_static_chunking_on_forward_count(self):
        """收益契约：连续准入的总前向次数 < 静态分批。

        静态分批（每 B 条一组跑到底）的前向次数 = Σ 各组 max(预算)。
        连续准入的前向次数 = 批量步数 + 补入前向次数。
        """
        budgets = [1, 5, 1, 5]
        reqs = [Request(i, torch.tensor([[5]]), n) for i, n in enumerate(budgets)]
        _, stats = continuous_generate(build_kv_forward(IncrementModel(), "cpu"),
                                       reqs, max_batch_size=2)

        static_forwards = sum(max(budgets[k:k + 2]) for k in range(0, len(budgets), 2))
        continuous_forwards = stats.steps + stats.admission_forwards
        assert continuous_forwards < static_forwards, (continuous_forwards, static_forwards)

    def test_empty_requests(self):
        outs, stats = continuous_generate(build_kv_forward(IncrementModel(), "cpu"), [])
        assert outs == [] and stats.steps == 0

    def test_prompts_not_mutated(self):
        reqs = [
            Request(req_id=0, prompt=torch.tensor([[5]]), max_new_tokens=2),
            Request(req_id=1, prompt=torch.tensor([[3, 4]]), max_new_tokens=3),
            Request(req_id=2, prompt=torch.tensor([[1, 2, 3]]), max_new_tokens=2),
        ]
        before = [r.prompt.clone() for r in reqs]
        continuous_generate(build_kv_forward(IncrementModel(), "cpu"), reqs, max_batch_size=2)
        for r, b in zip(reqs, before):
            assert torch.equal(r.prompt, b)
