"""M0 验收测试：autoregressive_generate 的行为契约。

跑法：source .venv/bin/activate && pytest tests/ -x
全部变绿 = M0 核心完成。
"""
import torch
import pytest

from engine.naive_decode import DecodingConfig, autoregressive_generate

VOCAB = 10
EOS = 0  # 测试里用 0 号 token 作为结束符


def make_increment_logits_fn(eos_after: int | None = None):
    """构造一个确定性的假模型：下一个 token 永远是 (上一个 token + 1) % VOCAB。

    若 eos_after 不为 None：当序列最后一个 token == eos_after 时，
    强制下一个 token 为 EOS（用来测提前停止）。
    """
    def logits_fn(ids: torch.LongTensor) -> torch.FloatTensor:
        # ids: [B, T] -> logits: [B, T, VOCAB]
        last = ids[:, -1]  # [B]
        next_tok = (last + 1) % VOCAB
        if eos_after is not None:
            hit = last == eos_after
            next_tok = torch.where(hit, torch.full_like(next_tok, EOS), next_tok)
        # 把"想要的下一个 token"编码成 one-hot logits（argmax 必然选中它）
        logits = torch.full((ids.shape[0], ids.shape[1], VOCAB), -10.0)
        # 必须用 rows 指明"第几行"：否则 logits[:, -1, next_tok] 会广播成 [B, B]，
        # 把每条序列的目标 token 写到所有行上（batch 用例就会互相串味）。
        rows = torch.arange(ids.shape[0])
        logits[rows, -1, next_tok] = 10.0
        return logits

    return logits_fn


class TestAutoregressiveGenerate:
    def test_generates_exactly_max_new_tokens(self):
        """无 EOS 时：输出长度 = prompt 长度 + max_new_tokens。"""
        prompt = torch.tensor([[5]])
        cfg = DecodingConfig(max_new_tokens=4, eos_token_id=None)
        out = autoregressive_generate(make_increment_logits_fn(), prompt, cfg)
        assert out.shape == (1, 5)
        # 5 -> 6 -> 7 -> 8 -> 9
        assert out[0].tolist() == [5, 6, 7, 8, 9]

    def test_stops_after_eos(self):
        """命中 EOS 时提前停止：5 -> 6 -> 7 -> (7 之后强制 EOS=0)。"""
        prompt = torch.tensor([[5]])
        cfg = DecodingConfig(max_new_tokens=100, eos_token_id=EOS)
        out = autoregressive_generate(make_increment_logits_fn(eos_after=7), prompt, cfg)
        # 5,6,7 走了 3 步后下一 token 是 0（EOS），包含 EOS，随后停止
        assert out[0].tolist() == [5, 6, 7, 0]

    def test_eos_is_respected_within_budget(self):
        """max_new_tokens 比需要的多，但 EOS 优先。"""
        prompt = torch.tensor([[8]])
        cfg = DecodingConfig(max_new_tokens=50, eos_token_id=EOS)
        out = autoregressive_generate(make_increment_logits_fn(eos_after=2), prompt, cfg)
        # 8->9->0? 注意 9+1=10 % 10 = 0，第一步就会命中 EOS=0
        assert out[0].tolist() == [8, 9, 0]

    def test_input_not_mutated(self):
        """不得原地修改调用方的输入张量。"""
        prompt = torch.tensor([[5]])
        before = prompt.clone()
        cfg = DecodingConfig(max_new_tokens=3, eos_token_id=None)
        autoregressive_generate(make_increment_logits_fn(), prompt, cfg)
        assert torch.equal(prompt, before)

    def test_batch_of_two(self):
        """批内两条序列同步解码（M0：整批同停）。"""
        prompt = torch.tensor([[5], [8]])  # [B=2, T=1]
        cfg = DecodingConfig(max_new_tokens=3, eos_token_id=None)
        out = autoregressive_generate(make_increment_logits_fn(), prompt, cfg)
        assert out.shape == (2, 4)
        assert out[0].tolist() == [5, 6, 7, 8]
        assert out[1].tolist() == [8, 9, 0, 1]  # 9+1=10 %10=0

    def test_longer_prompt(self):
        """prompt 长度 > 1 时，只看最后一个位置决定下一步。"""
        prompt = torch.tensor([[1, 2, 5]])  # T=3
        cfg = DecodingConfig(max_new_tokens=2, eos_token_id=None)
        out = autoregressive_generate(make_increment_logits_fn(), prompt, cfg)
        # 最后 token 是 5，所以从 6 继续；prompt 原样保留
        assert out[0].tolist() == [1, 2, 5, 6, 7]

    def test_vocabulary_wraparound(self):
        """词表回绕：(9+1)%10=0。注意 eos_token_id=None 时 0 只是普通 token。"""
        prompt = torch.tensor([[9]])
        cfg = DecodingConfig(max_new_tokens=2, eos_token_id=None)
        out = autoregressive_generate(make_increment_logits_fn(), prompt, cfg)
        assert out[0].tolist() == [9, 0, 1]
