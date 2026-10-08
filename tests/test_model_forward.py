"""M4b Step 1 验收：自研 GPT-2 前向与 HF 数值一致。

跑法：pytest tests/test_model_forward.py -x

核心契约：`gpt2_forward` 的 logits 与 HF `model(ids).logits` 逐元素一致（< 1e-4）。
另外验证两件 M4b 后续步骤要依赖的性质：
  · 因果性 —— 只用前 k 个 token 前向 == 整条前向的第 k 位（增量解码的依据）
  · attention 可替换 —— `attention_fn` 钩子会被逐层调用（Step 2 接分页的接缝）

不需要 GPU：gpt2 在 CPU 上几秒跑完，所以本文件不设 skipif。
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.model_forward import (_causal_attention, gpt2_forward,
                                  load_gpt2_weights)
from engine.paged_kv import PagedAttentionHook, PagedKVPool, allocate_for


@pytest.fixture(scope="module")
def gpt2():
    return AutoModelForCausalLM.from_pretrained("gpt2").eval(), \
        AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def weights(gpt2):
    return load_gpt2_weights(gpt2[0])


class TestGPT2Forward:
    def test_matches_hf_logits(self, gpt2, weights):
        """★ 核心契约：与 HF 的 logits 逐元素一致。"""
        model, tok = gpt2
        ids = tok("Hello world, this is a test", return_tensors="pt").input_ids
        with torch.no_grad():
            ref = model(ids).logits
        got = gpt2_forward(ids, weights)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-4), \
            f"max|diff| = {(got - ref).abs().max().item()}"

    def test_greedy_next_token_matches(self, gpt2, weights):
        """端到端：贪心取的第一个生成 token 必须一致。"""
        model, tok = gpt2
        ids = tok("The meaning of life is", return_tensors="pt").input_ids
        with torch.no_grad():
            ref_tok = int(model(ids).logits[0, -1].argmax())
        got_tok = int(gpt2_forward(ids, weights)[0, -1].argmax())
        assert got_tok == ref_tok

    def test_prefix_matches_full_forward(self, gpt2, weights):
        """因果性：只用前 k 个 token 前向，其最后一位 logits == 整条前向的第 k-1 位。

        这正是增量解码成立的依据（M4b Step 3 要用）。
        """
        _, tok = gpt2
        ids = tok("Hello world, this is a test", return_tensors="pt").input_ids
        S = ids.shape[1]
        full = gpt2_forward(ids, weights)
        for k in (1, 4, S):
            part = gpt2_forward(ids[:, :k], weights)
            assert torch.allclose(part[0, -1], full[0, k - 1], atol=1e-4), \
                f"k={k} 不成立"

    def test_explicit_position_ids_equivalent(self, gpt2, weights):
        """显式给 position_ids 应与默认 0..S-1 完全一致（增量解码靠它）。"""
        _, tok = gpt2
        ids = tok("Hello world", return_tensors="pt").input_ids
        S = ids.shape[1]
        default = gpt2_forward(ids, weights)
        explicit = gpt2_forward(ids, weights,
                                position_ids=torch.arange(S).unsqueeze(0))
        assert torch.allclose(default, explicit, atol=1e-6)

    def test_attention_fn_hook_is_used(self, gpt2, weights):
        """★ 为 M4b Step 2 预留的接缝：自定义 attention_fn 会被逐层调用。"""
        model, tok = gpt2
        ids = tok("Hello", return_tensors="pt").input_ids
        n_head, D = model.config.n_head, model.config.n_embd
        seen = []

        def spy(q, k, v, layer_idx):
            seen.append((layer_idx, tuple(q.shape)))
            return _causal_attention(q, k, v)

        got = gpt2_forward(ids, weights, attention_fn=spy)
        assert len(seen) == model.config.n_layer, "每层都应调用一次"
        assert seen[0] == (0, (1, n_head, ids.shape[1], D // n_head))
        assert torch.allclose(got, gpt2_forward(ids, weights), atol=1e-6)

    def test_batched_input_matches_single(self, gpt2, weights):
        """把两行拼成一个 batch，逐行结果应与单独跑一致。"""
        _, tok = gpt2
        a = tok("Hello world", return_tensors="pt").input_ids
        b = tok("Hello there", return_tensors="pt").input_ids
        assert a.shape == b.shape, "本用例要求两行长度相同"
        batched = gpt2_forward(torch.cat([a, b], dim=0), weights)
        assert torch.allclose(batched[0], gpt2_forward(a, weights)[0], atol=1e-4)
        assert torch.allclose(batched[1], gpt2_forward(b, weights)[0], atol=1e-4)


class TestPagedForward:
    """M4b Step 2：把 attention 换成“分页版”后，logits 必须不变。

    这就是 PagedAttention 的“无损”声明在【真模型】上的验证 ——
    M4a 只在孤立张量上验过，这里第一次跑在 12 层 GPT-2 上。
    """

    def test_paged_forward_matches_hf(self, gpt2, weights):
        model, tok = gpt2
        cfg = model.config
        ids = tok("Hello world, this is a test", return_tensors="pt").input_ids
        S = ids.shape[1]
        pool = PagedKVPool(num_blocks=32, block_size=4, num_layers=cfg.n_layer,
                           num_heads=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head)
        hook = PagedAttentionHook(pool, allocate_for(pool, S), S)

        got = gpt2_forward(ids, weights, attention_fn=hook)
        with torch.no_grad():
            ref = model(ids).logits

        assert torch.allclose(got, ref, atol=1e-4), \
            f"max|diff| = {(got - ref).abs().max().item()}"
        assert hook.kv_writes == cfg.n_layer * S, "每层每个 token 都应写一次池"

    @pytest.mark.parametrize("block_size", [1, 2, 4, 7, 64])
    def test_block_size_does_not_affect_result(self, gpt2, weights, block_size):
        """分页粒度（每块多大）不应影响结果 —— “无损”的更强形式。"""
        model, tok = gpt2
        cfg = model.config
        ids = tok("Hello world, this is a test", return_tensors="pt").input_ids
        S = ids.shape[1]
        pool = PagedKVPool(num_blocks=S + 4, block_size=block_size,
                           num_layers=cfg.n_layer, num_heads=cfg.n_head,
                           head_dim=cfg.n_embd // cfg.n_head)
        hook = PagedAttentionHook(pool, allocate_for(pool, S), S)

        got = gpt2_forward(ids, weights, attention_fn=hook)
        with torch.no_grad():
            ref = model(ids).logits
        assert torch.allclose(got, ref, atol=1e-4), \
            f"block_size={block_size} 时 max|diff| = {(got - ref).abs().max().item()}"
