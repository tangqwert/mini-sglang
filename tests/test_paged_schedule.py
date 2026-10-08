"""M4b Step 4 验收：分页 KV 池接进调度器（连续准入 + 零填充）。

跑法：pytest tests/test_paged_schedule.py -x

核心契约：
  ① 输出逐 token == 每条请求单独跑朴素解码（真 gpt2，逐 token 相等）
  ② padded_positions == 0 —— 分页之后不再需要左填充（Step 4 的核心收益）
  ③ 请求完成即归还全部分页显存（free_blocks 复原）
  ④ 同一轮的多条补入合并进一次前向（补入不再"每条一次独立前向"）

对比对象是 M2.5 的 continuous_generate：那里 padded_positions > 0、
补入得填充到当前批长 S、admission_forwards == admissions。
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.batching import Request
from engine.model_forward import gpt2_forward, load_gpt2_weights
from engine.naive_decode import DecodingConfig, autoregressive_generate
from engine.paged_kv import PagedKVPool
from engine.paged_schedule import paged_continuous_generate

POOL_BLOCKS = 256


@pytest.fixture(scope="module")
def gpt2():
    return AutoModelForCausalLM.from_pretrained("gpt2").eval(), \
        AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def weights(gpt2):
    return load_gpt2_weights(gpt2[0])


def make_pool(cfg, block_size: int = 8) -> PagedKVPool:
    return PagedKVPool(num_blocks=POOL_BLOCKS, block_size=block_size,
                       num_layers=cfg.n_layer, num_heads=cfg.n_head,
                       head_dim=cfg.n_embd // cfg.n_head)


def naive(prompt, weights, budget, eos_token_id=None):
    """基线：单条请求的朴素解码，但用同一个自研前向（每步重算整条序列）。"""
    return autoregressive_generate(
        lambda ids: gpt2_forward(ids, weights), prompt,
        DecodingConfig(max_new_tokens=budget, eos_token_id=eos_token_id))


class TestPagedContinuous:
    def test_matches_per_request_naive(self, gpt2, weights):
        """★ 核心契约：批式分页调度 == 每条单独朴素解码（逐 token 相等）。"""
        model, tok = gpt2
        texts = ["The meaning of life is", "Hello world, this is a test",
                 "I love", "Once upon a time"]
        budgets = [3, 5, 2, 4]
        reqs = [Request(req_id=i, prompt=tok(t, return_tensors="pt").input_ids,
                        max_new_tokens=b)
                for i, (t, b) in enumerate(zip(texts, budgets))]

        outs, stats = paged_continuous_generate(weights, reqs, make_pool(model.config),
                                               max_batch_size=2)

        assert len(outs) == len(reqs)
        for r, got in zip(reqs, outs):
            want = naive(r.prompt, weights, r.max_new_tokens)
            assert torch.equal(got, want), (got.tolist(), want.tolist())
        assert stats.padded_positions == 0

    @pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16])
    def test_independent_of_block_size(self, gpt2, weights, block_size):
        """分页粒度不该影响结果（页表是逻辑抽象，与物理块大小无关）。"""
        model, tok = gpt2
        reqs = [Request(req_id=i, prompt=tok(t, return_tensors="pt").input_ids,
                        max_new_tokens=3)
                for i, t in enumerate(["Hello world, this is a test", "Hi"])]

        outs, _ = paged_continuous_generate(weights, reqs,
                                            make_pool(model.config, block_size),
                                            max_batch_size=2)
        for r, got in zip(reqs, outs):
            assert torch.equal(got, naive(r.prompt, weights, r.max_new_tokens)), \
                f"block_size={block_size}"

    def test_no_padding_and_memory_fully_returned(self, gpt2, weights):
        """②+③：零填充；跑完后分页显存完整归还（没有泄漏的 block）。"""
        model, tok = gpt2
        texts = ["Hello world, this is a test", "The meaning of life is",
                 "Hi", "Once upon a time"]
        reqs = [Request(req_id=i, prompt=tok(t, return_tensors="pt").input_ids,
                        max_new_tokens=3)
                for i, t in enumerate(texts)]
        pool = make_pool(model.config)

        # 跑之前先记录"峰值占用"：每条序列最长会到 L + budget，且最后一步的
        # K/V 还没进池，所以理论上限 = Σ ceil((L_i + budget_i - 1) / bs)
        peak = sum(-(-(r.prompt.numel() + r.max_new_tokens - 1) // pool.block_size)
                   for r in reqs)
        assert peak < POOL_BLOCKS, "池子要够大，否则测试会因为耗尽而失败"

        _, stats = paged_continuous_generate(weights, reqs, pool, max_batch_size=3)

        assert stats.padded_positions == 0, "分页路径不该产生任何填充位置"
        assert stats.admission_forwards == 1, "两条补入合并成一次前向"
        assert len(pool.free_blocks) == POOL_BLOCKS, "收工后分页显存应全部归还"

    def test_admissions_are_batched_into_one_forward(self, gpt2, weights):
        """④ 4 条同预算请求、B=2：两条补入合并进【一次】前向（M2.5 是每条一次）。

        "Hi" 是 1 个 token，预算 2 → prefill 出首 token（剩 1），再 decode 一步收工。
        所以两步就把 4 条全部跑完，两个槽位同时空出 → 补入可以合并。
        """
        model, tok = gpt2
        reqs = [Request(req_id=i, prompt=tok("Hi", return_tensors="pt").input_ids,
                        max_new_tokens=2)
                for i in range(4)]
        pool = make_pool(model.config)

        outs, stats = paged_continuous_generate(weights, reqs, pool, max_batch_size=2)

        assert stats.admissions == 2
        assert stats.admission_forwards == 1, "两个槽位同时空出 → 补入应合并成一次前向"
        assert stats.steps == 2
        assert stats.idle_slot_steps == 0, "本安排下批始终满载"
        assert len(pool.free_blocks) == POOL_BLOCKS, "收工后显存应全部归还"
        for r, got in zip(reqs, outs):
            assert torch.equal(got, naive(r.prompt, weights, r.max_new_tokens))

    def test_eos_stops_only_that_request(self, gpt2, weights):
        """EOS 逐条独立：把 2 号请求的 EOS 设成它自己会生成的 token。"""
        model, tok = gpt2
        prompt = tok("The meaning of life is", return_tensors="pt").input_ids
        full = naive(prompt, weights, 4)                    # 先看它自然会生成什么
        eos = int(full[0, prompt.shape[1] + 1])             # 第 2 个生成 token 当作 EOS

        reqs = [Request(req_id=0, prompt=prompt, max_new_tokens=4, eos_token_id=eos),
                Request(req_id=1, prompt=prompt, max_new_tokens=4)]
        outs, _ = paged_continuous_generate(weights, reqs, make_pool(model.config),
                                            max_batch_size=2)

        want0 = naive(prompt, weights, 4, eos_token_id=eos)
        want1 = naive(prompt, weights, 4)
        assert torch.equal(outs[0], want0), (outs[0].tolist(), want0.tolist())
        assert torch.equal(outs[1], want1)
        assert outs[0].shape[1] < outs[1].shape[1], "命中 EOS 的那条应更短"

    def test_empty_requests(self, gpt2, weights):
        model, _ = gpt2
        outs, stats = paged_continuous_generate(weights, [], make_pool(model.config))
        assert outs == [] and stats.steps == 0

    def test_prompts_not_mutated(self, gpt2, weights):
        """分页路径不得修改调用方传入的 prompt（与 M2/M2.5 同一契约）。"""
        model, tok = gpt2
        reqs = [Request(req_id=i, prompt=tok(t, return_tensors="pt").input_ids,
                        max_new_tokens=2)
                for i, t in enumerate(["Hello world", "Hi", "The meaning of life"])]
        before = [r.prompt.clone() for r in reqs]

        outs, _ = paged_continuous_generate(weights, reqs, make_pool(model.config),
                                           max_batch_size=2)

        for r, b in zip(reqs, before):
            assert torch.equal(r.prompt, b)
        for r, got in zip(reqs, outs):          # 输出必须以【原始】 prompt 开头
            assert torch.equal(got[0, : r.prompt.shape[1]], r.prompt[0])
