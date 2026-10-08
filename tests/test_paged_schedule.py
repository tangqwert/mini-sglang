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
from engine.paged_schedule import (chunked_prefill_generate,
                                   paged_continuous_generate)

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

    def test_survives_slot_zero_finishing_first(self, gpt2, weights):
        """★ 回归：槽位 0 先完成、只剩槽位 1 活跃时，bases 必须按【活跃子集】对齐。

        Step 4 早期版本把整张 `length` 表传给了只含活跃槽位的钩子 —— 于是
        "A 的起点" 配上了 "B 的页表"，写入位置静默错位。它与开头那两条
        契约测试（active 恰好含槽位 0）擦肩而过：**盲区还是在"顺序/边界"**，
        M1 的 EOS 边界、M3 的 full-hit 都是同一类。已在钩子里加长度一致性断言。
        """
        model, tok = gpt2
        reqs = [Request(req_id=0, prompt=tok("Hi", return_tensors="pt").input_ids,
                        max_new_tokens=1),                  # 首步即完成 → 槽位 0 空出
                Request(req_id=1, prompt=tok("Hello world, this is a test",
                                             return_tensors="pt").input_ids,
                        max_new_tokens=4)]                  # 之后独自活跃于槽位 1
        outs, _ = paged_continuous_generate(weights, reqs, make_pool(model.config),
                                           max_batch_size=2)
        for r, got in zip(reqs, outs):
            assert torch.equal(got, naive(r.prompt, weights, r.max_new_tokens)), \
                (got.tolist(), naive(r.prompt, weights, r.max_new_tokens).tolist())

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


LONG_TEXT = ("In a world where artificial intelligence shapes everything, "
             "the race between capability and responsibility defines our era. ")


class TestChunkedPrefill:
    """M6：Chunked Prefill —— prefill 与 decode 合进同一次前向。

    与 Step 4 的差别只在调度器怎么组批：varlen 批次里同时放
    “活跃序列各 1 个 token” 与 “新序列的一大块 prompt”。
    """

    def _reqs(self, tok, texts, budgets):
        return [Request(req_id=i, prompt=tok(t, return_tensors="pt").input_ids,
                        max_new_tokens=b)
                for i, (t, b) in enumerate(zip(texts, budgets))]

    @pytest.mark.parametrize("chunk", [1, 2, 3, 7, 512])
    def test_matches_naive_across_chunk_sizes(self, gpt2, weights, chunk):
        """★ 核心契约：分块大小任意（含 1）都应逐 token 等于朴素解码。

        分块的正确性靠两点：绝对位置跨块连续 + 后续块能看到前面块。
        把 chunk 设成 1 是最狠的压力测试（每个 token 单独一步去喂）。
        """
        model, tok = gpt2
        reqs = self._reqs(tok, ["Hello world", "Hi", "The meaning of life"],
                          [2, 3, 2])

        outs, _ = chunked_prefill_generate(weights, reqs, make_pool(model.config),
                                          max_batch_size=2, max_prefill_tokens=chunk)
        for r, got in zip(reqs, outs):
            assert torch.equal(got, naive(r.prompt, weights, r.max_new_tokens)), \
                f"chunk={chunk} 请求 {r.req_id}: {got.tolist()}"

    def test_no_dedicated_admission_forward(self, gpt2, weights):
        """★ M6 的核心收益：补入不再需要独立前向，且确实与 decode 同批。"""
        model, tok = gpt2
        reqs = self._reqs(tok, ["Hi", "Hello world, this is a test",
                                "Hi", "The meaning of life is"],
                          [1, 3, 1, 3])

        outs, stats = chunked_prefill_generate(weights, reqs, make_pool(model.config),
                                              max_batch_size=2)

        assert stats.admission_forwards == 0, "M6 不该有专门为补入开的前向"
        assert stats.admissions > 0, "本负载应确实发生补入"
        assert stats.mixed_steps > 0, "应至少有一步同时含 prefill 与 decode"
        assert stats.padded_positions == 0
        for r, got in zip(reqs, outs):
            assert torch.equal(got, naive(r.prompt, weights, r.max_new_tokens))

    def test_long_prompt_is_split_into_chunks(self, gpt2, weights):
        """分块生效：长 prompt 被切成多块，而非一步吃下。"""
        model, tok = gpt2
        reqs = self._reqs(tok, [LONG_TEXT, "Hi"], [2, 2])
        L = reqs[0].prompt.numel()
        chunk = 8

        outs, stats = chunked_prefill_generate(weights, reqs, make_pool(model.config),
                                              max_batch_size=2, max_prefill_tokens=chunk)

        assert L > chunk, f"测试前提：prompt 要长于一块（L={L}）"
        assert stats.prefill_tokens == L + reqs[1].prompt.numel(), \
            "prefill 处理的 token 总数应恰等于两个 prompt 之和（没有多余计算）"
        assert stats.prefill_chunks > stats.admissions, \
            f"分块数 {stats.prefill_chunks} 应多于补入次数 {stats.admissions}"
        for r, got in zip(reqs, outs):
            assert torch.equal(got, naive(r.prompt, weights, r.max_new_tokens))

    def test_single_chunk_matches_step4_exactly(self, gpt2, weights):
        """不分块（chunk 足够大）时，M6 应与 Step 4 逐 token 一致。"""
        model, tok = gpt2
        reqs = self._reqs(tok, [LONG_TEXT, "Hi", "The meaning of life"], [3, 2, 4])
        pool_a, pool_b = make_pool(model.config), make_pool(model.config)

        out_a, _ = paged_continuous_generate(weights, reqs, pool_a, max_batch_size=2)
        out_b, stats = chunked_prefill_generate(weights, reqs, pool_b,
                                               max_batch_size=2,
                                               max_prefill_tokens=10 ** 6)

        for a, b in zip(out_a, out_b):
            assert torch.equal(a, b)
        assert stats.mixed_steps > 0, "即使不分块，补入也应与 decode 合流"

    def test_memory_returned_and_prompts_untouched(self, gpt2, weights):
        model, tok = gpt2
        reqs = self._reqs(tok, [LONG_TEXT, "Hi", "Hello world"], [3, 2, 2])
        pool = make_pool(model.config)
        before = [r.prompt.clone() for r in reqs]

        _, stats = chunked_prefill_generate(weights, reqs, pool, max_batch_size=2,
                                            max_prefill_tokens=16)

        assert len(pool.free_blocks) == POOL_BLOCKS, "收工后分页显存应全部归还"
        assert stats.padded_positions == 0
        for r, b in zip(reqs, before):
            assert torch.equal(r.prompt, b)

    def test_empty_requests(self, gpt2, weights):
        model, _ = gpt2
        outs, stats = chunked_prefill_generate(weights, [], make_pool(model.config))
        assert outs == [] and stats.steps == 0
