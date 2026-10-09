"""M7 验收：CUDA Graph 的 decode 步与 eager 路径逐 token 一致。

跑法：pytest tests/test_graph_decode.py -x
（需要 CUDA + triton；缺任一则整个文件跳过）

核心契约：CUDA Graph 只是"把同一串 kernel 打包成一次 launch"，**不改数学**。
所以必须验证三件事：
  ① 图路径 == 朴素解码（逐 token）
  ② 批大小不在捕获集合里时（如 3 条 → 用 4 条的图 + 1 行 padding）结果不受影响
  ③ **padding 行不得污染真实序列** —— 它们照常写 KV，必须被赶到 scratch 页去
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.batching import Request
from engine.graph_decode import PagedDecodeGraphs, graph_decode
from engine.model_forward import gpt2_forward, load_gpt2_weights
from engine.naive_decode import DecodingConfig, autoregressive_generate
from engine.paged_kv import PagedKVPool, ensure_blocks
from engine.paged_schedule import chunked_prefill_generate

triton_paged = pytest.importorskip("engine.triton_paged")
if not triton_paged.HAS_TRITON:                       # pragma: no cover
    pytest.skip("未安装 triton", allow_module_level=True)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="CUDA Graph 需要 CUDA")

PROMPTS = ["The meaning of life is", "Hello world", "Once upon a time",
           "In a shocking finding"]


@pytest.fixture(scope="module")
def gpu():
    model = AutoModelForCausalLM.from_pretrained("gpt2").cuda().eval()
    return model, AutoTokenizer.from_pretrained("gpt2"), load_gpt2_weights(model)


def make_pool(cfg, block_size=16, num_blocks=128):
    return PagedKVPool(num_blocks=num_blocks, block_size=block_size,
                       num_layers=cfg.n_layer, num_heads=cfg.n_head,
                       head_dim=cfg.n_embd // cfg.n_head, device="cuda")


def prefill(weights, pool, prompt, table, max_new):
    """用 eager 路径做 prefill，返回第一个新 token（图路径从这一步接管）。"""
    from engine.triton_paged import TritonPagedAttentionHook
    L = prompt.shape[1]
    ensure_blocks(pool, table, L + max_new)
    hook = TritonPagedAttentionHook(pool, [table], [0], [L])
    logits = gpt2_forward(prompt, weights, attention_fn=hook)
    return int(logits[0, -1].argmax())


def naive(weights, prompt, max_new):
    return autoregressive_generate(lambda ids: gpt2_forward(ids, weights),
                                   prompt, DecodingConfig(max_new_tokens=max_new))


@requires_cuda
class TestGraphDecode:
    def _run(self, gpu, texts, max_new, batch_sizes=(1, 2, 4), max_blocks=8):
        model, tok, weights = gpu
        pool = make_pool(model.config)
        prompts = [tok(t, return_tensors="pt").input_ids.cuda() for t in texts]
        tables = [[] for _ in prompts]
        firsts = [prefill(weights, pool, p, tb, max_new)
                  for p, tb in zip(prompts, tables)]
        # ⚠️ 必须把 prefill 用过的 tables 交给图（KV 已经在那些块里了）
        gen = graph_decode(weights, pool, tables,
                           [p.shape[1] for p in prompts], firsts, max_new,
                           batch_sizes=batch_sizes, max_blocks=max_blocks)
        return prompts, gen

    def test_matches_naive_single_sequence(self, gpu):
        """★ 最基础契约：单序列图解码 == 朴素解码。"""
        prompts, gen = self._run(gpu, [PROMPTS[0]], max_new=5)
        want = naive(gpu[2], prompts[0], 5)[0, prompts[0].shape[1]:].tolist()
        assert gen[0] == want, (gen[0], want)

    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_matches_naive_exact_batch_sizes(self, gpu, n):
        """批大小正好落在捕获集合里（1/2/4）。"""
        texts = PROMPTS[:n]
        prompts, gen = self._run(gpu, texts, max_new=4)
        for p, g in zip(prompts, gen):
            want = naive(gpu[2], p, 4)[0, p.shape[1]:].tolist()
            assert g == want, (g, want)

    def test_padding_row_does_not_corrupt(self, gpu):
        """★ 3 条序列走 4 条的图 → 1 行 padding。padding 行会照常写 KV，
        必须被赶到 scratch 页，否则会静默污染真实序列。"""
        texts = PROMPTS[:3]
        prompts, gen = self._run(gpu, texts, max_new=5, batch_sizes=(4,))
        for p, g in zip(prompts, gen):
            want = naive(gpu[2], p, 5)[0, p.shape[1]:].tolist()
            assert g == want, f"prompt={p.tolist()} 图={g} 朴素={want}"

    def test_padding_identical_to_exact_size(self, gpu):
        """同样的 3 条序列：走 3 档（不存在→用 4 档 + padding）与逐条对照都要一致。

        换个说法：padding 是否引入差异，用"和单条跑"对比来判定。
        """
        texts = PROMPTS[:3]
        _, gen_pad = self._run(gpu, texts, max_new=4, batch_sizes=(4,))
        _, gen_each = self._run(gpu, [texts[0]], max_new=4, batch_sizes=(1,))
        assert gen_pad[0] == gen_each[0], (gen_pad[0], gen_each[0])

    def test_replay_is_repeatable(self, gpu):
        """同一份缓冲区连播两次结果相同（图不该有跨次污染）。"""
        model, tok, weights = gpu
        pool = make_pool(model.config)
        prompt = tok(PROMPTS[1], return_tensors="pt").input_ids.cuda()
        table = []
        first = prefill(weights, pool, prompt, table, 4)

        graphs = PagedDecodeGraphs(weights, pool, batch_sizes=(1,),
                                   max_blocks=8)
        L = prompt.shape[1]
        ids = torch.tensor([first], device="cuda")
        pos = torch.tensor([L], device="cuda")
        a = graphs.step(ids, pos, [table], [L]).clone()
        b = graphs.step(ids, pos, [table], [L]).clone()
        assert torch.equal(a, b), (a.tolist(), b.tolist())

    def test_batch_size_over_capacity_raises(self, gpu):
        model, tok, weights = gpu
        pool = make_pool(model.config)
        graphs = PagedDecodeGraphs(weights, pool, batch_sizes=(1, 2), max_blocks=8)
        with pytest.raises(ValueError):
            graphs.step(torch.zeros(3, dtype=torch.long, device="cuda"),
                        torch.zeros(3, dtype=torch.long, device="cuda"),
                        [[0]] * 3, [0] * 3)

    def test_scratch_block_is_reserved(self, gpu):
        """scratch 页必须从池子里被拿走后不再归还（否则会被真实序列占用）。"""
        model, _, weights = gpu
        pool = make_pool(model.config)
        before = len(pool.free_blocks)
        PagedDecodeGraphs(weights, pool, batch_sizes=(1,), max_blocks=8)
        assert len(pool.free_blocks) == before - 1


M7_TEXTS = ["The meaning of life is", "Hello world, this is a test",
            "I love", "Once upon a time"]
M7_BUDGETS = [3, 5, 2, 4]


@requires_cuda
class TestSchedulerWithGraph:
    """M7.5：把图接进 `chunked_prefill_generate` 后，调度结果不许变。"""

    def _reqs(self, tok):
        return [Request(req_id=i, prompt=tok(t, return_tensors="pt").input_ids.cuda(),
                        max_new_tokens=b)
                for i, (t, b) in enumerate(zip(M7_TEXTS, M7_BUDGETS))]

    def _run(self, gpu, use_graph):
        model, tok, weights = gpu
        reqs = self._reqs(tok)
        outs, stats = chunked_prefill_generate(weights, reqs, make_pool(model.config),
                                              max_batch_size=2, use_graph=use_graph)
        return reqs, outs, stats

    def test_matches_eager_scheduler(self, gpu):
        """★ 图只改"怎么发 kernel"，不该改任何调度决策或输出。"""
        _, out_e, st_e = self._run(gpu, False)
        _, out_g, st_g = self._run(gpu, True)

        for a, b in zip(out_e, out_g):
            assert torch.equal(a, b), (a.tolist(), b.tolist())
        assert st_e.graph_steps == 0
        # 除了 graph_steps，其余统计量必须逐项相等
        for field in ("steps", "admissions", "admission_forwards", "prefill_steps",
                      "mixed_steps", "prefill_chunks", "prefill_tokens",
                      "padded_positions", "idle_slot_steps"):
            assert getattr(st_e, field) == getattr(st_g, field), field

    def test_graph_only_covers_decode_steps(self, gpu):
        """只有纯 decode 步能走图 —— 含 prefill 的步形状是变的。"""
        _, _, st = self._run(gpu, True)
        assert st.graph_steps > 0, "本负载应该有纯 decode 步"
        assert st.graph_steps == st.steps - st.prefill_steps, (
            st.graph_steps, st.steps, st.prefill_steps)

    def test_still_matches_naive_decode(self, gpu):
        """加了图之后，最终契约（与 M0 朴素解码逐 token 一致）仍成立。"""
        reqs, outs, _ = self._run(gpu, True)
        for r, got in zip(reqs, outs):
            want = autoregressive_generate(
                lambda ids: gpt2_forward(ids, gpu[2]), r.prompt,
                DecodingConfig(max_new_tokens=r.max_new_tokens))
            assert torch.equal(got, want), (got.tolist(), want.tolist())

    def test_batch_size_one_and_three(self, gpu):
        """批大小 1 与 3：后者不在 {1,2,4} 里，会走 4 档的图（多一行 padding）。"""
        model, tok, weights = gpu
        for bs, n in ((1, 1), (3, 3)):
            reqs = self._reqs(tok)[:n]
            outs, st = chunked_prefill_generate(
                weights, reqs, make_pool(model.config), max_batch_size=bs,
                use_graph=True)
            assert st.graph_steps > 0
            for r, got in zip(reqs, outs):
                want = autoregressive_generate(
                    lambda ids: gpt2_forward(ids, weights), r.prompt,
                    DecodingConfig(max_new_tokens=r.max_new_tokens))
                assert torch.equal(got, want), (bs, got.tolist(), want.tolist())
