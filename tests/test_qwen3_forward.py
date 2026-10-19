"""M8 验收：Qwen3 前向（GQA + RoPE + RMSNorm + SwiGLU + QK-Norm）。

跑法：pytest tests/test_qwen3_forward.py -x
（第一次跑需要下载 Qwen/Qwen3-0.6B ≈ 1.2GB；无 CUDA 时部分用例跳过）

为什么用 Qwen3-0.6B：它与官方 mini-sglang 的 bench 同款配置，
所以本项目的数字可以直接和官方对照。同时它覆盖了 GPT-2 没有的**全部**现代组件：

    LayerNorm → RMSNorm ｜ 学到的 wpe → RoPE ｜ MHA → **GQA** ｜ GELU → SwiGLU ｜ +QK-Norm

核心契约：`qwen3_forward` 与 HF 输出逐元素一致；且**接入分页路径后逐 token 仍与
朴素解码一致** —— 这才是"架构无关的引擎"该有的验收标准。
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.model_forward import paged_generate
from engine.naive_decode import DecodingConfig, autoregressive_generate
from engine.paged_kv import PagedKVPool, allocate_for
from engine.qwen3_forward import load_qwen3_weights, qwen3_forward

MODEL = "Qwen/Qwen3-0.6B"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")


@pytest.fixture(scope="module")
def qwen3():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval().to(dev)
    tok = AutoTokenizer.from_pretrained(MODEL)
    return model, tok, load_qwen3_weights(model)


def make_pool(cfg, block_size=16, num_blocks=32, device="cuda"):
    return PagedKVPool(num_blocks=num_blocks, block_size=block_size,
                       num_layers=cfg.num_hidden_layers,
                       num_heads=cfg.num_key_value_heads,      # ← GQA：池子只存 KV 头
                       head_dim=cfg.head_dim, device=device)


class TestQwen3Forward:
    def test_config_is_gqa(self, qwen3):
        """先把架构事实钉住：16 个 q 头 / 8 个 kv 头 / head_dim 128。"""
        _, _, w = qwen3
        assert (w.n_head, w.n_kv_head, w.head_dim) == (16, 8, 128)
        assert w.n_head != w.n_kv_head, "本用例必须覆盖 GQA"

    def test_matches_hf_logits(self, qwen3):
        """★ 核心契约：与 HF 的 logits 逐元素一致。"""
        model, tok, w = qwen3
        ids = tok("The meaning of life is", return_tensors="pt").input_ids.to(w.embed.device)
        with torch.no_grad():
            ref = model(ids).logits
        got = qwen3_forward(ids, w)
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-4), \
            f"max|diff| = {(got - ref).abs().max().item()}"

    def test_greedy_next_token_matches(self, qwen3):
        """端到端：贪心取的第一个生成 token 必须一致。"""
        model, tok, w = qwen3
        ids = tok("Hello world, this is a test", return_tensors="pt").input_ids.to(w.embed.device)
        with torch.no_grad():
            ref = int(model(ids).logits[0, -1].argmax())
        got = int(qwen3_forward(ids, w)[0, -1].argmax())
        assert got == ref

    def test_prefix_matches_full_forward(self, qwen3):
        """因果性 —— 增量解码成立的依据（也是 RoPE 位置参数正确性的体检）。"""
        _, tok, w = qwen3
        ids = tok("The meaning of life is", return_tensors="pt").input_ids.to(w.embed.device)
        full = qwen3_forward(ids, w)
        k = 4
        part = qwen3_forward(ids[:, :k], w)          # 不传 position_ids，用 0..k-1
        assert torch.allclose(part[0, -1], full[0, k - 1], atol=1e-4), \
            "只用前 k 个 token 前向，末位 logits 应等于整条前向的第 k-1 位"


@needs_cuda
class TestQwen3Paged:
    """把 Qwen3 接进分页路径 —— 这才是"架构无关的引擎"的验收。"""

    def test_paged_generate_matches_naive(self, qwen3):
        """★ 分页 + GQA 的增量解码 == 每条单独跑朴素解码（逐 token）。"""
        model, tok, w = qwen3
        cfg = model.config
        prompt = tok("The meaning of life is", return_tensors="pt").input_ids.cuda()
        T, N = prompt.shape[1], 5

        pool = make_pool(cfg)
        got = paged_generate(w, prompt, N, pool, allocate_for(pool, T + N),
                            forward_fn=qwen3_forward)
        want = autoregressive_generate(lambda ids: qwen3_forward(ids, w), prompt,
                                       DecodingConfig(max_new_tokens=N))
        assert torch.equal(got, want), (got.tolist(), want.tolist())

    def test_paged_generate_with_triton_backend(self, qwen3):
        """换成 Triton kernel（含 GQA 索引）后结果不变。"""
        triton_paged = pytest.importorskip("engine.triton_paged")
        if not triton_paged.HAS_TRITON:              # pragma: no cover
            pytest.skip("未安装 triton")
        from engine.triton_paged import TritonPagedAttentionHook

        model, tok, w = qwen3
        cfg = model.config
        prompt = tok("Hello world, this is a test", return_tensors="pt").input_ids.cuda()
        T, N = prompt.shape[1], 5

        pool = make_pool(cfg)
        got = paged_generate(w, prompt, N, pool, allocate_for(pool, T + N),
                            forward_fn=qwen3_forward, hook_cls=TritonPagedAttentionHook)
        want = autoregressive_generate(lambda ids: qwen3_forward(ids, w), prompt,
                                       DecodingConfig(max_new_tokens=N))
        assert torch.equal(got, want), (got.tolist(), want.tolist())

    def test_pool_uses_kv_head_count(self, qwen3):
        """池子必须按 **KV 头数** 开，而不是 q 头数 —— 否则显存白翻一倍。"""
        model, _, w = qwen3
        pool = make_pool(model.config)
        assert pool.num_heads == 8
        assert pool.keys.shape[3] == 8

    def test_full_engine_matches_naive(self, qwen3):
        """★ 端到端：Qwen3-0.6B 走**完整调度器**（连续准入 + Chunked Prefill），
        两个注意力后端都与朴素解码逐 token 一致。

        这条测试才是"架构无关的引擎"的真正验收 ——
        它同时跨过了 RoPE / RMSNorm / SwiGLU / GQA / 分页 / 两个 kernel 后端。
        """
        from engine.batching import Request
        from engine.paged_schedule import chunked_prefill_generate
        from engine.triton_paged import TritonPagedAttentionHook

        model, tok, w = qwen3
        cfg = model.config
        texts = ["The meaning of life is", "Hello world, this is a test",
                 "I love", "Once upon a time"]
        reqs = [Request(req_id=i, prompt=tok(t, return_tensors="pt").input_ids.cuda(),
                        max_new_tokens=b)
                for i, (t, b) in enumerate(zip(texts, [3, 5, 2, 4]))]

        def run(hook_cls):
            pool = PagedKVPool(num_blocks=128, block_size=16,
                               num_layers=cfg.num_hidden_layers,
                               num_heads=cfg.num_key_value_heads,
                               head_dim=cfg.head_dim, device="cuda")
            return chunked_prefill_generate(w, reqs, pool, max_batch_size=2,
                                            hook_cls=hook_cls,
                                            forward_fn=qwen3_forward)[0]

        outs_torch = run(None)
        outs_triton = run(TritonPagedAttentionHook)
        for a, b in zip(outs_torch, outs_triton):
            assert torch.equal(a, b), "两个注意力后端必须逐 token 一致"
        for r, got in zip(reqs, outs_torch):
            want = autoregressive_generate(lambda ids: qwen3_forward(ids, w),
                                           r.prompt,
                                           DecodingConfig(max_new_tokens=r.max_new_tokens))
            assert torch.equal(got, want), (got.tolist(), want.tolist())
