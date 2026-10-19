"""M5 端到端验收：把注意力实现换成 Triton kernel，输出与调度决策都不许变。

跑法：pytest tests/test_triton_schedule.py -x
（需要 CUDA；无 GPU 时整个文件跳过）

为什么这条测试最关键：M5 换掉的是整个注意力的实现（gather + 因果掩码 +
softmax + 加权求和 → 一个 kernel）。调度器只通过 `hook_cls` 这一个参数得知区别，
所以正确的判据是**两条路径的输出逐 token 相同、统计量完全相同**。
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

triton_paged = pytest.importorskip("engine.triton_paged")
if not triton_paged.HAS_TRITON:                       # pragma: no cover
    pytest.skip("未安装 triton", allow_module_level=True)
TritonHook = triton_paged.TritonPagedAttentionHook

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="Triton kernel 需要 CUDA")

TEXTS = ["The meaning of life is", "Hello world, this is a test",
         "I love", "Once upon a time"]
BUDGETS = [3, 5, 2, 4]


@pytest.fixture(scope="module")
def gpu():
    model = AutoModelForCausalLM.from_pretrained("gpt2").cuda().eval()
    return model, AutoTokenizer.from_pretrained("gpt2"), load_gpt2_weights(model)


def make_pool(cfg, block_size=8):
    return PagedKVPool(num_blocks=256, block_size=block_size,
                       num_layers=cfg.n_layer, num_heads=cfg.n_head,
                       head_dim=cfg.n_embd // cfg.n_head, device="cuda")


def build(gpu, n=len(TEXTS), budgets=None):
    _, tok, _ = gpu
    budgets = budgets or BUDGETS[:n]
    return [Request(req_id=i, prompt=tok(t, return_tensors="pt").input_ids.cuda(),
                    max_new_tokens=b)
            for i, (t, b) in enumerate(zip(TEXTS[:n], budgets))]


@pytest.mark.parametrize("block_size", [1, 4, 8, 16])
def test_chunked_triton_matches_torch_backend(gpu, block_size):
    """★ 换掉注意力实现，输出与统计量必须完全一致。"""
    model, _, weights = gpu
    reqs_a, reqs_b = build(gpu), build(gpu)

    out_a, st_a = chunked_prefill_generate(weights, reqs_a, make_pool(model.config, block_size),
                                           max_batch_size=2)
    out_b, st_b = chunked_prefill_generate(weights, reqs_b, make_pool(model.config, block_size),
                                           max_batch_size=2, hook_cls=TritonHook)

    for a, b, r in zip(out_a, out_b, reqs_a):
        assert torch.equal(a, b), (r.req_id, a.tolist(), b.tolist())
    assert st_a == st_b, "调度决策不该因为 attention 实现不同而变化"


def test_chunked_triton_matches_naive_decode(gpu):
    """Triton 路径仍须与 M0 朴素解码逐 token 一致（最终契约）。"""
    model, _, weights = gpu
    reqs = build(gpu)
    outs, _ = chunked_prefill_generate(weights, reqs, make_pool(model.config),
                                       max_batch_size=2, hook_cls=TritonHook)
    for r, got in zip(reqs, outs):
        want = autoregressive_generate(
            lambda ids: gpt2_forward(ids, weights), r.prompt,
            DecodingConfig(max_new_tokens=r.max_new_tokens))
        assert torch.equal(got, want), (r.req_id, got.tolist(), want.tolist())


def test_step4_triton_matches_torch_backend(gpu):
    """Step 4 调度器（无 chunking）也要能用 Triton 后端替换。"""
    model, _, weights = gpu
    reqs_a, reqs_b = build(gpu), build(gpu)

    out_a, st_a = paged_continuous_generate(weights, reqs_a, make_pool(model.config),
                                            max_batch_size=2)
    out_b, st_b = paged_continuous_generate(weights, reqs_b, make_pool(model.config),
                                            max_batch_size=2, hook_cls=TritonHook)

    for a, b in zip(out_a, out_b):
        assert torch.equal(a, b)
    assert st_a == st_b


@pytest.mark.parametrize("chunk", [1, 3, 512])
def test_triton_backend_under_chunked_prefill(gpu, chunk):
    """chunked prefill（含 chunk=1 的极端分块）下 Triton 后端也要正确。"""
    model, _, weights = gpu
    reqs = build(gpu)
    outs, _ = chunked_prefill_generate(weights, reqs, make_pool(model.config),
                                       max_batch_size=2, max_prefill_tokens=chunk,
                                       hook_cls=TritonHook)
    for r, got in zip(reqs, outs):
        want = autoregressive_generate(
            lambda ids: gpt2_forward(ids, weights), r.prompt,
            DecodingConfig(max_new_tokens=r.max_new_tokens))
        assert torch.equal(got, want), f"chunk={chunk}"


def test_triton_backend_releases_memory(gpu):
    """Triton 路径同样要把分页显存还干净。"""
    model, _, weights = gpu
    pool = make_pool(model.config)
    _, stats = chunked_prefill_generate(weights, build(gpu), pool, max_batch_size=2,
                                        hook_cls=TritonHook)
    assert len(pool.free_blocks) == pool.num_blocks
    assert stats.padded_positions == 0
    assert stats.admission_forwards == 0
