"""真模型集成测试：用真 gpt2 验证 naive 与 kv 两条解码路径输出一致。

与 tests/test_kv_cache.py 的区别：
  那里用确定性假模型（快、可精确断言内部行为）；
  这里用真模型（慢、需要 GPU/权重），专测假模型覆盖不到的东西：
    - 真 HF 模型的 past_key_values 语义（DynamicCache）
    - 真实注意力下，增量式前向与整条序列前向的数值等价性
    - 设备搬运（prompt 必须 .cuda()，cache 住在模型侧）

跑法：pytest tests/test_real_model.py -v
注意：需要 GPU + 已下载的 gpt2 权重；比单元测试慢（要真前向）。
没有 CUDA 时自动跳过（CI 或 CPU 机器不会误报红）。
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.naive_decode import (
    DecodingConfig,
    autoregressive_generate,
    build_logits_fn,
)
from engine.kv_cache import build_kv_forward, kv_generate

MODEL_NAME = "gpt2"
PROMPT = "The meaning of life is"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="真模型集成测试需要 GPU",
)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def model():
    """module 级 fixture：整个文件只加载一次权重（约 500MB，别重复加载）。"""
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()
    m.eval()
    return m


@pytest.fixture(scope="module")
def prompt_ids(tok):
    ids = tok(PROMPT, return_tensors="pt").input_ids
    return ids.cuda()  # 序列栖息地在 GPU（M0 的教训）


def test_kv_output_equals_naive(model, prompt_ids):
    """核心验收：两条解码路径逐 token 一致。

    这个测试的存在意义：假模型测试曾放过"返回旧 cache"的 bug
    （假模型的输出只依赖最后一个 token，丢 cache 也"对"）。
    真模型依赖完整上下文，任何 cache 传递错误都会在这里现形。
    """
    cfg = DecodingConfig(max_new_tokens=20, eos_token_id=None)
    out_naive = autoregressive_generate(build_logits_fn(model, "cuda"), prompt_ids, cfg)
    out_kv = kv_generate(build_kv_forward(model, "cuda"), prompt_ids, cfg)
    assert torch.equal(out_naive, out_kv)


def test_kv_generates_coherent_text(model, tok, prompt_ids):
    """冒烟：生成结果应该包含可读文本，而不是退化复读。

    丢 cache 的典型症状是 "the the the the..."——最后一个生成 token 的
    唯一性是个廉价的健全性指标。
    """
    cfg = DecodingConfig(max_new_tokens=16, eos_token_id=None)
    out = kv_generate(build_kv_forward(model, "cuda"), prompt_ids, cfg)
    generated = out[0, prompt_ids.shape[1]:].tolist()
    assert len(set(generated)) > 1, f"生成退化为复读: {generated}"
