"""engine.model_forward — M4b Step 1：从零手写 GPT-2 前向。

═══════════════════════════════════════════════════════════════════
 为什么需要它（M4a → M4b 的关键一跳）：
   HF 的 attention 内部会先把 past_k / past_v 拼成 [B, H, S, D] 再算注意力 ——
   分页存储"不必连续"的优势，在那一刻就被抹掉了。所以想真正用上 M4a 的
   gather，必须【接管 attention 本身】；而最干净的做法就是自己写前向。

 顺带一句：它也是理解 Transformer 推理的最佳入口 —— 整个前向只有 6 步：
   embed → 12 × (LN₁ → 多头注意力 → 残差 → LN₂ → MLP → 残差) → LN_f → lm_head
═══════════════════════════════════════════════════════════════════
"""
from dataclasses import dataclass, field

import torch

from engine.paged_kv import PagedAttentionHook


@dataclass
class GPT2Weights:
    """从 HF 模型里抽出来的裸权重（之后不再依赖 transformers 的 forward）。

    Attributes:
        wte: token embedding   [vocab, D]
        wpe: 位置 embedding    [ctx, D]
        n_head: 注意力头数
        layers: 每层一个 dict，键名见 load_gpt2_weights
    """

    wte: torch.Tensor
    wpe: torch.Tensor
    n_head: int
    ln_f_w: torch.Tensor
    ln_f_b: torch.Tensor
    layers: list = field(default_factory=list)


def load_gpt2_weights(model) -> GPT2Weights:
    """把 HF 的 GPT2LMHeadModel 权重抽成裸张量。

    HF 用 Conv1D 存线性层：权重形状是 [in, out]，前向写成 `x @ W + b`。
    我们在 gpt2_forward 里保持同样的约定，所以数值才能对上。
    """
    p = {name: param.detach() for name, param in model.named_parameters()}

    layers = []
    for i in range(model.config.n_layer):
        pre = f"transformer.h.{i}"
        layers.append({
            "ln1_w": p[f"{pre}.ln_1.weight"],    "ln1_b": p[f"{pre}.ln_1.bias"],
            "attn_w": p[f"{pre}.attn.c_attn.weight"], "attn_b": p[f"{pre}.attn.c_attn.bias"],
            "proj_w": p[f"{pre}.attn.c_proj.weight"], "proj_b": p[f"{pre}.attn.c_proj.bias"],
            "ln2_w": p[f"{pre}.ln_2.weight"],    "ln2_b": p[f"{pre}.ln_2.bias"],
            "fc_w": p[f"{pre}.mlp.c_fc.weight"], "fc_b": p[f"{pre}.mlp.c_fc.bias"],
            "mproj_w": p[f"{pre}.mlp.c_proj.weight"], "mproj_b": p[f"{pre}.mlp.c_proj.bias"],
        })

    return GPT2Weights(
        wte=p["transformer.wte.weight"],
        wpe=p["transformer.wpe.weight"],
        n_head=model.config.n_head,
        ln_f_w=p["transformer.ln_f.weight"],
        ln_f_b=p["transformer.ln_f.bias"],
        layers=layers,
    )


# ─────────────────────────── 三个基础零件 ───────────────────────────

def _layernorm(x, w, b, eps: float = 1e-5):
    """GPT-2 的 LayerNorm（权重/bias 形状都是 [D]，逐位置归一化）。"""
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * w + b


def _split_heads(x, n_head: int):
    """[B, S, D] → [B, H, S, Dh]（把最后一维切成 H 个头）"""
    B, S, D = x.shape
    return x.view(B, S, n_head, D // n_head).transpose(1, 2)


def _merge_heads(x):
    """[B, H, S, Dh] → [B, S, D]"""
    B, H, S, Dh = x.shape
    return x.transpose(1, 2).reshape(B, S, H * Dh)


def _causal_attention(q, k, v, layer_idx: int | None = None):
    """标准因果缩放点积注意力（M4a 的 paged_attention 就是它的"分页版"）。

    q/k/v 都是 [B, H, S, Dh] → 返回 [B, H, S, Dh]。

    注意 Sq 与 Sk 可以不等（增量解码时 Sq=1、Sk=全长）：
      i 行 j 列的因果掩码 = 上三角（j > i+Sk-Sq 时屏蔽）。
      用 `triu(diagonal=1)` 在 [Sq, Sk] 上直接构造即可，两种形态都正确。

    `layer_idx` 只是为了与其它 attention_fn 统一签名（分页版靠它定位层），
    连续版用不上。
    """
    Dh = q.shape[-1]
    scores = q @ k.transpose(-1, -2) / Dh ** 0.5
    Sq, Sk = scores.shape[-2], scores.shape[-1]
    causal = torch.triu(
        torch.full((Sq, Sk), float("-inf"), dtype=scores.dtype, device=scores.device),
        diagonal=1)
    return torch.softmax(scores + causal, dim=-1) @ v


# ─────────────────────────── 主前向 ───────────────────────────

def gpt2_forward(input_ids: torch.LongTensor,
                 weights: GPT2Weights,
                 position_ids: torch.LongTensor | None = None,
                 attention_fn=None) -> torch.Tensor:
    """GPT-2 前向（不经过 transformers 的 forward）。

    Args:
        input_ids: LongTensor[B, S]。
        weights: `load_gpt2_weights` 的产物。
        position_ids: LongTensor[B, S]；None 时用 0..S-1。
            ⚠️ 增量解码【必须】显式传 —— 此时 S=1，但它是第 pos 个 token。
        attention_fn: (q, k, v, layer_idx) -> out，各为 [B, H, S, Dh]。
            默认连续版 `_causal_attention`；**M4b Step 2 传入分页版**
            （PagedAttentionHook），这就是替换 attention 的接缝。

    Returns:
        logits [B, S, vocab]。
    """
    B, S = input_ids.shape
    D = weights.wte.shape[1]
    if position_ids is None:
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
    attention_fn = attention_fn or _causal_attention

    with torch.no_grad():
        # ── ① embedding：token + 位置 ──
        x = weights.wte[input_ids] + weights.wpe[position_ids]

        # ── ② 逐层 ──
        for layer_idx, W in enumerate(weights.layers):
            h = _layernorm(x, W["ln1_w"], W["ln1_b"])
            qkv = h @ W["attn_w"] + W["attn_b"]                 # [B, S, 3D]
            q, k, v = (_split_heads(t, weights.n_head)
                       for t in qkv.split(D, dim=2))            # 各 [B, H, S, Dh]
            y = attention_fn(q, k, v, layer_idx)
            x = x + _merge_heads(y) @ W["proj_w"] + W["proj_b"]  # 残差

            h = _layernorm(x, W["ln2_w"], W["ln2_b"])
            h = torch.nn.functional.gelu(h @ W["fc_w"] + W["fc_b"], approximate="tanh")
            x = x + h @ W["mproj_w"] + W["mproj_b"]             # 残差

        # ── ③ 收尾：LN + 权重共享的输出投影（lm_head = wteᵀ）──
        x = _layernorm(x, weights.ln_f_w, weights.ln_f_b)
        return x @ weights.wte.T


# ─────────────────── M4b Step 3：分页增量解码 ───────────────────

def paged_generate(weights: GPT2Weights,
                   prompt_ids: torch.LongTensor,
                   n_new: int,
                   pool,
                   block_table: list) -> torch.LongTensor:
    """用自研前向 + 分页 KV 池做贪心解码。

    结构 = prefill 一步 + decode (n_new - 1) 步：
      prefill：整条 prompt 一次前向（绝对位置 0..T-1），K/V 落进池
      decode ：每步只喂 1 个新 token（绝对位置 T+k），K/V 追加进池，
               注意力从池里 gather（`total` 就是"全部过去"，所以无需因果截断）

    ⚠️ decode 步【必须】显式给 position_ids —— 此时 S=1，但它是第 pos 个 token。
       漏了就退化成"一直按位置 0 算"，输出全错（M1 那轮踩过同类坑）。

    Returns:
        LongTensor[1, T + n_new]，prompt + 生成部分（含预 prompt）。
    """
    hook = PagedAttentionHook(pool, block_table)
    device = prompt_ids.device
    out = prompt_ids[0].tolist()

    # ── prefill ──
    logits = gpt2_forward(prompt_ids, weights, attention_fn=hook)
    nxt = int(logits[0, -1].argmax())
    out.append(nxt)

    # ── decode：每步只喂 1 个 token ──
    for _ in range(n_new - 1):
        pos = len(out) - 1                       # 待喂 token 的绝对位置
        logits = gpt2_forward(
            torch.tensor([[nxt]], device=device), weights, attention_fn=hook,
            position_ids=torch.tensor([[pos]], device=device))
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)

    return torch.tensor([out], device=device)
