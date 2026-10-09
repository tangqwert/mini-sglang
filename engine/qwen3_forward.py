"""engine.qwen3_forward — M8：Qwen3 前向（GQA + RoPE + RMSNorm + SwiGLU + QK-Norm）。

═══════════════════════════════════════════════════════════════════
 为什么加它：从"能跑 GPT-2"到"能跑真模型"

 GPT-2 与 Qwen3 的差距，正好覆盖了 2023 年之后所有主流 LLM 的全部区别：

   组件            GPT-2                Qwen3（LLaMA 系）
   ─────────────────────────────────────────────────────────────
   归一化          LayerNorm            **RMSNorm**（去均值、无 bias）
   位置编码        学到的 wpe           **RoPE**（旋转，作用在 q/k 上）
   注意力头        MHA（Hq == Hkv）     **GQA**（16 q-heads / 8 kv-heads）
   MLP             c_fc→GELU→c_proj     **SwiGLU**（gate/up/down，silu）
   额外            —                    **QK-Norm**（对 q/k 各加一层 RMSNorm）
   线性层 bias     有                   **无**

 所以补上这五项之后，`engine/` 就能说自己支持**架构无关的前向**，而不是"只会 GPT-2"。

 实测配置（Qwen/Qwen3-0.6B，与官方 mini-sglang 的 bench 同款）：
   28 层 / hidden 1024 / 16 q-heads / **8 kv-heads** / head_dim 128
   vocab 151936 / intermediate 3072 / rms_eps 1e-6 / rope_theta 1e6 / 权重绑定

 权重命名（从真实 state_dict 抄下来的，311 个 key）::

   model.embed_tokens.weight
   model.layers.{i}.input_layernorm.weight
   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
   model.layers.{i}.self_attn.{q,k}_norm.weight
   model.layers.{i}.post_attention_layernorm.weight
   model.layers.{i}.mlp.{gate,up,down}_proj.weight
   model.norm.weight / lm_head.weight
═══════════════════════════════════════════════════════════════════
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from engine.model_forward import _merge_heads


@dataclass
class Qwen3Weights:
    """从 HF Qwen3 模型里抽出来的裸权重。

    Attributes:
        embed: [vocab, D]（权重绑定，也当 lm_head 用）
        n_head / n_kv_head: q 与 kv 的头数（GQA 下 n_kv_head < n_head）
        head_dim: 每个头的维度（**注意不等于 D / n_head**：Qwen3-0.6B 是
                  D=1024、16 头、head_dim=128，所以 q_proj 输出是 2048 维）
        rms_eps / rope_theta: RMSNorm 的 eps 与 RoPE 的 base
    """

    embed: torch.Tensor
    n_head: int
    n_kv_head: int
    head_dim: int
    rms_eps: float
    rope_theta: float
    norm_w: torch.Tensor
    lm_head: torch.Tensor
    layers: list = field(default_factory=list)


def load_qwen3_weights(model) -> Qwen3Weights:
    """把 HF 的 Qwen3ForCausalLM 权重抽成裸张量。

    HF 的 `nn.Linear` 权重是 `[out_features, in_features]`，所以前向写成 `x @ W.T`。
    （注意这与 GPT-2 的 Conv1D `[in, out]` + `x @ W` 相反 —— 别串了。）
    """
    p = {name: param.detach() for name, param in model.named_parameters()}
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    layers = []
    for i in range(cfg.num_hidden_layers):
        pre = f"model.layers.{i}"
        layers.append({
            "ln1_w": p[f"{pre}.input_layernorm.weight"],
            "q_w": p[f"{pre}.self_attn.q_proj.weight"],
            "k_w": p[f"{pre}.self_attn.k_proj.weight"],
            "v_w": p[f"{pre}.self_attn.v_proj.weight"],
            "o_w": p[f"{pre}.self_attn.o_proj.weight"],
            "q_norm_w": p[f"{pre}.self_attn.q_norm.weight"],
            "k_norm_w": p[f"{pre}.self_attn.k_norm.weight"],
            "ln2_w": p[f"{pre}.post_attention_layernorm.weight"],
            "gate_w": p[f"{pre}.mlp.gate_proj.weight"],
            "up_w": p[f"{pre}.mlp.up_proj.weight"],
            "down_w": p[f"{pre}.mlp.down_proj.weight"],
        })

    rope = getattr(cfg, "rope_parameters", None) or {}
    return Qwen3Weights(
        embed=p["model.embed_tokens.weight"],
        n_head=cfg.num_attention_heads,
        n_kv_head=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        head_dim=head_dim,
        rms_eps=getattr(cfg, "rms_norm_eps", 1e-6),
        rope_theta=rope.get("rope_theta", getattr(cfg, "rope_theta", 1000000.0)),
        norm_w=p["model.norm.weight"],
        lm_head=p.get("lm_head.weight", p["model.embed_tokens.weight"]),
        layers=layers,
    )


# ─────────────────────────── 三个基础零件 ───────────────────────────

def _rmsnorm(x, w, eps: float):
    """RMSNorm：**不减均值**、不除标准差，只除以均方根。

    与 LayerNorm 的区别就这一点，但省掉了一次全量均值规约 ——
    这就是为什么现在的模型几乎都用它。

    ⚠️ `w.to(x.dtype)`：HF 会把 norm 权重**留在 fp32**（即使模型是 bf16），
    不对齐的话 q/k 会被提升成 fp32 而 v 还是 bf16 → 后面 `@` 直接报 dtype 错。
    """
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w.to(x.dtype)


def _rope_freqs(position_ids: torch.LongTensor, head_dim: int, theta: float):
    """预计算 RoPE 的 cos/sin，形状 [B, 1, S, head_dim]（1 维便于对 head 广播）。"""
    inv_freq = 1.0 / (theta ** (torch.arange(
        0, head_dim, 2, dtype=torch.float32, device=position_ids.device) / head_dim))
    freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]   # [B,S,D/2]
    emb = torch.cat([freqs, freqs], dim=-1)                             # [B,S,D]
    return emb.cos()[:, None, :, :], emb.sin()[:, None, :, :]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x: [B, H, S, D]。`rotate_half` 是 HF 约定：后半段取负搬到前面。

    直觉：把 D 维看成 D/2 个二维平面，每个平面按位置角旋转。
    这样 q·k 只依赖**相对位置差** —— 这是 RoPE 相对位置编码的本质。
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


# ─────────────────────────── 主前向 ───────────────────────────

def qwen3_forward(input_ids: torch.LongTensor,
                  weights: Qwen3Weights,
                  position_ids: torch.LongTensor | None = None,
                  attention_fn=None) -> torch.Tensor:
    """Qwen3 前向（不经过 transformers 的 forward）。

    Args:
        input_ids: LongTensor[B, S]
        position_ids: LongTensor[B, S]；None 时用 0..S-1。
            ⚠️ 增量解码【必须】显式传绝对位置 —— 与 gpt2_forward 同一个坑。
        attention_fn: (q, k, v, layer_idx) -> out。
            q 是 [B, Hq, S, Dh]，而 k/v 是 [B, Hkv, S, Dh]（**GQA 下头数不同**）；
            分页版钩子内部按 `Hq // Hkv` 分组索引。

    Returns:
        logits [B, S, vocab]
    """
    B, S = input_ids.shape
    D = weights.embed.shape[1]
    Hq, Hkv, Dh = weights.n_head, weights.n_kv_head, weights.head_dim
    if position_ids is None:
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
    if attention_fn is None:
        from engine.model_forward import _causal_attention
        attention_fn = _causal_attention

    with torch.no_grad():
        # ── ① embedding（无位置 embedding：位置信息由 RoPE 注入）──
        x = weights.embed[input_ids]
        cos, sin = _rope_freqs(position_ids, Dh, weights.rope_theta)

        for layer_idx, W in enumerate(weights.layers):
            h = _rmsnorm(x, W["ln1_w"], weights.rms_eps)

            # q 有 Hq 个头、k/v 只有 Hkv 个（GQA）
            q = (h @ W["q_w"].T).view(B, S, Hq, Dh)
            k = (h @ W["k_w"].T).view(B, S, Hkv, Dh)
            v = (h @ W["v_w"].T).view(B, S, Hkv, Dh)

            # ② QK-Norm：对每个头自己的 Dh 维做 RMSNorm（**在 RoPE 之前**）
            q = _rmsnorm(q, W["q_norm_w"], weights.rms_eps).transpose(1, 2)
            k = _rmsnorm(k, W["k_norm_w"], weights.rms_eps).transpose(1, 2)
            v = v.transpose(1, 2)

            # ③ RoPE 只作用在 q / k 上，v 不动
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

            y = attention_fn(q, k, v, layer_idx)          # [B, Hq, S, Dh]
            x = x + _merge_heads(y) @ W["o_w"].T          # 残差

            # ④ SwiGLU：silu(gate(x)) * up(x)，再 down 回 D
            h = _rmsnorm(x, W["ln2_w"], weights.rms_eps)
            h = F.silu(h @ W["gate_w"].T) * (h @ W["up_w"].T)
            x = x + h @ W["down_w"].T                     # 残差

        x = _rmsnorm(x, weights.norm_w, weights.rms_eps)
        return x @ weights.lm_head.T.to(x.dtype)
