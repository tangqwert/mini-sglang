"""engine.kv_cache — M1：KV Cache 增量解码。

═══════════════════════════════════════════════════════════════════
 你的任务：实现下面标注 TODO 的函数，使 tests/test_kv_cache.py 全绿，
 且 kv_generate 的输出与 M0 的 autoregressive_generate【逐 token 一致】。
 允许：torch 的任何操作、HF 的 past_key_values 机制。
═══════════════════════════════════════════════════════════════════

背景知识（M0 的痛点 → M1 的解法）：
  M0 每生成一个 token，都要把【整条序列】重新前向一遍。
  但注意力机制有个数学性质：位置 t 的 K/V 只由 token 0..t 决定，与后来者无关。
  所以只要把每层的 K/V 缓存下来（past_key_values），下一步就只需：
    1. 把【新的那 1 个 token】喂给模型（不是整条序列！）
    2. 模型内部：新 token 的 Q 与【缓存的全部 K/V】做注意力
    3. 返回新 token 位置的 logits，并把新的 K/V 追加进缓存

  成本对比（这就是 bench 里要看到的数字）：
    M0: 每步喂 T 个 token，共 n 步       → 总喂入 ≈ n*T + n²/2
    M1: 首步喂 T 个（prefill），之后每步喂 1 个 → 总喂入 = T + n - 1

设计约束（与 M0 一脉相承）：
  - 解码循环不认识"模型"，只认识 kv_forward 这个 callable。
  - kv_forward 返回 (logits, new_cache)，循环负责把 new_cache 传给下一步。
  - EOS 语义与 M0 完全一致：先拼接、后判定；批内任一命中即整批停。
"""
import torch

from engine.naive_decode import DecodingConfig  # noqa: F401  (契约与 M0 共用)


def build_kv_forward(model, device: torch.device):
    """把 HF 因果语言模型包装成"增量式前向"函数。

    Args:
        model: transformers 的 CausalLM，已 .to(device)。
        device: torch.device。

    Returns:
        callable: kv_forward(ids, cache) -> (logits, new_cache)
            ids:   LongTensor[B, T_in]——prefill 时是整条 prompt，解码时只有 1 个新 token
            cache: 上一步返回的 past_key_values；第一次调用传 None
            logits: FloatTensor[B, T_in, V]（T_in=1 时即 [B, 1, V]）
            new_cache: 更新过 K/V 的缓存对象

    提示:
        - 与 M0 的 build_logits_fn 唯一区别：要多传 past_key_values 和 use_cache=True。
        - cache 的搬运问题想一想：ids 要 .to(device)，那 cache 呢？
          （它第一次由模型创建，之后一直住在哪？）
    """
    def kv_forward(ids, cache, attention_mask=None, position_ids=None):
        # 唯一碰 model(...) 的入口：no_grad 与设备搬运都钉在这一层，
        # 调用方不可能漏。cache 由模型创建、一直住在 model 的设备上，无需搬运。
        with torch.no_grad():
            ids = ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if position_ids is not None:
                position_ids = position_ids.to(device)
            output = model(
                input_ids=ids,
                past_key_values=cache,
                use_cache=True,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return output.logits, output.past_key_values
    return kv_forward


def kv_generate(kv_forward, input_ids: torch.LongTensor, config: DecodingConfig) -> torch.LongTensor:
    """KV Cache 增量解码（贪心）。

    Args:
        kv_forward: build_kv_forward 返回的 callable。
        input_ids: [B, T] 的 prompt（CPU 上来，第一次 forward 前由 kv_forward 搬运）。
        config: 解码配置（与 M0 共用）。

    Returns:
        LongTensor[B, T + n]，与 autoregressive_generate 的输出【完全一致】。

    契约:
        - 喂入时间表（性能契约，测试会精确断言）：
            第 1 次前向（prefill）：喂整条 prompt，得到第 1 个新 token
            第 k 次前向（k>=2）  ：只喂上一步的 1 个新 token
        - EOS 语义与 M0 逐字相同：新 token 先拼进结果，再判定是否提前停止；
          批内任一序列命中即整批停。
        - 不修改 input_ids 本身。

    提示:
        - 统一循环：feed 第 1 轮是整条 prompt，之后是上一步的 1 个新 token。
          这样 EOS 判定天然只有一处——prefill 产出的第一个 token 也会被检查。
        - 和 M0 循环体的唯一区别：喂的是 1 个 token 而非整条序列。
        - torch.cat 的返回值记得接住（你踩过两次的坑 😉）。
    """
    generated = input_ids.clone()
    cache = None
    feed = generated                  # 第 1 轮喂整条 prompt
    for _ in range(config.max_new_tokens):
        logits, cache = kv_forward(feed, cache)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        if config.eos_token_id is not None and torch.any(next_token == config.eos_token_id):
            break
        feed = next_token             # 之后只喂 1 个新 token
    return generated