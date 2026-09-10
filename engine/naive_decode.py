"""engine.naive_decode — M0：朴素自回归解码循环。

═══════════════════════════════════════════════════════════════════
 你的任务：实现下面标注 TODO 的函数，使 tests/test_naive_decode.py 全绿。
 禁止：使用 transformers 的 model.generate()（那正是我们要手搓的东西）。
 允许：torch 的任何操作。
═══════════════════════════════════════════════════════════════════

背景知识（写之前先想清楚）：
  自回归解码 = 一个 for 循环：
    1. 把当前序列喂给模型，拿到"每个词表位置"的分数（logits）
    2. 只取【最后一个位置】的 logits
    3. 贪心解码：argmax 选出下一个 token
    4. 把新 token 拼到序列尾部，回到 1
  结束条件：生成了 eos_token_id，或达到 max_new_tokens。

设计约束（为什么这么设计，想明白会在面试被问）：
  - logits_fn 与解码循环解耦：循环不认识"模型"，只认识"给序列返回 logits 的函数"。
    这样 M2 的 batching、M4 的 kernel 替换都不用改循环本身。
  - 全程带 batch 维度 [B, T]：M0 就按批量接口写，M2 才不用重写。
"""
from dataclasses import dataclass

import torch


@dataclass
class DecodingConfig:
    """解码配置。

    Attributes:
        max_new_tokens: 最多生成多少个新 token（不含 prompt）。
        eos_token_id: 结束符 id。为 None 时只能靠 max_new_tokens 停。
    """

    max_new_tokens: int = 64
    eos_token_id: int | None = None


def autoregressive_generate(
    logits_fn,
    input_ids: torch.LongTensor,
    config: DecodingConfig,
) -> torch.LongTensor:
    """朴素自回归解码（贪心）。

    Args:
        logits_fn: callable, 签名 logits_fn(ids: LongTensor[B, T]) -> FloatTensor[B, T, V]。
            给定当前序列，返回每个位置的词表分数。
        input_ids: [B, T] 的 prompt。
        config: 解码配置。

    Returns:
        LongTensor[B, T + n]，即 prompt + 生成部分（含 prompt，不含 eos 之后的任何东西；
        eos 本身是否包含——【包含】eos，它与生成内容同属一步）。

    约束:
        - 不得修改 input_ids 本身（调用方的张量不能被原地改写）。
        - batch 内任意一条序列命中 eos 后，该序列不再追加新 token
          （M0 允许整批一起停：只要有一条命中 eos 就全部停止，并 pad 无效位置——
          不，更简单：M0 不做 per-request 停止，批内第一条命中 eos 即整体停止。
          per-request 停止是 M2 continuous batching 的课题。）
        - 时间复杂度说明（写进你的笔记）：每步都对整条序列重算一遍前向。
          这就是 M1 KV cache 要优化的点——先体会它有多慢。
    """
    generated = input_ids.clone()
    for _ in range(config.max_new_tokens):
        logits = logits_fn(generated)                             # [B, T_cur, V]
        # 只取最后一个位置的分数 → [B, V]，argmax 得 [B]；keepdim 直接给 [B, 1]
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # 先拼进去再判断：契约规定 EOS 本身也要保留在结果里
        generated = torch.cat([generated, next_token], dim=1)
        if config.eos_token_id is not None and (next_token == config.eos_token_id).any():
            break
    return generated


def build_logits_fn(model, device: torch.device):
    """把 HuggingFace 因果语言模型包装成循环所需的 logits_fn。

    Args:
        model: transformers 的 CausalLM（如 AutoModelForCausalLM 加载的对象），
            已 .to(device)。
        device: torch.device。

    Returns:
        callable: logits_fn(ids: LongTensor[B, T]) -> FloatTensor[B, T, V]。
            内部应：torch.no_grad()、把 ids 搬到 device、取 model(...).logits。

    提示:
        - torch.no_grad() 忘了加会怎样？（内存泄漏+变慢——想想 autograd 在攒什么）
        - HF 模型返回的是 ModelOutput 对象，logits 是它的字段。
        - dtype 不用管，M0 用默认 fp32/fp16 都行。
    """
    def logits_fn(ids):
        with torch.no_grad():
            ids = ids.to(device)
            output = model(input_ids=ids)
            return output.logits

    return logits_fn
