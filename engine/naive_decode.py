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
    3. 选 token：默认 argmax（贪心）；M8 起可通过 sampling 注入
       temperature / top_k / top_p 采样
    4. 把新 token 拼到序列尾部，回到 1
  结束条件：生成了 eos_token_id，或达到 max_new_tokens。

设计约束（为什么这么设计，想明白会在面试被问）：
  - logits_fn 与解码循环解耦：循环不认识"模型"，只认识"给序列返回 logits 的函数"。
    这样 M2 的 batching、M4 的 kernel 替换都不用改循环本身。
  - 全程带 batch 维度 [B, T]：M0 就按批量接口写，M2 才不用重写。
"""
from dataclasses import dataclass, field

import torch

from engine.sampling import SamplingParams, sample_token, token_stops


@dataclass
class DecodingConfig:
    """解码配置。

    Attributes:
        max_new_tokens: 最多生成多少个新 token（不含 prompt）。
        eos_token_id: 结束符 id。为 None 时只能靠 max_new_tokens 停。
        sampling: M8 采样参数。默认 temperature=0（贪心），与旧版逐 token 一致。
    """

    max_new_tokens: int = 64
    eos_token_id: int | None = None
    sampling: SamplingParams = field(default_factory=SamplingParams)


def autoregressive_generate(
    logits_fn,
    input_ids: torch.LongTensor,
    config: DecodingConfig,
    on_step=None,
) -> torch.LongTensor:
    """朴素自回归解码（默认贪心，可通过 config.sampling 启用采样）。

    Args:
        logits_fn: callable, 签名 logits_fn(ids: LongTensor[B, T]) -> FloatTensor[B, T, V]。
            给定当前序列，返回每个位置的词表分数。
        input_ids: [B, T] 的 prompt。
        config: 解码配置。
        on_step: 可选回调，签名 on_step(step_idx, new_tokens)。每生成一个
            token（已拼进结果、EOS 判定之前）调用一次；benchmark 用它记录
            TTFT / TPOT 的时间戳，默认 None 不产生额外开销。

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
    generator = None
    for step in range(config.max_new_tokens):
        logits = logits_fn(generated)
        if generator is None and config.sampling.seed is not None:
            # 惰性创建：此时才知道 logits 在 CPU 还是 GPU
            generator = config.sampling.make_generator(logits.device)
        new_tokens = sample_token(logits[:, -1, :], config.sampling, generator)
        generated = torch.cat((generated, new_tokens), dim=1)
        # M0 的批语义：任一行命中 EOS / stop token 即整批停止
        stop = any(token_stops(config.sampling, config.eos_token_id, int(t))
                   for t in new_tokens[:, 0])
        if on_step is not None:
            # 放在 int(t) 之后：GPU 上这一步已同步，时间戳是 token 真正可用的时刻。
            on_step(step, new_tokens)
        if stop:
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
        # no_grad 与设备搬运都钉在这一层：这是唯一碰 model(...) 的入口，
        # 绑死后任何调用方都不可能漏（漏了会攒计算图 → 显存翻倍甚至 OOM）。
        with torch.no_grad():
            input_ids = ids.to(device)
            output = model(input_ids)
        return output.logits
    return logits_fn
