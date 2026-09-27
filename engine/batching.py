"""engine.batching — M2：Continuous Batching（动态退出批解码）。

═══════════════════════════════════════════════════════════════════
 你的任务：实现 batched_generate，使 tests/test_batching.py 全绿。
 核心契约：每条请求有独立的生命周期（预算/EOS），完成的行冻结，
 其余继续；全部完成后返回每条请求各自的完整序列。
═══════════════════════════════════════════════════════════════════

背景知识（M1 → M2 的跨越）：
  M1 一条序列独占模型。M2 让 N 条请求拼成一个 batch 一起解码：
    - 每步一次前向服务 N 条请求 → 权重搬运成本被 N 条分摊（batching 的经济账）
    - 但 prompt 长度不同 → 必须左填充（left-pad）成统一长度
    - pad token 不能被注意力看到 → 需要 attention_mask
    - pad 会打乱位置编号 → 需要显式 position_ids
    - 某条请求完成（EOS / 预算用尽）→ 冻结该行，其余继续 ← 这就是"动态退出"

  M2 vs 真正的 continuous batching：
    本里程碑不做"槽位补新请求"（refill）——那需要 PagedAttention 的
    显存搬移能力（M3.5）。这里实现的是：动态退出 + per-request 生命周期。

设计约束：
  - 调度循环只认识 kv_forward（M1 的接口，需扩展 mask/position 参数）。
  - 填充与位置编号的机械活由 pad_left / build_position_ids 提供（已实现），
    你专注写调度器本体。
"""
from dataclasses import dataclass

import torch

from engine.naive_decode import DecodingConfig  # noqa: F401


@dataclass
class Request:
    """一条解码请求。

    Attributes:
        req_id: 请求标识（返回结果按它对齐）。
        prompt: LongTensor[T]，该请求自己的 prompt。
        max_new_tokens: 该请求自己的生成预算。
        eos_token_id: 该请求自己的结束符；None 表示只受预算约束。
    """

    req_id: int
    prompt: torch.LongTensor
    max_new_tokens: int
    eos_token_id: int | None = None


# ─────────────────────────── 已提供的工具 ───────────────────────────

def pad_left(requests: list[Request], pad_token_id: int) -> tuple[torch.LongTensor, torch.LongTensor]:
    """把所有 prompt 左填充到统一长度。

    为什么是"左"填充：真实 token 挤在右侧，每行的【最后一个位置】都是
    真实 token——prefill 之后每行最后位置直接可取 argmax，省心。

    Returns:
        padded: LongTensor[N, Tmax]，短 prompt 的左边填 pad_token_id
        mask:   LongTensor[N, Tmax]，真实 token=1，pad=0
    """
    prompts = [r.prompt.reshape(1, -1) for r in requests]  # 统一形状
    tmax = max(p.shape[1] for p in prompts)                # 目标宽度
    n = len(requests)                                      # 行数
    padded = torch.full((n, tmax), pad_token_id, dtype=torch.long)
    mask = torch.zeros((n, tmax), dtype=torch.long)
    for i, p in enumerate(prompts):
        padded[i, tmax - p.shape[1]:] = p
        mask[i, tmax - p.shape[1]:] = 1
    return padded, mask


def build_position_ids(mask: torch.LongTensor) -> torch.LongTensor:
    """从 attention_mask 推出每行的位置编号。

    位置 = 该位置之前（含自己）的真实 token 数 - 1。
    左填充的 pad 在真实 token 之前，若不显式给 position_ids，
    模型会把 pad 也数进去，导致位置整体偏移 → 生成质量劣化。

    Returns:
        LongTensor[N, T]，真实 token 位置 0,1,2,...；pad 位置为 0（反正被 mask 屏蔽）
    """
    return (mask.cumsum(dim=-1) - 1).clamp(min=0)


# ─────────────────────────── 你来实现 ───────────────────────────

def batched_generate(
    kv_forward,
    requests: list[Request],
    pad_token_id: int = 0,
) -> list[torch.LongTensor]:
    
    """N 条请求共享 batch 的动态退出解码（贪心）。

    Args:
        kv_forward: M1 的 build_kv_forward 产物，需支持
            kv_forward(ids, cache, attention_mask=None, position_ids=None)
        requests: 请求列表（不要修改它们！prompt 原样保留）。
        pad_token_id: 填充符 id（测试假模型用 0，恰好加和不变）。

    Returns:
        list[LongTensor]，顺序与 requests 一致，每条为该请求的
        prompt + 生成部分（含 EOS，若命中），【无 pad】。
        第 i 条长度 = prompt_i 长度 + 该请求实际生成的 token 数。

    契约:
        - prefill 一次：整批 padded prompt 一起前向。
        - 之后每步一次批量前向：每行喂【该行上一步刚生成的 token】；
          已冻结的行喂 pad_token_id 占位（浪费一点算力，换 batch 形状不变）。
        - 每行独立判定完成：拼入新 token 后，
          ① 命中该行的 eos_token_id，或 ② 该行已生成的数量达到 max_new_tokens
          → 冻结（不再向该行的结果追加任何 token）。
        - 全部行冻结 → 循环结束，返回结果。

    提示（调度器的状态清单，写之前先想清楚每个是什么形状）:
        - generated_ids: dict[int, list[int]]，按 req_id 记录每行已生成 token
        - remaining:     list[int]，每行剩余预算
        - frozen:        list[bool]，每行是否已完成
        - last_tokens:   LongTensor[N, 1]，每行上一步生成的 token（喂下一步）
        - mask:          每步要往前 cat 一列 1（喂进去的 token 都是"真实"前向）
        - position_ids:  每步 [N, 1]——每行的下一个位置 = 该行已有的真实 token 数
          （= prompt 真实长度 + 已生成数；注意冻结行的 dummy 也会进 cache，
           但它们永远不会被返回，所以位置算不算都行——按"每行+1"处理最简单）

    提示（结构）:
        1. pad_left → prefill（kv_forward(padded, None, attention_mask, position_ids)）
        2. argmax 得每行第一个新 token，登记进 generated_ids，做第一轮 EOS/预算判定
        3. while 有未冻结的行：构造 feed（冻结行喂 pad）→ 批量前向 → argmax →
           只给未冻结的行追加 → 判定 → 更新 last_tokens
        4. 收尾：按 requests 顺序返回 prompt 拼生成（注意用原始 prompt，
           不是 padded！padded 里混着 pad）
    """
    n = len(requests)
    padded, mask = pad_left(requests, pad_token_id)

    generated = {r.req_id: [] for r in requests}
    remaining = [r.max_new_tokens for r in requests]
    frozen = [False]*n

    #prefill
    position_ids = build_position_ids(mask)
    logits, cache = kv_forward(padded, None,
                               attention_mask = mask, position_ids = position_ids)
    last_tokens = torch.argmax(logits[:,-1,:], dim=-1, keepdim=True)

    for i, r in enumerate(requests):
        tok = last_tokens[i,0].item()
        generated[r.req_id].append(tok)
        remaining[i] -= 1
        if (r.eos_token_id is not None and tok == r.eos_token_id) or remaining[i] ==0:
            frozen[i] = True
    # 存在活着的行
    while not all(frozen):
        feed = torch.where(torch.tensor(frozen, device=last_tokens.device).unsqueeze(1), torch.full_like(last_tokens, pad_token_id), last_tokens)

        mask = torch.cat([mask, torch.ones((n,1), dtype=torch.long)], dim=1)

        position_ids = (mask.sum(dim=1,keepdim=True)-1).clamp(min=0)

        logits, cache = kv_forward(feed, cache,
                               attention_mask = mask, position_ids = position_ids)
        next_tokens = torch.argmax(logits[:,-1,:],dim=-1,keepdim=True)

        for i, r in enumerate(requests):
            if frozen[i]:
                continue
            tok = next_tokens[i,0].item()
            generated[r.req_id].append(tok)
            remaining[i] -= 1
            if (r.eos_token_id is not None and tok == r.eos_token_id) or remaining[i] == 0:
                frozen[i] = True
        last_tokens =  next_tokens
    outs = []
    for i, r in enumerate(requests):
        gen = torch.tensor([generated[r.req_id]], dtype=torch.long, device=r.prompt.device)
        outs.append(torch.cat([r.prompt.reshape(1,-1),gen], dim=1))
    return outs

