"""engine.batching — M2：Continuous Batching（动态退出批解码）。
                         M2.5：调度器 + 槽位连续准入。

═══════════════════════════════════════════════════════════════════
 M2   → batched_generate   ：请求集合固定，某行冻结后**空转到批结束**。
 M2.5 → continuous_generate：维护 pending 队列 + 固定槽位数；某行冻结就**立刻
                             补入下一条请求**，让批尽量保持满载。
                             （这就是 continuous admission，真 continuous
                               batching 的核心，也是 vLLM/SGLang 的卖点之一）
═══════════════════════════════════════════════════════════════════
 你的任务：实现 batched_generate（M2）与 continuous_generate（M2.5），
 使 tests/test_batching.py 全绿。
 核心契约：每条请求有独立的生命周期（预算/EOS），完成的行冻结；
 其余继续 —— M2 空转到批结束，M2.5 则立刻补入下一条请求。
═══════════════════════════════════════════════════════════════════

背景知识（M1 → M2 的跨越）：
  M1 一条序列独占模型。M2 让 N 条请求拼成一个 batch 一起解码：
    - 每步一次前向服务 N 条请求 → 权重搬运成本被 N 条分摊（batching 的经济账）
    - 但 prompt 长度不同 → 必须左填充（left-pad）成统一长度
    - pad token 不能被注意力看到 → 需要 attention_mask
    - pad 会打乱位置编号 → 需要显式 position_ids
    - 某条请求完成（EOS / 预算用尽）→ 冻结该行，其余继续 ← 这就是"动态退出"

  M2.5 的实现要点与诚实声明：
    补入一条"长度不同"的新请求，靠的是【左填充到当前批长 → 单独 prefill →
    把那行拷进批 cache 的空槽】。因果注意力逐行独立，所以数值上与"它一开始
    就在批里"完全等价（实测 logits 差 < 4e-5，float32 噪声）。
    代价：① 填充位置永久占在 cache 里（显存碎片）；② 补入需要一次独立前向。
    这正是 PagedAttention（M4a）存在的理由 —— 分页后每行页表独立，
    既不需要填充，也能把补入和 decode 合进同一个 kernel。

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


# ─────────────────── M2.5：调度器 + 槽位连续准入 ───────────────────

def copy_row(dst_cache, src_cache, dst_row: int, src_row: int = 0) -> None:
    """把 src_cache 的第 src_row 行原地拷进 dst_cache 的第 dst_row 行。

    前提：两个 cache 的序列长度相同 —— 由调用方保证（新请求的 prompt 会左填充到当前批长）。

    为什么需要它：HF 的批 cache 要求**所有行等长**，所以"槽位空出后补入一条长度不同的
    新请求"没法直接做。技巧是：把新请求的 prompt 左填充到当前批长，先单独 prefill 一遍
    拿到它自己的 K/V，再把这一行拷进批 cache 的空槽。因果注意力是逐行独立的，所以
    数值上与"它一开始就在批里"完全等价（实测 logits 差 < 4e-5，float32 噪声）。

    代价：那 S-L 个填充位置会永久占在 cache 里 → 显存碎片。
    这正是 PagedAttention（M4a）要解决的问题：分页后每行页表独立，无需填充。
    """
    if hasattr(dst_cache, "layers"):          # 真 HF DynamicCache：[B, H, S, D] 逐层拷行
        for dst_layer, src_layer in zip(dst_cache.layers, src_cache.layers):
            dst_layer.keys[dst_row] = src_layer.keys[src_row]
            dst_layer.values[dst_row] = src_layer.values[src_row]
    else:                                     # 假缓存（测试用）：拷"已见序列"那一行
        dst_cache.ids[dst_row] = src_cache.ids[src_row]


@dataclass
class SchedulerStats:
    """调度统计：连续准入的收益与代价（写进 README 的诚实数据）。

    Attributes:
        steps:              批量前向次数（decode 步）。
        admissions:         槽位补入次数。
        admission_forwards: 补入附带的独立 prefill 前向次数。
        idle_slot_steps:    Σ 每步空闲槽位数 —— 批利用率的损失。
        padded_positions:   Σ 左填充产生的位置数 —— 显存碎片。
    """
    steps: int = 0
    admissions: int = 0
    admission_forwards: int = 0
    idle_slot_steps: int = 0
    padded_positions: int = 0


def continuous_generate(
    kv_forward,
    requests: list[Request],
    max_batch_size: int = 4,
    pad_token_id: int = 0,
) -> tuple[list[torch.LongTensor], SchedulerStats]:
    """M2.5：固定槽位的连续准入解码（槽位一空出，pending 队列立刻补入）。

    与 M2 的 `batched_generate` 的区别：
      M2   ：请求集合固定，某行冻结后**空转到批结束**（槽位浪费）。
      M2.5 ：维护 pending 队列与固定数量的槽位；某行冻结 → 立刻补入下一条请求，
             使批尽量保持满载 —— 这就是 continuous admission，真 continuous
             batching 的核心。

    契约（与 batched_generate 一致的部分不再重复）：
      - 输出逐 token 等价于每条请求单独跑朴素解码（顺序同 requests）。
      - 每条请求的 EOS / 预算独立生效。
      - 不修改 requests 里的 prompt。

    简化声明（诚实）：
      - 真实服务中请求**陆续到达**、长度未知；此处假设请求集合已知，故取
        S0 = 全局最大 prompt 长度，保证任意时刻补入的请求都能装进当前批长。
      - 补入一行需要**一次独立的 prefill 前向**（batch=1 或 k），这是没有
        PagedAttention 的代价；真引擎把它和 decode 合进同一个 kernel。
        所以本里程碑真正的结论是**收益与代价的对照**，见 SchedulerStats。
    """
    stats = SchedulerStats()
    if not requests:
        return [], stats

    B = min(max_batch_size, len(requests))
    device = requests[0].prompt.device
    S0 = max(r.prompt.numel() for r in requests)

    pending = list(requests)
    slot: list[Request | None] = [None] * B
    generated: dict[int, list[int]] = {}
    remaining = [0] * B

    mask = torch.zeros((B, S0), dtype=torch.long, device=device)
    last_tokens = torch.full((B, 1), pad_token_id, dtype=torch.long, device=device)
    cache = None
    S = S0                       # 当前批序列长度（cache 宽度），自己维护、不依赖缓存内部

    def register(i: int, r: Request, tok: int) -> bool:
        """登记第 i 行刚产出的 token，返回该行是否已完成（需冻结）。"""
        generated[r.req_id].append(tok)
        remaining[i] -= 1
        done = remaining[i] <= 0 or (r.eos_token_id is not None and tok == r.eos_token_id)
        if done:
            slot[i] = None                     # 释放槽位，交还给调度器
        return done

    # ── 初始批：一次批量 prefill ──
    for i, r in enumerate(pending[:B]):
        slot[i] = r
        generated[r.req_id] = []
        remaining[i] = r.max_new_tokens
        L = r.prompt.numel()
        mask[i, S0 - L:] = 1
        stats.padded_positions += S0 - L
    padded = torch.full((B, S0), pad_token_id, dtype=torch.long, device=device)
    for i, r in enumerate(pending[:B]):
        L = r.prompt.numel()
        padded[i, S0 - L:] = r.prompt
    pending = pending[B:]

    logits, cache = kv_forward(padded, None, attention_mask=mask,
                               position_ids=build_position_ids(mask))
    first = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    for i, r in enumerate(slot):
        if r is not None:
            register(i, r, int(first[i, 0]))
            last_tokens[i, 0] = int(first[i, 0])

    def admit_free_slots() -> None:
        """把 pending 里的请求补进空槽：左填充到当前批长 → 独立 prefill → 拷行。"""
        nonlocal cache, last_tokens, mask
        for i in range(B):
            if slot[i] is not None or not pending:
                continue
            r = pending.pop(0)
            L = r.prompt.numel()
            row = torch.full((1, S), pad_token_id, dtype=torch.long, device=device)
            row[0, S - L:] = r.prompt
            m = torch.zeros((1, S), dtype=torch.long, device=device)
            m[0, S - L:] = 1
            lg, new_cache = kv_forward(row, None, attention_mask=m,
                                       position_ids=build_position_ids(m))
            copy_row(cache, new_cache, i, 0)
            mask[i] = m[0]
            last_tokens[i, 0] = int(lg[0, -1, :].argmax())
            slot[i] = r
            generated[r.req_id] = []
            remaining[i] = r.max_new_tokens
            stats.admissions += 1
            stats.admission_forwards += 1
            stats.padded_positions += S - L
            register(i, r, int(lg[0, -1, :].argmax()))

    # ── 主循环：补入 → decode → 冻结 ──
    # 补入放在循环开头，而不是末尾：否则"初始批在第一步就全部完成"时，
    # 循环条件（无活跃行）会立刻为假 → pending 永远补不进来。
    while True:
        admit_free_slots()                     # 空出的槽位立刻补入
        active = torch.tensor([s is not None for s in slot], device=device).unsqueeze(1)
        if not bool(active.any()):
            break                              # 没有活跃行、也没有 pending 可补 → 收工

        feed = torch.where(active, last_tokens,
                           torch.full_like(last_tokens, pad_token_id))
        mask = torch.cat([mask, active.long()], dim=1)
        position_ids = (mask.sum(dim=1, keepdim=True) - 1).clamp(min=0)

        logits, cache = kv_forward(feed, cache,
                                   attention_mask=mask, position_ids=position_ids)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for i, r in enumerate(slot):
            if r is not None:
                register(i, r, int(next_tokens[i, 0]))
        last_tokens = next_tokens
        stats.steps += 1
        stats.idle_slot_steps += int((~active).sum())
        S += 1                                 # cache 每步长 1 列

    outs = []
    for r in requests:
        gen = torch.tensor([generated[r.req_id]], dtype=torch.long,
                           device=r.prompt.device)
        outs.append(torch.cat([r.prompt.reshape(1, -1), gen], dim=1))
    return outs, stats


