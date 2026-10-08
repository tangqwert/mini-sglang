"""engine.paged_schedule — M4b Step 4：把分页 KV 池接进调度器。

═══════════════════════════════════════════════════════════════════
 与 engine.batching 的分工（有意为之）：
   batching.py        通用调度器：只认识注入的 kv_forward，与模型解耦。
   paged_schedule.py  分页专用调度器：分页前向的调用形状不同（拍平 varlen +
                      每序列一张页表），沿用不了同一个 callable 接口，
                      于是显式多出一层"集成层"。

 对比 M2.5 的 continuous_generate，换来三件事：
   ① 左填充彻底消失：padded_positions 恒为 0 —— 没有一列算力/显存被浪费
   ② 补入成本从"填充到当前批长 S"降到"该请求自己的 L"，
      且同一轮的【多条补入可以合并进一次前向】
   ③ KV 显存精确按需分配（ensure_blocks 按需增长、完成即 free），无碎片

 M2.5 实测 1.25x（理想 2.00x），差距来源就是①+②；Step 4 就是把它们拿掉。
═══════════════════════════════════════════════════════════════════
"""
import torch

from engine.batching import Request, SchedulerStats
from engine.model_forward import GPT2Weights, gpt2_forward
from engine.paged_kv import (BatchedPagedAttentionHook, PagedKVPool,
                             ensure_blocks)


def paged_continuous_generate(
    weights: GPT2Weights,
    requests: list[Request],
    pool: PagedKVPool,
    max_batch_size: int = 4,
) -> tuple[list[torch.LongTensor], SchedulerStats]:
    """分页版连续准入解码（准入策略同 M2.5，KV 管理换成页表 + varlen 前向）。

    与 `continuous_generate` 的关键差异：
      · 每条序列有自己的长度，**从不左填充** → `stats.padded_positions == 0`
      · prefill 把 K 条请求的 prompt 拍平成一次前向（K 可以 > 1，补入也能合并）
      · decode 每步一次前向：活跃行各喂 1 个 token，同样是拍平布局

    契约：
      · 输出逐 token == 每条请求单独跑朴素解码（顺序同 requests）。
      · 每条请求的 EOS / 预算独立生效；不修改 requests 的 prompt。
      · 请求完成即归还它的全部分页显存。

    诚实声明：
      · 逐序列注意力仍是 Python 循环（真引擎用 varlen kernel 一次算完）——
        这是吞吐上的差距，M5 才补；本步兑现的是**零填充 + 精确显存**。
      · 仍假设请求集合已知（同 M2.5），故允许一次性把 pending 全部纳入调度。
    """
    stats = SchedulerStats()
    if not requests:
        return [], stats

    B = min(max_batch_size, len(requests))
    device = requests[0].prompt.device

    pending = list(requests)
    slots: list[Request | None] = [None] * B       # 固定槽位
    tables: list[list | None] = [None] * B         # 每槽位一张页表（block id 列表）
    length = [0] * B                               # 槽位已写入池的 token 数
    last = [0] * B                                 # 槽位待喂的 token（其绝对位置 = length）
    generated: dict[int, list[int]] = {}
    remaining = [0] * B

    def register(i: int, r: Request) -> None:
        """登记第 i 行刚产出的 token；完成则释放槽位并归还分页显存。"""
        tok = last[i]
        generated[r.req_id].append(tok)
        remaining[i] -= 1
        if remaining[i] <= 0 or (r.eos_token_id is not None and tok == r.eos_token_id):
            for b in tables[i]:
                pool.free(b)
            slots[i] = None
            tables[i] = None

    def prefill(pairs: list[tuple[int, Request]], is_admission: bool) -> None:
        """把若干请求的 prompt 拍平成【一次】前向，各写各自的页表。

        注意这里没有 pad —— 每条序列在拍平张量里占自己那一段（varlen）。
        """
        flat, pos, lens = [], [], []
        for i, r in pairs:
            L = r.prompt.numel()
            tables[i] = []
            ensure_blocks(pool, tables[i], L)
            flat.append(r.prompt.reshape(-1))
            pos.append(torch.arange(L, device=device))
            lens.append(L)

        hook = BatchedPagedAttentionHook(
            pool, [tables[i] for i, _ in pairs], [0] * len(pairs), lens)
        logits = gpt2_forward(torch.cat(flat).unsqueeze(0), weights,
                              attention_fn=hook,
                              position_ids=torch.cat(pos).unsqueeze(0))

        ends = torch.tensor(lens).cumsum(0) - 1        # 各序列最后一个位置的拍平下标
        for j, (i, r) in enumerate(pairs):
            slots[i] = r
            generated[r.req_id] = []
            remaining[i] = r.max_new_tokens
            length[i] = lens[j]                        # 池里只有 prompt（生成的首 token 下一步才进池）
            last[i] = int(logits[0, ends[j]].argmax())
            register(i, r)                             # 首 token 计入生成并判预算/EOS

        if is_admission:
            stats.admissions += len(pairs)
            stats.admission_forwards += 1              # 多条补入合并 → 只算一次前向

    # ── 主循环：补入 → decode → 冻结 ──
    # 补入放在循环开头（同 M2.5 的 bug 修复）：否则"初始批第一步就全完成"时，
    # 循环条件立刻为假、pending 永远补不进来。
    first = True
    while True:
        free = [i for i in range(B) if slots[i] is None]
        if free and pending:
            prefill([(i, pending.pop(0)) for i in free if pending],
                    is_admission=not first)
        first = False

        active = [i for i in range(B) if slots[i] is not None]
        if not active:
            break

        feed, pos = [], []
        for i in active:
            ensure_blocks(pool, tables[i], length[i] + 1)
            feed.append(last[i])
            pos.append(length[i])                      # 绝对位置 = 池里已有的 token 数
        hook = BatchedPagedAttentionHook(
            pool, [tables[i] for i in active], list(length), [1] * len(active))
        logits = gpt2_forward(torch.tensor([feed], device=device), weights,
                              attention_fn=hook,
                              position_ids=torch.tensor([pos], device=device))
        for j, i in enumerate(active):
            last[i] = int(logits[0, j].argmax())
            length[i] += 1                             # 这一步的 K/V 已进池
            register(i, slots[i])

        stats.steps += 1
        stats.idle_slot_steps += B - len(active)

    outs = []
    for r in requests:
        gen = torch.tensor([generated[r.req_id]], dtype=torch.long,
                           device=r.prompt.device)
        outs.append(torch.cat([r.prompt.reshape(1, -1), gen], dim=1))
    return outs, stats
