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
from engine.radix_cache_paged import PagedRadixCache  # noqa: F401 (类型/文档)
from engine.sampling import sample_tokens


def paged_continuous_generate(
    weights: GPT2Weights,
    requests: list[Request],
    pool: PagedKVPool,
    max_batch_size: int = 4,
    hook_cls=None,
    forward_fn=None,
    on_token=None,
) -> tuple[list[torch.LongTensor], SchedulerStats]:
    """分页版连续准入解码（准入策略同 M2.5，KV 管理换成页表 + varlen 前向）。

    Args:
        hook_cls: 注意力实现（默认纯 PyTorch 的 BatchedPagedAttentionHook）。
            M5 可换成 `TritonPagedAttentionHook` —— 调度器不需要知道区别，
            这就是“循环只认识可调用对象”的又一次兼现。
        forward_fn: 模型前向（默认 `gpt2_forward`）。换架构时传 `qwen3_forward`。

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
    hook_cls = hook_cls or BatchedPagedAttentionHook
    forward_fn = forward_fn or gpt2_forward

    B = min(max_batch_size, len(requests))
    device = requests[0].prompt.device

    pending = list(requests)
    slots: list[Request | None] = [None] * B       # 固定槽位
    tables: list[list | None] = [None] * B         # 每槽位一张页表（block id 列表）
    length = [0] * B                               # 槽位已写入池的 token 数
    last = [0] * B                                 # 槽位待喂的 token（其绝对位置 = length）
    generated: dict[int, list[int]] = {}
    generators: dict[int, torch.Generator | None] = {}

    def gen_for(r: Request, device) -> torch.Generator | None:
        if r.req_id not in generators:
            generators[r.req_id] = r.sampling.make_generator(device)
        return generators[r.req_id]

    def register(i: int, r: Request) -> None:
        """登记第 i 行刚产出的 token；完成则释放槽位并归还分页显存。"""
        tok = last[i]
        generated[r.req_id].append(tok)
        if on_token is not None:
            on_token(r.req_id, tok, len(generated[r.req_id]))
        if r.finished(tok, len(generated[r.req_id])):
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

        hook = hook_cls(
            pool, [tables[i] for i, _ in pairs], [0] * len(pairs), lens)
        logits = forward_fn(torch.cat(flat).unsqueeze(0), weights,
                           attention_fn=hook,
                           position_ids=torch.cat(pos).unsqueeze(0))

        ends = torch.tensor(lens).cumsum(0) - 1        # 各序列最后一个位置的拍平下标
        produced = sample_tokens(
            logits[0, ends],
            [r.sampling for _, r in pairs],
            [gen_for(r, logits.device) for _, r in pairs])
        for j, (i, r) in enumerate(pairs):
            slots[i] = r
            generated[r.req_id] = []
            length[i] = lens[j]                        # 池里只有 prompt（生成的首 token 下一步才进池）
            last[i] = int(produced[j, 0])
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
        hook = hook_cls(
            pool, [tables[i] for i in active], [length[i] for i in active],
            [1] * len(active))
        logits = forward_fn(torch.tensor([feed], device=device), weights,
                           attention_fn=hook,
                           position_ids=torch.tensor([pos], device=device))
        produced = sample_tokens(
            logits[0],
            [slots[i].sampling for i in active],
            [gen_for(slots[i], logits.device) for i in active])
        for j, i in enumerate(active):
            last[i] = int(produced[j, 0])
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


# ─────────────────── M6：Chunked Prefill（prefill 与 decode 合流）───────────────────

def chunked_prefill_generate(
    weights: GPT2Weights,
    requests: list[Request],
    pool: PagedKVPool,
    max_batch_size: int = 4,
    max_prefill_tokens: int = 512,
    hook_cls=None,
    use_graph: bool = False,
    forward_fn=None,
    radix: PagedRadixCache | None = None,
    on_token=None,
    decode_graphs=None,
) -> tuple[list[torch.LongTensor], SchedulerStats]:
    """M6：Chunked Prefill —— 把 prefill 塞进 decode 的同一次前向。

    Args:
        use_graph: M7.5 —— 纯 decode 步改走 CUDA Graph（见 `engine/graph_decode.py`）。
            只有【本步没有任何 prefill 工作】时才走图，因为图的形状必须固定；
            含 prefill 的步（尤其是分块的）形状是变的，仍走 eager。
            `stats.graph_steps` 记录了多少步真的走了图。
        radix: M3.5b —— 传入 `PagedRadixCache` 时启用前缀缓存：
            准入时先查树，命中的完整 block 直接接进页表（零拷贝），
            只对未命中的后缀做 prefill；请求完成时把整条序列回写进树。
            命中长度按 block 对齐，且最多到 `prompt_len - 1`（最后一个 token
            必须现场算一次，否则拿不到下一个词的 logits）。
            统计见 `stats.radix_hit_requests / radix_hit_tokens`。
        on_token: 可选回调 `on_token(req_id, token, n_generated)`，每产出一个 token
            调用一次；benchmark 用它记录每请求 TTFT / TPOT。
        decode_graphs: 可选的**外部预捕获** `PagedDecodeGraphs`。传入时本函数不
            重新捕获、也不负责 close；用于 benchmark / serving 把捕获成本移到
            计时区间之外，并在多个请求间复用图。

    ── 要解决的最后一个痛点 ──
    M4b Step 4 把"补入时的填充计算"去掉了（喂入 token 降为 1/6.6），但**前向次数
    一个没少**：补入一条新请求仍然要单独跑一次 prefill 前向。所以 1.25x（理想 2.00x）
    只修好了一半。M6 修另一半。

    ── 怎么做 ──
    varlen 批次本来就不要求各序列长度相同！于是同一次前向里可以混着：
        活跃序列各贡献 1 个 token（decode）
        + 刚补入的序列贡献它的前 L 个 token（prefill）
    对 `BatchedPagedAttentionHook` 来说这只是 `new_lens = [1, 1, L]` 而已 —— 钩子
    不用改，改的是调度器怎么组这个批。真引擎（vLLM/SGLang）做的就是这件事，
    它们的术语叫 **mixed batch / prefill-decode 合流**。

    ── 为什么还要"chunked" ──
    如果一条 4096-token 的 prompt 独占一步，这一步的算力就是别人的 4096 倍，
    其他请求的 ITL（inter-token latency，逐 token 延迟）会被这一个"巨无霸步"打爆。
    所以把 prompt 切成若干块、每步只喂 `max_prefill_tokens` 个（vLLM 里叫
    `max_num_batched_tokens`），把单步成本摊平 —— 这就是 Chunked Prefill 的名字来源。
    分块的正确性依赖两点：① 位置编号必须是**绝对位置**（跨块连续）；
    ② 后续块的 query 必须能看到前面块 —— 前面块已在池里，gather 得到，天然满足。

    ── 关键不变量 ──
    · `stats.admission_forwards == 0` —— 不存在"专门为补入开的独立前向"
    · `stats.mixed_steps > 0` —— 确实发生了 prefill/decode 同批（M2.5/Step4 恒为 0）
    · `stats.padded_positions == 0` —— 仍然是零填充

    契约（其余同 `paged_continuous_generate`）：
      · 输出逐 token == 每条请求单独跑朴素解码（顺序同 requests）。
      · 每条请求的 EOS / 预算独立生效；不修改 requests 的 prompt。
    """
    assert max_prefill_tokens >= 1, "max_prefill_tokens 至少要能喂 1 个 token，否则无法推进"
    stats = SchedulerStats()
    if not requests:
        return [], stats

    # 采样需要随机数，而 CUDA Graph 捕获的是确定性的 argmax；教学版直接退化到 eager。
    # （真引擎会把采样 kernel 也捕获进图，或采用“图只算 logits、图外采样”。）
    if use_graph and any(r.sampling.temperature > 0 for r in requests):
        use_graph = False

    hook_cls = hook_cls or BatchedPagedAttentionHook
    forward_fn = forward_fn or gpt2_forward

    B = min(max_batch_size, len(requests))
    device = requests[0].prompt.device

    pending = list(requests)
    slot: list[Request | None] = [None] * B
    tables: list[list | None] = [None] * B
    length = [0] * B                 # 已写入池的 token 数
    phase = ["free"] * B             # 'free' | 'prefill' | 'decode'
    last = [0] * B                   # decode 阶段待喂的 token（绝对位置 = length）
    generated: dict[int, list[int]] = {}
    adopted: list[list[int]] = [[] for _ in range(B)]   # M3.5b：每槽位共享的 Radix 块
    generators: dict[int, torch.Generator | None] = {}

    def gen_for(r: Request, device) -> torch.Generator | None:
        if r.req_id not in generators:
            generators[r.req_id] = r.sampling.make_generator(device)
        return generators[r.req_id]

    # ── M7.5：预捕获若干张 decode 图（按批大小）──
    graphs = decode_graphs
    owns_graphs = False
    if use_graph and decode_graphs is None:
        # 惰性导入：graph_decode 依赖 triton，不该让本模块硬依赖它
        from engine.graph_decode import PagedDecodeGraphs
        # 一条序列最长会到 L + max_new（且最后一步的 K/V 不进池）；留一格余量
        longest = max(r.prompt.numel() + r.max_new_tokens for r in requests)
        max_blocks = -(-longest // pool.block_size) + 1
        cap = sorted({s for s in (1, 2, 4, 8) if s <= B} | {B})
        graphs = PagedDecodeGraphs(weights, pool, batch_sizes=cap,
                                   max_blocks=max_blocks, forward_fn=forward_fn)
        owns_graphs = True

    def register(i: int, tok: int) -> None:
        """登记第 i 行刚产出的 token；完成则插树、释放槽位并归还分页显存。"""
        r = slot[i]
        generated[r.req_id].append(tok)
        if on_token is not None:
            on_token(r.req_id, tok, len(generated[r.req_id]))
        if r.finished(tok, len(generated[r.req_id])):
            if radix is not None:
                # 池里已写 length[i] 个 token 的 KV；最后生成的 token 还没喂过，
                # 所以回写序列 = prompt + generated[:-1]。
                seq = r.prompt.reshape(-1).tolist() + generated[r.req_id][:-1]
                n_full = min(length[i] // pool.block_size, len(tables[i]))
                unused = radix.insert(seq[:n_full * pool.block_size],
                                      tables[i][:n_full])
                # 树没接收的：未写满的尾块 + 重复块，都还给池子
                for b in tables[i][n_full:]:
                    pool.free(b)
                for b in unused:
                    pool.free(b)
                radix.release_blocks(adopted[i])
                adopted[i] = []
            else:
                for b in tables[i]:
                    pool.free(b)
            slot[i], tables[i], phase[i] = None, None, "free"

    while True:
        # ── 组混合批：活跃槽位各 1 token 的 decode + 空槽位补入并喂一块 prefill ──
        flat, pos, bases, sizes, owners, kinds = [], [], [], [], [], []
        budget = max_prefill_tokens

        for i in range(B):
            if slot[i] is None:
                if not pending or budget <= 0:
                    continue
                r = pending.pop(0)
                slot[i], tables[i], length[i], phase[i] = r, [], 0, "prefill"
                generated[r.req_id] = []
                stats.admissions += 1

                # ── M3.5b：准入时查 Radix 前缀缓存 ──
                # 命中块直接接进页表（共享物理块，零拷贝），prefill 只喂后缀。
                if radix is not None:
                    prompt_ids = r.prompt.reshape(-1).tolist()
                    match = radix.match_prefix(prompt_ids)
                    # 至少留 1 个 token 现场算 logits；再向下对齐到完整 block
                    n_cap = max((len(prompt_ids) - 1) // pool.block_size, 0)
                    n_adopt = min(len(match.blocks), n_cap)
                    if n_adopt > 0:
                        adopted[i] = match.blocks[:n_adopt]
                        radix.acquire_blocks(adopted[i])
                        tables[i] = list(adopted[i])
                        length[i] = n_adopt * pool.block_size
                        stats.radix_hit_requests += 1
                        stats.radix_hit_tokens += length[i]

            if phase[i] == "prefill":
                L = slot[i].prompt.numel()
                take = min(L - length[i], budget)      # 本步这块的大小
                if take <= 0:
                    continue                           # 预算被前面的槽位用完了，下一步再来
                ensure_blocks(pool, tables[i], length[i] + take)
                flat.append(slot[i].prompt.reshape(-1)[length[i]:length[i] + take])
                # ★ 绝对位置：跨块必须连续（第 2 块要从 take 开始，而不是从 0）
                pos.append(torch.arange(length[i], length[i] + take, device=device))
                bases.append(length[i])
                sizes.append(take)
                budget -= take
                stats.prefill_tokens += take
                stats.prefill_chunks += 1
            else:                                      # decode：1 个 token
                ensure_blocks(pool, tables[i], length[i] + 1)
                flat.append(torch.tensor([last[i]], device=device))
                pos.append(torch.tensor([length[i]], device=device))
                bases.append(length[i])
                sizes.append(1)

            owners.append(i)
            kinds.append(phase[i])

        if not owners:
            break                                      # 没活跃、也没得补 → 收工

        n_pre = kinds.count("prefill")
        stats.prefill_steps += 1 if n_pre else 0
        stats.mixed_steps += 1 if n_pre and n_pre < len(kinds) else 0

        # ── 一次前向同时服务 prefill 与 decode ──
        # M7.5：纯 decode 步走 CUDA Graph（形状固定）；含 prefill 的步仍走 eager
        # （分块的 prefill 形状是变的，图的静态形状装不下）。
        if use_graph and n_pre == 0:
            ids_t = torch.tensor([last[i] for i in owners], device=device)
            pos_t = torch.tensor(bases, device=device)
            produced = [int(t) for t in graphs.step(ids_t, pos_t,
                                                    [tables[i] for i in owners], bases)]
            stats.graph_steps += 1
        else:
            hook = hook_cls(
                pool, [tables[i] for i in owners], bases, sizes)
            logits = forward_fn(torch.cat(flat).unsqueeze(0), weights,
                               attention_fn=hook,
                               position_ids=torch.cat(pos).unsqueeze(0))
            ends = torch.tensor(sizes).cumsum(0) - 1   # 各序列在拍平维度的末位
            produced = [None] * len(owners)
            # 只有“本步会产出 token”的行才采样：
            #   decode 行一定产出；prefill 行只有喂完最后一块才产出。
            # 不在中间块上白采样，才能保证“同一 seed 在不同 chunk 大小下可复现”。
            need = [j for j, i in enumerate(owners)
                    if kinds[j] == "decode"
                    or (kinds[j] == "prefill"
                        and length[i] + sizes[j] == slot[i].prompt.numel())]
            if need:
                sampled = sample_tokens(
                    logits[0, ends[need], :],
                    [slot[owners[j]].sampling for j in need],
                    [gen_for(slot[owners[j]], logits.device) for j in need])
                for k, j in enumerate(need):
                    produced[j] = int(sampled[k, 0])

        for j, i in enumerate(owners):
            length[i] += sizes[j]                      # 本步的 K/V 已进池
            if kinds[j] == "prefill":
                if length[i] == slot[i].prompt.numel():
                    # 最后一块喂完 → 本步末位 logits 就是第一个生成 token，转入 decode
                    phase[i] = "decode"
                    last[i] = produced[j]
                    register(i, last[i])
                # 还没喂完的块：不产出 token，继续 prefill
            else:
                last[i] = produced[j]
                register(i, last[i])

        stats.steps += 1
        stats.idle_slot_steps += B - len(owners)

    if owns_graphs and graphs is not None:
        graphs.close()

    outs = []
    for r in requests:
        gen = torch.tensor([generated[r.req_id]], dtype=torch.long,
                           device=r.prompt.device)
        outs.append(torch.cat([r.prompt.reshape(1, -1), gen], dim=1))
    return outs, stats
