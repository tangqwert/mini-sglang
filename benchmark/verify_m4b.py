"""benchmark.verify_m4b — M4b Step 4 对比：分页调度 vs M2.5 连续准入。

同一批请求、同一个 gpt2、同一个 batch 上限，对比两条调度路径：
  M2.5  engine.batching.continuous_generate        整块 DynamicCache + 左填充
  Step4 engine.paged_schedule.paged_continuous_generate  分页池 + varlen（零填充）

要量化的两笔账：
  ① padded_positions —— M2.5 为补入一条短请求，得把它左填充到【当前批长 S】；
     分页路径恒为 0
  ② tokens_fed —— 喂进模型的真实 token 数（算力代理指标）。
     M2.5 补入要前向 S 个位置（绝大多数是 pad），分页只前向该请求自己的 L 个

工作负载刻意选成"长请求占住一个槽位 + 短请求不断补入"，把 ① ② 放到最大。

跑法：python -m benchmark.verify_m4b
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import engine.paged_schedule as paged_schedule
from engine.batching import Request, continuous_generate
from engine.kv_cache import build_kv_forward
from engine.model_forward import gpt2_forward, load_gpt2_weights
from engine.naive_decode import DecodingConfig, autoregressive_generate
from engine.paged_kv import PagedKVPool
from engine.paged_schedule import paged_continuous_generate

LONG_BUDGET, SHORT_BUDGET, N_SHORT, BATCH = 30, 2, 8, 2


def build_requests(tok, device):
    long_prompt = ("In a world where artificial intelligence shapes everything, "
                   "the race between capability and responsibility defines our era. ") * 8
    reqs = [Request(req_id=0, prompt=tok(long_prompt, return_tensors='pt').input_ids.to(device),
                    max_new_tokens=LONG_BUDGET, eos_token_id=None)]
    for i in range(N_SHORT):
        reqs.append(Request(req_id=i + 1,
                            prompt=tok(f"Question {i}: hi", return_tensors='pt').input_ids.to(device),
                            max_new_tokens=SHORT_BUDGET, eos_token_id=None))
    return reqs


class Counter:
    def __init__(self):
        self.forwards = 0
        self.tokens = 0

    def __repr__(self):
        return f"forwards={self.forwards}, tokens_fed={self.tokens}"


def run_m25(model, reqs):
    """M2.5：整块 DynamicCache + 左填充。"""
    cnt = Counter()
    kv_forward = build_kv_forward(model, 'cuda')

    def counting(ids, cache, attention_mask=None, position_ids=None):
        cnt.forwards += 1
        cnt.tokens += ids.shape[0] * ids.shape[1]
        return kv_forward(ids, cache, attention_mask, position_ids)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs, stats = continuous_generate(counting, reqs, max_batch_size=BATCH)
    torch.cuda.synchronize()
    return outs, stats, cnt, time.perf_counter() - t0


def run_paged(weights, reqs, model):
    """Step 4：分页池 + varlen 前向（零填充）。"""
    cfg = model.config
    pool = PagedKVPool(num_blocks=512, block_size=16,
                       num_layers=cfg.n_layer, num_heads=cfg.n_head,
                       head_dim=cfg.n_embd // cfg.n_head, device='cuda')
    cnt = Counter()
    peak = [0]

    orig_forward = paged_schedule.gpt2_forward
    orig_alloc = pool.allocate

    def counting_forward(input_ids, w, position_ids=None, attention_fn=None):
        cnt.forwards += 1
        cnt.tokens += input_ids.shape[0] * input_ids.shape[1]
        return orig_forward(input_ids, w, position_ids=position_ids,
                            attention_fn=attention_fn)

    def counting_alloc():
        b = orig_alloc()
        peak[0] = max(peak[0], pool.num_blocks - len(pool.free_blocks))
        return b

    paged_schedule.gpt2_forward = counting_forward
    pool.allocate = counting_alloc
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs, stats = paged_continuous_generate(weights, reqs, pool,
                                                max_batch_size=BATCH)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    finally:
        paged_schedule.gpt2_forward = orig_forward
        pool.allocate = orig_alloc
    tokens_in_pool = peak[0] * pool.block_size
    return outs, stats, cnt, elapsed, tokens_in_pool, pool


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2').cuda().eval()
    weights = load_gpt2_weights(model)
    reqs = build_requests(tok, 'cuda')
    long_len = reqs[0].prompt.shape[1]

    # 预热（CUDA 冷启动会污染第一组计时）
    warm_ids = tok('warmup', return_tensors='pt').input_ids.cuda()
    run_paged(weights, [Request(0, warm_ids.clone(), 2)], model)

    outs_m25, s25, c25, t25 = run_m25(model, reqs)
    outs_pg, spg, cpg, tpg, tokens_pool, pool = run_paged(weights, reqs, model)

    # ── 正确性：两条调度路径必须逐 token 一致 ──
    same = all(torch.equal(a, b) for a, b in zip(outs_m25, outs_pg))
    naive_ok = True
    for r, got in zip(reqs, outs_pg):
        want = autoregressive_generate(lambda ids: gpt2_forward(ids, weights),
                                       r.prompt, DecodingConfig(max_new_tokens=r.max_new_tokens))
        naive_ok &= torch.equal(got, want)

    print(f"\n工作负载：1 条长请求（prompt {long_len} tok，预算 {LONG_BUDGET}）"
          f" + {N_SHORT} 条短请求（预算 {SHORT_BUDGET}），槽位 B={BATCH}")
    print(f"正确性：两条调度路径逐 token 一致 = {same}；分页路径 vs 朴素解码一致 = {naive_ok}\n")

    head = f"{'':<26}{'M2.5(整块)':>14}{'Step4(分页)':>14}"
    print(head)
    print("-" * len(head))
    rows = [
        ("批量步数 steps", s25.steps, spg.steps),
        ("补入次数 admissions", s25.admissions, spg.admissions),
        ("补入前向 forwards", s25.admission_forwards, spg.admission_forwards),
        ("总前向次数", c25.forwards, cpg.forwards),
        ("喂入 token 数", c25.tokens, cpg.tokens),
        ("填充位置 padded", s25.padded_positions, spg.padded_positions),
        ("空转槽位 idle", s25.idle_slot_steps, spg.idle_slot_steps),
        ("墙钟 ms", f"{t25 * 1e3:.0f}", f"{tpg * 1e3:.0f}"),
    ]
    for name, a, b in rows:
        print(f"{name:<26}{a:>14}{b:>14}")

    if c25.tokens:
        print(f"\n喂入 token 数之比：M2.5 / Step4 = {c25.tokens / max(cpg.tokens, 1):.2f}x"
              "  （分页把补入的填充计算彻底去掉）")

    # ── KV 显存：M2.5 整块缓存要按【批宽】×【槽位数】开，填充位置也占着 → 碎片 ──
    H = model.config.n_head
    D = model.config.n_embd // H
    bpe = 4                                       # float32
    s25_width = long_len + s25.steps               # M2.5 的缓存批宽 = S0 + 步数
    m25_mb = 2 * BATCH * s25_width * H * D * bpe / 1e6
    pg_mb = 2 * tokens_pool * H * D * bpe / 1e6
    print("\nKV 显存（K + V，fp32）：")
    print(f"  M2.5  批宽锁死为 S={s25_width} × {BATCH} 行 → {m25_mb:.1f} MB（填充位置也在内）")
    print(f"  Step4 峰值实际占用 {tokens_pool} 个 token → {pg_mb:.1f} MB（精确，完成即归还）")
    print(f"        （池子按 {pool.num_blocks} 块预分配；真引擎会随需扩容，此处为教学版简化）")
    print(f"        收工后已归还全部 {len(pool.free_blocks)} 块\n")


if __name__ == '__main__':
    main()
