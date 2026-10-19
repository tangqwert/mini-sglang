"""benchmark.verify_m6 — 三方对照：M2.5 / Step 4 / M6 Chunked Prefill。

同一批请求、同一个 gpt2、同一个 batch 上限，量三条路径递进地拆掉三笔开销：

  开销                      M2.5   Step4   M6
  ① 补入时的填充计算         有      消除    消除
  ② 补入要一次独立前向       有       有     消除   ← M6 修的
  ③ 单步成本峰值无上限       有       有     可限   ← max_prefill_tokens（M6 的本意）

①②合起来就是 M2.5 那个"1.25x（理想 2.00x）"的全部差距；③是 Chunked Prefill 名字的来源：
一条 4096-token 的 prompt 独占一步，会把其他请求的 ITL 打爆，所以要切块摊平。

跑法：python -m benchmark.verify_m6
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
from engine.paged_schedule import chunked_prefill_generate, paged_continuous_generate

LONG_REPEAT, LONG_BUDGET, SHORT_BUDGET, N_SHORT, BATCH = 8, 30, 2, 8, 2


class Meter:
    """计量器：前向次数 + 喂入 token 数 + 单步 token 数（成本峰值）。"""

    def __init__(self):
        self.forwards = 0
        self.tokens = 0
        self.step_sizes = []

    def note(self, n_tokens: int):
        self.forwards += 1
        self.tokens += n_tokens
        self.step_sizes.append(n_tokens)

    @property
    def max_step(self) -> int:
        return max(self.step_sizes) if self.step_sizes else 0


def build_requests(tok, device):
    long_prompt = ("In a world where artificial intelligence shapes everything, "
                   "the race between capability and responsibility defines our era. ") * LONG_REPEAT
    reqs = [Request(req_id=0,
                    prompt=tok(long_prompt, return_tensors='pt').input_ids.to(device),
                    max_new_tokens=LONG_BUDGET)]
    for i in range(N_SHORT):
        reqs.append(Request(req_id=i + 1,
                            prompt=tok(f"Question {i}: hi", return_tensors='pt').input_ids.to(device),
                            max_new_tokens=SHORT_BUDGET))
    return reqs


def run_m25(model, reqs, repeat=3):
    """M2.5：整块 DynamicCache + 左填充 + 补入独立前向。

    墙钟重复 repeat 次取最短（笔记本 GPU 有降频，单次测量不可信）；
    计量只做第一遍，否则前向次数/token 数会被重复累加。
    """
    meter = Meter()
    kv_forward = build_kv_forward(model, 'cuda')
    state = {'count': True}

    def counting(ids, cache, attention_mask=None, position_ids=None):
        if state['count']:
            meter.note(ids.shape[0] * ids.shape[1])
        return kv_forward(ids, cache, attention_mask, position_ids)

    times, outs, stats = [], None, None
    for k in range(repeat):
        state['count'] = (k == 0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs, stats = continuous_generate(counting, reqs, max_batch_size=BATCH)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return outs, stats, meter, min(times)


def run_paged(fn, weights, reqs, model, repeat=3, **kw):
    """分页路径通用跑法（fn = Step 4 或 M6），用 monkeypatch 计量喂入 token。"""
    cfg = model.config
    pool = PagedKVPool(num_blocks=512, block_size=16, num_layers=cfg.n_layer,
                       num_heads=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
                       device='cuda')
    meter = Meter()
    orig = paged_schedule.gpt2_forward
    state = {'count': True}

    def counting_forward(input_ids, w, position_ids=None, attention_fn=None):
        if state['count']:
            meter.note(input_ids.shape[0] * input_ids.shape[1])
        return orig(input_ids, w, position_ids=position_ids, attention_fn=attention_fn)

    paged_schedule.gpt2_forward = counting_forward
    times, outs, stats = [], None, None
    try:
        for k in range(repeat):
            state['count'] = (k == 0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs, stats = fn(weights, reqs, pool, max_batch_size=BATCH, **kw)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    finally:
        paged_schedule.gpt2_forward = orig
    assert len(pool.free_blocks) == pool.num_blocks, "显存应全部归还"
    return outs, stats, meter, min(times)


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2').cuda().eval()
    weights = load_gpt2_weights(model)
    reqs = build_requests(tok, 'cuda')
    long_len = reqs[0].prompt.numel()

    # 预热（CUDA 冷启动会污染第一组计时）
    warm = [Request(0, tok('warmup', return_tensors='pt').input_ids.cuda(), 2)]
    run_paged(paged_continuous_generate, weights, warm, model)
    run_paged(chunked_prefill_generate, weights, warm, model)

    r25 = run_m25(model, reqs)
    r4 = run_paged(paged_continuous_generate, weights, reqs, model)
    r6 = run_paged(chunked_prefill_generate, weights, reqs, model,
                   max_prefill_tokens=10 ** 6)

    # ── 正确性：三条路径必须逐 token 一致，且都等于朴素解码 ──
    same = all(torch.equal(a, b) and torch.equal(a, c)
               for a, b, c in zip(r25[0], r4[0], r6[0]))
    naive_ok = all(torch.equal(got, autoregressive_generate(
        lambda ids: gpt2_forward(ids, weights), r.prompt,
        DecodingConfig(max_new_tokens=r.max_new_tokens))) for r, got in zip(reqs, r6[0]))

    print(f"\n工作负载：1 条长请求（prompt {long_len} tok，预算 {LONG_BUDGET}）"
          f" + {N_SHORT} 条短请求（预算 {SHORT_BUDGET}），槽位 B={BATCH}")
    print(f"正确性：三条路径逐 token 一致 = {same}；M6 vs 朴素解码一致 = {naive_ok}\n")

    cols = ['M2.5(整块)', 'Step4(分页)', 'M6(合流)']
    runs = [r25, r4, r6]
    head = f"{'':<28}" + "".join(f"{c:>14}" for c in cols)
    print(head)
    print("-" * len(head))

    def row(name, fn):
        vals = [fn(r) for r in runs]
        print(f"{name:<28}" + "".join(f"{str(v):>14}" for v in vals))

    row("总前向次数", lambda r: r[2].forwards)
    row(" └ 补入独立前向", lambda r: r[1].admission_forwards)
    row("喂入 token 数", lambda r: r[2].tokens)
    row("单步 token 峰值", lambda r: r[2].max_step)
    row("填充位置 padded", lambda r: r[1].padded_positions)
    row("prefill/decode 同批步", lambda r: r[1].mixed_steps)
    row("prefill 分块数", lambda r: r[1].prefill_chunks)
    row("空转槽位 idle", lambda r: r[1].idle_slot_steps)
    row("墙钟 ms", lambda r: f"{r[3] * 1e3:.0f}")

    print(f"\n前向次数：M2.5 {r25[2].forwards} → Step4 {r4[2].forwards} → M6 {r6[2].forwards}"
          f"（Step4 消掉了填充计算，M6 才消掉额外的前向）")
    print(f"喂入 token：{r25[2].tokens} → {r4[2].tokens} → {r6[2].tokens}")

    # ── ③ Chunked Prefill 的本意：给单步成本设上限 ──
    print("\n【Chunked Prefill 的代价/收益】给单步 token 数设上限（长 prompt 会被切块）：")
    head2 = f"{'max_prefill_tokens':<22}{'总前向':>10}{'单步峰值':>12}{'墙钟 ms':>10}"
    print(head2)
    print("-" * len(head2))
    for chunk in (10 ** 6, 64, 16):
        _, st, mt, el = run_paged(chunked_prefill_generate, weights, reqs, model,
                                 max_prefill_tokens=chunk)
        label = "不限" if chunk > 10 ** 5 else str(chunk)
        print(f"{label:<22}{mt.forwards:>10}{mt.max_step:>12}{el * 1e3:>10.0f}")
    print("\n  上限越紧 → 单步峰值越低（ITL 更稳），但前向次数变多（总计算量不变，"
          "甚至因调度开销略增）。\n  这就是 Chunked Prefill 的核心权衡。")


if __name__ == '__main__':
    main()
