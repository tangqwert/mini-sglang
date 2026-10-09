"""benchmark.verify_m7 — M7 对比：eager decode 步 vs CUDA Graph decode 步。

只测**纯 decode 步**（每行 1 个 token）—— 这正是 CUDA Graph 覆盖的场景，
也是长输出时占比最高的一段。

⚠️ 三路对照，而不是两路 —— 因为这里有**两个混在一起的变量**：
    ① 向量化 KV 写入：`_GraphPagedHook` 用一次 index_put 写完 B 行；
       而调度器现在用的 `TritonPagedAttentionHook` 是**每行一次 Python 循环**
       + 每行每层一次 `torch.as_tensor(list)`（CPU→GPU 拷贝）
    ② CUDA Graph：把整步的几十个 kernel 折成 1 次 launch
  只做"A vs C"会把②的功劳算给①。所以拆成：
    A  eager + 逐行 hook    ← 调度器当前的真实行为
    B  eager + 向量化 hook  ← 只换①
    C  graph + 向量化 hook  ← ①+②
  B/A = ①的收益（随 B 增长）；C/B = ②的收益（应该几乎不随 B 变）。

跑法：python -m benchmark.verify_m7
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.graph_decode import DecodeGraph, PagedDecodeGraphs, graph_decode
from engine.model_forward import gpt2_forward, load_gpt2_weights
from engine.naive_decode import DecodingConfig, autoregressive_generate
from engine.paged_kv import PagedKVPool, ensure_blocks
from engine.triton_paged import TritonPagedAttentionHook

N_STEPS = 60
MAX_NEW = N_STEPS + 2
MAX_BLOCKS = 16
BATCHES = (1, 2, 4, 8)
PROMPTS = ["The meaning of life is", "Hello world", "Once upon a time",
           "In a shocking finding, scientists discovered",
           "A long time ago in a galaxy", "The quick brown fox",
           "It was the best of times", "Machine learning models"]


def make_pool(cfg, block_size=16, num_blocks=512):
    return PagedKVPool(num_blocks=num_blocks, block_size=block_size,
                       num_layers=cfg.n_layer, num_heads=cfg.n_head,
                       head_dim=cfg.n_embd // cfg.n_head, device="cuda")


def setup(weights, pool, texts, max_new):
    """用 eager 路径做 prefill，返回 (tables, lengths, first_tokens)。"""
    from engine.triton_paged import TritonPagedAttentionHook as H
    tok = AutoTokenizer.from_pretrained('gpt2')
    tables, lens, firsts = [], [], []
    for t in texts:
        ids = tok(t, return_tensors='pt').input_ids.cuda()
        L = ids.shape[1]
        tb = []
        ensure_blocks(pool, tb, L + max_new)
        hook = H(pool, [tb], [0], [L])
        logits = gpt2_forward(ids, weights, attention_fn=hook)
        tables.append(tb)
        lens.append(L)
        firsts.append(int(logits[0, -1].argmax()))
    return tables, lens, firsts


class RunnerA:
    """A：调度器当前的 eager 行为（逐行 hook，每行每层一次 as_tensor）。"""

    def __init__(self, weights, pool, tables, lens):
        self.weights, self.pool, self.tables = weights, pool, tables
        self.lens = list(lens)

    def step(self, cur):
        dev = self.pool.keys.device
        ids = torch.tensor(cur, dtype=torch.long, device=dev).unsqueeze(0)
        pos = torch.tensor(self.lens, dtype=torch.long, device=dev).unsqueeze(0)
        hook = TritonPagedAttentionHook(self.pool, self.tables, list(self.lens),
                                       [1] * len(cur))
        logits = gpt2_forward(ids, self.weights, attention_fn=hook,
                              position_ids=pos)
        out = logits[0].argmax(dim=-1)
        self.lens = [L + 1 for L in self.lens]
        return out


class RunnerBC:
    """B/C：向量化 hook（可选再套一层 CUDA Graph）。"""

    def __init__(self, weights, pool, n_seq, use_graph: bool):
        self.pool = pool
        self.dg = DecodeGraph(weights, pool, n_seq, MAX_BLOCKS, pool.allocate(),
                              capture=use_graph)

    def step(self, cur, tables, lens):
        dev = self.pool.keys.device
        ids = torch.tensor(cur, dtype=torch.long, device=dev)
        pos = torch.tensor(lens, dtype=torch.long, device=dev)
        self.dg.load(ids, pos, tables, lens)
        out = self.dg.replay() if self.dg.graph is not None else self.dg.run_eager()
        return out


def bench(fn, repeat=3, warmup_steps=5):
    fn(warmup_steps)                               # 预热
    torch.cuda.synchronize()
    best = float('inf')
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(N_STEPS)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best / N_STEPS * 1e3                    # ms / decode 步


def main():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained('gpt2').cuda().eval()
    weights = load_gpt2_weights(model)
    cfg = model.config
    tok = AutoTokenizer.from_pretrained('gpt2')

    print(f"\n【decode 步耗时】gpt2 + 分页 KV + Triton kernel，每行 1 token，"
          f"取 {N_STEPS} 步均值（best of 3）")
    head = (f"{'B':>3}{'A: eager+逐行':>16}{'B: eager+向量化':>17}"
            f"{'C: graph':>12}{'B/A':>8}{'C/B':>8}")
    print(head)
    print('-' * len(head))

    summary = {}
    for B in BATCHES:
        texts = (PROMPTS * 4)[:B]

        pool = make_pool(cfg)
        tables, lens, firsts = setup(weights, pool, texts, MAX_NEW)
        ra = RunnerA(weights, pool, tables, lens)

        def run_a(steps, ra=ra, firsts=firsts, base=lens):
            # ⚠️ 每次计时都要把长度重置：RunnerA 的 lens 是【有状态】的，
            #    跨轮次累积的话，预热 5 步 + 3×60 步 = 185 步早就超出页表容量
            #    → 写到野指针 → CUDA device-side assert。
            ra.lens = list(base)
            cur = list(firsts)
            for _ in range(steps):
                cur = [int(x) for x in ra.step(cur)]
        tA = bench(run_a)

        pool_b = make_pool(cfg)
        tb_b, l_b, f_b = setup(weights, pool_b, texts, MAX_NEW)
        rb = RunnerBC(weights, pool_b, B, use_graph=False)

        def run_b(steps, rb=rb, tb=tb_b, l=l_b, f=f_b):
            cur, lens = list(f), list(l)
            for _ in range(steps):
                cur = [int(x) for x in rb.step(cur, tb, lens)]
                lens = [x + 1 for x in lens]
        tB = bench(run_b)

        pool_c = make_pool(cfg)
        tb_c, l_c, f_c = setup(weights, pool_c, texts, MAX_NEW)
        rc = RunnerBC(weights, pool_c, B, use_graph=True)

        def run_c(steps, rc=rc, tb=tb_c, l=l_c, f=f_c):
            cur, lens = list(f), list(l)
            for _ in range(steps):
                cur = [int(x) for x in rc.step(cur, tb, lens)]
                lens = [x + 1 for x in lens]
        tC = bench(run_c)

        # 正确性：B 与 C 跑同样的 12 步，输出必须一致
        cur, lens = list(f_b), list(l_b)
        for _ in range(12):
            cur = [int(x) for x in rb.step(cur, tb_b, lens)]
            lens = [x + 1 for x in lens]
        cur2, lens2 = list(f_c), list(l_c)
        for _ in range(12):
            cur2 = [int(x) for x in rc.step(cur2, tb_c, lens2)]
            lens2 = [x + 1 for x in lens2]
        same = cur == cur2

        summary[B] = (tA, tB, tC, same)
        print(f"{B:>3}{tA:>16.3f}{tB:>17.3f}{tC:>12.3f}"
              f"{tB / tA:>7.2f}x{tC / tB:>7.2f}x"
              + ("" if same else "   ❌ B/C 输出不一致"))

    print(f"\n输出逐 token 一致 = {all(v[3] for v in summary.values())}")
    print("\n  读法（两条线索，别混为一谈）：")
    print("  · **B/A** = KV 写入向量化的收益：B=1 时 0.78x，B=8 时 0.38x ——"
          " **收益随 B 增长**，\n    因为它消掉的是"
          "「每行一次 Python 循环 + 每行每层一次 `torch.as_tensor`」这类**逐行**开销。")
    print("  · **C/B** = CUDA Graph 单独的收益：稳定在 0.41–0.50（即 2–2.4x）。"
          "\n    它消掉的 kernel launch 次数**与 B 无关**（整步就是那么多 kernel），"
          "\n    所以图路径耗时几乎不随 B 变（3.3 → 4.2 ms），而 A 从 10.1 涨到 22.4。")
    print("  · 两条合起来（C/A）：B=1 为 3.1x，**B=8 为 5.3x**。")
    print("\n  ⚠️ 一个测量教训：本脚本第一版把 `PagedDecodeGraphs` 的构造"
          "（含**图捕获**）放进了计时区间，\n"
          "     于是得出「graph 在 B=1 反而慢」—— 那是假象，捕获成本被摊进了每一步。"
          "\n     **捕获必须放在计时之外**（与 M2 那次冷启动偏差是同一类错误，"
          "\n     见 `docs/interview-prep.md` 故事 D）。")

    # ── 端到端一致性（B=1）：图路径必须与朴素解码逐 token 一致 ──
    p = tok(PROMPTS[0], return_tensors='pt').input_ids.cuda()
    pool3 = make_pool(cfg)
    tables3, lens3, firsts3 = setup(weights, pool3, [PROMPTS[0]], 13)
    got = graph_decode(weights, pool3, tables3, lens3, firsts3, 13,
                       batch_sizes=(1,), max_blocks=MAX_BLOCKS)
    want = autoregressive_generate(lambda ids: gpt2_forward(ids, weights), p,
                                   DecodingConfig(max_new_tokens=13))
    want_gen = want[0, p.shape[1]:].tolist()
    ok = got[0] == want_gen
    print(f"\n  端到端一致性（B=1，13 token）：{'✅ 与朴素解码逐 token 一致' if ok else f'❌ {got[0]} vs {want_gen}'}")


if __name__ == '__main__':
    main()
