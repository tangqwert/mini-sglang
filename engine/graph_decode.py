"""engine.graph_decode — M7：把 decode 步捕获成 CUDA Graph。

═══════════════════════════════════════════════════════════════════
 为什么做它（动机是被量出来的，不是"听说 CUDA Graph 很酷"）：
   M6 结束时量到：喂入 token 降 6.6 倍、墙钟反而慢 2 倍 → 定位到 Python 侧的
   kernel 启动开销（12 层 × 30 步 ≈ 7000 次启动）。M5 用 Triton 把每层的
   十来个 kernel 折成 1 个；M7 再把**整步**的几十个 kernel 折成 1 次 launch。

 概念验证（gpt2、decode 形态、单序列）：
     eager gpt2_forward  7.125 ms
     graph replay        2.807 ms   → 2.54x，且数值完全一致（max|diff| = 0）
   ⇒ 这个数字同时证明：**单次 forward 约 60% 的时间在 CPU 侧**，不是 GPU 算得慢。
     （早先用 CUDA event 测出的"GPU ≈ 墙钟"是误读：event 间隔包含 GPU 空等
       CPU 发 kernel 的间隙。）

 为什么分页池天然适合 CUDA Graph：
   捕获要求**地址固定**。分页池是一次性预分配的大张量、页表是固定宽度的缓冲区，
   序列变长只是改页表内容（`ensure_blocks` 写进已有缓冲区），**KV 的地址从头到尾不变**。
   连续存储方案则要随长度 realloc —— 那是 CUDA Graph 最讨厌的东西。
   → **"分页的又一个好处"**，这条值得单独记。

 本模块的边界（诚实声明）：
   · 只捕获**纯 decode 步**（每行恰好 1 个 token）。prefill / chunked prefill 的
     形状是变的，仍走 eager 路径 —— 这与真引擎一致（vLLM 的 graph 也只覆盖 decode）。
   · 批大小必须落在预捕获的集合里（默认 {1,2,4,8}）；不足时**用 scratch 页补齐**，
     多出来的行结果丢弃。padding 行写进专用的 scratch 块，不会污染真实序列。
═══════════════════════════════════════════════════════════════════
"""
import torch

from engine.paged_kv import PagedKVPool
from engine.triton_paged import _paged_attn_kernel

DEFAULT_BATCH_SIZES = (1, 2, 4, 8)


class _GraphPagedHook:
    """专供 CUDA Graph 的纯 decode 分页注意力钩子。

    与 `TritonPagedAttentionHook` 的三点区别（都是为了"可捕获"）：
      ① 页表 / 起点 / 总长全部来自**外部预分配缓冲区**，图内只读，不新建张量
      ② 写入按 B 条序列**一次性向量化**（1 个 index_put，而不是每序列一个）
      ③ 不做任何 `torch.as_tensor(list)` —— 那种 CPU→GPU 拷贝在图内是非法的
    """

    def __init__(self, pool: PagedKVPool, bt_t: torch.Tensor, base_t: torch.Tensor,
                 totalk_t: torch.Tensor, cu_t: torch.Tensor, n_seq: int,
                 block_m: int = 16, block_n: int = 64):
        self.pool = pool
        self.bt_t, self.base_t, self.totalk_t, self.cu_t = bt_t, base_t, totalk_t, cu_t
        self.n_seq = n_seq
        self.rows = torch.arange(n_seq, device=pool.keys.device)   # 预建，图内复用
        self.block_m, self.block_n = block_m, block_n

    def __call__(self, q, k, v, layer_idx: int):
        B, H, S, Dh = k.shape
        assert B == 1 and S == self.n_seq, \
            f"decode 图要求拍平成 [1, {self.n_seq}]，收到 [{B}, {S}]"
        Hq = q.shape[1]
        G = Hq // H                       # GQA：Hq / Hkv（GPT-2 为 1）
        bs = self.pool.block_size

        # ① 写：B 个 token 一次性 scatter（绝对位置 → 块 + 槽位）
        blk = self.bt_t[self.rows, self.base_t // bs]
        slot = self.base_t % bs
        self.pool.keys[layer_idx, blk, slot] = k[0].transpose(0, 1)
        self.pool.values[layer_idx, blk, slot] = v[0].transpose(0, 1)

        # ② 注意力：一个 kernel 算完全部序列 × 全部 head
        out = torch.empty_like(q)
        keys = self.pool.keys[layer_idx]
        vals = self.pool.values[layer_idx]
        qv, ov = q[0], out[0]
        _paged_attn_kernel[(self.n_seq, 1, Hq)](
            qv, keys, vals, ov,
            self.bt_t, self.cu_t, self.base_t, self.totalk_t,
            Dh ** -0.5,
            qv.stride(0), qv.stride(1),
            keys.stride(0), keys.stride(1), keys.stride(2),
            ov.stride(0), ov.stride(1),
            self.bt_t.stride(0),
            H=Hq, D=Dh,
            BLOCK_M=self.block_m, BLOCK_N=self.block_n,
            BLOCK_SIZE=bs,
            G=G,
        )
        return out


class DecodeGraph:
    """固定批大小的一张 decode 图（含它自己的静态缓冲区）。"""

    def __init__(self, weights, pool: PagedKVPool, n_seq: int,
                 max_blocks: int, scratch_block: int,
                 block_m: int = 16, block_n: int = 64, warmup: int = 3,
                 capture: bool = True):
        self.weights, self.pool, self.n_seq = weights, pool, n_seq
        self.scratch_block, self.max_blocks = scratch_block, max_blocks
        dev = pool.keys.device

        H, D = pool.num_heads, pool.head_dim
        # ── 静态缓冲区：地址固定，图内只读/只写 ──
        self.ids = torch.zeros(1, n_seq, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, n_seq, dtype=torch.long, device=dev)
        self.bt = torch.full((n_seq, max_blocks), scratch_block,
                             dtype=torch.int32, device=dev)
        self.base = torch.zeros(n_seq, dtype=torch.int32, device=dev)
        self.totalk = torch.zeros(n_seq, dtype=torch.int32, device=dev)
        self.cu = torch.arange(n_seq + 1, dtype=torch.int32, device=dev)   # 静态
        self.out_tok = torch.zeros(n_seq, dtype=torch.long, device=dev)

        self.hook = _GraphPagedHook(pool, self.bt, self.base, self.totalk, self.cu,
                                    n_seq, block_m, block_n)
        self.graph = self._capture(warmup) if capture else None

    def _step(self):
        from engine.model_forward import gpt2_forward          # 避免循环导入
        logits = gpt2_forward(self.ids, self.weights, self.pos,
                              attention_fn=self.hook)
        self.out_tok.copy_(logits[0].argmax(dim=-1))

    def _capture(self, warmup: int) -> torch.cuda.CUDAGraph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                    # 预热必须在旁路流上做
            for _ in range(warmup):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step()
        return g

    def load(self, ids, positions, block_tables: list, seq_lens) -> None:
        """把这一步的输入写进静态缓冲区（图外操作）。不足的行用 scratch 页补齐。

        ⚠️ padding 行【必须】指向 scratch 块：否则它们会拿上一轮的旧页表，
        从而把 K/V **写进真实序列的块里**、静默污染结果。
        （padding 行的输出会被丢弃，但它们照常写 KV —— 这是最容易漏的地方。）
        """
        n = len(block_tables)
        assert n <= self.n_seq
        self.bt[n:] = self.scratch_block            # 补齐行 → 专用 scratch 页
        for i in range(n):
            t = block_tables[i]
            assert len(t) <= self.max_blocks, \
                f"序列 {i} 需要 {len(t)} 个 block，超过 max_blocks={self.max_blocks}"
            self.bt[i, :len(t)] = torch.as_tensor(t, dtype=torch.int32,
                                                  device=self.bt.device)
        self.base[:n] = torch.as_tensor(seq_lens, dtype=torch.int32,
                                        device=self.base.device)
        self.base[n:] = 0                           # padding 行：位置 0 → 只看 scratch
        self.totalk.copy_(self.base + 1)
        self.ids[0, :n] = ids
        self.ids[0, n:] = 0
        self.pos[0, :n] = positions
        self.pos[0, n:] = 0

    def replay(self) -> torch.Tensor:
        """图路径：replay 一次 = 一次 launch 完成整步。"""
        self.graph.replay()
        return self.out_tok

    def run_eager(self) -> torch.Tensor:
        """同一条 hook、同一份静态缓冲区，但**逐步 launch**。

        存在的意义是**消融实验**：把"向量化写入"与"CUDA Graph"两个变量拆开。
        否则无法回答"收益到底来自图，还是来自写入向量化"。
        """
        assert self.graph is None, "用 capture=False 构造才能跑 eager 模式"
        self._step()
        return self.out_tok


class PagedDecodeGraphs:
    """按批大小管理多张图 —— 调度器只调 `step()`。

    批大小落在 `batch_sizes` 里就直接用；否则**向上取最近的一档**，多出来的行
    用 scratch 页补齐（结果丢弃）。
    """

    def __init__(self, weights, pool: PagedKVPool, batch_sizes=DEFAULT_BATCH_SIZES,
                 max_blocks: int = 64, **kw):
        self.pool = pool
        # 预留一个 scratch 块给 padding 行：它有自己的 K/V 空间，永不与真实序列冲突
        self.scratch_block = pool.allocate()
        self.max_blocks = max_blocks
        self.sizes = sorted(batch_sizes)
        self.graphs = {n: DecodeGraph(weights, pool, n, max_blocks,
                                      self.scratch_block, **kw)
                       for n in self.sizes}

    def pick(self, n: int) -> DecodeGraph:
        for s in self.sizes:
            if s >= n:
                return self.graphs[s]
        raise ValueError(f"批大小 {n} 超过已捕获的最大值 {self.sizes[-1]}")

    def step(self, ids, positions, block_tables: list, seq_lens) -> torch.Tensor:
        g = self.pick(len(block_tables))
        g.load(ids, positions, block_tables, seq_lens)
        return g.replay()[: len(block_tables)].clone()


def graph_decode(weights, pool: PagedKVPool, tables: list, prompt_lens: list,
                 first_tokens: list, max_new: int,
                 batch_sizes=DEFAULT_BATCH_SIZES,
                 max_blocks: int = 64) -> list[list[int]]:
    """端到端小工具：从「已 prefill 好的状态」开始，用图跑完 decode。

    ⚠️ `tables` 必须与 prefill 用的是**同一批页表** —— 因为 prompt 的 K/V 已经
       写进那些块了。如果这里重新分配一套表，图会去读一块**空的**显存，
       表现是：首 token（eager prefill 算的）正确，之后每步都吐同一个 token。
       调用方需先用 `ensure_blocks(pool, tables[i], prompt_lens[i] + max_new)`
       预留容量。

    Args:
        tables: 每条序列的页表（与 prefill 共用）
        prompt_lens: 每条序列 prompt 的长度 = 第一个 decode 步的绝对位置
        first_tokens: 每条序列 prefill 产出的第一个新 token
        max_new: 每条序列一共生成几个 token（含 first_tokens）

    Returns:
        list[list[int]]，每条序列生成的 token（长度 max_new）。
    """
    n = len(tables)
    assert len(prompt_lens) == len(first_tokens) == n
    lengths = list(prompt_lens)
    gen = [[first_tokens[i]] for i in range(n)]
    cur = list(first_tokens)
    graphs = PagedDecodeGraphs(weights, pool, batch_sizes, max_blocks)
    dev = pool.keys.device

    for _ in range(max_new - 1):
        ids = torch.tensor(cur, dtype=torch.long, device=dev)
        pos = torch.tensor(lengths, dtype=torch.long, device=dev)
        nxt = graphs.step(ids, pos, tables, lengths)
        for i in range(n):
            lengths[i] += 1                    # 本步的 K/V 已进池
            cur[i] = int(nxt[i])
            gen[i].append(cur[i])
    return gen
