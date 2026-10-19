"""engine.radix_cache_paged — M3.5b：分页版 Radix 前缀缓存（block 粒度）。

═══════════════════════════════════════════════════════════════════
M3 的 radix_cache.py 是"教学版"：
  · 节点 = 一段 token，cache = 【整条路径】的 DynamicCache 克隆
  · 不做节点分裂、不做驱逐、不做引用计数
  · 复用的代价是 clone 一大块 KV（M3 实测只有 0.82x）

M3.5b 把树搬进 M4a 的分页世界：
  · 节点 = 一串【完整的物理 block】（edge 按 block 压缩）
  · 复用 = 共享 block id（零拷贝）
  · 分裂 = 把 block_ids 一分为二（不拷贝 KV）
  · 驱逐 = 把叶子的 block 还给池子（LRU）
  · 引用计数 = 每个 block 一份 ref_count（而不是每个节点一份 lock_ref）

为什么"只共享完整 block"：
  分页池里，一条序列的最后一个 block 通常没写满。如果把它共享出去，
  下一条请求继续写同一个 block 的后续槽位时，就会改掉别人正在读的 KV。
  vLLM 的 prefix caching 也采用同一规则：只缓存/共享整块。
  代价是命中长度向下对齐到 block_size；收益是共享的块从此【只读】，
  写新 token 永远落在新块上，不需要 copy-on-write。

为什么 refcount 落在 block 而不是节点：
  SGLang 的节点持有 KV indices，并给节点加 lock_ref；M3.5b 在插入时会
  【分裂节点】。一旦分裂，原来"一个节点"的块会被拆到两个节点上，
  节点级锁要么漏保护、要么永久泄漏。block 是所有结构变化中唯一稳定的
  身份，所以引用计数必须跟着 block 走 —— 这也是 vLLM block pool 的做法。

与分页池的配合：
  · PagedKVPool.allocate() 在池子耗尽时会回调 PagedRadixCache.evict()，
    自动按 LRU 归还"没有被活跃请求引用"的块。
  · 活跃请求通过 acquire_blocks() 给复用的块 +1，结束时 release_blocks()。
    ref_count == 1 表示"只有树持有"（可驱逐）；> 1 表示"树 + 活跃请求"（跳过）。

一棵树的形状示例（block_size=4，每个字母 = 一个物理 block）::

    root
     └── [1,2,3,4] (A)          ← 两分支共享第一个 block（同一份物理 KV）
          ├── [5,6,7,8] (B)
          └── [9,10,11,12] (C)

  两条序列 [1..4]+[5..8] 与 [1..4]+[9..12] 只在分叉的 block 上各存一份。
  若之后插入 [1,2,3,4]，会命中 4 个 token（一个完整 block）。
  若插入 [1,2,3]，命中 0 个 token —— 因为该 block 还没被写满，
  共享它会让后续写入污染别的序列（见上面的"只共享完整 block"）。
═══════════════════════════════════════════════════════════════════
"""
import time
from dataclasses import dataclass, field

from engine.paged_kv import PagedKVPool


def _as_int_list(seq) -> list[int]:
    """接受 list[int] 或 torch.Tensor，统一成 Python int 列表。"""
    return seq.tolist() if hasattr(seq, "tolist") else list(seq)


@dataclass(eq=False)
class BlockNode:
    """Radix 树节点：一段 token + 它独占的物理 block。

    约定：
      · ``tokens`` 的长度 == ``len(block_ids) * block_size``，且按 block 对齐。
      · ``block_ids`` 只属于【本节点这条边】；子树里的块不在这里。
      · 同一路径上的块拼起来，就是一条序列的完整页表前缀。
      · ``eq=False``：节点按对象身份哈希（要放进 evictable 集合）。
    """
    tokens: list[int]
    block_ids: list[int]
    parent: "BlockNode | None" = None
    children: dict[tuple, "BlockNode"] = field(default_factory=dict)
    last_access: float = field(default_factory=time.monotonic)


@dataclass
class PagedMatch:
    """match_prefix 的返回值。

    Attributes:
        hit_tokens: 命中的 token 数，**总是 block_size 的整数倍**。
        blocks:     命中路径上的物理 block id（按逻辑顺序，可直接当页表前缀）。
        node:       最深的命中节点（用于观测/调试；锁定在 block 上，不用它）。
    """
    hit_tokens: int
    blocks: list[int]
    node: BlockNode | None


class PagedRadixCache:
    """分页版 Radix 前缀缓存。

    典型用法（调度器里）::

        radix = PagedRadixCache(pool)          # 自动安装 pool.eviction_hook

        match = radix.match_prefix(prompt_ids)
        adopt = match.blocks[:n_blocks]
        radix.acquire_blocks(adopt)            # 请求开始共享这些块
        table = list(adopt)                    # 页表 = 共享前缀 + 后续新块
        ...
        unused = radix.insert(seq_tokens, table[:n_full])
        radix.release_blocks(adopt)            # 请求结束，树的引用仍在
        for b in unused:
            pool.free(b)                       # 重复块由调用方归还
    """

    def __init__(self, pool: PagedKVPool):
        self.pool = pool
        self.bs = pool.block_size
        self.root = BlockNode(tokens=[], block_ids=[])
        # 叶子节点集合：只有叶子可被驱逐（父节点的块被子树路径共享）。
        self.evictable: set[BlockNode] = set()
        # 统计（测试/benchmark 用）
        self.cached_blocks = 0        # 树当前持有的物理块总数
        self.hit_blocks_total = 0     # 累计命中的块数
        self.evicted_blocks = 0       # 累计驱逐的块数
        # 池子耗尽 → 自动按 LRU 驱逐
        pool.eviction_hook = self.evict

    # ─────────────────────────── 查树 ───────────────────────────

    def match_prefix(self, token_ids) -> PagedMatch:
        """沿树走，返回最长 block 对齐前缀。

        命中长度只统计【完整 block】：查询结尾若停在某个 block 中间，
        该 block 不命中（它可能还没写满，见模块文档）。
        """
        token_ids = _as_int_list(token_ids)
        bs = self.bs
        n_blocks = len(token_ids) // bs
        if n_blocks == 0:
            return PagedMatch(0, [], None)

        now = time.monotonic()
        node = self.root
        blocks: list[int] = []
        bi = 0
        last: BlockNode | None = None

        while bi < n_blocks:
            key = tuple(token_ids[bi * bs:(bi + 1) * bs])
            child = node.children.get(key)
            if child is None:
                break

            # 节点可能压缩了多个 block，逐个 block 比较
            k = 0
            limit = min(len(child.block_ids), n_blocks - bi)
            while k < limit:
                off = k * bs
                if token_ids[(bi + k) * bs:(bi + k + 1) * bs] != child.tokens[off:off + bs]:
                    break
                k += 1
            # k >= 1：第一个 block 是查 children 的 key，必然相等

            if k < len(child.block_ids):
                child = self._split(child, k)      # 查询在节点中间分叉/结束
            blocks.extend(child.block_ids)
            bi += k
            last = child
            node = child

        if last is not None:                        # LRU：命中路径整体刷新
            nd = last
            while nd is not None:
                nd.last_access = now
                nd = nd.parent
        self.hit_blocks_total += len(blocks)
        return PagedMatch(len(blocks) * bs, list(blocks), last)

    # ─────────────────────────── 插树 ───────────────────────────

    def insert(self, token_ids, block_ids) -> list[int]:
        """把一条已算完 KV 的序列（只取完整 block）并入树。

        Args:
            token_ids: 序列 token（长度 >= len(block_ids) * block_size）。
            block_ids: 与 token 前缀一一对应的物理块；**所有权移交给树**。

        Returns:
            ``unused``：树里已经有相同 token 内容、但物理 id 不同的重复块。
            调用方应把它们 ``pool.free()``。若返回空列表，说明传入的块
            要么被树接收，要么本来就是树里的块（adopted）。
        """
        token_ids = _as_int_list(token_ids)
        block_ids = list(block_ids)
        bs = self.bs
        n = len(block_ids)
        if n == 0:
            return []
        if len(token_ids) < n * bs:
            raise ValueError(
                f"block_ids 需要 {n * bs} 个 token，实际只有 {len(token_ids)} 个")

        tokens = token_ids[:n * bs]
        now = time.monotonic()
        node = self.root
        bi = 0
        unused: list[int] = []

        while bi < n:
            key = tuple(tokens[bi * bs:(bi + 1) * bs])
            child = node.children.get(key)
            if child is None:
                # 剩余整段挂成子节点（edge 压缩）
                tail = BlockNode(tokens=tokens[bi * bs:],
                                 block_ids=block_ids[bi:],
                                 parent=node,
                                 last_access=now)
                node.children[key] = tail
                self.evictable.discard(node)   # 从叶子变成内部节点
                self.evictable.add(tail)
                self.cached_blocks += n - bi
                break

            # 逐 block 比较（child 可能压缩了多个 block）
            k = 0
            limit = min(len(child.block_ids), n - bi)
            while k < limit:
                off = k * bs
                if tokens[(bi + k) * bs:(bi + k + 1) * bs] != child.tokens[off:off + bs]:
                    break
                k += 1

            if k < len(child.block_ids):
                child = self._split(child, k)

            # 所有权对账：树里已覆盖的这段，若物理块不同 → 调用方的是重复块
            for i in range(k):
                if child.block_ids[i] != block_ids[bi + i]:
                    unused.append(block_ids[bi + i])

            bi += k
            node = child
            node.last_access = now

        # 路径上的祖先也刷新（LRU 以"最近被走过"为准）
        nd = node
        while nd is not None:
            nd.last_access = now
            nd = nd.parent
        return unused

    def _split(self, child: BlockNode, k: int) -> BlockNode:
        """把 child 的前 k 个 block 切成一个独立节点，返回它（父节点不变）。"""
        bs = self.bs
        parent = child.parent
        assert parent is not None and 1 <= k < len(child.block_ids)

        prefix = BlockNode(tokens=child.tokens[:k * bs],
                           block_ids=child.block_ids[:k],
                           parent=parent,
                           last_access=child.last_access)

        child.tokens = child.tokens[k * bs:]
        child.block_ids = child.block_ids[k:]
        child.parent = prefix
        parent.children[tuple(prefix.tokens[:bs])] = prefix
        prefix.children[tuple(child.tokens[:bs])] = child

        # prefix 有子节点 → 不是叶子；child 的叶子状态不变（仍在 evictable 里）
        return prefix

    # ───────────────────── 引用计数（block 级）─────────────────────

    def acquire_blocks(self, block_ids) -> None:
        """请求开始共享这些块：每块 ref_count += 1。"""
        for b in block_ids:
            self.pool.acquire(b)

    def release_blocks(self, block_ids) -> None:
        """请求结束共享：每块 ref_count -= 1（树仍持有，所以不会回 free list）。"""
        for b in block_ids:
            self.pool.release(b)

    # ─────────────────────────── 驱逐 ───────────────────────────

    def evict(self, n_blocks: int = 1) -> int:
        """按 LRU 驱逐叶子节点，直到释放 >= n_blocks 个块。

        跳过任何被活跃请求共享的节点（块 ref_count > 1）。
        返回实际释放的块数；0 表示"没有可驱逐的东西"（池子真的满了）。
        """
        freed = 0
        while freed < n_blocks:
            victim = None
            for nd in sorted(self.evictable, key=lambda x: x.last_access):
                if nd.parent is None or not nd.block_ids:
                    continue
                if any(self.pool.ref_count[b] > 1 for b in nd.block_ids):
                    continue                      # 有活跃请求在用，不能动
                victim = nd
                break
            if victim is None:
                break

            parent = victim.parent
            del parent.children[tuple(victim.tokens[:self.bs])]
            for b in victim.block_ids:
                self.pool.free(b)
            self.cached_blocks -= len(victim.block_ids)
            self.evicted_blocks += len(victim.block_ids)
            freed += len(victim.block_ids)

            self.evictable.discard(victim)
            victim.parent = None
            if parent is not self.root and not parent.children:
                self.evictable.add(parent)         # 父节点变成新叶子
        return freed
