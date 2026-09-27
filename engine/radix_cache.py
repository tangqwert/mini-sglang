"""engine.radix_cache — M3：Radix 前缀缓存复用。

═══════════════════════════════════════════════════════════════════
 你的任务：实现 RadixCache（查树/插树）与 cached_generate（继承式解码），
 使 tests/test_radix_cache.py 全绿。
 核心契约：带缓存的输出必须与不带缓存完全一致（逐 token），
 但共享前缀的请求只算一次前缀。
═══════════════════════════════════════════════════════════════════

背景知识（M1/M2 → M3 的跨越）：
  M1 的 KV cache 是"一条序列自己的草稿纸"，用完即弃。
  M3 把草稿纸存进一棵 Radix 树（前缀树），跨请求共享：

      请求A: [1,2,3,4,5, ...生成]   → 算完，把整条序列的 KV 存进树
      请求B: [1,2,3,4,9, ...生成]   → 查树：命中前缀 [1,2,3,4]
                                    → 只需现场算 [9]，前 4 个位置的 KV 直接继承
      请求C: [7,8, ...]             → 查树未命中 → 全额计算（和 M1 一样）

  为什么叫 Radix：树按 token 段压缩存储（一段可能多个 token），
  查找 = 沿树走，命中长度 = 走过的 token 数。

mini 版的诚实声明（与 SGLang 的差距）：
  本实现的 insert 【不做节点分裂】。若新序列与已存节点"部分重叠但分叉"，
  则放弃缓存该序列（宁可不缓存，不能存错——KV 和 token 对不上是致命 bug）。
  SGLang 用节点分裂解决（把重叠段切成独立节点，KV 用 clone_cache_prefix 裁剪）
  ——那是 M3.5 的课题。

设计约束：
  - 树里存的 cache 对象是 kv_forward 的产物（真模型=DynamicCache，假模型亦然），
    树不关心它的内部结构——克隆用本文件提供的 clone_cache_prefix。
"""
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from engine.batching import Request  # noqa: F401  (与 M2 共用请求类型)


# ─────────────────────────── 已提供的工具 ───────────────────────────

def clone_cache_prefix(cache, length: int):
    """把 cache 的【前 length 个位置】的 KV 拷贝进一个全新的 DynamicCache。

    为什么全用 .clone()：切片只是视图，会拖着原 cache 的大张量不放；
    clone 彻底断开依赖，原 cache 可以安全地继续被树持有。

    前置条件：length <= cache 当前长度（调用方保证）。
    """
    new = DynamicCache()
    for i, layer in enumerate(cache.layers):
        new.update(layer.keys[:, :, :length, :].clone(),
                   layer.values[:, :, :length, :].clone(), i)
    return new


# ─────────────────────────── 你来实现 ───────────────────────────

@dataclass
class RadixNode:
    """树节点 = 一段 token + 该段完整路径的 KV cache + 子节点表。

    约定：node.tokens[0] 必须等于"父节点到本节点的边的 token"（插入时保证）。
    约定：node.cache 覆盖【根到本节点末尾】的完整路径的 KV（整条链一起存，mini 版）。
    """
    tokens: list
    cache: object = None
    children: dict = field(default_factory=dict)   # {首 token: RadixNode}


class RadixCache:
    """前缀树。你实现 match_prefix 与 insert 两个方法。"""

    def __init__(self):
        self.root = RadixNode(tokens=[])

    def match_prefix(self, token_ids: list) -> tuple[int, object]:
        """查树：返回 (命中长度 hit, 命中节点的 cache)。

        算法（沿树走，greedy 最长前缀）：
          注意：child.tokens 本身就包含"边"上的第一个 token（tokens[0] == 查字典的 key），
          所以不需要单独 hit += 1，一段比下来即可。
          node, hit, cache = root, 0, None
          while hit < len(token_ids):
              child = node.children.get(token_ids[hit])   # 用下一个 token 找分叉
              若无 child → 停
              take = child.tokens 与 token_ids[hit:] 的公共前缀长度（逐位比较的循环）
              hit += take
              cache = child.cache（它覆盖的路径 ⊇ hit，裁剪交给调用方）
              若 take < len(child.tokens)（部分重叠/ids 耗尽）→ 停
              否则 node = child（整段吃下），继续循环
          hit == 0 时返回 (0, None)。

        Returns:
            (hit, cache)：hit=0 时 cache 必为 None；
            hit>0 时 cache 覆盖的路径长度 >= hit（多出的部分由调用方裁剪）。
        """
        node, hit, cache = self.root, 0, None
        while hit < len(token_ids):
            child = node.children.get(token_ids[hit])
            if child is None:
                break
            take = 0
            while take < len(child.tokens) and hit + take < len(token_ids) and child.tokens[take] == token_ids[hit+take]:
                take += 1
            hit +=take
            cache = child.cache
            if take < len(child.tokens):
                break
            node = child
        return (hit, cache) if hit > 0 else (0, None)


    def insert(self, token_ids: list, cache):
        """插树：把一条已算完 KV 的序列存进树。

        mini 版规则（不做分裂）：
          1. 按 match_prefix 的走法走到最深的、【整段都被吃下】的节点
          2. 剩余段 token_ids[hit:] 非空 → 挂成该节点的子节点（key=剩余段[0]）
          3. 剩余段为空（序列已存在/是其前缀）→ 用新 cache 覆盖该节点
          ⚠️ 若走树时最后一步是"部分重叠"（take < len(child.tokens)），
             本实现直接放弃插入（宁缺毋错）——这是 mini 版与 SGLang 的差距。
         insert [1,2,3]     hit=3=len  → 剩余段空 → 只覆盖 cache（规则②）
         insert [1,2,3,7,8] hit=3      → 剩余段 [7,8] → 挂子节点（规则③）
         insert [1,2,9]     走到 [2,3] 时 take=1 部分重叠 → partial → 放弃（mini 规则⚠️）
        """
        node, hit = self.root, 0
        partial = False
        while hit < len(token_ids):
            child = node.children.get(token_ids[hit])
            if child is None:
                break
            take = 0
            while take < len(child.tokens) and hit + take < len(token_ids) \
                    and child.tokens[take] == token_ids[hit+take]:
                take += 1
            hit += take
            if take < len(child.tokens):
                partial = True
                break
            node = child
        if partial:
            return
        if hit == len(token_ids):
            node.cache = cache
        else:
            rest = token_ids[hit:]
            node.children[rest[0]] = RadixNode(tokens=rest, cache=cache)


def cached_generate(kv_forward, requests: list[Request], radix: RadixCache) -> list:
    """带前缀缓存的逐条解码（贪心）。


    与 M1 kv_generate 的区别：每条请求开工前先查树，命中则继承 KV，
    只现场计算未命中的后缀；收工后把整条序列回写进树。

    Args:
        kv_forward: M1/M2 的前向闭包（含 attention_mask/position_ids 参数）。
        requests: 请求列表（逐条处理，非批量）。
        radix: 已建好的（可为空的）RadixCache。

    Returns:
        list[LongTensor]，顺序同 requests，每条 = prompt + 生成部分。

    契约（每条请求的处理流程）:
        1. ids = prompt 的 token 列表（int list）
        2. hit, stored = radix.match_prefix(ids)
        3. use = min(hit, len(ids) - 1)
           ── 为什么 -1：哪怕整条 prompt 全命中，也必须现场算【至少 1 个 token】
              才能拿到"下一个词"的 logits（缓存里只有 KV，没有 logits）
        4. use > 0 → cache = clone_cache_prefix(stored, use)；否则 cache = None
        5. prefill：kv_forward(tensor([ids[use:]]), cache) —— 只喂未命中的后缀！
        6. M1 式解码循环（该请求自己的 max_new_tokens / eos_token_id）
        7. 收工：radix.insert(ids + 本条生成的 token, cache)
           （注意：cache 只覆盖到 ids+generated[:-1]——最后一个 token 的 K/V
+             要等下一次 forward 才会进 cache。所以这里也必须少记 1 个。）

    提示:
        - 张量与 list 的转换：prompt 是 [1, T] 张量 → ids = prompt[0].tolist()
        - 喂后缀时 tensor([后缀列表], dtype=long, device=prompt.device) —— 设备跟随输入
        - 第 6 步的循环体和 M1 的 kv_generate 几乎一样（argmax/cat/EOS），
          这是你第三次写它——写完你就真的会了
    """
    outs=[]
    for r in requests:
        ids = r.prompt[0].tolist()

        # 查树
        hit, stored = radix.match_prefix(ids)
        use = min(hit, len(ids)-1)
        if use > 0:
            cache = clone_cache_prefix(stored, use)
        else:
            cache = None

        # 
        suffix = ids[use:]
        feed = torch.tensor([suffix], dtype = torch.long, device=r.prompt.device)
        logits, cache = kv_forward(feed, cache)
        next_token = torch.argmax(logits[:,-1,:], dim=-1, keepdim=True)
        generated =  [next_token[0,0].item()]
        if (r.eos_token_id is not None and generated[-1] == r.eos_token_id):
            frozen = True
        else:
            frozen = r.max_new_tokens <= 1
        while not frozen:
            logits, cache = kv_forward(torch.tensor([[generated[-1]]], device=r.prompt.device), cache)
            next_token = torch.argmax(logits[:,-1,:],dim=-1,keepdim=True)
            generated.append(next_token[0,0].item())
            frozen = (r.eos_token_id is not None and generated[-1] == r.eos_token_id) \
                        or len(generated) >= r.max_new_tokens
        radix.insert(ids + generated[:-1], cache)
        outs.append(torch.cat([r.prompt.reshape(1, -1),
                               torch.tensor([generated], dtype=torch.long,
                                            device=r.prompt.device)], dim=1))
    return outs
