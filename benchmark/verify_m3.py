"""benchmark.verify_m3 — M3 真模型验证：Radix 前缀缓存。

验证两件事：
  1. 正确性：cached_generate（查树+继承 KV）与逐条 kv_generate 输出逐 token 一致
  2. 收益：3 条共享长前缀的请求，cached 版墙钟时间 < 逐条版

跑法：python -m benchmark.verify_m3
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.batching import Request
from engine.kv_cache import build_kv_forward, kv_generate
from engine.naive_decode import DecodingConfig
from engine.radix_cache import RadixCache, cached_generate


def main():
    tok = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2').cuda().eval()
    kv_forward = build_kv_forward(model, 'cuda')

    # 3 条请求共享同一段长前缀（模拟"同一个 system prompt"），尾巴各不相同
    # 前缀要足够长，收益才盖得住计时噪声（13 token 时省的计算量太小）
    shared = ("In a world where artificial intelligence shapes everything, "
              "the race between capability and responsibility defines our era. ") * 6
    tails = ["the future belongs to", "we must remember that", "history will judge"]
    prompts = [shared + t for t in tails]

    reqs = [Request(req_id=i, prompt=tok(p, return_tensors='pt').input_ids.cuda(),
                    max_new_tokens=20, eos_token_id=tok.eos_token_id)
            for i, p in enumerate(prompts)]
    n_prompt = reqs[0].prompt.shape[1]

    # 预热：CUDA 冷启动（kernel 编译/显存池分配）可高达秒级，不预热会污染第一组计时
    ids_warm = tok("warmup", return_tensors='pt').input_ids.cuda()
    warm = Request(req_id=99, prompt=ids_warm, max_new_tokens=4, eos_token_id=None)
    cached_generate(kv_forward, [warm], RadixCache())
    kv_generate(kv_forward, ids_warm, DecodingConfig(max_new_tokens=4))
    torch.cuda.synchronize()

    # ── cached（M3 引擎）──
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs_cached = cached_generate(kv_forward, reqs, RadixCache())
    torch.cuda.synchronize()
    t_cached = time.perf_counter() - t0

    # ── 逐条（M1 引擎，正确性基线 + 耗时对照）──
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    outs_single = []
    for r in reqs:
        cfg = DecodingConfig(max_new_tokens=r.max_new_tokens, eos_token_id=r.eos_token_id)
        outs_single.append(kv_generate(kv_forward, r.prompt.clone(), cfg))
    torch.cuda.synchronize()
    t_single = time.perf_counter() - t1

    print(f'\n共享前缀长度: {n_prompt} tokens x 3 条请求')
    print('=== 正确性：cached vs 逐条 ===')
    all_ok = True
    for i, (c, s) in enumerate(zip(outs_cached, outs_single)):
        same = torch.equal(c, s)
        all_ok &= same
        print(f'req{i}: {"✅ 一致" if same else "❌ 不一致!"}  '
              f'文本: {tok.decode(c[0], skip_special_tokens=True)[:40]!r}')

    print('\n=== 耗时 ===')
    print(f'cached  (查树+继承): {t_cached * 1000:.0f} ms')
    print(f'sequential(逐条全算): {t_single * 1000:.0f} ms')
    print(f'radix 收益: {t_single / t_cached:.2f}x')
    print('\n' + ('🎉 M3 真模型验证通过' if all_ok else '❌ 存在不一致，回查 clone/match'))


if __name__ == '__main__':
    main()
