"""benchmark.verify_m2 — M2 真模型验证：批量解码 vs 逐条解码。

验证两件事：
  1. 正确性：batched_generate 的每条结果 == 该请求单独跑 kv_generate 的结果
  2. 收益：4 条请求一批的墙钟时间 vs 逐条跑 4 次的墙钟时间

跑法：python -m benchmark.verify_m2
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.batching import Request, batched_generate
from engine.kv_cache import build_kv_forward, kv_generate
from engine.naive_decode import DecodingConfig


def main():
    tok = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2').cuda().eval()
    kv_forward = build_kv_forward(model, 'cuda')

    prompts = [
        "The meaning of life is",
        "Once upon a time",
        "In a shocking finding, scientists discovered",
        "Hello world",
    ]
    reqs = [Request(req_id=i, prompt=tok(p, return_tensors='pt').input_ids.cuda(),
                    max_new_tokens=24, eos_token_id=tok.eos_token_id)
            for i, p in enumerate(prompts)]

    # ── 批量跑（M2 引擎）──
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs_batch = batched_generate(kv_forward, reqs)
    torch.cuda.synchronize()
    t_batch = time.perf_counter() - t0

    # ── 逐条跑（M1 引擎，正确性基线 + 耗时对照）──
    cfgs = [DecodingConfig(max_new_tokens=r.max_new_tokens, eos_token_id=r.eos_token_id)
            for r in reqs]
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    outs_single = [kv_generate(kv_forward, r.prompt.clone(), cfg)
                   for r, cfg in zip(reqs, cfgs)]
    torch.cuda.synchronize()
    t_single = time.perf_counter() - t1

    print('\n=== 正确性：batch vs 逐条 ===')
    all_ok = True
    for i, (b, s) in enumerate(zip(outs_batch, outs_single)):
        same = torch.equal(b, s)
        all_ok &= same
        print(f'req{i}: {"✅ 一致" if same else "❌ 不一致!"}  len={b.shape[1]}  '
              f'文本: {tok.decode(b[0], skip_special_tokens=True)[:42]!r}')

    print('\n=== 耗时 ===')
    print(f'batched   (4 条一批): {t_batch * 1000:.0f} ms')
    print(f'sequential(逐条 4 次): {t_single * 1000:.0f} ms')
    print(f'batching 收益: {t_single / t_batch:.2f}x')
    print('\n' + ('🎉 M2 真模型验证通过' if all_ok else '❌ 存在不一致，回查调度器'))


if __name__ == '__main__':
    main()
