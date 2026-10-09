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

    # ── 正确性：批量结果 vs 逐条结果 ──
    outs_batch = batched_generate(kv_forward, reqs)
    cfgs = [DecodingConfig(max_new_tokens=r.max_new_tokens, eos_token_id=r.eos_token_id)
            for r in reqs]
    outs_single = [kv_generate(kv_forward, r.prompt.clone(), cfg)
                   for r, cfg in zip(reqs, cfgs)]

    # ── 耗时 ──
    # ⚠️ 必须先预热。本脚本早期版本让 batched 路径跑第一个，于是 CUDA 上下文创建 /
    # cuBLAS handle / kernel 编译的冷启动开销【全部落在它头上】——实测首次 581ms、
    # 预热后只有 183ms。修正后收益从 1.18x 变成 ~3.0x。
    # 教训：**冷启动偏差会系统性地惩罚"先跑的那条路径"**，两个被测对象必须等价预热。
    def run_batched():
        return batched_generate(kv_forward, reqs)

    def run_single():
        for r, cfg in zip(reqs, cfgs):
            kv_generate(kv_forward, r.prompt.clone(), cfg)

    def timed(fn, repeat: int = 3) -> float:
        fn()                                     # 预热（不计入）
        torch.cuda.synchronize()
        best = float('inf')
        for _ in range(repeat):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
        return best * 1e3                        # 取最短（笔记本 GPU 有降频）

    t_batch = timed(run_batched)
    t_single = timed(run_single)

    print('\n=== 正确性：batch vs 逐条 ===')
    all_ok = True
    for i, (b, s) in enumerate(zip(outs_batch, outs_single)):
        same = torch.equal(b, s)
        all_ok &= same
        print(f'req{i}: {"✅ 一致" if same else "❌ 不一致!"}  len={b.shape[1]}  '
              f'文本: {tok.decode(b[0], skip_special_tokens=True)[:42]!r}')

    print('\n=== 耗时（已预热，best of 3）===')
    n_forw_b = 1 + 23        # 1 次 prefill + 23 次 decode（预算 24）
    print(f'batched   (4 条一批): {t_batch:.0f} ms   ({t_batch / n_forw_b:.1f} ms / forward)')
    print(f'sequential(逐条 4 次): {t_single:.0f} ms   ({t_single / (4 * n_forw_b):.1f} ms / forward)')
    print(f'batching 收益: {t_single / t_batch:.2f}x（理想 4.00x）')
    print(f'  读法：批内每次 forward 是单条的 {t_batch / n_forw_b / (t_single / (4 * n_forw_b)):.2f}x'
          '（填充行 + mask 拼接 + Python 记账），')
    print('        但它一次服务 4 条请求 —— 所以净收益低于理想的 4x。')
    print('\n' + ('🎉 M2 真模型验证通过' if all_ok else '❌ 存在不一致，回查调度器'))


if __name__ == '__main__':
    main()
