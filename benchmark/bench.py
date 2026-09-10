"""benchmark.bench — M0 baseline 吞吐测量。

用法（先确认 autoregressive_generate 的测试全绿）：
    python benchmark/bench.py --model Qwen/Qwen2.5-0.5B --max-new-tokens 128

你将得到两个核心指标（记住它们，M1 之后就靠打败它们吃饭）：
  - TTFT  (Time To First Token)：从提交 prompt 到第一个 token 出来的时间
  - Decode 吞吐 (tokens/s)：稳态下每秒生成多少 token

观察点（写进笔记）：
  1. prompt 越长，TTFT 越长 —— 因为 naive 解码每一步都要对整条序列重算前向
  2. 第 2 个 token 之后每步的耗时几乎恒定 —— 想想为什么是"几乎"而不是"完全"
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.naive_decode import DecodingConfig, autoregressive_generate, build_logits_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out", default="benchmark/results/m0_naive.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    # 关键：必须搬到 device 上。否则 generated 留在 CPU，而 logits_fn 算出的
    # next_token 在 GPU，循环里的 torch.cat 会因设备不一致报错。
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)  # [1, T]
    logits_fn = build_logits_fn(model, device)
    cfg = DecodingConfig(max_new_tokens=args.max_new_tokens)

    # 预热 1 次（首步含 CUDA kernel 编译/显存分配，不预热会污染数据）
    autoregressive_generate(logits_fn, input_ids, cfg)
    torch.cuda.synchronize()

    # 正式测量
    start = time.perf_counter()
    output = autoregressive_generate(logits_fn, input_ids, cfg)
    torch.cuda.synchronize()  # 关键：异步执行下，不同步就测不准
    elapsed = time.perf_counter() - start

    n_new = output.shape[1] - input_ids.shape[1]
    ttft_proxy = elapsed / n_new  # M0 的朴素近似：逐 token 计时在 M1 实现更精确
    throughput = n_new / elapsed

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n=== 生成文本 ===\n{text}\n")
    print(f"prompt tokens      : {input_ids.shape[1]}")
    print(f"new tokens         : {n_new}")
    print(f"total time         : {elapsed:.2f} s")
    print(f"avg time/token     : {ttft_proxy*1000:.1f} ms")
    print(f"decode throughput  : {throughput:.1f} tokens/s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "prompt_tokens": input_ids.shape[1],
        "new_tokens": n_new,
        "total_s": elapsed,
        "tokens_per_s": throughput,
    }, indent=2))
    print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
