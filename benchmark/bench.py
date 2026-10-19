"""benchmark.bench — M0 baseline 吞吐测量。

用法（先确认 autoregressive_generate 的测试全绿）：
    python benchmark/bench.py --model Qwen/Qwen2.5-0.5B --max-new-tokens 128

你将得到四个核心指标（记住它们，M1 之后就靠打败它们吃饭）：
  - TTFT  (Time To First Token)：从提交 prompt 到第一个 token 出来的时间
  - TPOT  (Time Per Output Token)：decode 段平均每个 token 的时间（不含 prefill）
  - Decode 吞吐 (tokens/s)：稳态下每秒生成多少 token（= 1 / TPOT）
  - 端到端吞吐：把 prefill 也算进去的平均吞吐（会和 TPOT 口径不同）

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
from engine.kv_cache import build_kv_forward, kv_generate


class StepTimer:
    """记录每次「产出一个 token」的墙钟时间，用于算 TTFT / TPOT。

    回调发生在 EOS 判定之后，而 EOS 判定里的 `int(token)` 已经同步了 GPU；
    因此时间戳是 token 真正可用的时刻，且与总耗时同口径。

    TPOT 的定义（与 vLLM 等一致）：
        (最后一个 token 的时间 - 第一个 token 的时间) / (n_new - 1)
    即 **decode 段每 token 的平均耗时，不含 prefill**；n_new == 1 时无定义。
    """

    def __init__(self):
        self.t0 = None
        self.times: list[float] = []

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def __call__(self, step: int, new_tokens) -> None:
        self.times.append(time.perf_counter() - self.t0)

    def finish(self, n_new: int) -> list[float]:
        """返回每个 token 相对 start 的累计毫秒数。"""
        return [t * 1000.0 for t in self.times[:n_new]]


def _timing_stats(step_ms: list[float]) -> dict:
    """从「每个 token 的累计时间」算 TTFT / TPOT。"""
    n = len(step_ms)
    ttft_ms = step_ms[0] if n else None
    if n >= 2:
        tpot_ms = (step_ms[-1] - step_ms[0]) / (n - 1)
        decode_tput = 1000.0 / tpot_ms
    else:
        tpot_ms = None
        decode_tput = None
    return {"ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
            "decode_tokens_per_s": decode_tput}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--prompt-repeat", type=int, default=1,
                        help="把 --prompt 重复 N 次再喂进去（长 prompt 实验用，"
                             "让“同一长度同一 prompt 下换模型”这类对照可复现）")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--engine", choices=["naive", "kv"], default="naive",
                        help="naive=M0 整条序列重算；kv=M1 KV cache 增量解码")
    parser.add_argument("--out", default=None,
                        help="默认按引擎自动命名: m0_naive.json / m1_kv.json")
    args = parser.parse_args()
    if args.out is None:
        args.out = "benchmark/results/m0_naive.json" if args.engine == "naive" \
            else "benchmark/results/m1_kv.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    # 关键：必须搬到 device 上。否则 generated 留在 CPU，而 logits_fn 算出的
    # next_token 在 GPU，循环里的 torch.cat 会因设备不一致报错。
    text = (args.prompt + " ") * args.prompt_repeat
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)  # [1, T]
    cfg = DecodingConfig(max_new_tokens=args.max_new_tokens)

    # 两条引擎路径对齐成同一个 generate(ids, on_step=None) 接口——测量代码完全复用，保证公平
    if args.engine == "naive":
        logits_fn = build_logits_fn(model, device)
        def generate(ids, on_step=None):
            return autoregressive_generate(logits_fn, ids, cfg, on_step=on_step)
    else:
        kv_forward = build_kv_forward(model, device)
        def generate(ids, on_step=None):
            return kv_generate(kv_forward, ids, cfg, on_step=on_step)

    # 预热 1 次（首步含 CUDA kernel 编译/显存分配，不预热会污染数据）
    generate(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # 正式测量：on_step 在每个 token 产出后记时间点
    timer = StepTimer()
    timer.start()
    start = timer.t0                  # 与 on_step 时间戳共用同一时间原点
    output = generate(input_ids, on_step=timer)
    if device.type == "cuda":
        torch.cuda.synchronize(device)  # 关键：异步执行下，不同步就测不准
    elapsed = time.perf_counter() - start

    n_new = output.shape[1] - input_ids.shape[1]
    step_ms = timer.finish(n_new)
    assert len(step_ms) == n_new, (len(step_ms), n_new)
    timing = _timing_stats(step_ms)
    avg_per_token = elapsed / n_new
    throughput = n_new / elapsed

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n=== 生成文本 ===\n{text}\n")
    print(f"prompt tokens      : {input_ids.shape[1]}")
    print(f"new tokens         : {n_new}")
    print(f"total time         : {elapsed:.2f} s")
    if timing["ttft_ms"] is not None:
        print(f"TTFT (首 token)    : {timing['ttft_ms']:.1f} ms")
    else:
        print("TTFT               : N/A（没有生成 token）")
    if timing["tpot_ms"] is not None:
        print(f"TPOT (decode/token): {timing['tpot_ms']:.1f} ms（不含 prefill）")
        print(f"decode throughput  : {timing['decode_tokens_per_s']:.1f} tokens/s（= 1/TPOT）")
    else:
        print("TPOT               : N/A（生成 token 数 < 2）")
    print(f"avg time/token     : {avg_per_token*1000:.1f} ms（整段生成平均，含 prefill）")
    print(f"end-to-end tput    : {throughput:.1f} tokens/s（含 prefill）")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "engine": args.engine,
        "prompt_tokens": input_ids.shape[1],
        "new_tokens": n_new,
        "total_s": elapsed,
        "tokens_per_s": throughput,
        "ttft_ms": timing["ttft_ms"],
        "tpot_ms": timing["tpot_ms"],
        "decode_tokens_per_s": timing["decode_tokens_per_s"],
    }, indent=2))
    print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
