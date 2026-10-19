"""benchmark.bench_serving — M10 HTTP 服务的并发压测（端到端延迟 + 吞吐）。

═══════════════════════════════════════════════════════════════════
 先启动服务：
   python -m engine.server --model gpt2 --port 8000

 再压测：
   python -m benchmark.bench_serving --num-prompts 32 --concurrency 1,2,4 --max-tokens 16

 说明（诚实边界）：
   · M10 是**非流式**：这里只能测 end-to-end latency，测不了 TTFT。
     真 TTFT 需要 `stream=true` + 首 token 回调。
   · 服务是**单进程锁串行**：并发升高时看到的是排队，不是 continuous batching。
   · prompt 默认是一组固定短句；要更真实的分布用 `--prompt-file` 传
     ShareGPT/LMSYS 的 prompt（每行一条）。
═══════════════════════════════════════════════════════════════════
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

DEFAULT_PROMPTS = [
    "The meaning of life is",
    "Hello world",
    "Once upon a time",
    "In a shocking finding, scientists discovered",
    "A long time ago in a galaxy",
    "The quick brown fox",
    "It was the best of times",
    "Machine learning models",
]


def percentile(sorted_values: list[float], q: float) -> float:
    """线性插值分位数（q ∈ [0,1]）。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def load_prompts(args) -> list[str]:
    if args.prompt_file:
        lines = [ln.strip() for ln in
                 Path(args.prompt_file).read_text(encoding="utf-8").splitlines()]
        prompts = [ln for ln in lines if ln]
        if not prompts:
            raise ValueError(f"{args.prompt_file} 里没有非空 prompt")
    else:
        prompts = list(DEFAULT_PROMPTS)
    # 按 num_prompts 循环取用：既支持单条 prompt 压测，也支持真实 prompt 集
    return [prompts[i % len(prompts)] for i in range(args.num_prompts)]


async def one_request(client: httpx.AsyncClient, url: str,
                      prompt: str, args) -> dict:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    t0 = time.perf_counter()
    resp = await client.post(url, json=payload)
    latency_ms = (time.perf_counter() - t0) * 1e3
    resp.raise_for_status()
    body = resp.json()
    usage = body.get("usage", {})
    return {
        "latency_ms": latency_ms,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "text": body["choices"][0]["message"]["content"],
    }


async def run_level(args, concurrency: int, prompts: list[str]) -> dict:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        # 预热：服务/模型/KV 池的首次请求不进入统计
        for _ in range(args.warmup):
            await one_request(client, url, prompts[0], args)

        async def worker(p: str):
            async with sem:
                return await one_request(client, url, p, args)

        t0 = time.perf_counter()
        raw = await asyncio.gather(*(worker(p) for p in prompts),
                                   return_exceptions=True)
        wall_s = time.perf_counter() - t0

    ok = [r for r in raw if not isinstance(r, Exception)]
    errors = [r for r in raw if isinstance(r, Exception)]
    if not ok:
        raise RuntimeError(f"全部请求失败；第一个错误：{errors[0]!r}")

    latencies = sorted(r["latency_ms"] for r in ok)
    out_tokens = sum(r["completion_tokens"] for r in ok)
    return {
        "concurrency": concurrency,
        "requests": len(raw),
        "success": len(ok),
        "errors": len(errors),
        "wall_s": wall_s,
        "requests_per_s": len(ok) / wall_s,
        "output_tokens": out_tokens,
        "output_tokens_per_s": out_tokens / wall_s,
        "mean_ms": sum(latencies) / len(latencies),
        "p50_ms": percentile(latencies, 0.50),
        "p90_ms": percentile(latencies, 0.90),
        "p99_ms": percentile(latencies, 0.99),
        "min_ms": latencies[0],
        "max_ms": latencies[-1],
    }


async def main_async(args) -> None:
    prompts = load_prompts(args)
    levels = []
    print(f"target = {args.base_url}")
    print(f"requests/concurrency = {args.num_prompts}, max_tokens = {args.max_tokens}")
    print(f"{'conc':>6}{'ok':>8}{'err':>5}{'req/s':>9}"
          f"{'out tok/s':>11}{'p50 ms':>9}{'p90 ms':>9}{'p99 ms':>9}")
    print("-" * 66)

    for c in args.concurrency:
        try:
            metrics = await run_level(args, c, prompts)
        except httpx.ConnectError as exc:
            print(f"\n连接失败：{exc}")
            print("请先启动服务：python -m engine.server --model gpt2 --port 8000")
            return
        levels.append(metrics)
        print(f"{c:>6}{metrics['success']:>8}{metrics['errors']:>5}"
              f"{metrics['requests_per_s']:>9.2f}"
              f"{metrics['output_tokens_per_s']:>11.1f}"
              f"{metrics['p50_ms']:>9.1f}{metrics['p90_ms']:>9.1f}"
              f"{metrics['p99_ms']:>9.1f}")

    result = {
        "base_url": args.base_url,
        "model": args.model,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "warmup": args.warmup,
        "note": ("非流式：latency 是端到端（没有 TTFT）；"
                 "服务是单进程锁串行，并发升高主要是排队；"
                 "backend/graph/radix 由启动 server 时的参数决定"),
        "levels": levels,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M10 HTTP serving benchmark")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--concurrency", default="1,2,4",
                   help="逗号分隔的并发级别，例如 1,2,4,8")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--prompt-file", default=None,
                   help="每行一条 prompt；默认用内置短句")
    p.add_argument("--output", default="benchmark/results/m10_serving.json")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.concurrency = [int(x) for x in args.concurrency.split(",") if x.strip()]
    if args.num_prompts < 1:
        raise ValueError("--num-prompts 至少为 1")
    if not args.concurrency or any(c < 1 for c in args.concurrency):
        raise ValueError("--concurrency 必须是正整数")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
