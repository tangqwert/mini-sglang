"""benchmark.bench_matrix — 单请求 / 多请求的统一评测矩阵（第一版）。

目标不是“再多一个 benchmark”，而是把**同一个 workload、同一套时间戳**用于：

  单请求：naive / KV / Paged-PyTorch / Paged-Triton / Paged-Triton-Graph
  多请求：顺序 KV / 静态 batching / Chunked+Triton（可选 Graph）

指标：
  · TTFT：首 token 时间
  · TPOT：(最后一个 token - 首 token) / (n_new - 1)，不含 prefill
  · 端到端：total wall；多请求另给 makespan / req/s / output tok/s / p50/p99

⚠️ 这是“系统级”对比，不是单变量消融。归因请配合 verify_m*.py：
    M0→M1  KV Cache
    M4b→M5 kernel
    eager→M7 Graph

跑法：
    python -m benchmark.bench_matrix --model gpt2 --single --multi
    python -m benchmark.bench_matrix --model Qwen/Qwen3-0.6B --single
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.batching import Request, batched_generate
from engine.kv_cache import build_kv_forward, kv_generate
from engine.model_forward import gpt2_forward, load_gpt2_weights
from engine.naive_decode import DecodingConfig, autoregressive_generate, build_logits_fn
from engine.paged_kv import BatchedPagedAttentionHook, PagedKVPool
from engine.paged_schedule import chunked_prefill_generate

try:
    from engine.triton_paged import HAS_TRITON, TritonPagedAttentionHook
except ImportError:                                   # pragma: no cover
    HAS_TRITON = False
    TritonPagedAttentionHook = None


class WallTimer:
    """记录每个请求每个 token 的墙钟时间（毫秒，相对 timer.start）。"""

    def __init__(self):
        self.t0 = None
        self.times = defaultdict(list)

    def start(self):
        self.t0 = time.perf_counter()

    def mark(self, req_id=0):
        if self.t0 is None:
            self.t0 = time.perf_counter()
        self.times[req_id].append((time.perf_counter() - self.t0) * 1000.0)


def _req_metrics(times_ms):
    n = len(times_ms)
    if n == 0:
        return {"ttft_ms": None, "tpot_ms": None, "e2e_ms": None, "n": 0}
    ttft = times_ms[0]
    e2e = times_ms[-1]
    tpot = (e2e - ttft) / (n - 1) if n >= 2 else None
    return {"ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": e2e, "n": n}


def _stats_dict(stats):
    if stats is None:
        return None
    return {
        "steps": stats.steps, "admissions": stats.admissions,
        "prefill_steps": stats.prefill_steps,
        "mixed_steps": stats.mixed_steps,
        "padded_positions": stats.padded_positions,
        "prefill_tokens": stats.prefill_tokens,
        "graph_steps": stats.graph_steps,
    }


def _multi_summary(per_req, wall, n_req, new_tokens, stats):
    return {
        "wall_s": wall,
        "req_per_s": n_req / wall,
        "output_tok_per_s": new_tokens / wall,
        "p50_e2e_ms": statistics.median(
            [m["e2e_ms"] for m in per_req if m["e2e_ms"] is not None]),
        "p99_e2e_ms": max(m["e2e_ms"] for m in per_req
                          if m["e2e_ms"] is not None),
        "p50_ttft_ms": statistics.median(
            [m["ttft_ms"] for m in per_req if m["ttft_ms"] is not None]),
        "p50_tpot_ms": statistics.median(
            [m["tpot_ms"] for m in per_req if m["tpot_ms"] is not None]),
        "per_req": per_req,
        "stats": _stats_dict(stats),
    }


def _single_summary(times, out, wall, prompt, stats):
    m = _req_metrics(times)
    m["wall_s"] = wall
    m["tokens"] = int(out.shape[1] - prompt.shape[1])
    m["graph_steps"] = getattr(stats, "graph_steps", 0)
    return m


def load_meta(model_name, device, dtype=torch.float32):
    """加载 HF 模型 + 自研前向所需的 weights / pool 配置。"""
    tok = AutoTokenizer.from_pretrained(model_name)
    if "qwen3" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype).to(device).eval()
        from engine.qwen3_forward import load_qwen3_weights, qwen3_forward
        weights = load_qwen3_weights(model)
        n_layers = model.config.num_hidden_layers
        n_heads = getattr(model.config, "num_key_value_heads",
                          model.config.num_attention_heads)
        head_dim = getattr(model.config, "head_dim",
                           model.config.hidden_size // model.config.num_attention_heads)
        forward_fn = qwen3_forward
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
        weights = load_gpt2_weights(model)
        n_layers = model.config.n_layer
        n_heads = model.config.n_head
        head_dim = model.config.n_embd // model.config.n_head
        forward_fn = gpt2_forward
    return {
        "model_name": model_name, "tokenizer": tok, "model": model,
        "weights": weights, "forward_fn": forward_fn,
        "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
        "device": device, "dtype": dtype,
    }


def make_pool(meta, num_blocks=None, block_size=16):
    return PagedKVPool(
        num_blocks=num_blocks or meta.get("num_blocks", 256), block_size=block_size,
        num_layers=meta["n_layers"], num_heads=meta["n_heads"],
        head_dim=meta["head_dim"], device=meta["device"],
        dtype=meta.get("dtype", torch.float32))


def triton_hook_for(meta):
    """Qwen3 的 head_dim=128；BLOCK_N=64 会把 shared memory 顶爆，降到 32。"""
    if meta["head_dim"] > 64:
        return partial(TritonPagedAttentionHook, block_n=32)
    return TritonPagedAttentionHook


def triton_block_n(meta):
    return 32 if meta["head_dim"] > 64 else 64


def single_prompt(tokenizer, device, repeat, model_name):
    # gpt2: 139 次约 696 token；Qwen3: 141 次约 706 token
    n = 141 if "qwen3" in model_name.lower() else 139
    n = repeat or n
    text = ("The meaning of life is " * n)
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def _run_single_mode(mode, meta, prompt, max_new, pool=None, decode_graphs=None):
    """执行一次单请求；返回 (times, output, wall_s, stats)。"""
    device = meta["device"]
    timer = WallTimer()
    req = Request(req_id=0, prompt=prompt, max_new_tokens=max_new,
                  eos_token_id=None)
    stats = None

    if mode == "naive":
        logits_fn = build_logits_fn(meta["model"], device)
        cfg = DecodingConfig(max_new_tokens=max_new)
        timer.start()
        out = autoregressive_generate(
            logits_fn, prompt, cfg, on_step=lambda step, _tok: timer.mark(0))
    elif mode == "kv":
        kv_forward = build_kv_forward(meta["model"], device)
        cfg = DecodingConfig(max_new_tokens=max_new)
        timer.start()
        out = kv_generate(
            kv_forward, prompt, cfg, on_step=lambda step, _tok: timer.mark(0))
    else:
        pool = pool or make_pool(meta)
        if mode in ("paged-triton", "paged-triton-graph") and not HAS_TRITON:
            raise RuntimeError("未安装 Triton")
        hook_cls = None if mode == "paged-torch" else triton_hook_for(meta)
        use_graph = mode == "paged-triton-graph"
        timer.start()
        outs, stats = chunked_prefill_generate(
            meta["weights"], [req], pool, max_batch_size=1,
            max_prefill_tokens=10 ** 9, hook_cls=hook_cls,
            use_graph=use_graph, forward_fn=meta["forward_fn"],
            on_token=lambda rid, _tok, _n: timer.mark(rid),
            decode_graphs=decode_graphs)
        out = outs[0]
    torch.cuda.synchronize() if device.type == "cuda" else None
    wall = (time.perf_counter() - timer.t0) if timer.t0 is not None else 0.0
    return timer.times[0], out, wall, stats


def run_single(meta, prompt, max_new, repeat):
    results = {}
    warm = prompt[:, : min(8, prompt.shape[1])]
    modes = ("naive", "kv", "paged-torch", "paged-triton")
    for mode in modes:
        if mode == "paged-triton" and not HAS_TRITON:
            continue
        # 预热：Triton 首次编译 / cuBLAS 初始化
        try:
            _run_single_mode(mode, meta, warm, 2)
        except Exception as exc:                    # pragma: no cover
            print(f"  [skip] {mode}: {exc}")
            continue
        runs = []
        for _ in range(repeat):
            times, out, wall, stats = _run_single_mode(mode, meta, prompt, max_new)
            runs.append(_single_summary(times, out, wall, prompt, stats))
        runs.sort(key=lambda r: (r["tpot_ms"] is None, r["tpot_ms"] or 1e9))
        results[mode] = runs[len(runs) // 2]

    # Graph 单独跑：图必须在计时前捕获好，否则捕获成本会污染 TTFT。
    if HAS_TRITON:
        mode = "paged-triton-graph"
        try:
            from engine.graph_decode import PagedDecodeGraphs
            pool = make_pool(meta)
            longest = prompt.shape[1] + max_new
            max_blocks = -(-longest // pool.block_size) + 1
            graphs = PagedDecodeGraphs(
                meta["weights"], pool, batch_sizes=(1,), max_blocks=max_blocks,
                forward_fn=meta["forward_fn"], block_n=triton_block_n(meta))
            try:
                # 预热图路径（不捕获，图已建好）
                _run_single_mode(mode, meta, warm, 2, pool=pool,
                                 decode_graphs=graphs)
                runs = []
                for _ in range(repeat):
                    times, out, wall, stats = _run_single_mode(
                        mode, meta, prompt, max_new, pool=pool,
                        decode_graphs=graphs)
                    runs.append(_single_summary(times, out, wall, prompt, stats))
                runs.sort(key=lambda r: r["tpot_ms"] or 1e9)
                results[mode] = runs[len(runs) // 2]
            finally:
                graphs.close()
        except Exception as exc:                    # pragma: no cover
            print(f"  [skip] {mode}: {exc}")
    return results


def build_multi_requests(tokenizer, device, n_short=8):
    long_prompt = ("In a world where artificial intelligence shapes everything, "
                   "the race between capability and responsibility defines our era. ") * 8
    reqs = [Request(0, tokenizer(long_prompt, return_tensors="pt").input_ids.to(device),
                    max_new_tokens=64, eos_token_id=None)]
    for i in range(n_short):
        ids = tokenizer(f"Question {i}: hi", return_tensors="pt").input_ids.to(device)
        reqs.append(Request(i + 1, ids, max_new_tokens=16, eos_token_id=None))
    return reqs


def _fresh_requests(reqs):
    """benchmark 重复跑时重建 Request，避免上一次的采样状态污染。"""
    return [Request(r.req_id, r.prompt.clone(), r.max_new_tokens,
                    eos_token_id=r.eos_token_id, sampling=r.sampling)
            for r in reqs]


def run_multi(meta, reqs, repeat, batch=2, max_prefill_tokens=64):
    results = {}
    device = meta["device"]
    kv_forward = build_kv_forward(meta["model"], device)

    def seq_kv():
        timer = WallTimer()
        outs, per_req = [], []
        t0 = time.perf_counter()
        for r in _fresh_requests(reqs):
            rt = WallTimer()
            rt.start()
            out = kv_generate(
                kv_forward, r.prompt, DecodingConfig(max_new_tokens=r.max_new_tokens),
                on_step=lambda step, _tok, rid=r.req_id: rt.mark(rid))
            torch.cuda.synchronize() if device.type == "cuda" else None
            outs.append(out)
            per_req.append(_req_metrics(rt.times[r.req_id]))
        wall = time.perf_counter() - t0
        return outs, per_req, wall, None

    def static_batch():
        timer = WallTimer()
        timer.start()
        outs = batched_generate(kv_forward, _fresh_requests(reqs),
                                on_token=lambda rid, _tok, _n: timer.mark(rid))
        torch.cuda.synchronize() if device.type == "cuda" else None
        wall = (time.perf_counter() - timer.t0)
        return outs, [_req_metrics(timer.times[r.req_id]) for r in reqs], wall, None

    def full():
        pool = make_pool(meta)
        timer = WallTimer()
        timer.start()
        outs, stats = chunked_prefill_generate(
            meta["weights"], _fresh_requests(reqs), pool, max_batch_size=batch,
            max_prefill_tokens=max_prefill_tokens, hook_cls=triton_hook_for(meta),
            use_graph=False, forward_fn=meta["forward_fn"],
            on_token=lambda rid, _tok, _n: timer.mark(rid))
        torch.cuda.synchronize() if device.type == "cuda" else None
        wall = time.perf_counter() - timer.t0
        return outs, [_req_metrics(timer.times[r.req_id]) for r in reqs], wall, stats

    runners = [("seq-kv", seq_kv), ("static-batch", static_batch), ("full", full)]
    for name, fn in runners:
        try:
            fn()                                    # 预热/验证
        except Exception as exc:                    # pragma: no cover
            print(f"  [skip] {name}: {exc}")
            continue
        runs, last = [], None
        for _ in range(repeat):
            outs, per_req, wall, stats = fn()
            new_tokens = sum(int(o.shape[1] - r.prompt.shape[1])
                             for o, r in zip(outs, reqs))
            last = _multi_summary(per_req, wall, len(reqs), new_tokens, stats)
            runs.append(last)
        runs.sort(key=lambda r: r["wall_s"])
        results[name] = runs[len(runs) // 2]
    return results


def run_multi_graph(meta, reqs, repeat, batch, max_prefill_tokens):
    """多请求 full + 预捕获 CUDA Graph（图复用，捕获不计入计时）。"""
    if not HAS_TRITON:
        return {}
    from engine.graph_decode import PagedDecodeGraphs
    pool = make_pool(meta)
    longest = max(r.prompt.numel() + r.max_new_tokens for r in reqs)
    max_blocks = -(-longest // pool.block_size) + 1
    cap = sorted({s for s in (1, 2, 4, 8) if s <= batch} | {batch})
    graphs = PagedDecodeGraphs(meta["weights"], pool, batch_sizes=cap,
                               max_blocks=max_blocks,
                               forward_fn=meta["forward_fn"],
                               block_n=triton_block_n(meta))
    try:
        def once():
            timer = WallTimer()
            timer.start()
            outs, stats = chunked_prefill_generate(
                meta["weights"], _fresh_requests(reqs), pool,
                max_batch_size=batch, max_prefill_tokens=max_prefill_tokens,
                hook_cls=triton_hook_for(meta), use_graph=True,
                forward_fn=meta["forward_fn"],
                on_token=lambda rid, _tok, _n: timer.mark(rid),
                decode_graphs=graphs)
            torch.cuda.synchronize() if meta["device"].type == "cuda" else None
            wall = time.perf_counter() - timer.t0
            per_req = [_req_metrics(timer.times[r.req_id]) for r in reqs]
            new_tokens = sum(int(o.shape[1] - r.prompt.shape[1])
                             for o, r in zip(outs, reqs))
            return wall, per_req, new_tokens, stats

        once()                                    # 预热（图已捕获，走同一路径）
        runs = []
        for _ in range(repeat):
            wall, per_req, new_tokens, stats = once()
            runs.append(_multi_summary(per_req, wall, len(reqs), new_tokens, stats))
        runs.sort(key=lambda r: r["wall_s"])
        return {"full-graph": runs[len(runs) // 2]}
    finally:
        graphs.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt-repeat", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num-blocks", type=int, default=256)
    ap.add_argument("--dtype", choices=("float32", "bfloat16", "float16"),
                    default="float32")
    ap.add_argument("--max-prefill-tokens", type=int, default=64)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.single and not args.multi:
        args.single = args.multi = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    meta = load_meta(args.model, device, dtype=dtype)
    meta["num_blocks"] = args.num_blocks
    prompt = single_prompt(meta["tokenizer"], device, args.prompt_repeat, args.model)
    print(f"model={args.model} device={device} prompt_tokens={prompt.shape[1]}")

    out = {"model": args.model, "device": str(device),
           "dtype": str(meta["dtype"]), "prompt_tokens": prompt.shape[1]}
    if args.single:
        out["single"] = run_single(meta, prompt, args.max_new_tokens, args.repeat)
    if args.multi:
        reqs = build_multi_requests(meta["tokenizer"], device)
        out["multi"] = run_multi(meta, reqs, args.repeat, args.batch,
                                 args.max_prefill_tokens)
        out["multi"].update(run_multi_graph(meta, reqs, args.repeat, args.batch,
                                            args.max_prefill_tokens))

    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"\n结果已写入 {p}")


if __name__ == "__main__":
    main()
