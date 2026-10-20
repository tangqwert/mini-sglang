# benchmark/results — 测量留档

本目录存放 `benchmark/verify_m*.py` 的可核对输出。用途是**数字可指认**，不是排名。

## 文件

| 文件 | 来源 | 说明 |
|---|---|---|
| `m0_naive*.json`、`m1_kv*.json` | `benchmark/bench.py` | 原始逐项计时（历史测量，含 prompt 长度） |
| `m0_naive_long_tpot.json`、`m1_kv_long_tpot.json` | `benchmark/bench.py` | gpt2 696-token 同 prompt 的 TTFT / TPOT（`--engine naive` vs `kv`） |
| `m0_naive_qwen3_long_tpot.json`、`m1_kv_qwen3_long_tpot.json` | `benchmark/bench.py` | Qwen3-0.6B 706-token 同 prompt 的 TTFT / TPOT（`--engine naive` vs `kv`） |
| `m2_batching.json` | `benchmark/verify_m2.py` | 批量 vs 逐条（4 请求 × 24 token） |
| `m3_radix.json` | `benchmark/verify_m3.py` | 118-token 共享前缀，cached vs 逐条 |
| `m4b_paged_schedule.json` | `benchmark/verify_m4b.py` | 1 长 + 8 短 / 2 槽位：M2.5 vs Step4 |
| `m5_triton.json` | `benchmark/verify_m5.py` | PyTorch 分页 vs Triton + 注意力微基准 |
| `m6_chunked_prefill.json` | `benchmark/verify_m6.py` | M2.5 / Step4 / M6 三方对照 + 分块上限扫描 |
| `m7_cuda_graph.json` | `benchmark/verify_m7.py` | eager 逐行 / eager 向量化 / CUDA Graph 三路对照 |
| `matrix_gpt2.json` | `benchmark/bench_matrix.py` | 单请求 5 模式 + 多请求 seq / static / full / full-graph（统一 workload） |
| `matrix_qwen3_bf16.json` | `benchmark/bench_matrix.py` | Qwen3-0.6B bf16 + bf16 pool，单/多请求统一矩阵（256 blocks） |
| `m10_serving.json` | `benchmark/bench_serving.py` | 并发打 M10 HTTP 服务（32 请求 × 16 token，并发 1/2/4） |

## 测量口径

- 全部在 **RTX 4070 Laptop (8GB)** 上测得，gpt2 fp32。
- `m0_naive*.json` / `m1_kv*.json` 的 `tokens_per_s` 与平均 ms/token 是**整段生成**
  （prefill + decode）摊销，**含 prefill，不是 decode TPOT**。
- `*_tpot.json` 是同一口径的拆分：`ttft_ms` 是首 token 时间；`tpot_ms` 是
  `(最后一个 token 时间 - 首 token 时间) / (new_tokens - 1)`，**不含 prefill**。
  696-token gpt2 实测（3 次中位数）：naive 27.5 ms → KV **5.9 ms（4.6x）**，TTFT 23.7 → 19.4 ms。
  Qwen3-0.6B 706-token 实测（3 次中位数）：naive 59.0 ms → KV **25.4 ms（2.3x）**，TTFT 44.3 → 39.1 ms。
- `m7_cuda_graph.json` 的 `C_over_A`（如 3.83x）是“KV 写入向量化 + CUDA Graph”合计；
  纯 Graph 收益看 `C_over_B` 的倒数（约 2.1–2.4x）。README 的 5.3x 是较早的代表性快照，
  不是纯 Graph 收益。
- `matrix_gpt2.json` 是**系统级**矩阵，不是单变量消融：
  - 单请求：naive / KV / paged-PyTorch / paged-Triton / paged-Triton-Graph；
    后三者是同一分页调度器换 hook / 执行方式，Graph 在计时前预捕获。
  - 多请求：1 长 + 8 短（预算 64/16），B=4，**所有请求同时到达**；
    `static-batch` 是强基线（HF 融合批式注意力），`full` 是 paged+Chunked+Triton，
    `full-graph` 复用预捕获图。
  - 关键不要混：`full`（无图）多请求 748 ms，`full-graph` 309 ms；两者的差异是
    **图捕获是否在计时内**，不是“图让 kernel 变快”这么简单。
- `matrix_qwen3_bf16.json` 是 Qwen3-0.6B、bf16 权重 + bf16 pool、256 blocks、706-token prompt；
  单请求 full-graph TPOT 95.0 → 17.5 ms（5.4x），多请求 6444 → 1126 ms（5.7x）。
  Triton kernel 内部把 q/k/v 转 fp32 计算，因此不是“bf16 精度逐元素对齐”的证明。
- 计时前**预热**，重复 3 次取最短（笔记本 GPU 降频会导致约 ±30% 的波动）。
- M2–M7 是独立复跑快照；README 的表格使用 `matrix_*.json` 同次测量值。
  不同批次会受笔记本 GPU 降频影响（可达 ±30%），优先看**同一次运行内**的相对关系，
  不要跨次比较绝对毫秒。
- 正确性字段（`correctness` / `all_matches_naive`）来自脚本内的逐 token 对比。
- M10 是非流式、单进程锁串行；`m10_serving.json` 的 latency 是端到端，不是 TTFT。
  并发 1/2/4 的吞吐稳定在 ~8.2 req/s，p50 线性增长，这就是“排队”而不是 continuous batching。
