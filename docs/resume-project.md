# 简历项目模板 — mini-sglang

> 本文件是简历"项目经历"栏的素材库与纪律清单。
> 原则：**每个数字都要能在 `benchmark/results/` 指认出处；每条 bullet 都要能扛住 10 分钟深挖。**

## 简历条目（投递版）

> **mini-sglang —— 从零实现的轻量级 LLM 推理引擎**（个人项目｜Python/PyTorch）
> 参考 SGLang 架构，以 TDD 方式从零实现推理系统核心路径：解码循环 → KV Cache → 批处理 → 前缀缓存。

- **KV Cache 增量解码**：基于注意力位置不变性实现 prefill + 增量前向，输出与朴素解码逐 token 一致；601-token 长 prompt 下每 token 耗时 29.2ms → 6.5ms（**4.5x**）
- **跨模型瓶颈分析**：对照实验发现 KV Cache 收益受权重搬运地板效应压缩（Qwen2.5-0.5B 仅 **1.95x**），建立"每步耗时 = 权重搬运 + 序列计算"成本模型并实测验证
- **Continuous Batching**：实现左填充 + attention mask + 位置编码对齐的多请求批解码与 per-request 动态退出，批量与逐条输出逐 token 一致
- **Radix 前缀缓存**：实现 Radix 树管理跨请求 KV 复用，共享前缀请求的 prefill 计算量从 O(T) 降为 O(后缀)；TDD 全程，31 项单元/集成测试

## Bullet ↔ 面试深挖对照表（每条都要能扛 10 分钟）

| Bullet | 高概率追问 | 我的答案要点 |
|---|---|---|
| KV Cache | 为什么 decode 是 memory-bound？ | 每步搬全部权重（gpt2 fp32 0.5GB）；M0 数据：序列越长每步越慢（5.6→28.7ms） |
| 4.5x 怎么测的 | 实验方法？ | 同一条 601-token prompt、预热 1 次、`torch.cuda.synchronize()` 包夹、naive/kv 同口径 |
| 跨模型分析 | 为什么 Qwen 收益小？ | 成本公式：每步=权重搬运(固定)+序列计算(∝长度)；Qwen 权重 4x → 地板 ~15ms，实测 14.8ms 吻合 |
| Batching | 为什么需要 attention mask / position_ids？ | 左填充的 pad 会污染注意力、打乱位置编号——踩过真 RuntimeError |
| Radix | "命中缓存"的严谨定义？ | token 序列的精确公共前缀（非语义相似）；位置不变性 ⇒ KV 无损复用（误差 0.0 实验） |
| Radix | 真 SGLang 和 mini 版差距？ | mini 不做节点分裂（部分重叠直接放弃插入）、不做 refill（需 PagedAttention 搬 KV）——知道差距在哪 |

## 上简历前 Checklist

- [ ] `pytest tests/` 全绿（M3 cached_generate 收官）
- [ ] M3 真模型验证：共享前缀请求的 prefill 确实只算后缀
- [ ] commit + push
- [ ] README 升级：定位 / 架构图 / 结果表 / 快速开始（数字与本文件一致）
- [ ] （可选）一键 demo：30 秒看到两条引擎对比

## 数据档案（防面试官复查，随里程碑更新）

| 实验 | 数据 | 出处 |
|---|---|---|
| M1 KV cache（gpt2, 601 tok） | 29.2 → 6.5 ms/token，4.5x | `m0_naive_long.json` / `m1_kv_long.json` |
| M1 KV cache（Qwen, 601 tok） | 28.9 → 14.8 ms/token，1.95x | `m0_naive_qwen_long_en.json` / `m1_kv_qwen_long.json` |
| M1 短 prompt（5 tok） | 5.8 → 6.1 ms/token，≈持平 | `m0_naive_short.json` / `m1_kv_short.json` |
| M2 batching（4 请求） | 批 516ms vs 逐条 606ms，1.18x | `benchmark/verify_m2.py` |
| M3 radix 正确性 | 3 请求共享 118-token 前缀，cached vs 逐条输出逐 token 一致 | `benchmark/verify_m3.py` |
| M3 radix 耗时 | 0.82x——小模型 prefill ≈5ms 被权重搬运主导，省的计算 < clone/Python 开销；真收益场景=大模型×长前缀×高命中率（SGLang 用 PagedAttention 零拷贝引用解决 clone） | `benchmark/verify_m3.py` |
