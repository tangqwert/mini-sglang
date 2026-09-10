# mini-sglang：受 SGLang 启发的轻量级 LLM 推理引擎

> 一个用于学习 LLM 推理系统的教学项目：从最朴素的解码循环开始，
> 逐步实现 KV Cache、Continuous Batching、Radix 前缀复用，
> 最终用自研 CUDA/Triton kernel 替换热点算子。
> 硬件：RTX 4070 Laptop (8GB) ｜ 模型：Qwen2.5-0.5B / GPT-2 124M

## 里程碑

- [ ] **M0** Naive 解码循环 + baseline 吞吐测量 ← 当前阶段
- [ ] **M1** KV Cache（正确性对比 + 加速比）
- [ ] **M2** Continuous Batching（多请求交织调度）
- [ ] **M3** Radix Tree 前缀缓存复用
- [ ] **M4** 自研 CUDA/Triton kernel 替换热点算子

## 开发方式

TDD / 规格先行：`tests/` 定义行为契约，`engine/` 中的实现由学习者完成。
每完成一个里程碑跑一次 benchmark，数据记入 `benchmark/results/`。

## 快速开始

```bash
source .venv/bin/activate
pytest tests/ -x            # 跑测试（M0 实现完成前应全红）
python benchmark/bench.py --model Qwen/Qwen2.5-0.5B --max-new-tokens 128
```

## 环境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers pytest
# 若报 "NVIDIA driver too old"，改装匹配驱动的版本：
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
