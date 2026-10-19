"""engine.llm — M10：统一入口类（模型 + tokenizer + 分页池 + 调度器）。

═══════════════════════════════════════════════════════════════════
 在 M10 之前，所有能力都是"函数式"的：调用方自己加载模型、建池、拼
 `Request`、调 `chunked_prefill_generate`。M10 把它们收进一个对象：

     LLMEngine
       ├── tokenizer / model_config / weights
       ├── PagedKVPool（+ 可选 PagedRadixCache）
       ├── forward_fn（gpt2_forward / qwen3_forward）
       ├── hook_cls（PyTorch 分页 / Triton）
       └── chat() / generate()

 这样 HTTP 服务（engine/server.py）只需要 `engine.chat(...)`，不关心
 下面用的是 GPT-2 还是 Qwen3、PyTorch 还是 Triton。

 并发说明（诚实声明）：
   `generate()` 用一把进程内锁串行化请求。模型/池/调度器都是单实例、
   非线程安全；M10 的目标是"能演示 + 接口正确"，不是高并发服务。
   真引擎用请求队列 + 连续批处理；那需要把 `chunked_prefill_generate`
   改成"在线到达"模型，是另一项工程。
═══════════════════════════════════════════════════════════════════
"""
import threading
import warnings
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.batching import Request
from engine.model_forward import gpt2_forward, load_gpt2_weights
from engine.paged_kv import BatchedPagedAttentionHook, PagedKVPool
from engine.paged_schedule import chunked_prefill_generate
from engine.qwen3_forward import load_qwen3_weights, qwen3_forward
from engine.radix_cache_paged import PagedRadixCache
from engine.sampling import SamplingParams


@dataclass
class LLMConfig:
    """服务启动配置。"""
    model_name: str = "gpt2"
    device: str = "auto"              # auto | cpu | cuda
    num_blocks: int = 512
    block_size: int = 16
    max_prefill_tokens: int = 512
    attention_backend: str = "torch"  # torch | triton
    use_graph: bool = False
    use_radix: bool = False


@dataclass
class ChatResult:
    """一次 chat 生成的结果（与 OpenAI usage 字段对齐）。"""
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str                 # "stop" | "length"
    generated_token_ids: list[int]


def format_chat(tokenizer, messages: list[dict]) -> str:
    """把 chat messages 变成一段 prompt。

    优先用 tokenizer 自带的 chat template（Qwen 系有）；GPT-2 没有模板，
    退化成 "User: ...\\nAssistant:" 的纯文本格式。两种都能让 M10 跑起来，
    但后者只是演示，不是生产级对话格式。
    """
    normalized = [{"role": str(m.get("role", "user")),
                   "content": str(m.get("content", ""))} for m in messages]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                normalized, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in normalized]
    lines.append("Assistant:")
    return "\n".join(lines)


def truncate_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    """在最早的 stop 字符串处截断；返回 (截断后文本, 是否命中)。"""
    cut = len(text)
    for s in stops:
        if not s:
            continue
        pos = text.find(s)
        if pos != -1:
            cut = min(cut, pos)
    return text[:cut], cut < len(text)


class LLMEngine:
    """一个可复用的推理引擎实例（线程内串行）。"""

    def __init__(self, config: LLMConfig):
        self.config = config
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # 显式 fp32：PagedKVPool 是 fp32；HF 5.x 会按 checkpoint 把 Qwen3 载成 bf16，
        # 直接混用会在 scatter_kv 处报 dtype mismatch（真实踩坑）。
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, dtype=torch.float32)
        model = model.to(self.device).eval()
        self.model_config = model.config

        model_type = getattr(model.config, "model_type", "")
        if model_type == "gpt2":
            self.weights = load_gpt2_weights(model)
            self.forward_fn = gpt2_forward
            n_layers = model.config.n_layer
            n_kv_heads = model.config.n_head
            head_dim = model.config.n_embd // model.config.n_head
        elif model_type == "qwen3":
            self.weights = load_qwen3_weights(model)
            self.forward_fn = qwen3_forward
            n_layers = model.config.num_hidden_layers
            n_kv_heads = getattr(model.config, "num_key_value_heads",
                                 model.config.num_attention_heads)
            head_dim = getattr(model.config, "head_dim",
                               model.config.hidden_size // model.config.num_attention_heads)
        else:
            raise ValueError(
                f"暂不支持 model_type={model_type!r}；支持 gpt2 / qwen3")

        eos = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(eos, (list, tuple)):
            eos = eos[0] if eos else None
        self.eos_token_id = eos or self.tokenizer.eos_token_id

        # 释放 HF 模型对象；后面只用自己的裸权重
        del model
        if self.device == "cuda":
            torch.cuda.empty_cache()

        self.pool = PagedKVPool(
            num_blocks=config.num_blocks, block_size=config.block_size,
            num_layers=n_layers, num_heads=n_kv_heads, head_dim=head_dim,
            device=self.device)
        self.radix = PagedRadixCache(self.pool) if config.use_radix else None
        self.hook_cls = self._resolve_hook()
        self.use_graph, self.graph_reason = self._resolve_graph()
        self._decode_graphs = None
        self._lock = threading.Lock()

    def _resolve_hook(self):
        backend = (self.config.attention_backend or "torch").lower()
        if backend == "triton":
            try:
                from engine.triton_paged import (HAS_TRITON,
                                                 TritonPagedAttentionHook)
                if HAS_TRITON and torch.cuda.is_available():
                    return TritonPagedAttentionHook
            except Exception:
                pass
            warnings.warn("Triton 不可用，回退到 PyTorch 分页注意力")
        return BatchedPagedAttentionHook

    def _resolve_graph(self) -> tuple[bool, str]:
        if not self.config.use_graph:
            return False, "未启用"
        if (self.config.attention_backend or "").lower() != "triton":
            return False, "CUDA Graph 需要 attention_backend=triton"
        try:
            from engine.triton_paged import HAS_TRITON
        except Exception:
            return False, "Triton 不可用"
        if not (HAS_TRITON and torch.cuda.is_available()):
            return False, "Triton/CUDA 不可用"
        # 图路径通过 PagedDecodeGraphs(forward_fn=...) 支持 gpt2 / qwen3，
        # 不再硬编码 GPT-2；这里无需按 model_type 额外禁用。
        return True, "启用"

    def _get_decode_graphs(self):
        """惰性预捕获单请求 decode 图；后续请求复用，避免重复捕获。"""
        if self._decode_graphs is None:
            from engine.graph_decode import PagedDecodeGraphs
            self._decode_graphs = PagedDecodeGraphs(
                self.weights, self.pool, batch_sizes=(1,),
                max_blocks=self.pool.num_blocks, forward_fn=self.forward_fn)
        return self._decode_graphs

    def close(self) -> None:
        """归还 Graph 的 scratch 块（可重复调用）。"""
        graphs, self._decode_graphs = self._decode_graphs, None
        if graphs is not None:
            graphs.close()

    def __del__(self):                                # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    # ─────────────────────────── 生成 ───────────────────────────

    def generate(self, prompt_ids: list[int], sampling: SamplingParams,
                 max_new_tokens: int) -> list[int]:
        """对一条 prompt 生成 token ids（不含 prompt）。"""
        if max_new_tokens <= 0:
            return []
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        req = Request(req_id=0, prompt=ids, max_new_tokens=max_new_tokens,
                      eos_token_id=self.eos_token_id, sampling=sampling)
        with self._lock:
            graphs = (self._get_decode_graphs()
                      if self.use_graph and sampling.is_greedy else None)
            outs, _ = chunked_prefill_generate(
                self.weights, [req], self.pool,
                max_batch_size=1,
                max_prefill_tokens=self.config.max_prefill_tokens,
                hook_cls=self.hook_cls,
                use_graph=self.use_graph,
                forward_fn=self.forward_fn,
                radix=self.radix,
                decode_graphs=graphs)
        return outs[0][0, ids.shape[1]:].tolist()

    def chat(self, messages: list[dict], max_new_tokens: int = 64,
             sampling: SamplingParams | None = None,
             stop: list[str] | None = None) -> ChatResult:
        """对话补全：格式化 messages → 生成 → 解码 + stop 截断。"""
        prompt = format_chat(self.tokenizer, messages)
        prompt_ids = self.tokenizer(
            prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            raise ValueError("prompt 为空：messages 没有产生任何 token")

        sampling = sampling or SamplingParams()
        stops = [s for s in (stop or []) if s]
        stop_ids = self._stop_token_ids(stops)
        if stop_ids:
            sampling = SamplingParams(
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                top_p=sampling.top_p,
                seed=sampling.seed,
                stop_token_ids=tuple(sampling.stop_token_ids) + stop_ids,
                ignore_eos=sampling.ignore_eos)

        generated = self.generate(prompt_ids, sampling, max_new_tokens)
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        text, hit_stop = truncate_at_stop(raw, stops)

        hit_eos = (not sampling.ignore_eos
                   and bool(generated) and generated[-1] == self.eos_token_id)
        hit_token_stop = any(t in sampling.stop_token_ids for t in generated)
        finish = "stop" if (hit_stop or hit_eos or hit_token_stop) else "length"

        return ChatResult(
            text=text.strip(),
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(generated),
            finish_reason=finish,
            generated_token_ids=generated)

    def _stop_token_ids(self, stops: list[str]) -> tuple[int, ...]:
        """把单 token 的 stop 字符串转成 stop token id（多 token 只做后处理）。"""
        ids = []
        for s in stops:
            enc = self.tokenizer.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        return tuple(ids)
