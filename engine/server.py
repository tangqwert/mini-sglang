"""engine.server — M10：OpenAI 兼容的最小 HTTP 服务。

═══════════════════════════════════════════════════════════════════
 目标不是做一个生产级 API Server，而是让项目"可演示"：
   ① `POST /v1/chat/completions`：OpenAI 兼容的请求/响应形状
   ② `GET  /health`、`GET /v1/models`：健康检查与模型列表
   ③ 采样参数（temperature / top_p / top_k / seed / stop）透传

 明确不做（诚实声明）：
   · **非流式**：`stream=true` 直接返回 400。真流式需要调度器逐 token 回调，
     那是另一项改造；本项目在 M10 只做"完整生成后一次返回"。
   · **单进程串行**：LLMEngine 内部一把锁；同时到达的请求排队执行。
   · 没有 tokenizer/detokenizer worker、没有多进程 ZMQ。
═══════════════════════════════════════════════════════════════════

跑法::

    python -m engine.server --model gpt2 --port 8000
    curl -s http://127.0.0.1:8000/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
      -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
"""
import argparse
import time
import uuid

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "M10 需要 fastapi / uvicorn / pydantic：pip install fastapi uvicorn") from exc

from engine.llm import LLMConfig, LLMEngine
from engine.sampling import SamplingParams


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=64, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = None
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    stream: bool = False


def _stops(req: ChatCompletionRequest) -> list[str]:
    if req.stop is None:
        return []
    return [req.stop] if isinstance(req.stop, str) else list(req.stop)


def create_app(llm: LLMEngine) -> FastAPI:
    """把 LLMEngine 包成 FastAPI app（便于测试时注入 fake engine）。"""
    app = FastAPI(title="mini-sglang", version="0.1.0")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": llm.config.model_name,
            "device": llm.device,
            "attention_backend": getattr(llm.config, "attention_backend", "unknown"),
            "graph": llm.use_graph,
            "radix": llm.radix is not None,
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{
                "id": llm.config.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "mini-sglang",
            }],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if req.stream:
            raise HTTPException(
                status_code=400,
                detail="stream=true 暂不支持：M10 是非流式最小服务")

        sampling = SamplingParams(
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            seed=req.seed,
            ignore_eos=req.ignore_eos,
        )
        try:
            result = llm.chat(
                [m.model_dump() for m in req.messages],
                max_new_tokens=req.max_tokens,
                sampling=sampling,
                stop=_stops(req),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - 真模型运行时才会走到
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or llm.config.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    return app


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="mini-sglang OpenAI-compatible server")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-blocks", type=int, default=512)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-prefill-tokens", type=int, default=512)
    p.add_argument("--backend", default="torch", choices=["torch", "triton"])
    p.add_argument("--graph", action="store_true",
                   help="纯 decode 步使用 CUDA Graph（需要 --backend triton）")
    p.add_argument("--radix", action="store_true",
                   help="启用分页 Radix 前缀缓存")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    llm = LLMEngine(LLMConfig(
        model_name=args.model,
        device=args.device,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_prefill_tokens=args.max_prefill_tokens,
        attention_backend=args.backend,
        use_graph=args.graph,
        use_radix=args.radix,
    ))
    app = create_app(llm)

    import uvicorn
    print(f"mini-sglang server: http://{args.host}:{args.port}")
    print(f"  model={args.model} device={llm.device} "
          f"backend={args.backend} graph={llm.use_graph} ({llm.graph_reason}) "
          f"radix={args.radix}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
