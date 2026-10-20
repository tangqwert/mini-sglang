"""engine.sampling — M8：采样策略（temperature / top_k / top_p）+ 停止条件。

═══════════════════════════════════════════════════════════════════
M0–M7 的所有解码循环都直接 `argmax` —— 只有贪心，没有采样。
M8 把"下一个 token 怎么选"抽成独立的一层：

    logits ──► sample_token() ──► token id

这样 5 条解码路径（naive / kv_cache / batching / paged / radix）只需把
`torch.argmax(...)` 换成 `sample_token(...)`，不需要各写一套采样逻辑。

默认 `temperature=0` 仍然走 argmax —— 现有全部测试的输出不变。
只有显式传 `SamplingParams(temperature>0)` 才会启用随机采样。

采样顺序（与 vLLM / HF 一致）：
    temperature 缩放 → top_k 截断 → top_p 截断 → softmax → multinomial
先截断再归一化，否则截断后概率和不为 1。

停止条件也收在这里（`token_stops` / `is_finished`）：
    max_new_tokens / eos_token_id / stop_token_ids / ignore_eos
避免四套判定散落在每个循环里。
═══════════════════════════════════════════════════════════════════
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    """单条请求的采样参数（与 vLLM 的 SamplingParams 同名同角色）。

    Attributes:
        temperature: 0 = 贪心（argmax）；>0 = 按 logits/temperature 采样。
        top_k: 只保留概率最高的 k 个 token；0 = 不限制。
        top_p: nucleus sampling，保留累计概率 >= top_p 的最小 token 集合；
            1.0 = 不限制。
        seed: 随机种子。None 表示用全局 RNG；给定后同一请求可复现。
        stop_token_ids: 命中这些 token 就停止（EOS 之外的自定义停止符）。
        ignore_eos: True 时忽略 eos_token_id（基准测试常用：强制生成固定长度）。
    """

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(f"temperature 不能为负：{self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k 不能为负：{self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p 必须在 (0, 1] 内：{self.top_p}")
        # 允许调用方传 list，统一冻结成 tuple（dataclass 是 frozen 的，用 object.__setattr__）
        if not isinstance(self.stop_token_ids, tuple):
            object.__setattr__(self, "stop_token_ids", tuple(self.stop_token_ids))

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0

    def make_generator(self, device) -> "torch.Generator | None":
        """按 seed 造一个与 logits 同设备的 RNG；seed=None 时返回 None。

        为什么按设备造：CUDA 的 multinomial 要求 generator 与概率张量同设备。
        """
        if self.seed is None:
            return None
        dev = torch.device(device)
        # 直接用完整 device（如 cuda:1），避免多卡时落到当前默认卡
        generator = torch.Generator(device=dev)
        generator.manual_seed(int(self.seed))
        return generator


def sample_token(logits: torch.Tensor, params: SamplingParams,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """从 logits 采样一个 token。

    Args:
        logits: [B, V]（也接受 [V]）。
        params: 采样参数。
        generator: 可选 RNG；必须与 logits 同设备类型。

    Returns:
        [B, 1]（输入为 [V] 时返回 [1]），LongTensor。
    """
    squeeze = logits.dim() == 1
    x = logits.unsqueeze(0) if squeeze else logits
    if x.dim() != 2:
        raise ValueError(f"logits 必须是 [B, V] 或 [V]，实际 {tuple(logits.shape)}")

    if params.is_greedy:
        out = x.argmax(dim=-1, keepdim=True)
    else:
        x = x / params.temperature

        # ① top_k：只留最大的 k 个（kth 值以下全部 -inf）
        if params.top_k > 0:
            k = min(params.top_k, x.size(-1))
            kth = torch.topk(x, k, dim=-1).values[..., -1:]
            x = x.masked_fill(x < kth, float("-inf"))

        # ② top_p：按概率降序累积，丢掉"累计已达标"之后的尾巴。
        #    drop 的条件用 cum - probs（即"当前 token 之前"的累计），
        #    保证第一个让累计概率跨过 top_p 的 token 被保留。
        if params.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(x, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1, dtype=torch.float32)
            cum = probs.cumsum(dim=-1)
            drop = (cum - probs) >= params.top_p
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            x = torch.full_like(x, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

        probs = torch.softmax(x, dim=-1, dtype=torch.float32)
        if generator is not None and generator.device.type != probs.device.type:
            raise ValueError(
                f"generator 在 {generator.device}，probs 在 {probs.device}；"
                "请用 SamplingParams.make_generator(logits.device) 创建")
        out = torch.multinomial(probs, num_samples=1, generator=generator)

    return out.squeeze(0) if squeeze else out


def sample_tokens(logits: torch.Tensor, params_list: list[SamplingParams],
                  generators: list[torch.Generator | None] | None = None) -> torch.Tensor:
    """逐行采样（每条请求可以有不同的 temperature / top_k / top_p）。

    Args:
        logits: [B, V]，行与 params_list 一一对应。
        params_list: 长度 B 的采样参数。
        generators: 长度 B 的 RNG（None 表示用全局 RNG）。

    Returns:
        [B, 1] LongTensor。

    全贪心时走批量 argmax 快路径；只要有一行非贪心，就逐行调用
    `sample_token`（B 很小，教学版不做按参数分组的优化）。
    """
    if logits.dim() != 2:
        raise ValueError(f"logits 必须是 [B, V]，实际 {tuple(logits.shape)}")
    b = logits.size(0)
    if len(params_list) != b:
        raise ValueError(f"参数行数 {len(params_list)} != batch {b}")
    if generators is None:
        generators = [None] * b
    if len(generators) != b:
        raise ValueError(f"generator 行数 {len(generators)} != batch {b}")

    if all(p.is_greedy for p in params_list):
        return logits.argmax(dim=-1, keepdim=True)

    rows = [sample_token(logits[i], params_list[i], generators[i]).reshape(1)
            for i in range(b)]
    return torch.stack(rows, dim=0)


def token_stops(params: SamplingParams, eos_token_id: int | None, token: int) -> bool:
    """该 token 是否触发停止（不含 max_new_tokens 预算判定）。"""
    if not params.ignore_eos and eos_token_id is not None and token == eos_token_id:
        return True
    return token in params.stop_token_ids


def is_finished(params: SamplingParams, max_new_tokens: int,
                eos_token_id: int | None, token: int, n_generated: int) -> bool:
    """完整的完成判定：预算 + EOS + 自定义停止符。"""
    return n_generated >= max_new_tokens or token_stops(params, eos_token_id, token)
