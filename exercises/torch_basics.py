"""torch 基础系统复习 — 断言式自测练习。

用法：
    1. 每个 exercise 里把 `____` 或 TODO 处补上（答案不唯一，断言只验性质）
    2. 运行：python exercises/torch_basics.py
    3. 全绿 = 这一块过关；报错的 assert 自带提示

设计说明：
    - 不教你"背 API"，教你**预测形状和值**——读推理引擎代码的核心能力
    - 每节对应 mini-sglang 里真实出现过的一类代码
    - 建议节奏：一节 10-15 分钟，写之前先在纸上预测答案
"""
import torch

PASS = []


def check(name, cond, hint=""):
    if cond:
        PASS.append(name)
    else:
        raise AssertionError(f"✗ {name}  {hint}")


# ═══════════════════════ 第 1 节：形状与 dtype ═══════════════════════
# 对应项目代码：pad_left 的 torch.full、input_ids 的 dtype

def sec1_shape_and_dtype():
    """张量的两大属性：shape（形状）和 dtype（元素类型）。"""
    # 1.1 造一个 [2, 3] 的全 0 整数张量（token id 的标准姿势）
    t = torch.zeros(2,3)  # 提示: torch.zeros / torch.full 都行；注意 dtype
    check("1.1 shape", t.shape == (2, 3), "目标形状 [2,3]")
    check("1.1 dtype", t.dtype == torch.long, "token id 要用整数类型 torch.long")

    # 1.2 不运行代码，先在纸上预测，再写出来验证：
    #     torch.tensor([[1, 2], [3, 4], [5, 6]]) 的 shape 是多少？
    predicted = (3,1)  # 填一个 tuple，如 (1, 2)
    real = torch.tensor([[1, 2], [3, 4], [5, 6]]).shape
    check("1.2 预测形状", predicted == real, f"真实是 {real}——纸上预测对了吗？")

    # 1.3 dtype 决定"能存什么"：float 张量里能存 3.14，long 里 3.14 会怎样？
    #     预测下面这行的行为（会报错？截断？保留小数？），然后取消注释验证：
    t = torch.tensor([1, 2, 3], dtype=torch.float32)
    print(t)  # 你预测打印出什么？
    check("1.3 思考题", True, "口头回答：long 转 float32 会不会丢精度？反过来呢？")

    # 1.4 项目链接：M0 打印过 gpt2 权重 wte.weight shape=(50257, 768)。
    #     这个张量总共多少个元素？（用 numel 心算 + 代码验证）
    n = 50257 * 768
    check("1.4 numel", n == 50257 * 768 and torch.empty(50257, 768).numel() == n)


# ═══════════════════════ 第 2 节：索引与切片 ═══════════════════════
# 对应项目代码：logits[:, -1, :]、last_tokens[i, 0]、k2[:, :, :5]

def sec2_indexing():
    t = torch.tensor([[10, 11, 12, 13],
                      [20, 21, 22, 23],
                      [30, 31, 32, 33]])   # [3, 4]

    # 2.1 取第 1 行（整体）——预测 shape 和值再写
    row = t[1,:,:]
    check("2.1 整行", row.tolist() == [20, 21, 22, 23] and row.shape == (4,))

    # 2.2 取单个元素：第 2 行第 3 列（值 = 32）——两种写法
    a = t[[2,3]]     # 二维一次索引
    b = t[2][2]     # 链式索引（list 风格）
    check("2.2 单元素", a.item() == 32 and b.item() == 32,
          "t[行, 列] 与 t[行][列] 等价，但前者是标准写法")

    # 2.3 切片：所有行的第 1~2 列（含头不含尾）
    cols = t[:, ____]
    check("2.3 列切片", cols.tolist() == [[11, 12], [21, 22], [31, 32]],
          "冒号=这一维全要；1:3=下标1和2")

    # 2.4 负索引：每行的最后一列
    last = t[____]
    check("2.4 负索引", last.tolist() == [13, 23, 33],
          "这就是 logits[:, -1, :] 里那个 -1 的含义")

    # 2.5 组合拳（M0/M1 的核心动作）：t 当作 [N=3行, T=4位置, 假想V=1]
    #     "取每行最后一个位置" → 对二维来说就是"取每行最后一列"
    #     要求结果 shape == (3, 1)，用切片 + 保持维度的方式写
    kept = t[:, ____]     # 提示: 用 start: 冒号切片（不是单个下标！）
    check("2.5 keepdim 效果", kept.shape == (3, 1) and kept[:, 0].tolist() == [13, 23, 33],
          "t[:, -1] 会得到 (3,) 降维；t[:, -1:] 才是 (3,1)——这就是 keepdim 的手工版")


# ═══════════════════════ 第 3 节：形状变换 ═══════════════════════
# 对应项目代码：r.prompt.reshape(1, -1)、frozen.unsqueeze(1)

def sec3_shape_ops():
    t = torch.tensor([1, 2, 3, 4, 5, 6])   # 一维 [6]

    # 3.1 变成 [2, 3]——元素总数必须守恒
    m = t.reshape(____)
    check("3.1 reshape", m.shape == (2, 3) and m[1].tolist() == [4, 5, 6],
          "reshape 按行优先顺序填：1,2,3 / 4,5,6")

    # 3.2 -1 占位符：变成 3 行，列数自动算
    m2 = t.reshape(3, ____)
    check("3.2 -1 占位", m2.shape == (3, 2), "6/3=2，自动算出来")

    # 3.3 [T] → [1, T]（加 batch 维，prompt 进 batch 的标准动作）
    b = torch.tensor([3, 4]).reshape(____)
    check("3.3 加 batch 维", b.shape == (1, 2) and b.tolist() == [[3, 4]])

    # 3.4 unsqueeze：在指定位置插一个长度 1 的维度
    v = torch.tensor([False, True])
    vv = v.unsqueeze(____)        # 变成 [2, 1]
    check("3.4 unsqueeze", vv.shape == (2, 1) and vv.dtype == torch.bool,
          "unsqueeze(1)=在最里层插维；这就是 M2 frozen 对齐 [N,1] 的动作")

    # 3.5 squeeze：unsqueeze 的反操作，删掉所有长度 1 的维度
    check("3.5 squeeze", vv.squeeze().shape == (2,))

    # 3.6 思考：reshape 会复制数据吗？（提示：打印 t 和 m 共享内存的验证）
    m[0, 0] = 999
    # print(t)  # t[0] 变了吗？
    check("3.6 视图语义", t[0].item() == 999,
          "reshape 返回视图（多数情况）——改一个另一个跟着变，这就是'只换看法不搬数据'")


# ═══════════════════════ 第 4 节：广播（broadcasting） ═══════════════════════
# 规则：两个形状从【右往左】逐维比较，每对维度要么相等、要么有一个是 1、要么缺失
# 对应项目代码：(full * attention_mask)、position_ids 的运算

def sec4_broadcasting():
    # 4.1 [3, 1] + [1, 4] → ？先在纸上画：行方向复制 4 份、列方向复制 3 份
    a = torch.ones(3, 1)
    b = torch.ones(1, 4)
    check("4.1 广播相加", (a + b).shape == (3, 4) and (a + b).sum().item() == 12,
          "3 行都变成 [1,1,1,1]，4 列都变成 [1,1,1]^T")

    # 4.2 [N, T] * [N, T]（同形）——逐元素，无广播
    x = torch.tensor([[1., 2.], [3., 4.]])
    check("4.2 逐元素乘", (x * x).tolist() == [[1., 4.], [9., 16.]],
          "注意：这是逐元素乘，不是矩阵乘！矩阵乘用 x @ x")

    # 4.3 M2 的真实场景：假模型的 (full * attention_mask).sum(dim=1)
    #     full 是 [2, 4] 的 token id，mask 是 [2, 4] 的 0/1——同形相乘=按位置过滤
    full = torch.tensor([[5, 0, 0, 6], [3, 4, 0, 8]])
    mask = torch.tensor([[1, 0, 0, 1], [1, 1, 0, 1]])
    masked_sum = ____
    check("4.3 mask 过滤求和", masked_sum.tolist() == [11, 15],
          "pad(0) 本身加和为零，但 mask 保证任何 pad id 都不计入")

    # 4.4 广播陷阱：[3] 和 [3] 相加是逐元素；[3,1] 和 [3] 呢？先预测！
    c = torch.ones(3, 1) + torch.ones(3)
    check("4.4 广播陷阱", c.shape == (3, 3),
          "[3,1] 和 [3]：右对齐比较 → (1 vs 3 广播, 缺失 vs 3 广播) → [3,3]"
          "——这就是为什么 M2 里到处 .unsqueeze(1)/keepdim=True：不是迷信，是防错位")

    # 4.5 广播报错：[2, 3] 和 [3, 2] 能相加吗？
    try:
        _ = torch.ones(2, 3) + torch.ones(3, 2)
        ok = False
    except RuntimeError:
        ok = True
    check("4.5 不兼容报错", ok, "从右往左：3 vs 2 不等且都非 1 → RuntimeError")


# ═══════════════════════ 第 5 节：归约（reduction）与 dim 语义 ═══════════════════════
# 核心心法：dim 参数 = "被消掉的维度"（sum/max）；cat 的 dim = "生长的方向"
# 对应项目代码：argmax(dim=-1, keepdim=True)、mask.sum(dim=1, keepdim=True)、cumsum

def sec5_reduction():
    t = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.]])       # [2, 3]

    # 5.1 sum(dim=0)：消掉行 → 每列一个和
    check("5.1 sum dim=0", t.sum(dim=0).tolist() == [5., 7., 9.])

    # 5.2 sum(dim=1)：消掉列 → 每行一个和（先预测！）
    check("5.2 sum dim=1", t.sum(dim=1).tolist() == [6., 15.])

    # 5.3 keepdim：消掉的维度留一个长度 1 的壳
    s = t.sum(dim=1, keepdim=True)
    check("5.3 keepdim", s.shape == (2, 1) and s[:, 0].tolist() == [6., 15.])

    # 5.4 argmax：最大值的【下标】；dim=-1 沿最后一维
    check("5.4 argmax", t.argmax(dim=-1).tolist() == [2, 2], "每行最大值都在下标 2")

    # 5.5 argmax + keepdim 组合（M1/M2 每步生成 token 的动作）
    a = t.argmax(dim=-1, keepdim=True)
    check("5.5 argmax keepdim", a.shape == (2, 1), "保持 [N,1] 才能当 batch 喂模型")

    # 5.6 cumsum：累加（build_position_ids 的核心）
    mask = torch.tensor([[0, 0, 1], [1, 1, 1]])
    c = mask.cumsum(dim=-1)
    check("5.6 cumsum", c.tolist() == [[0, 0, 1], [1, 2, 3]],
          "第 i 个位置 = 前 i+1 个数之和；再 -1.clamp(0) 就是真实 token 的位置编号")

    # 5.7 dim 语义大统一：预测下面两个的结果形状（先纸上写，再验证）
    s0 = t.sum(dim=0)
    s1 = t.sum(dim=1)
    check("5.7 形状预测", s0.shape == (3,) and s1.shape == (2,),
          "消谁谁的长度就没了")


# ═══════════════════════ 第 6 节：组合操作 ═══════════════════════
# 对应项目代码：torch.cat、torch.where、.item()、torch.tensor([...])

def sec6_composition():
    # 6.1 cat 沿 dim=1 横向生长（M2 的 mask 扩列）
    mask = torch.tensor([[0, 1], [1, 1]])
    mask2 = torch.cat([mask, torch.ones(2, 1, dtype=torch.long)], dim=1)
    check("6.1 cat 扩列", mask2.shape == (2, 3) and mask2[:, 2].tolist() == [1, 1])

    # 6.2 cat 沿 dim=0 竖向堆叠——预测形状！
    st = torch.cat([torch.ones(2, 3), torch.zeros(1, 3)], dim=0)
    check("6.2 cat 堆叠", st.shape == (3, 3))

    # 6.3 where 三元选择（M2 的 feed 构造）
    cond = torch.tensor([[False], [True]])
    a = torch.tensor([[6.], [8.]])
    b = torch.tensor([[0.], [0.]])
    feed = torch.where(cond, b, a)     # 注意参数顺序：真取第2个，假取第3个
    check("6.3 where", feed.tolist() == [[6.], [0.]],
          "cond 真取 b(pad)，假取 a(last_token)——和 M2 的 feed 一模一样")

    # 6.4 item()：单元素张量 → Python 数（张量世界与 Python 世界的渡船）
    v = torch.tensor([[7]])
    check("6.4 item", v[0, 0].item() == 7 and isinstance(v[0, 0].item(), int))

    # 6.5 list[int] → 张量：外面那层 [] 决定维度（M2 收尾的动作）
    one_d = torch.tensor([6, 2], dtype=torch.long)
    two_d = torch.tensor([[6, 2]], dtype=torch.long)
    check("6.5 一维 vs 二维", one_d.shape == (2,) and two_d.shape == (1, 2),
          "torch.cat 要求两边维度数相同：prompt 是 [1,T]，gen 也必须是 [1,n]")

    # 6.6 终极组合：模拟 M2 收尾——prompt [1,2] 和 gen [1,2] 拼成完整序列
    prompt = torch.tensor([[3, 4]])
    gen = torch.tensor([[8, 6, 2]], dtype=torch.long)
    full = torch.cat([prompt, gen], dim=1)
    check("6.6 完整序列", full.tolist() == [[3, 4, 8, 6, 2]])


# ═══════════════════════ 第 7 节：no_grad 与设备（概念自测） ═══════════════════════

def sec7_grad_and_device():
    # 7.1 no_grad 里算的东西没有 grad_fn——推理引擎不建计算图（省内存+提速）
    x = torch.tensor([1.0], requires_grad=True)
    with torch.no_grad():
        y = x * 2
    check("7.1 no_grad 不建图", y.requires_grad is False and y.grad_fn is None)

    # 7.2 开着会怎样：z 有 grad_fn（autograd 在攒计算图）
    z = x * 2
    check("7.2 默认建图", z.grad_fn is not None,
        "所以推理代码必须包 no_grad——不然每步攒图直到 OOM")

    # 7.3 设备：CPU 张量和 GPU 张量是两个世界的居民（M0 炸过的 RuntimeError）
    check("7.3 概念", True,
        "口头回答：为什么 .to(device) 的返回值必须接住？不接住会怎样？")


if __name__ == "__main__":
    sections = [sec1_shape_and_dtype, sec2_indexing, sec3_shape_ops,
                sec4_broadcasting, sec5_reduction, sec6_composition,
                sec7_grad_and_device]
    total = 0
    for sec in sections:
        before = len(PASS)
        try:
            sec()
        except AssertionError as e:
            print(f"\n❌ 停在 {sec.__name__}: {e}")
            print(f"   已通过 {before} 项。修好这里再重跑。\n")
            raise SystemExit(1)
        except NameError:
            print(f"\n✏️ {sec.__name__} 里还有 ____ 没填（或变量名打错了）。"
                  f"已通过 {before} 项。\n")
            raise SystemExit(1)
        total = len(PASS)
        print(f"✅ {sec.__name__}  (累计 {total} 项)")
    print(f"\n🎉 全部 {total} 项通过——torch 基础过关，回项目继续 M2！")
