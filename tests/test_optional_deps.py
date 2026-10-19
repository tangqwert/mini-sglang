"""可选依赖的导入路径：没有 Triton 时也必须能 import + collect 测试。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_graph_decode_imports_without_triton():
    # 把 triton 置为 None，模拟"未安装 triton"：engine.triton_paged 会捕获
    # ImportError，graph_decode 也必须降级为 HAS_TRITON=False 而不是在导入期崩溃。
    code = (
        "import sys; sys.modules['triton'] = None; "
        "import engine.graph_decode as g; "
        "assert g.HAS_TRITON is False, g.HAS_TRITON"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
