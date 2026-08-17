# 测试命令:uv run --package sub0 --group test pytest sub0/tests/test_computer.py
# * dev dependencyt不含pytest
# * default-groups不含test dependency group

from sub0.computer import sumx


def test_sumx() -> None:
    assert sumx(1, 2) == 3
