# 测试命令:python -m pytest .\tests\test_main.py
# * dev dependencyt包含pytest

from main import main


def test_main_output(capsys) -> None:
    main()

    captured = capsys.readouterr()
    assert captured.out == "Hello from uv-example!\n"
