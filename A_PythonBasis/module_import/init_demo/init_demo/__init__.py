# 导入路径简化前:from init_demo.math_ops import add, mul
from .math_ops import add, mul  # 导入路径简化后:from init_demo import add, mul

# 导入路径简化前:from init_demo.str_ops.ops import shout, whisper
from .str_ops.ops import shout, whisper  # 导入路径简化后:from init_demo import shout, whispe

# `from init_demo import *`导入时只导出add、mul和shout
__all__ = ["add", "mul", "shout"] 