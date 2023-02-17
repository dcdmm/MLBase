print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))
import os
import sys

print(":::", os.path.abspath(".."))
sys.path.append(os.path.abspath(".."))

# dir_relative中的代码存在相对路径导入
# # # --->dir_relative.demo.py调用dir_relative.core.config.py
# # # --->dir_relative.demos_0.demo0.py调用dir_relative.core.config.py
# # # --->dir_relative.demos_0.demos_1.demo1.py调用dir_relative.core.config.py
# 默认模块搜索路径为:A_PythonBasis.other.module_import.dir_relative
# 添加模块搜索路径`A_PythonBasis.other.module_import`,从而可以从dir_relative处开始导入(即目录结构与dir_relative平级)
from dir_relative.demos_0.demo0 import hello_demo0
from dir_relative.demos_0.demos_1.demo1 import hello_demo1
from dir_relative.demo import hello_demo

hello_demo()
print('****************************************')
hello_demo0()
print('****************************************')
hello_demo1()