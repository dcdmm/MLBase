print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))
# __file__:当前*.py文件的路径
# __package__:When it is present, relative imports will be based on this attribute rather than the module __name__ attribute;当前文件__package__属性为None

# dir_relative中的代码存在相对路径导入
# # # --->dir_relative.demo.py调用dir_relative.core.config.py
# # # --->dir_relative.demos_0.demo0.py调用dir_relative.core.config.py
# # # --->dir_relative.demos_0.demos_1.demo1.py调用dir_relative.core.config.py
# 解决1:从dir_relative处开始导入(即目录结构与dir_relative平级)
# 解决2:从比dir_relative目录结构更高的位置开始导入(即目录结构比dir_relative更高)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 解决1:
from dir_relative.demos_0.demo0 import hello_demo0
from dir_relative.demos_0.demos_1.demo1 import hello_demo1
from dir_relative.demo import hello_demo

hello_demo()
print('****************************************')
hello_demo0()
print('****************************************')
hello_demo1()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 解决2:
# # # 方式一:在比dir_relative目录结构更高的位置创建python文件再进行导入
# # # 方式二:通过sys.path.append添加临时模块搜索路径
import os
import sys

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath(".." + os.sep + ".."))

from module_import.dir_relative.demos_0.demo0 import hello_demo0  # 方式二(目录结构从`model_import`开始)
from other.module_import.dir_relative.demos_0.demo0 import hello_demo0  # 方式二(目录结构从`other`开始)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
