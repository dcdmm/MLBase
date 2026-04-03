print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))
# __file__:当前*.py文件的路径
# __package__:When it is present, relative imports will be based on this attribute rather than the module __name__ attribute;当前文件__package__属性值为None

# 搜索路径sys.path默认包括(当前文件所在目录): ???\MLBase\A_PythonBasis\module_import

import base_rect_area

print(base_rect_area.__doc__)  # 模块文档

# 1. 若源文件发生改变,运行test_base.py即刻可以生效
# 2. 可以通过ctrl + 鼠标左击 进入导入库的源代码
area = base_rect_area.Rect_Area(1, 2)
print(area.area())

print('########################################################################')

from base_rect_volumes import *

# 1. (逐层)可以找到base_rect_volumes.py
# 2. base_rect_volumes.py中__package__="",不能使用相对导入


volum = Rect_volumes(area, 3)
print(volum.volumes())
volum.print_hello()

print('########################################################################')

from base.print_hello import print_hello

print_hello("python")

from base.print_hello1 import print_hello1

# 1. (逐层)可以找到base/print_hello1.py
# 2. base/print_hello1.py中__package__="base",可以使用相对导入.
# * * print_hello1.py文件路径: ???\MLBase\A_PythonBasis\module_import\base\print_hello1.py
# * * 模块搜索路径: ???\MLBase\A_PythonBasis\module_import
# * * print_hello1.py文件名: print_hello1.py
# * * __package__(其余部分): base
# * * 完整模块名(__package__ + 文件名): base.print_hello1
# * * * 原始路径: from .config import * 
# * * * .回退: base + . = base(不能为空)
# * * * 真实路径: from base.config import *
print_hello1("c++")

from base.inner0.print_hello2 import print_hello2

# 1. (逐层)可以找到base/inner0/print_hello2.py
# 2. base/inner0/print_hello2.py中__package__="base.inner0",可以使用相对导入..
# * * print_hello2.py文件路径: ???\MLBase\A_PythonBasis\module_import\base\inner0\print_hello2.py
# * * 模块搜索路径: ???\MLBase\A_PythonBasis\module_import
# * * print_hello2.py文件名: print_hello2.py
# * * __package__(其余部分): base.inner0
# * * 完整模块名(__package__ + 文件名): base.inner0.print_hello2
# * * * 原始路径: from ..config import * 
# * * * ..回退: base.inner0 + .. = base(不能为空)
# * * * 真实路径: from base.config import *
print_hello2("c++")
