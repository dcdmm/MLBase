print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))
# __file__:当前*.py文件的路径
# __package__:When it is present, relative imports will be based on this attribute rather than the module __name__ attribute;当前文件__package__属性值为None

# ok:搜索路径sys.path包含: C:\Users\dcdmm\Music\GitHubProjects\MLNote
# import A_PythonBasis.module_import.base_rect_area

# ok:搜索路径sys.path包含: C:\Users\dcdmm\Music\GitHubProjects\MLNote\A_PythonBasis
# import module_import.base_rect_area

# ok:搜索路径sys.path包含: C:\Users\dcdmm\Music\GitHubProjects\MLNote\A_PythonBasis\module_import
import base_rect_area

print(base_rect_area.__doc__)  # 模块文档

# 1. 若源文件发生改变,运行test_base.py即刻可以生效
# 2. 可以通过ctrl + 鼠标左击 进入导入库的源代码
area = base_rect_area.Rect_Area(1, 2)
print(area.area())

print('########################################################################')

from base_rect_volumes import *

# 1. 搜索路径sys.path包含: C:\Users\dcdmm\Music\GitHubProjects\MLNote\A_PythonBasis\module_import
# 2. (逐层)可以找到base_rect_volumes.py
# 3. base_rect_volumes.py中__package__="",不能使用相对导入(. or .. or ...)


volum = Rect_volumes(area, 3)
print(volum.volumes())
volum.print_hello()

print('########################################################################')

from base.print_hello import print_hello

print_hello("python")

from base.print_hello1 import print_hello1

# 1. 搜索路径sys.path包含: C:\Users\dcdmm\Music\GitHubProjects\MLNote\A_PythonBasis
# 2. (逐层)可以找到base/print_hello1.py
# 3. base/print_hello1.py中__package__="base",可以使用相对导入(. or .. or ...)
print_hello1("c++")  # 包括相对路径(. or .. or ...)导入的代码;此时也能正常导入

from base.inner0.print_hello2 import print_hello2

# 1. 搜索路径sys.path包含: C:\Users\dcdmm\Music\GitHubProjects\MLNote\A_PythonBasis
# 2. (逐层)可以找到base/print_hello2.py
# 3. base/print_hello1.py中__package__="base.inner0",可以使用相对导入(. or .. or ...)
print_hello1("c++")
