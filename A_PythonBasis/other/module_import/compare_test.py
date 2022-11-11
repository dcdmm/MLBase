# from A_PythonBasis.other.module_import.compare_rect_area import Rect_Area  # 绝对路径导入

# 相对路径导入
# compare_reac_area.py与compare_test.py同目录结构下(平级)
from compare_rect_area import Rect_Area

# 1. 若源文件发生改变,运行compare_test.py文件立马可以得到改变
# 2. 可以通过ctrl + 鼠标左击 进入导入库的源代码
area = Rect_Area(1, 2)
print(area.area())

print('########################################################################')

# 相对路径导入
# compare_reac_area.py与compare_test.py同目录结构下(平级)
# 在与compare_reac_area.py平级的的文件中导入compare_reac_area.pyy,要求compare_reac_area.py不能包含相对路径(. or .. or ...)
from compare_rect_volumes import *

volum = Rect_volumes(area, 3)
print(volum.volumes())
volum.print_hello()

print('########################################################################')

# 相对路径导入
from compare.print_hello import print_hello  # compare_reac_area.py目录结构比print_hello.py高

print_hello("python")  # 不包含相对路径(. or .. or ...)导入的代码;正常导入

from compare.print_hello1 import print_hello1

print_hello1("c++")  # 包括相对路径(./../...)导入的代码;此时也能正常导入
