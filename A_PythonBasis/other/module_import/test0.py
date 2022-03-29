# from A_PythonBasis.other.module_import.import_model_test0 import Rect_Area # 绝对路径导入
from rect_area import Rect_Area  # 模块的相对导入(reac_area.py与test0.py位于同目录结构下)
# 在与rect_volumes.py平级的的文件中执行rect_volumns.py,要求rect_volumns.py不能包含相对路径(./../...)
from rect_volumes import *

# 1. 若源文件发生改变,运行test.py文件立马可以得到改变
# 2. 可以通过ctrl + 鼠标左击 进入导入库的源代码

area = Rect_Area(1, 2)

volum = Rect_volumes(area, 3)
print(volum.volumes())
volum.print_hello()

from dir_example.print_hello import print_hello  # 模块的相对导入(print_hello.py目录结构比test.py低)

print_hello("python")  # 不包含相对路径(./../...)导入的代码

from dir_example.print_hello1 import print_hello1

print_hello1("c++")  # 包括相对路径(./../...)导入的代码
