# from A_PythonBasis.Other.modul_import.import_model_test0 import Rect_Area # 绝对路径导入
from rect_area import Rect_Area  # 模块的相对导入(reac_area.py与test.py位于同目录结构下)
from rect_volumes import *

# 1. 若源文件发生改变,运行test.py文件立马可以得到改变
# 2. 可以通过ctrl + 鼠标左击 进入导入库的源代码

area = Rect_Area(1, 2)

volum = Rect_volumes(area, 3)
print(volum.volumes())
volum.print_hello()

from dir_test.print_hello import print_hello  # 模块的相对导入(dir_test与test.py位于同目录结构下)

print_hello("python")
