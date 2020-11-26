from import_model_test0 import Rect_Area # 同目录结构下模块的相对导入
from import_modul_test1 import Rect_volumes

# 直接导入-->
# 1. 若源文件发生改变,运行test.py文件立马可以得到改变
# 2. 可以通过ctrl + 鼠标左击 进入导入库的源代码

area = Rect_Area(1, 2)
volum = Rect_volumes(area, 3)

print(volum.volumes())

