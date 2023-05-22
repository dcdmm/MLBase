import os
import sys

path = os.path.abspath("..")
print("path:", path)
sys.path.append(path)
print(os.path.abspath("../.."))
sys.path.append(os.path.abspath(".." + os.sep + ".."))

# test.py与print_hello.py位于不同目录结构下
# test.py的目录结构为compare_test
# print_hello.py的目录结构为compare
# ===>解决1:从compare处开始导入(即目录结构与compare平级)
from base.print_hello import print_hello
from base.print_hello1 import print_hello1
print_hello("php")
print_hello1('C++')

# ===>解决2:从比compare目录结构更高的位置开始导入(即目录结构比compare更高)
from module_import.base.print_hello1 import print_hello1

print('#' * 100)

# 存在相对路径导入的代码不能直接执行
from ..base.print_hello import print_hello
