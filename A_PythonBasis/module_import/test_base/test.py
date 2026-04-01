# test.py与print_hello.py位于不同目录结构下
# test.py的目录结构为test_base
# print_hello.py的目录结构为base
# ===>解决1:从base处开始导入(即目录结构与base平级)
from base.print_hello import print_hello
from base.print_hello1 import print_hello1

print_hello("php")
print_hello1('C++')

# ===>解决2:从比base目录结构更高的位置开始导入(即目录结构比base更高)
from module_import.base.print_hello1 import print_hello1

print('#' * 100)