import os

print(os.getcwd())  # 获取当前工作路径

path = os.path.abspath("../dir_example")  # 要临时添加到path路径的第三方模块的路径
print(path)

import sys

# 使用>import<语句导入一个第三方的模块时
# Python解析器默认会在当前目录和path路径中搜索所有已安装的内置模块和第三方模块
# 若导入的第三方模块不在当前目录下,则需要第三方模块的路径临时加入到path路径中,如下所示:
sys.path.append(path)

for i in sys.path:
    print(i)

from print_hello import print_hello  # test1.py与print_hello.py位于不同目录结构下

print_hello("php")

# 存在相对路径导入的代码不能直接执行
from ..dir_example.print_hello import print_hello
