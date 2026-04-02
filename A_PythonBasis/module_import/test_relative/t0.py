import sys
import os

r"""
搜索路径sys.path默认包括(当前文件所在目录):???\MLBase\A_PythonBasis\module_import\test_relative
"""

# 解决方案1: 
# 临时将上一级目录(???\MLBase\A_PythonBasis\module_import\)添加到搜索路径sys.path中,然后python直接运行
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for i in sys.path:
    print(i)

# python直接运行报错(ModuleNotFoundError: No module named 'base')
from base.print_hello1 import *
from base.print_hello import *

print_hello("python")
print_hello1("rust")

# 解决方案2:
# module_import目录下执行`python -m test_relative.t0`(临时添加当前工作目录???\MLBase\A_PythonBasis\module_import\到搜索路径sys.path中)

# 解决方案3:
# *.pth文件中写入???\MLBase\A_PythonBasis\module_import\路径,然后python直接运行

# 解决方案4:
# 上一级目录(???\MLBase\A_PythonBasis\module_import\)创建.py文件并写入`import test_relative.t0`,然后python直接运行该.py文件

# 解决方案5:
# module_import目录下执行`python -m test_relative.t0`(临时添加当前工作目录???\MLBase\A_PythonBasis\module_import\到搜索路径sys.path中)

# 解决方案6:
