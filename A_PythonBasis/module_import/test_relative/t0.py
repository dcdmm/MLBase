print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))

import sys
import os

for i in sys.path:
    print(i)

r"""
搜索路径sys.path默认包括(当前文件所在目录):???\MLBase\A_PythonBasis\module_import\test_relative
"""

# python直接运行报错(ModuleNotFoundError: No module named 'base')
# from base.print_hello import *
# from base.print_hello1 import *
# print_hello('java')
# print_hello1('python')

# 解决方案1:
# 临时将上一级目录(???\MLBase\A_PythonBasis\module_import\)添加到搜索路径sys.path中,然后python直接运行
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from base.print_hello import *
# from base.print_hello1 import *
# print_hello('java')
# print_hello1('python')

# 解决方案2:
# module_import目录下执行`python -m test_relative.t0`(临时添加当前工作目录???\MLBase\A_PythonBasis\module_import\到搜索路径sys.path中)
# from base.print_hello import *
# from base.print_hello1 import *
# print_hello('java')
# print_hello1('python')

# 解决方案3:
# A_PythonBasis目录下执行`python -m module_import.test_relative.t0`(临时添加当前工作目录???\MLBase\A_PythonBasis到搜索路径sys.path中)
# * __package__: module_import.test_relative
# * * 原始路径: from ..base.print_hello import * 
# * * 回退: module_import.test_relative + .. = module_import
# * * 真实路径: from module_import.base.print_hello import *
# from ..base.print_hello import *
# from ..base.print_hello1 import *
# print_hello('java')
# print_hello1('python')

# 解决方案4:
# *.pth文件中写入???\MLBase\A_PythonBasis\module_import\路径,然后python直接运行
# from base.print_hello import *
# from base.print_hello1 import *
# print_hello('java')
# print_hello1('python')

# 解决方案5:
# 上一级目录(???\MLBase\A_PythonBasis\module_import\)创建.py文件并写入`import test_relative.t0`,然后python直接运行该.py文件
# from base.print_hello import *
# from base.print_hello1 import *
# print_hello('java')
# print_hello1('python')


# 解决方案6:
# base_pack根目录下执行: pip install -e .
# * site-packages下只生成链接/元数据文件,源码不会被复制
# * 代码变更后不需要重新安装,直接生效
from base_pack.print_hi import *
print_hi('python')


# 解决方案7:
# uv workspace