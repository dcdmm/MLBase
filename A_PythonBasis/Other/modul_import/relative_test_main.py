print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))
# __file__:当前*.py文件的路径
# __package__:When it is present, relative imports will be based on this attribute rather than the module __name__ attribute;当前文件__package__属性为None

from relative_test.demos_0.demo0 import hello_demo0
from relative_test.demos_0.demos_1.demo1 import hello_demo1
from relative_test.demo import hello_demo

# 项目relative_test中包含相对路径(./../...)导入的代码
# 只能在与项目relative_test目录结构平级或更高(即比项目relative_test中的代码目录结构更高)的python文件中执行relative_test中包含相对路径(./../...)导入的代码
hello_demo()
print('****************************************')
hello_demo0()
print('****************************************')
hello_demo1()
