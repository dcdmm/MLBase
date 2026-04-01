print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))
from ..core.config import *  # ..表示父级目录


def hello_demo0():
    print("The value of config.count is {0}".format(count))
    print_info()
