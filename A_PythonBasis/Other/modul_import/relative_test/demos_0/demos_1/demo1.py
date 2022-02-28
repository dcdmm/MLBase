print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))
from ... import config  # ...表示父级目录的父级目录

print("The value of config.count is {0}".format(config.count))


def hello_demo1():
    print(config.count)
