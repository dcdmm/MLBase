from a import *

# 向上继承自a中的root logger
logger = logging.getLogger(__name__)


def func_b():
    logging.info('func_b!')


if __name__ == '__main__':
    func_b()
