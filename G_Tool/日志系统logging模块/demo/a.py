import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(lineno)d - %(message)s ',
                    filename='demo.log')  # root logger


def func_a():
    logging.info('func_a!')


if __name__ == '__main__':
    func_a()
