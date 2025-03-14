from logging_info import logger  # 初始化根日志记录器
from a import a_func
from b import b_func
from c import c_func


def main():
    # 使用根日志记录器(日志信息写入到root.log中)
    logger.debug(msg="main debug")
    logger.info(msg="main info")
    logger.warning(msg="main warning")
    logger.error(msg="main error")
    logger.critical(msg="main critical")
    return a_func() + b_func() + c_func() + "main"


if __name__ == '__main__':
    print(main())
