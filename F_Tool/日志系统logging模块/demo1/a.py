import logging

logger = logging.getLogger(__name__)  # 当前模块的日志记录器


def a_func():
    # 继承根日志记录器的配置(没有显式配置)(日志信息写入到root.log中)
    logger.debug(msg="a debug")
    logger.info(msg="a info")
    logger.warning(msg="a warning")
    logger.error(msg="a error")
    logger.critical(msg="a critical")
    return "a"
