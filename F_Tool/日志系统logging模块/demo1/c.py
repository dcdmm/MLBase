import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
handler = logging.FileHandler(filename='c.log', mode='w', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 禁止子日志系统将日志消息传递给根日志系统(日志信息仅写入到root.log和b.log中)
logger.propagate = False


def c_func():
    logger.debug(msg="c debug")
    logger.info(msg="c info")
    logger.warning(msg="c warning")
    logger.error(msg="c error")
    logger.critical(msg="c critical")
    return "c"
