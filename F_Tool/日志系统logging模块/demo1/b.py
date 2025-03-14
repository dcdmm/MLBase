import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.FileHandler(filename='b.log', mode='w', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def b_func():
    # 子日志系统默认会将日志消息传递给根日志系统(日志信息写入到root.log和b.log中)
    logger.debug(msg="b debug")
    logger.info(msg="b info")
    logger.warning(msg="b warning")
    logger.error(msg="b error")
    logger.critical(msg="b critical")
    return "b"
