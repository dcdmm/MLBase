import logging

logger = logging.getLogger()  # 根日志记录器
logger.setLevel(logging.INFO)
handler = logging.FileHandler(filename='root.log', mode='w', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
