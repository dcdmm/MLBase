import logging

# Return a logger with the specified name, creating it if necessary.
# If no name is specified, return the root logger.
logger = logging.getLogger()

# Set the logging level of this logger.  level must be an int or a str.
logger.setLevel(logging.INFO)
print(logger)

# Initialize the handler.
# If stream is not specified, sys.stderr is used.
s_handler = logging.StreamHandler()
# Set the logging level of this handler.  level must be an int or a str.
s_handler.setLevel(logging.DEBUG)  # hander日记级别未设置或低于logger日志级别:使用logger日志级别
print(s_handler)

# Open the specified file and use it as the stream for logging.
f_handler = logging.FileHandler(
    # 日志保存位置(绝对 or 相对)
    filename='senior.log',
    mode='w',
    encoding='utf-8')
f_handler.setLevel(logging.WARNING)  # hander日记级别高于logger日志级别:使用hander日志级别
print(f_handler)

formatter0 = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
formatter1 = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)d - %(message)s')

# Set the formatter for this handler.
s_handler.setFormatter(formatter0)
f_handler.setFormatter(formatter1)

# Add the specified handler to this logger.
logger.addHandler(s_handler)
logger.addHandler(f_handler)

logger.debug(msg="debug")
logger.info(msg="info")
logger.warning(msg="warning")
logger.error(msg="error")
logger.critical(msg="critical")
