import logging

'''
Does basic configuration for the logging system by creating a StreamHandler with a default Formatter and adding it to the root logger. 
The functions debug(), info(), warning(), error() and critical() will call basicConfig() automatically if no handlers are defined for the root logger.

This function does nothing if the root logger already has handlers configured, unless the keyword argument force is set to True.
'''
logging.basicConfig(
    # Specifies that a FileHandler be created, using the specified filename, rather than a StreamHandler.
    filename='basic.log',  # 日志保存位置(绝对 or 相对;默认输出到控制台)
    # If filename is specified, open the file in this mode. Defaults to 'a'.
    filemode='w',  # 日志文件的打开模式
    # Set the root logger level to the specified level.
    level=logging.NOTSET,  # 日志级别
    # se the specified format string for the handler.
    # Defaults to attributes levelname, name and message separated by colons.
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    # Use the specified date/time format, as accepted by time.strftime().
    datefmt='%Y-%m-%d  %H:%M:%S %a'
)

logging.debug(msg="msg1")
logging.info(msg="msg2")
logging.warning(msg="msg3")
logging.error(msg="msg4")
logging.critical(msg="msg5")

a = 1
b = 0
try:
    c = a / b
except ZeroDivisionError as reason:
    logging.error(reason)
    '''
    Logs a message with level ERROR on the root logger. The arguments are interpreted as for debug(). Exception info is added to the logging message. 
    This function should only be called from an exception handler.
    '''
    logging.exception(reason)  # 带报错具体信息(推荐)
