import logging.config
import logging

# Read the logging configuration from a ConfigParser-format file.
logging.config.fileConfig('loggging.conf')

logger = logging.getLogger('log01')

logger.debug(msg="log1 debug")
logger.info(msg="log1 info")
logger.warning(msg="log1 warning")
logger.error(msg="log1 error")
logger.critical(msg="log1 critical")