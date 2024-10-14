import logging.config
import logging

# Read the logging configuration from a ConfigParser-format file.
logging.config.fileConfig('loggging.conf')

logger = logging.getLogger()

logger.debug(msg="root debug")
logger.info(msg="root info")
logger.warning(msg="root warning")
logger.error(msg="root error")
logger.critical(msg="root critical")