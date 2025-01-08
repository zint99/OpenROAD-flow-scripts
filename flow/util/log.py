import logging


class ModuleFilter(logging.Filter):
    def filter(self, record):
        return record.name == "root"  # only logging for root


class Log:
    def __init__(self, file_name, logger_name=None, level=logging.INFO):
        self.file_name = file_name
        self.level = level
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(self.level)
        self.init_log()

    def init_log(self):
        # FileHandler
        handler = logging.FileHandler(self.file_name, mode="w+")
        handler.setLevel(self.level)
        formatter = logging.Formatter(
            "[%(asctime)s - %(filename)s - %(levelname)s - %(lineno)d]: %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.level)
        stream_handler.setFormatter(formatter)
        # add filter
        stream_handler.addFilter(ModuleFilter())
        self.logger.addHandler(stream_handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)
