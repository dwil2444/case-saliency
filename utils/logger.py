import logging
import sys


class CustomLogger:
    def __init__(self, name: str):
        """
        Args:
            name: Name of the logger

        Returns: None: Instantiates Logger
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Check if the logger already has handlers to avoid adding duplicates
        if not self.logger.hasHandlers():
            logging.getLogger().handlers.clear()
            handler = logging.StreamHandler(sys.stdout)
            # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)
