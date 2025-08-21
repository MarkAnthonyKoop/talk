import logging
import os
from typing import Optional

class Logger:
    """A logging utility for recording events and errors."""

    def __init__(self, name: str, level: str = "INFO", log_file: Optional[str] = None):
        """Initializes the Logger.

        Args:
            name: The name of the logger.
            level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            log_file: The path to the log file (optional).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.upper())

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler (optional)
        if log_file:
            try:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)
            except Exception as e:
                print(f"Error creating file handler: {e}")

    def debug(self, message: str) -> None:
        """Logs a debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Logs an info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Logs a warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Logs an error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Logs a critical message."""
        self.logger.critical(message)

# Example usage (singleton pattern could be used for a global logger)
# logger = Logger(__name__, level="DEBUG", log_file="my_app.log")
# logger.info("Application started.")
# logger.debug("Debugging information.")
# logger.warning("A potential issue.")
# logger.error("An error occurred.")
# logger.critical("The application is crashing.")