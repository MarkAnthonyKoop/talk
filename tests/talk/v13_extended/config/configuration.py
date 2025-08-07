import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Configuration:
    """
    Loads configuration settings from environment variables.

    This class provides access to configuration settings such as API keys,
    database connection details, and other environment-specific parameters.
    """

    def __init__(self):
        """
        Initializes the Configuration by loading settings from environment variables.
        """
        self.database_url: Optional[str] = os.environ.get("DATABASE_URL")
        self.api_key: Optional[str] = os.environ.get("API_KEY")
        self.debug_mode: bool = os.environ.get("DEBUG_MODE", "False").lower() == "true"
        self.log_level: str = os.environ.get("LOG_LEVEL", "INFO").upper()

        logging.basicConfig(level=self.log_level)

        logging.info("Configuration loaded from environment variables.")
        logging.debug(f"Database URL: {self.database_url}")
        logging.debug(f"API Key: {'[REDACTED]' if self.api_key else None}")  # Redact sensitive info
        logging.debug(f"Debug Mode: {self.debug_mode}")
        logging.debug(f"Log Level: {self.log_level}")

    def get_database_url(self) -> str:
        """
        Returns the database URL.

        Returns:
            str: The database URL.

        Raises:
            ValueError: If the DATABASE_URL environment variable is not set.
        """
        if not self.database_url:
            logging.error("DATABASE_URL environment variable not set.")
            raise ValueError("DATABASE_URL environment variable not set.")
        return self.database_url

    def get_api_key(self) -> str:
        """
        Returns the API key.

        Returns:
            str: The API key.

        Raises:
            ValueError: If the API_KEY environment variable is not set.
        """
        if not self.api_key:
            logging.error("API_KEY environment variable not set.")
            raise ValueError("API_KEY environment variable not set.")
        return self.api_key


# Global configuration instance
config = Configuration()


if __name__ == '__main__':
    # Example Usage
    # Set environment variables (e.g., in your .env file or shell)
    # export DATABASE_URL="postgresql://user:password@host:port/database"
    # export API_KEY="your_secret_api_key"
    # export DEBUG_MODE="True"
    # export LOG_LEVEL="DEBUG"

    try:
        print(f"Database URL: {config.get_database_url()}")
        print(f"API Key: [REDACTED]")  # Always redact API keys in output
        print(f"Debug Mode: {config.debug_mode}")
        print(f"Log Level: {config.log_level}")
    except ValueError as e:
        print(f"Error: {e}")