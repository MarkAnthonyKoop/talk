import bcrypt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HashingService:
    """
    A service for hashing and verifying passwords using bcrypt.

    This class provides methods for securely hashing passwords before storing them
    and for verifying passwords against their stored hashes during authentication.
    """

    def __init__(self, rounds: int = 12):
        """
        Initializes the HashingService with a specified number of bcrypt rounds.

        Args:
            rounds (int): The number of rounds to use for bcrypt hashing.
                           Higher values increase security but also increase computation time.
                           Defaults to 12.
        """
        self.rounds = rounds
        logging.info(f"HashingService initialized with bcrypt rounds: {self.rounds}")

    def hash_password(self, password: str) -> str:
        """
        Hashes the given password using bcrypt.

        Args:
            password (str): The password to hash.

        Returns:
            str: The bcrypt hash of the password, as a string.

        Raises:
            TypeError: If the password is not a string.
            ValueError: If the password is empty or too short.
        """
        if not isinstance(password, str):
            logging.error("Password must be a string.")
            raise TypeError("Password must be a string.")

        if not password:
            logging.error("Password cannot be empty.")
            raise ValueError("Password cannot be empty.")

        if len(password) < 8:
            logging.warning("Password is shorter than 8 characters. Consider increasing password complexity requirements.")


        try:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(self.rounds))
            logging.debug("Password successfully hashed.")
            return hashed_password.decode('utf-8')  # Store as string for easier handling
        except Exception as e:
            logging.error(f"Error hashing password: {e}")
            raise

    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verifies the given password against the stored bcrypt hash.

        Args:
            password (str): The password to verify.
            password_hash (str): The stored bcrypt hash to compare against.

        Returns:
            bool: True if the password matches the hash, False otherwise.

        Raises:
            TypeError: If either password or password_hash is not a string.
            ValueError: If either password or password_hash is empty.
        """
        if not isinstance(password, str) or not isinstance(password_hash, str):
            logging.error("Password and password_hash must be strings.")
            raise TypeError("Password and password_hash must be strings.")

        if not password or not password_hash:
            logging.error("Password and password_hash cannot be empty.")
            raise ValueError("Password and password_hash cannot be empty.")

        try:
            is_valid = bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
            logging.debug(f"Password verification result: {is_valid}")
            return is_valid
        except ValueError as e:
            logging.error(f"Error verifying password: {e}")
            # bcrypt.checkpw raises ValueError if the hash is invalid
            return False
        except Exception as e:
            logging.error(f"Unexpected error during password verification: {e}")
            return False


if __name__ == '__main__':
    # Example Usage
    hashing_service = HashingService()
    password = "mysecretpassword"
    hashed_password = hashing_service.hash_password(password)
    print(f"Hashed password: {hashed_password}")

    is_valid = hashing_service.verify_password(password, hashed_password)
    print(f"Password is valid: {is_valid}")

    wrong_password = "wrongpassword"
    is_valid = hashing_service.verify_password(wrong_password, hashed_password)
    print(f"Wrong password is valid: {is_valid}")

    try:
        hashing_service.hash_password(123)
    except TypeError as e:
        print(f"Expected Error: {e}")

    try:
        hashing_service.verify_password("test", "invalid_hash")
    except Exception as e:
        print(f"Verification Error: {e}")