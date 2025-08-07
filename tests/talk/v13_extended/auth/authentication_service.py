import logging
from typing import Optional

from core.user_model import User
from core.hashing_service import HashingService
from core.storage_engine import StorageEngine  # Assuming a storage engine interface/class exists

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AuthenticationService:
    """
    Provides authentication logic for user registration, login, and logout.

    This service interacts with the user model, hashing service, and storage engine
    to manage user accounts and authentication.
    """

    def __init__(self, hashing_service: HashingService, storage_engine: StorageEngine):
        """
        Initializes the AuthenticationService with a hashing service and storage engine.

        Args:
            hashing_service (HashingService): The hashing service for password management.
            storage_engine (StorageEngine): The storage engine for user data persistence.
        """
        self.hashing_service = hashing_service
        self.storage_engine = storage_engine
        logging.info("AuthenticationService initialized.")

    def register_user(self, username: str, password: str, email: str) -> Optional[User]:
        """
        Registers a new user in the system.

        Args:
            username (str): The desired username for the new user.
            password (str): The password for the new user.
            email (str): The email address for the new user.

        Returns:
            Optional[User]: The newly created User object if registration is successful,
                            None otherwise.

        Raises:
            ValueError: If the username or email already exists.
            Exception: If an unexpected error occurs during registration.
        """
        try:
            if self.storage_engine.get_user_by_username(username):
                logging.warning(f"Username '{username}' already exists.")
                raise ValueError("Username already exists.")

            if self.storage_engine.get_user_by_email(email):
                logging.warning(f"Email '{email}' already exists.")
                raise ValueError("Email already exists.")


            hashed_password = self.hashing_service.hash_password(password)
            new_user = User(username=username, password_hash=hashed_password, email=email)
            self.storage_engine.save_user(new_user)  # Assuming save_user adds the user.
            logging.info(f"User '{username}' registered successfully.")
            return new_user
        except ValueError as e:
            logging.error(f"Registration failed: {e}")
            raise
        except Exception as e:
            logging.exception("Unexpected error during registration.")
            raise

    def login_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticates a user and logs them in.

        Args:
            username (str): The username of the user attempting to log in.
            password (str): The password of the user attempting to log in.

        Returns:
            Optional[User]: The User object if login is successful, None otherwise.
        """
        try:
            user = self.storage_engine.get_user_by_username(username)
            if user and self.hashing_service.verify_password(password, user.password_hash):
                logging.info(f"User '{username}' logged in successfully.")
                return user
            else:
                logging.warning(f"Invalid credentials for user '{username}'.")
                return None
        except Exception as e:
            logging.exception(f"Error during login for user '{username}': {e}")
            return None

    def logout_user(self, username: str) -> bool:
        """
        Logs out a user.  Currently, this is a placeholder and doesn't perform any actions.

        Args:
            username (str): The username of the user to log out.

        Returns:
            bool: Always returns True, indicating successful logout (for now).
        """
        logging.info(f"User '{username}' logged out.")
        return True  # In a real system, this might invalidate a session or token.


if __name__ == '__main__':
    # Example Usage (requires a mock StorageEngine)
    class MockStorageEngine(StorageEngine):  # type: ignore # (Fixes pylance error)
        def __init__(self):
            self.users = {}

        def save_user(self, user: User):
            self.users[user.username] = user

        def get_user_by_username(self, username: str):
            return self.users.get(username)

        def get_user_by_email(self, email: str):
            for user in self.users.values():
                if user.email == email:
                    return user
            return None


    hashing_service = HashingService()
    storage_engine = MockStorageEngine()
    auth_service = AuthenticationService(hashing_service, storage_engine)

    try:
        # Register a user
        new_user = auth_service.register_user("testuser", "password123", "test@example.com")
        if new_user:
            print(f"Registered user: {new_user}")

        # Login the user
        logged_in_user = auth_service.login_user("testuser", "password123")
        if logged_in_user:
            print(f"Logged in user: {logged_in_user}")

        # Attempt to login with incorrect password
        incorrect_login = auth_service.login_user("testuser", "wrongpassword")
        if not incorrect_login:
            print("Login failed as expected.")

        # Logout the user
        auth_service.logout_user("testuser")

    except ValueError as e:
        print(f"Error: {e}")