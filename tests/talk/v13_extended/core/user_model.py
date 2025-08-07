import uuid
import logging
from typing import Optional

from pydantic import BaseModel, Field, validator, EmailStr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class User(BaseModel):
    """
    Represents a user in the system.

    Attributes:
        user_id (UUID): Unique identifier for the user.  Defaults to a randomly generated UUID.
        username (str): The user's username. Must be between 3 and 50 characters.
        password_hash (str): Hashed password for the user.  Should not be stored in plaintext.
        email (EmailStr): The user's email address. Validated to be a valid email format.
        is_active (bool): Whether the user account is active. Defaults to True.
        created_at (datetime): Timestamp of when the user account was created. Auto-generated.
        updated_at (datetime): Timestamp of when the user account was last updated. Auto-generated.
    """
    user_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for the user.")
    username: str = Field(..., min_length=3, max_length=50, description="The user's username.")
    password_hash: str = Field(..., description="Hashed password for the user.")
    email: EmailStr = Field(..., description="The user's email address.")
    is_active: bool = Field(default=True, description="Whether the user account is active.")
    created_at: Optional[float] = Field(default_factory=lambda: time.time(), description="Timestamp of account creation")
    updated_at: Optional[float] = Field(default_factory=lambda: time.time(), description="Timestamp of last update")


    @validator("username")
    def validate_username(cls, username: str) -> str:
        """
        Validates the username to ensure it meets the required criteria.

        Args:
            username (str): The username to validate.

        Returns:
            str: The validated username.

        Raises:
            ValueError: If the username is invalid.
        """
        if not (3 <= len(username) <= 50):
            logging.error(f"Invalid username length: {len(username)}")
            raise ValueError("Username must be between 3 and 50 characters.")
        return username

    @validator("password_hash")
    def validate_password_hash(cls, password_hash: str) -> str:
        """
        Validates that the password hash is not empty.  Additional validation (e.g., length, complexity)
        should be performed during password hashing, not here.

        Args:
            password_hash (str): The password hash to validate.

        Returns:
            str: The validated password hash.

        Raises:
            ValueError: If the password hash is empty.
        """
        if not password_hash:
            logging.error("Password hash cannot be empty.")
            raise ValueError("Password hash cannot be empty.")
        return password_hash

    def __repr__(self):
        return f"User(user_id='{self.user_id}', username='{self.username}', email='{self.email}')"

    def update_timestamp(self):
        """Updates the updated_at field to the current timestamp."""
        self.updated_at = time.time()



if __name__ == '__main__':
    import time
    # Example Usage
    try:
        user = User(username="testuser", password_hash="hashed_password", email="test@example.com")
        print(user)
        print(f"User ID: {user.user_id}")
        print(f"Created at: {user.created_at}")
        time.sleep(1)
        user.update_timestamp()
        print(f"Updated at: {user.updated_at}")
        print(user.model_dump_json(indent=2))


    except ValueError as e:
        logging.error(f"Error creating user: {e}")

    try:
        invalid_user = User(username="te", password_hash="hashed_password", email="invalid-email")
    except ValueError as e:
        logging.error(f"Expected error creating user: {e}")