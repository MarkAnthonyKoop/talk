import unittest
from unittest.mock import patch
from typing import Dict

from core.user_model import User
from core.hashing_service import HashingService
from core.storage_engine import StorageEngine
from auth.authentication_service import AuthenticationService


class MockStorageEngine(StorageEngine):  # type: ignore
    def __init__(self):
        super().__init__(storage_file=":memory:")  # In-memory storage for testing
        self.users: Dict[str, User] = {}
        self.emails: Dict[str, User] = {}

    def save_user(self, user: User):
        self.users[user.username] = user
        self.emails[user.email] = user

    def get_user_by_username(self, username: str):
        return self.users.get(username)

    def get_user_by_email(self, email: str):
        for user in self.users.values():
            if user.email == email:
                return user
        return None

class UserAuthenticationTests(unittest.TestCase):
    """
    Unit and integration tests for user authentication functionality.
    """

    def setUp(self):
        """
        Set up test environment before each test.
        """
        self.hashing_service = HashingService()
        self.storage_engine = MockStorageEngine()
        self.auth_service = AuthenticationService(self.hashing_service, self.storage_engine)

    def test_register_user_success(self):
        """
        Test successful user registration.
        """
        new_user = self.auth_service.register_user("testuser", "password123", "test@example.com")
        self.assertIsNotNone(new_user)
        self.assertEqual(new_user.username, "testuser")
        self.assertNotEqual(new_user.password_hash, "password123")  # Ensure password is hashed
        self.assertEqual(self.storage_engine.get_user_by_username("testuser"), new_user)

    def test_register_user_duplicate_username(self):
        """
        Test registration failure due to duplicate username.
        """
        self.auth_service.register_user("testuser", "password123", "test@example.com")
        with self.assertRaises(ValueError):
            self.auth_service.register_user("testuser", "anotherpassword", "another@example.com")

    def test_register_user_duplicate_email(self):
        """
        Test registration failure due to duplicate email.
        """
        self.auth_service.register_user("testuser", "password123", "test@example.com")
        with self.assertRaises(ValueError):
            self.auth_service.register_user("anotheruser", "anotherpassword", "test@example.com")

    def test_login_user_success(self):
        """
        Test successful user login.
        """
        self.auth_service.register_user("testuser", "password123", "test@example.com")
        logged_in_user = self.auth_service.login_user("testuser", "password123")
        self.assertIsNotNone(logged_in_user)
        self.assertEqual(logged_in_user.username, "testuser")

    def test_login_user_incorrect_password(self):
        """
        Test login failure due to incorrect password.
        """
        self.auth_service.register_user("testuser", "password123", "test@example.com")
        logged_in_user = self.auth_service.login_user("testuser", "wrongpassword")
        self.assertIsNone(logged_in_user)

    def test_login_user_nonexistent_user(self):
        """
        Test login failure due to nonexistent user.
        """
        logged_in_user = self.auth_service.login_user("nonexistentuser", "password123")
        self.assertIsNone(logged_in_user)

    def test_logout_user(self):
        """
        Test user logout (currently a placeholder).
        """
        self.assertTrue(self.auth_service.logout_user("testuser"))

    def test_password_hashing(self):
        """
        Test password hashing functionality.
        """
        password = "testpassword"
        hashed_password = self.hashing_service.hash_password(password)
        self.assertNotEqual(password, hashed_password)
        self.assertTrue(self.hashing_service.verify_password(password, hashed_password))
        self.assertFalse(self.hashing_service.verify_password("wrongpassword", hashed_password))


if __name__ == '__main__':
    unittest.main()