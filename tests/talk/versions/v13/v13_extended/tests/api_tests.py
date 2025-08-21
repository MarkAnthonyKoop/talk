import unittest
from fastapi.testclient import TestClient

from api.user_routes import create_user_routes
from api.protected_routes import create_protected_routes
from auth.authentication_service import AuthenticationService
from core.hashing_service import HashingService
from core.storage_engine import StorageEngine
from core.user_model import User


class MockStorageEngine(StorageEngine):  # type: ignore
    def __init__(self):
        super().__init__(storage_file=":memory:")  # In-memory storage for testing
        self.users = {}
        self.emails = {}

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

class APITests(unittest.TestCase):
    """
    Tests for API endpoints, including protected routes.
    """

    def setUp(self):
        """
        Set up test environment before each test.
        """
        self.hashing_service = HashingService()
        self.storage_engine = MockStorageEngine()
        self.authentication_service = AuthenticationService(self.hashing_service, self.storage_engine)

        self.user_app = create_user_routes(self.authentication_service, self.storage_engine)
        self.protected_app = create_protected_routes(self.authentication_service)

        self.user_client = TestClient(self.user_app)
        self.protected_client = TestClient(self.protected_app)

    def test_register_endpoint(self):
        """
        Test the /register endpoint.
        """
        response = self.user_client.post("/register", json={"username": "testuser", "password": "password123", "email": "test@example.com"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["username"], "testuser")

    def test_login_endpoint(self):
        """
        Test the /login endpoint.
        """
        # First register a user
        self.user_client.post("/register", json={"username": "testuser", "password": "password123", "email": "test@example.com"})

        response = self.user_client.post("/login", json={"username": "testuser", "password": "password123"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json())

    def test_protected_endpoint_success(self):
        """
        Test the /protected endpoint with a valid token.
        """
        response = self.protected_client.get("/protected", headers={"Authorization": "Bearer valid_token"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("protected resource", response.text)

    def test_protected_endpoint_unauthorized(self):
        """
        Test the /protected endpoint with an invalid token.
        """
        response = self.protected_client.get("/protected", headers={"Authorization": "Bearer invalid_token"})
        self.assertEqual(response.status_code, 401)

    def test_admin_endpoint_success(self):
        """
        Test the /admin endpoint with a valid admin user.
        """
        # Create an admin user
        admin_user = User(username="admin", password_hash="hashed", email="admin@example.com")
        self.storage_engine.save_user(admin_user)

        response = self.protected_client.get("/admin", headers={"Authorization": "Bearer valid_token"})
        # Assuming valid_token corresponds to user "admin"
        self.assertEqual(response.status_code, 200)
        self.assertIn("admin-only resource", response.text)

    def test_admin_endpoint_forbidden(self):
        """
        Test the /admin endpoint with a non-admin user.
        """
        # Create a non-admin user
        non_admin_user = User(username="testuser", password_hash="hashed", email="test@example.com")
        self.storage_engine.save_user(non_admin_user)

        response = self.protected_client.get("/admin", headers={"Authorization": "Bearer valid_token"})
        # Assuming valid_token corresponds to user "testuser"
        self.assertEqual(response.status_code, 403)


if __name__ == '__main__':
    unittest.main()