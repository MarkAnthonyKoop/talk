import logging
from typing import Any

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from auth.authentication_service import AuthenticationService
from auth.authorization_middleware import AuthorizationMiddleware
from core.user_model import User

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

security = HTTPBearer()


def create_protected_routes(authentication_service: AuthenticationService) -> FastAPI:  # type: ignore
    """
    Creates and configures protected API routes that require authentication.

    Args:
        authentication_service (AuthenticationService): The authentication service for user authentication.

    Returns:
        FastAPI: The FastAPI application with protected routes configured.
    """
    app = FastAPI()

    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
        """
        Retrieves the current user from the token (placeholder).

        In a real application, you would validate the JWT and retrieve the user from the database.
        """
        if credentials.credentials == "valid_token":
            return User(username="testuser", password_hash="hashed", email="test@example.com")  # Mock user
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/protected", response_model=str)
    async def protected_route(current_user: User = Depends(get_current_user)):
        """
        A protected route that requires authentication.
        """
        return f"Hello, {current_user.username}! This is a protected resource."

    @app.get("/admin", response_model=str)
    async def admin_route(current_user: User = Depends(get_current_user)):
        """
        An admin route that requires authentication and potentially specific roles.
        """
        # In a real application, you would check the user's roles/permissions here.
        if current_user.username == "admin":  # Example: Check if the user is an admin.
            return "Welcome, Admin! This is an admin-only resource."
        else:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient privileges.")

    return app


if __name__ == '__main__':
    # Example Usage (requires a mock AuthenticationService)
    from core.hashing_service import HashingService
    from core.storage_engine import StorageEngine

    class MockStorageEngine(StorageEngine):  # type: ignore
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
    authentication_service = AuthenticationService(hashing_service, storage_engine)

    app = create_protected_routes(authentication_service)

    # To run this, you would need to use a real ASGI server like uvicorn:
    # uvicorn api.protected_routes:app --reload
    # Then, you can send requests to the endpoints using a tool like curl or Postman.
    # Example:
    # curl -H "Authorization: Bearer valid_token" http://localhost:8000/protected
    # curl -H "Authorization: Bearer invalid_token" http://localhost:8000/protected