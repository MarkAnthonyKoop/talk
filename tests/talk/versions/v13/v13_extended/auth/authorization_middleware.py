import logging
from typing import Callable, Any

from auth.authentication_service import AuthenticationService
from core.user_model import User

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AuthorizationMiddleware:
    """
    Middleware to protect routes based on user authentication status.

    This middleware checks for a valid user session or token and authorizes access to protected routes.
    It integrates with the authentication service to verify user credentials.
    """

    def __init__(self, authentication_service: AuthenticationService, get_user_from_request: Callable[[Any], User | None], unauthorized_callback: Callable[[Any], Any]):
        """
        Initializes the AuthorizationMiddleware with an authentication service and a function to extract
        the user from the request.

        Args:
            authentication_service (AuthenticationService): The authentication service for verifying user credentials.
            get_user_from_request (Callable[[Any], User | None]): A function that takes a request object and returns
                                                                  the User object if authenticated, or None otherwise.
            unauthorized_callback (Callable[[Any], Any]): A function to call if the user is unauthorized.  Takes the request as input.
        """
        self.authentication_service = authentication_service
        self.get_user_from_request = get_user_from_request
        self.unauthorized_callback = unauthorized_callback
        logging.info("AuthorizationMiddleware initialized.")

    def __call__(self, next_handler: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """
        The middleware callable.  This wraps the next handler in the chain.

        Args:
            next_handler (Callable[[Any], Any]): The next handler in the chain.

        Returns:
            Callable[[Any], Any]: A wrapped handler that performs authorization checks.
        """
        async def middleware(request: Any) -> Any:
            """
            The actual middleware logic.
            """
            user = self.get_user_from_request(request)

            if user:
                logging.debug(f"User '{user.username}' is authenticated.")
                return await next_handler(request)  # type: ignore # (Pylance issue with async)
            else:
                logging.warning("Unauthorized access attempt.")
                return self.unauthorized_callback(request)

        return middleware


if __name__ == '__main__':
    # Example Usage (requires a mock AuthenticationService and request object)
    class MockAuthenticationService:
        def __init__(self):
            pass

        def login_user(self, username, password):
            if username == "testuser" and password == "password123":
                return User(username="testuser", password_hash="hashed", email="test@example.com")
            return None

    class MockRequest:
        def __init__(self, headers):
            self.headers = headers

    def get_user_from_request(request):
        auth_header = request.headers.get("Authorization")
        if auth_header == "Bearer valid_token":
            return User(username="testuser", password_hash="hashed", email="test@example.com")
        return None

    def unauthorized_callback(request):
        return "Unauthorized"

    async def next_handler(request):
        return "Authorized"

    auth_service = MockAuthenticationService()
    middleware = AuthorizationMiddleware(auth_service, get_user_from_request, unauthorized_callback)

    # Test with valid token
    request_valid = MockRequest({"Authorization": "Bearer valid_token"})
    wrapped_handler = middleware(next_handler)
    import asyncio
    result = asyncio.run(wrapped_handler(request_valid))
    print(f"Valid token test: {result}")

    # Test with invalid token
    request_invalid = MockRequest({"Authorization": "Bearer invalid_token"})
    wrapped_handler = middleware(next_handler)
    result = asyncio.run(wrapped_handler(request_invalid))
    print(f"Invalid token test: {result}")