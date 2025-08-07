import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from fastapi.exceptions import RequestValidationError

from api.user_routes import create_user_routes
from api.protected_routes import create_protected_routes
from config.configuration import config
from utils.error_handling import register_exception_handlers
from auth.authentication_service import AuthenticationService
from auth.authorization_middleware import AuthorizationMiddleware
from core.hashing_service import HashingService
from core.storage_engine import StorageEngine
from core.user_model import User

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Initialize components
hashing_service = HashingService()
storage_engine = StorageEngine()
authentication_service = AuthenticationService(hashing_service, storage_engine)

# Configure routes
app.mount("/users", create_user_routes(authentication_service, storage_engine))
app.mount("/protected", create_protected_routes(authentication_service))

# Error handling
register_exception_handlers(app)

# Authorization Middleware (Example - needs proper request context handling)
security = HTTPBearer()

async def get_current_user_stub(request: Request) -> User | None:
    """Stub function to simulate getting user from request (e.g., JWT)."""
    auth_header = request.headers.get("Authorization")
    if auth_header == "Bearer valid_token":  # Simulate valid token
        return User(username="testuser", password_hash="hashed", email="test@example.com")
    return None

def unauthorized_callback_stub(request: Request) -> JSONResponse:
    """Stub function for handling unauthorized requests."""
    return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

authorization_middleware = AuthorizationMiddleware(
    authentication_service=authentication_service,
    get_user_from_request=get_current_user_stub,
    unauthorized_callback=unauthorized_callback_stub
)

# Wrap routes that need authorization (This is just an example, not true middleware)
# In a real application, you'd use Depends or similar for per-route protection

# Example endpoint that would be protected
@app.get("/admin")
async def admin_route(request: Request):
    """Example admin route."""
    wrapped_handler = authorization_middleware(lambda r: "Admin route accessed" if r else "Unauthorized")
    return await wrapped_handler(request) # type: ignore # (Pylance problem with async)


@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the API"}


if __name__ == "__main__":
    # This is only for local development. In a production environment, you would use an ASGI server
    # like uvicorn to run the application.
    # Example: uvicorn main.app:app --reload
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)