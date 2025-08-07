import logging
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CustomException(Exception):
    """Base class for custom exceptions."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


class AuthenticationError(CustomException):
    """Exception for authentication failures."""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class AuthorizationError(CustomException):
    """Exception for authorization failures."""
    def __init__(self, detail: str = "Unauthorized"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class ResourceNotFoundError(CustomException):
    """Exception for resource not found."""
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class ValidationError(CustomException):
    """Exception for validation errors."""
    def __init__(self, detail: str = "Validation error"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def register_exception_handlers(app: FastAPI):
    """Registers custom exception handlers for the FastAPI application."""

    @app.exception_handler(CustomException)
    async def custom_exception_handler(request: Request, exc: CustomException):
        """Handles custom exceptions."""
        logging.error(f"Custom exception: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handles request validation errors."""
        logging.error(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handles generic exceptions (fallback)."""
        logging.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )


if __name__ == '__main__':
    # Example Usage
    from fastapi import Depends, HTTPException

    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/test_auth_error")
    async def test_auth_error():
        raise AuthenticationError()

    @app.get("/test_validation_error")
    async def test_validation_error():
        raise ValidationError(detail="Invalid input")

    @app.get("/test_generic_error")
    async def test_generic_error():
        raise ValueError("Something went wrong")

    # To run this, use uvicorn:
    # uvicorn utils.error_handling:app --reload
    # Then, access the endpoints in your browser or with curl.
    # Example:
    # curl http://localhost:8000/test_auth_error