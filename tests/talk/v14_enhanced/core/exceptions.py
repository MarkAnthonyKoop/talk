"""Custom exceptions for the core package."""

class CoreException(Exception):
    """Base exception for all core-related errors."""
    pass

class InvalidOrderException(CoreException):
    """Raised when order validation fails."""
    pass

class InventoryException(CoreException):
    """Raised when inventory operations fail."""
    pass

class PaymentException(CoreException):
    """Raised when payment processing fails."""
    pass

class SecurityException(CoreException):
    """Raised when security validation fails."""
    pass

class ConfigurationException(CoreException):
    """Raised when configuration validation fails."""
    pass