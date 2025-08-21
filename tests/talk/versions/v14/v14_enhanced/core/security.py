"""Security utilities for core components."""

import hashlib
import hmac
import os
from typing import Optional
from cryptography.fernet import Fernet
from .exceptions import SecurityException

class Security:
    """Security manager for encryption and validation."""
    
    def __init__(self, encryption_key: Optional[str] = None, api_key: Optional[str] = None):
        self.encryption = Fernet(encryption_key.encode()) if encryption_key else None
        self.api_key = api_key
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not self.encryption:
            raise SecurityException("Encryption key not configured")
        return self.encryption.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt encrypted data."""
        if not self.encryption:
            raise SecurityException("Encryption key not configured")
        return self.encryption.decrypt(encrypted_data.encode()).decode()
    
    def validate_api_key(self, provided_key: str) -> bool:
        """Validate API key using constant time comparison."""
        if not self.api_key:
            raise SecurityException("API key not configured")
        return hmac.compare_digest(provided_key, self.api_key)
    
    @staticmethod
    def generate_encryption_key() -> str:
        """Generate new Fernet encryption key."""
        return Fernet.generate_key().decode()
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using strong algorithm."""
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return f"{salt.hex()}:{key.hex()}"