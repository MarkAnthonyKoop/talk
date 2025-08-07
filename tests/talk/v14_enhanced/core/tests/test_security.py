"""Tests for security utilities."""

import pytest
from ..security import Security
from ..exceptions import SecurityException

def test_encryption():
    """Test data encryption and decryption."""
    key = Security.generate_encryption_key()
    security = Security(encryption_key=key)
    
    data = "sensitive data"
    encrypted = security.encrypt(data)
    decrypted = security.decrypt(encrypted)
    
    assert data == decrypted
    assert encrypted != data

def test_api_key_validation():
    """Test API key validation."""
    security = Security(api_key="test_key")
    
    assert security.validate_api_key("test_key")
    assert not security.validate_api_key("wrong_key")

def test_password_hashing():
    """Test password hashing."""
    password = "secure_password"
    hashed = Security.hash_password(password)
    
    assert ":" in hashed
    assert len(hashed) > 32
    assert hashed != password