"""Tests for validation utilities."""

import pytest
from datetime import datetime
from ..validation import Validator
from ..models import Order, PaymentDetails, InventoryItem, OrderStatus
from ..exceptions import InvalidOrderException

@pytest.fixture
def valid_item():
    return InventoryItem(
        id="test1",
        name="Test Item",
        quantity=10,
        price=9.99
    )

@pytest.fixture
def valid_payment():
    return PaymentDetails(
        amount=99.99,
        payment_method="credit_card"
    )

def test_order_validation(valid_item, valid_payment):
    """Test order validation."""
    order = Order(
        id="order1",
        items=[(valid_item, 2)],
        status=OrderStatus.PENDING,
        created_at=datetime.now(),
        payment_details=valid_payment
    )
    
    Validator.validate_order(order)  # Should not raise

def test_invalid_order():
    """Test invalid order validation."""
    with pytest.raises(InvalidOrderException):
        order = Order(
            id="",
            items=[],
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
        Validator.validate_order(order)