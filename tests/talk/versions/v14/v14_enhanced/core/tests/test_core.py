"""Unit tests for core functionality."""

import pytest
import redis
from datetime import datetime
from unittest.mock import Mock, patch
from ..core import (
    OrderProcessor,
    InventoryManager,
    PaymentProcessor,
    Order,
    OrderStatus,
    PaymentDetails,
    InventoryItem,
    InvalidOrderException
)
from ..config import CoreConfig

@pytest.fixture
def config():
    return CoreConfig()

@pytest.fixture
def inventory_manager(config):
    return InventoryManager(config)

@pytest.fixture
def payment_processor(config):
    return PaymentProcessor(config)

@pytest.fixture
def order_processor(config):
    return OrderProcessor(config)

@pytest.fixture
def sample_item():
    return InventoryItem(
        id="item1",
        name="Test Item",
        quantity=10,
        price=9.99
    )

@pytest.fixture
def sample_order(sample_item):
    return Order(
        id="order1",
        items=[(sample_item, 2)],
        status=OrderStatus.PENDING,
        created_at=datetime.now(),
        payment_details=PaymentDetails(
            amount=19.98,
            payment_method="credit_card"
        )
    )

def test_order_validation(sample_order):
    sample_order.validate()  # Should not raise
    
    # Test invalid order
    with pytest.raises(InvalidOrderException):
        invalid_order = Order(
            id="",
            items=[],
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
        invalid_order.validate()

def test_inventory_manager(inventory_manager, sample_item):
    with patch.object(redis.Redis, 'get', return_value=None):
        with patch.object(redis.Redis, 'set', return_value=True):
            inventory_manager.update_quantity(sample_item.id, -1)

def test_payment_processor(payment_processor):
    payment_details = PaymentDetails(amount=99.99, payment_method="credit_card")
    transaction_id = payment_processor.process_payment(payment_details)
    assert transaction_id is not None

def test_order_processor(order_processor, sample_order):
    with patch.object(InventoryManager, 'update_quantity'):
        with patch.object(PaymentProcessor, 'process_payment'):
            status = order_processor.process_order(sample_order)
            assert status == OrderStatus.COMPLETED
