"""
Core package for enterprise-grade order processing and inventory management.

This package provides production-ready components for processing orders,
managing inventory, and handling payments with full thread safety, monitoring,
and security features.

Example:
    >>> from core import OrderProcessor, CoreConfig
    >>> config = CoreConfig.from_yaml('config.yml')
    >>> processor = OrderProcessor(config)
    >>> status = processor.process_order(order)
"""

from .core import OrderProcessor, InventoryManager, PaymentProcessor
from .models import Order, PaymentDetails, InventoryItem, OrderStatus
from .exceptions import (
    CoreException, 
    InvalidOrderException,
    InventoryException, 
    PaymentException,
    SecurityException,
    ConfigurationException
)
from .config import CoreConfig
from .monitoring import MetricsCollector
from .security import Security
from .validation import Validator
from .database import Database

__version__ = "1.1.0"
__all__ = [
    "OrderProcessor",
    "InventoryManager",
    "PaymentProcessor",
    "Order",
    "PaymentDetails", 
    "InventoryItem",
    "OrderStatus",
    "CoreException",
    "InvalidOrderException",
    "InventoryException",
    "PaymentException",
    "SecurityException",
    "ConfigurationException",
    "CoreConfig",
    "MetricsCollector",
    "Security",
    "Validator",
    "Database"
]