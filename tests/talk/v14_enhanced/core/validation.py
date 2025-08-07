"""Validation utilities for core components."""

from typing import Dict, Any
from .models import Order, PaymentDetails, InventoryItem
from .exceptions import InvalidOrderException

class Validator:
    """Validation utilities for core components."""
    
    @staticmethod
    def validate_order(order: Order) -> None:
        """Validate order data."""
        if not order.id:
            raise InvalidOrderException("Order ID required")
        if not order.items:
            raise InvalidOrderException("Order must contain items")
        for item, quantity in order.items:
            if quantity <= 0:
                raise InvalidOrderException(f"Invalid quantity for item {item.id}")
            Validator.validate_inventory_item(item)
        if order.payment_details:
            Validator.validate_payment_details(order.payment_details)
    
    @staticmethod
    def validate_inventory_item(item: InventoryItem) -> None:
        """Validate inventory item data."""
        if not item.id or not isinstance(item.id, str):
            raise ValueError("Invalid item ID")
        if item.quantity < 0:
            raise ValueError("Quantity cannot be negative")
        if item.price < 0:
            raise ValueError("Price cannot be negative")
        if not item.name:
            raise ValueError("Item name is required")
    
    @staticmethod
    def validate_payment_details(payment: PaymentDetails) -> None:
        """Validate payment details."""
        if payment.amount <= 0:
            raise ValueError("Payment amount must be positive")
        if not payment.payment_method:
            raise ValueError("Payment method required")
        if payment.payment_method not in ["credit_card", "debit_card", "bank_transfer"]:
            raise ValueError("Invalid payment method")