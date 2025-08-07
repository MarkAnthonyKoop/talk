"""Data models for core components."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional
from enum import Enum

class OrderStatus(Enum):
    """Enumeration of possible order statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class InventoryItem:
    """Represents an inventory item."""
    id: str
    name: str
    quantity: int
    price: float
    sku: Optional[str] = None
    category: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class PaymentDetails:
    """Payment processing details."""
    amount: float
    payment_method: str
    transaction_id: Optional[str] = None
    encrypted_card_data: Optional[str] = None
    payment_status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Order:
    """Represents an order in the system."""
    id: str
    items: List[Tuple[InventoryItem, int]]
    status: OrderStatus
    created_at: datetime
    payment_details: Optional[PaymentDetails] = None
    customer_id: Optional[str] = None
    notes: Optional[str] = None
    updated_at: datetime = field(default_factory=datetime.now)