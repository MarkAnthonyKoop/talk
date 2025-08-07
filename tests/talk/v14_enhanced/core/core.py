"""
Core business logic implementation for order processing system.
Provides thread-safe, monitored, and secure order processing capabilities.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import redis
from ratelimit import limits

from .config import CoreConfig
from .models import Order, OrderStatus, InventoryItem, PaymentDetails
from .exceptions import (
    CoreException, 
    InvalidOrderException,
    InventoryException,
    PaymentException
)
from .monitoring import MetricsCollector
from .security import Security
from .validation import Validator
from .database import Database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InventoryManager:
    """Thread-safe inventory management with caching and monitoring."""
    
    def __init__(self, config: CoreConfig, db: Database):
        self.config = config
        self.db = db
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            password=config.redis_password,
            ssl=config.redis_ssl,
            decode_responses=True
        )
        self._lock = threading.Lock()
        self.metrics = MetricsCollector()
    
    def get_item(self, item_id: str) -> Optional[InventoryItem]:
        """Get inventory item with caching."""
        try:
            # Try cache first
            cached_data = self.redis_client.get(f"inventory:{item_id}")
            if cached_data:
                return InventoryItem.from_json(cached_data)
            
            # Fall back to database
            item_data = self.db.get_inventory_item(item_id)
            if item_data:
                item = InventoryItem.from_dict(item_data)
                # Update cache
                self.redis_client.set(
                    f"inventory:{item_id}",
                    item.to_json(),
                    ex=3600
                )
                return item
            return None
            
        except Exception as e:
            logger.error(f"Error fetching inventory item: {str(e)}")
            self.metrics.record_error("inventory_fetch")
            raise InventoryException(f"Failed to fetch item {item_id}")
    
    def update_quantity(self, item_id: str, quantity_change: int) -> None:
        """Update item quantity with thread safety."""
        try:
            with self._lock:
                item = self.get_item(item_id)
                if not item:
                    raise InventoryException(f"Item {item_id} not found")
                
                new_quantity = item.quantity + quantity_change
                if new_quantity < 0:
                    raise InventoryException(f"Insufficient inventory for item {item_id}")
                
                item.quantity = new_quantity
                item.updated_at = datetime.now()
                
                # Update database and cache atomically
                self.db.save_inventory_item(item_id, item.to_dict())
                self.redis_client.set(
                    f"inventory:{item_id}",
                    item.to_json(),
                    ex=3600
                )
                
                self.metrics.record_inventory_update()
                logger.info(f"Updated inventory for item {item_id}: {quantity_change}")
                
        except Exception as e:
            logger.error(f"Error updating inventory: {str(e)}")
            self.metrics.record_error("inventory_update")
            raise

class PaymentProcessor:
    """Secure payment processing with retry logic and rate limiting."""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.security = Security(config.encryption_key)
        self.metrics = MetricsCollector()
    
    @limits(calls=100, period=60)
    def process_payment(self, payment: PaymentDetails) -> str:
        """Process payment with rate limiting and retries."""
        start_time = time.time()
        
        try:
            Validator.validate_payment_details(payment)
            
            for attempt in range(self.config.max_retries):
                try:
                    # Encrypt sensitive data
                    if payment.card_data:
                        payment.encrypted_card_data = self.security.encrypt(
                            payment.card_data
                        )
                    
                    # Process payment (mock implementation)
                    transaction_id = self.security.generate_transaction_id()
                    payment.transaction_id = transaction_id
                    payment.payment_status = "completed"
                    
                    processing_time = time.time() - start_time
                    self.metrics.record_payment_processed(processing_time)
                    logger.info(f"Payment processed: {transaction_id}")
                    
                    return transaction_id
                    
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise PaymentException(f"Payment processing failed: {str(e)}")
                    time.sleep(self.config.retry_delay)
                    
        except Exception as e:
            logger.error(f"Payment processing error: {str(e)}")
            self.metrics.record_error("payment")
            raise

class OrderProcessor:
    """Main order processing coordinator with full monitoring and security."""
    
    def __init__(self, config: CoreConfig):
        self.config = config
        self.db = Database(config.db_path)
        self.inventory_manager = InventoryManager(config, self.db)
        self.payment_processor = PaymentProcessor(config)
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.metrics = MetricsCollector()
    
    def process_order(self, order: Order) -> OrderStatus:
        """Process order with validation, monitoring and error handling."""
        start_time = time.time()
        
        try:
            # Validate order
            Validator.validate_order(order)
            order.status = OrderStatus.PROCESSING
            logger.info(f"Processing order {order.id}")
            
            # Save initial order state
            self.db.save_order(order.id, order.to_dict())
            
            # Check and reserve inventory
            for item, quantity in order.items:
                self.inventory_manager.update_quantity(item.id, -quantity)
            
            # Process payment
            if order.payment_details:
                transaction_id = self.payment_processor.process_payment(
                    order.payment_details
                )
                logger.info(f"Payment processed: {transaction_id}")
            
            order.status = OrderStatus.COMPLETED
            order.updated_at = datetime.now()
            
            # Save final order state
            self.db.save_order(order.id, order.to_dict())
            
            processing_time = time.time() - start_time
            self.metrics.record_order_processed(processing_time)
            
            return order.status
            
        except Exception as e:
            order.status = OrderStatus.FAILED
            order.updated_at = datetime.now()
            self.db.save_order(order.id, order.to_dict())
            
            logger.error(f"Order processing failed: {str(e)}")
            self.metrics.record_error("order_processing")
            raise