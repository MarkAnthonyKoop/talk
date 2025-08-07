"""Monitoring and metrics collection for core components."""

import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass, field
from prometheus_client import Counter, Histogram, start_http_server

@dataclass
class Metrics:
    """Container for component metrics."""
    total_orders: Counter = field(default_factory=lambda: Counter('total_orders', 'Total orders processed'))
    order_processing_time: Histogram = field(default_factory=lambda: Histogram('order_processing_seconds', 'Time spent processing orders'))
    inventory_updates: Counter = field(default_factory=lambda: Counter('inventory_updates', 'Total inventory updates'))
    payment_processing_time: Histogram = field(default_factory=lambda: Histogram('payment_processing_seconds', 'Time spent processing payments'))
    errors: Counter = field(default_factory=lambda: Counter('processing_errors', 'Total processing errors', ['type']))

class MetricsCollector:
    """Centralized metrics collection and monitoring."""
    
    _instance: Optional['MetricsCollector'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.metrics = Metrics()
                # Start Prometheus metrics server
                start_http_server(8000)
            return cls._instance
    
    def record_order_processed(self, processing_time: float):
        """Record order processing metrics."""
        self.metrics.total_orders.inc()
        self.metrics.order_processing_time.observe(processing_time)
    
    def record_inventory_update(self):
        """Record inventory update metric."""
        self.metrics.inventory_updates.inc()
    
    def record_payment_processed(self, processing_time: float):
        """Record payment processing metrics."""
        self.metrics.payment_processing_time.observe(processing_time)
    
    def record_error(self, error_type: str):
        """Record processing error."""
        self.metrics.errors.labels(type=error_type).inc()