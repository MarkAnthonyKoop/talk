"""Command-line interface for core functionality."""

import click
import json
from datetime import datetime
from .core import OrderProcessor, Order, PaymentDetails, InventoryItem, OrderStatus
from .config import CoreConfig

@click.group()
def cli():
    """Core management CLI."""
    pass

@cli.command()
@click.argument('order_file')
def process_order(order_file):
    """Process order from JSON file."""
    try:
        with open(order_file) as f:
            order_data = json.load(f)
        
        # Create order object
        items = [
            (InventoryItem(**item), qty)
            for item, qty in order_data['items']
        ]
        
        order = Order(
            id=order_data['id'],
            items=items,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            payment_details=PaymentDetails(**order_data['payment'])
        )
        
        # Process order
        config = CoreConfig()
        processor = OrderProcessor(config)
        status = processor.process_order(order)
        
        click.echo(f"Order processed successfully: {status.value}")
        
    except Exception as e:
        click.echo(f"Error processing order: {str(e)}", err=True)

if __name__ == '__main__':
    cli()