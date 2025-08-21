"""Database management for core components."""

import sqlite3
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import json

class Database:
    """SQLite database manager."""
    
    def __init__(self, db_path: str = "core.db"):
        self.db_path = db_path
        self.initialize_db()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context management."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def initialize_db(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS inventory (
                    id TEXT PRIMARY KEY,
                    data JSON NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    data JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS payments (
                    id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    data JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (order_id) REFERENCES orders (id)
                );
            """)
    
    def save_inventory_item(self, item_id: str, data: Dict[str, Any]):
        """Save inventory item to database."""
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO inventory (id, data) VALUES (?, ?)",
                (item_id, json.dumps(data))
            )
            conn.commit()
    
    def get_inventory_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get inventory item from database."""
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT data FROM inventory WHERE id = ?",
                (item_id,)
            ).fetchone()
            return json.loads(result['data']) if result else None