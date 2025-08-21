import logging
import os
from typing import List, Dict, Any, Optional

from core.engine import SQLEngine  # Assuming SQLEngine is the main class in core/engine.py
from core.parser import SQLParser
from core.query_planner import QueryPlanner
from core.execution_engine import ExecutionEngine
from core.transaction_manager import TransactionManager
from core.lock_manager import LockManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime%)s - %(levelname)s - %(message)s')

class ClientAPIError(Exception):
    """Base class for client API-related exceptions."""
    pass

class DatabaseClient:
    """Client API for interacting with the database engine."""

    def __init__(self, db_path: str):
        """Initializes the DatabaseClient.

        Args:
            db_path: The path to the database file.
        """
        self.db_path = db_path
        self.engine: Optional[SQLEngine] = None
        self.parser: Optional[SQLParser] = None
        self.query_planner: Optional[QueryPlanner] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.transaction_manager: Optional[TransactionManager] = None
        self.lock_manager: Optional[LockManager] = None
        self.connected = False

    def connect(self) -> None:
        """Connects to the database."""
        try:
            self.lock_manager = LockManager()
            self.engine = SQLEngine(self.db_path, self.lock_manager)
            self.parser = SQLParser()
            self.query_planner = QueryPlanner(self.engine)
            self.execution_engine = ExecutionEngine(self.engine)
            self.transaction_manager = TransactionManager(self.engine)
            self.engine.load_database()  # Load existing database or create a new one
            self.connected = True
            logging.info(f"Connected to database at {self.db_path}")
        except Exception as e:
            logging.error(f"Failed to connect to database: {e}")
            raise ClientAPIError(f"Failed to connect to database: {e}")

    def close(self) -> None:
        """Closes the connection to the database."""
        if self.connected:
            try:
                self.engine.close()
                self.connected = False
                logging.info(f"Disconnected from database at {self.db_path}")
            except Exception as e:
                logging.error(f"Failed to close database connection: {e}")
                raise ClientAPIError(f"Failed to close database connection: {e}")

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Executes the given SQL query.

        Args:
            sql: The SQL query to execute.

        Returns:
            A list of dictionaries, where each dictionary represents a row in the result set.

        Raises:
            ClientAPIError: If an error occurs during query execution.
        """
        if not self.connected:
            raise ClientAPIError("Not connected to the database. Call connect() first.")

        try:
            parsed_query = self.parser.parse(sql)
            plan = self.query_planner.plan(parsed_query)
            result = self.execution_engine.execute(plan)
            return result
        except Exception as e:
            logging.error(f"Failed to execute query: {e}")
            raise ClientAPIError(f"Failed to execute query: {e}")

    def begin_transaction(self) -> int:
        """Begins a new transaction."""
        if not self.connected:
            raise ClientAPIError("Not connected to the database. Call connect() first.")
        try:
            transaction_id = self.transaction_manager.begin_transaction()
            return transaction_id
        except Exception as e:
            logging.error(f"Failed to begin transaction: {e}")
            raise ClientAPIError(f"Failed to begin transaction: {e}")

    def commit_transaction(self, transaction_id: int) -> None:
        """Commits the given transaction."""
        if not self.connected:
            raise ClientAPIError("Not connected to the database. Call connect() first.")
        try:
            self.transaction_manager.commit_transaction(transaction_id)
        except Exception as e:
            logging.error(f"Failed to commit transaction: {e}")
            raise ClientAPIError(f"Failed to commit transaction: {e}")

    def rollback_transaction(self, transaction_id: int) -> None:
        """Rollbacks the given transaction."""
        if not self.connected:
            raise ClientAPIError("Not connected to the database. Call connect() first.")
        try:
            self.transaction_manager.rollback_transaction(transaction_id)
        except Exception as e:
            logging.error(f"Failed to rollback transaction: {e}")
            raise ClientAPIError(f"Failed to rollback transaction: {e}")