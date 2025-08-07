import logging
import threading
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TransactionError(Exception):
    """Base class for transaction-related exceptions."""
    pass

class Transaction:
    """Represents a transaction."""

    def __init__(self, transaction_id: int):
        self.transaction_id = transaction_id
        self.operations: List[Dict[str, Any]] = []  # List of operations performed in the transaction
        self.status = "ACTIVE"  # Can be ACTIVE, COMMITTED, ROLLEDBACK

    def add_operation(self, operation: Dict[str, Any]) -> None:
        """Adds an operation to the transaction's log."""
        self.operations.append(operation)

    def commit(self) -> None:
        """Marks the transaction as committed."""
        if self.status != "ACTIVE":
            raise TransactionError("Cannot commit a transaction that is not active.")
        self.status = "COMMITTED"

    def rollback(self) -> None:
        """Marks the transaction as rolled back."""
        if self.status != "ACTIVE":
            raise TransactionError("Cannot rollback a transaction that is not active.")
        self.status = "ROLLEDBACK"

    def __repr__(self):
        return f"Transaction(id={self.transaction_id}, status='{self.status}')"

class TransactionManager:
    """Manages transactions, ensuring ACID properties."""

    def __init__(self, db_engine):
        self.db_engine = db_engine
        self.transactions: Dict[int, Transaction] = {}
        self.next_transaction_id = 1
        self.lock = threading.Lock()  # For thread safety

    def begin_transaction(self) -> int:
        """Starts a new transaction.

        Returns:
            The ID of the new transaction.
        """
        with self.lock:
            transaction_id = self.next_transaction_id
            self.next_transaction_id += 1
            transaction = Transaction(transaction_id)
            self.transactions[transaction_id] = transaction
            logging.info(f"Transaction {transaction_id} started.")
            return transaction_id

    def commit_transaction(self, transaction_id: int) -> None:
        """Commits the specified transaction.

        Args:
            transaction_id: The ID of the transaction to commit.
        """
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction:
                raise TransactionError(f"Transaction {transaction_id} not found.")

            if transaction.status != "ACTIVE":
                raise TransactionError(f"Transaction {transaction_id} is not active.")

            try:
                # Apply the changes to the database
                for operation in transaction.operations:
                    op_type = operation["type"]
                    if op_type == "INSERT":
                        self.db_engine.insert_row(operation["table"], operation["values"])
                    elif op_type == "UPDATE":
                        self.db_engine.update_row(operation["table"], operation["row_id"], operation["new_values"])
                    elif op_type == "DELETE":
                        self.db_engine.delete_row(operation["table"], operation["row_id"])
                    else:
                        raise TransactionError(f"Unsupported operation type: {op_type}")

                transaction.commit()
                logging.info(f"Transaction {transaction_id} committed.")
                del self.transactions[transaction_id]

            except Exception as e:
                logging.error(f"Error during commit of transaction {transaction_id}: {e}")
                self.rollback_transaction(transaction_id) #Rollback if commit fails
                raise

    def rollback_transaction(self, transaction_id: int) -> None:
        """Rolls back the specified transaction.

        Args:
            transaction_id: The ID of the transaction to roll back.
        """
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction:
                raise TransactionError(f"Transaction {transaction_id} not found.")

            if transaction.status != "ACTIVE":
                raise TransactionError(f"Transaction {transaction_id} is not active.")

            try:
                # Undo the operations performed by the transaction
                # (This is a simplified example; a real implementation would need more sophisticated logging
                # and undo mechanisms)
                # In this simple implementation, we don't have full undo capabilities,
                # so we just log the rollback.  A more robust system would need to maintain
                # undo logs.
                logging.warning(f"Rollback for transaction {transaction_id} is incomplete.  Full undo not implemented.")

                transaction.rollback()
                logging.info(f"Transaction {transaction_id} rolled back.")
                del self.transactions[transaction_id]

            except Exception as e:
                logging.error(f"Error during rollback of transaction {transaction_id}: {e}")
                raise

    def record_operation(self, transaction_id: int, operation: Dict[str, Any]) -> None:
        """Records an operation within the specified transaction.

        Args:
            transaction_id: The ID of the transaction.
            operation: A dictionary describing the operation.
        """
        with self.lock:
            transaction = self.transactions.get(transaction_id)
            if not transaction:
                raise TransactionError(f"Transaction {transaction_id} not found.")

            transaction.add_operation(operation)
            logging.debug(f"Operation recorded for transaction {transaction_id}: {operation}")

    def get_transaction_status(self, transaction_id: int) -> str:
         """Retrieves the status of a transaction.

         Args:
             transaction_id: The ID of the transaction.

         Returns:
             The status of the transaction (ACTIVE, COMMITTED, ROLLEDBACK, or None if not found).
         """
         with self.lock:
             transaction = self.transactions.get(transaction_id)
             if transaction:
                 return transaction.status
             else:
                 return None