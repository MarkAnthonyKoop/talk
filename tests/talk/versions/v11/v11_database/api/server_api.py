import logging
import socket
import threading
from typing import List, Dict, Any
import json

from api.client_api import DatabaseClient, ClientAPIError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 65432
DATABASE_PATH = "my_database.db"  # Adjust as needed

class ServerAPIError(Exception):
    """Base class for server API-related exceptions."""
    pass

class DatabaseServer:
    """Server API for handling client connections and processing SQL queries."""

    def __init__(self, host: str = SERVER_HOST, port: int = SERVER_PORT, db_path: str = DATABASE_PATH):
        """Initializes the DatabaseServer."""
        self.host = host
        self.port = port
        self.db_path = db_path
        self.server_socket = None
        self.running = False

    def start(self) -> None:
        """Starts the server and listens for client connections."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow address reuse
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen()
            self.running = True
            logging.info(f"Server listening on {self.host}:{self.port}")

            while self.running:
                conn, addr = self.server_socket.accept()
                logging.info(f"Accepted connection from {addr}")
                client_thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                client_thread.start()

        except OSError as e:
            logging.error(f"Failed to start server: {e}")
            self.stop()
            raise ServerAPIError(f"Failed to start server: {e}")

        finally:
            if self.server_socket:
                self.server_socket.close()

    def stop(self) -> None:
        """Stops the server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
            logging.info("Server stopped.")

    def handle_client(self, conn: socket.socket, addr: tuple) -> None:
        """Handles communication with a single client.

        Args:
            conn: The socket connection to the client.
            addr: The address of the client.
        """
        db_client = DatabaseClient(self.db_path)
        try:
            db_client.connect()
            while True:
                try:
                    data = conn.recv(4096)  # Receive data from the client
                    if not data:
                        break # Client disconnected

                    try:
                        request = json.loads(data.decode('utf-8'))
                        request_type = request.get("type")

                        if request_type == "execute_query":
                            sql_query = request.get("sql")
                            if not sql_query:
                                raise ValueError("SQL query is missing.")
                            result = db_client.execute_query(sql_query)
                            response = {"status": "success", "result": result}
                        elif request_type == "begin_transaction":
                            transaction_id = db_client.begin_transaction()
                            response = {"status": "success", "transaction_id": transaction_id}
                        elif request_type == "commit_transaction":
                            transaction_id = request.get("transaction_id")
                            if not transaction_id:
                                raise ValueError("Transaction ID is missing.")
                            db_client.commit_transaction(transaction_id)
                            response = {"status": "success"}
                        elif request_type == "rollback_transaction":
                            transaction_id = request.get("transaction_id")
                            if not transaction_id:
                                raise ValueError("Transaction ID is missing.")
                            db_client.rollback_transaction(transaction_id)
                            response = {"status": "success"}
                        else:
                            response = {"status": "error", "message": "Invalid request type."}

                    except (json.JSONDecodeError, ValueError) as e:
                        response = {"status": "error", "message": f"Invalid request format: {e}"}
                    except ClientAPIError as e:
                         response = {"status": "error", "message": str(e)}
                    except Exception as e:
                         response = {"status": "error", "message": f"Server error: {e}"}

                    conn.sendall(json.dumps(response).encode('utf-8'))

                except ConnectionResetError:
                    logging.info(f"Client {addr} disconnected abruptly.")
                    break
                except Exception as e:
                    logging.error(f"Error handling client {addr}: {e}")
                    break

        finally:
            db_client.close()
            conn.close()
            logging.info(f"Connection to {addr} closed.")