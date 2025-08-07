"""
Storage engine for persisting key-value data to disk.
Uses JSON format for storage.
"""

def __init__(self, data_dir: str = "data", filename: str = "database.json"):
    """
    Initializes the StorageEngine.

    Args:
        data_dir: Directory to store the database file.
        filename: Name of the database file.
    """
    self.data_dir = data_dir
    self.filename = filename
    self.filepath = os.path.join(self.data_dir, self.filename)
    self._lock = threading.Lock()  # Use a lock for thread-safe file access
    self._ensure_data_dir()

def _ensure_data_dir(self):
    """
    Creates the data directory if it doesn't exist.
    """
    if not os.path.exists(self.data_dir):
        try:
            os.makedirs(self.data_dir)
            logging.info(f"Created data directory: {self.data_dir}")
        except OSError as e:
            logging.error(f"Failed to create data directory: {e}")
            raise

def load_data(self) -> Dict[str, Any]:
    """
    Loads data from the JSON file.

    Returns:
        A dictionary containing the loaded data.  Returns an empty dict if loading fails or file doesn't exist.
    """
    with self._lock:
        try:
            if not os.path.exists(self.filepath):
                return {}

            with open(self.filepath, "r") as f:
                data = json.load(f)
                logging.debug(f"Loaded data from {self.filepath}")
                return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Error loading data from {self.filepath}: {e}. Returning empty dict.")
            return {}
        except Exception as e:
            logging.error(f"Unexpected error loading data: {e}")
            return {}  # Return empty dict on error

def save_data(self, data: Dict[str, Any]):
    """
    Saves data to the JSON file.

    Args:
        data: The dictionary containing the data to save.
    """
    with self._lock:
        try:
            with open(self.filepath, "w") as f:
                json.dump(data, f, indent=4)
                logging.debug(f"Saved data to {self.filepath}")
        except Exception as e:
            logging.error(f"Error saving data to {self.filepath}: {e}")

