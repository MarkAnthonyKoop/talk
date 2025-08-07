"""
In-memory cache for storing key-value pairs.
Uses an OrderedDict for LRU eviction.
"""

def __init__(self, capacity: int = 1000):
    """
    Initializes the Cache.

    Args:
        capacity: The maximum number of items the cache can hold.
    """
    self.capacity = capacity
    self.cache: OrderedDict[str, Any] = OrderedDict()
    self._lock = threading.Lock()  # Use a lock for thread-safe cache access

def get(self, key: str) -> Optional[Any]:
    """
    Retrieves a value from the cache.

    Args:
        key: The key to retrieve.

    Returns:
        The value associated with the key, or None if the key is not in the cache.
    """
    with self._lock:
        try:
            if key in self.cache:
                self.cache.move_to_end(key)  # Move to end to mark as recently used
                logging.debug(f"Cache hit for key: {key}")
                return self.cache[key]
            else:
                logging.debug(f"Cache miss for key: {key}")
                return None
        except Exception as e:
            logging.error(f"Error getting key {key} from cache: {e}")
            return None

def set(self, key: str, value: Any):
    """
    Sets a value in the cache.

    Args:
        key: The key to set.
        value: The value to associate with the key.
    """
    with self._lock:
        try:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)  # Remove the least recently used item
            logging.debug(f"Cache set for key: {key}")
        except Exception as e:
            logging.error(f"Error setting key {key} in cache: {e}")

def delete(self, key: str):
    """
    Deletes a key from the cache.

    Args:
        key: The key to delete.
    """
    with self._lock:
        try:
            if key in self.cache:
                del self.cache[key]
                logging.debug(f"Cache delete for key: {key}")
        except KeyError:
            logging.debug(f"Key {key} not found in cache during delete operation.")
        except Exception as e:
            logging.error(f"Error deleting key {key} from cache: {e}")

def clear(self):
    """
    Clears the entire cache.
    """
    with self._lock:
        try:
            self.cache.clear()
            logging.debug("Cache cleared")
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")

