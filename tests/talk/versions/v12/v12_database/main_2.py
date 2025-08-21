"""
A key-value database with a storage engine and caching layer.
"""

def __init__(self, storage_engine: StorageEngine, cache: Cache):
    """
    Initializes the KeyValueDatabase.

    Args:
        storage_engine: The storage engine to use.
        cache: The cache to use.
    """